from __future__ import annotations

import os
import gc

import torch
import torch_npu  # type: ignore
import torch.distributed as dist
from torch import nn
import torch.nn.functional as F

import nanovllm.ops as ascend_ops
from nanovllm.engine.dsa_offload import (
    OFFLOAD_FUSE,
    OFFLOAD_NONE,
    format_lidu_miss_count_report,
    parse_lidu_miss_count_layers,
)
from nanovllm.engine.full_decode_graph import (
    MLAGraphTask,
    is_full_decode_graph_capturing,
    record_mla_graph_task,
)
from nanovllm.models.dsa_indexer_project import (
    dsa_indexer_project,
    dsa_indexer_project_query_only,
)
from nanovllm.models.glm_moe_dsa_config import GlmMoeDsaConfig
from nanovllm.models.dsa_offload_ops import (
    LIDU_MTP_UNION_CAPACITY,
    LIDU_TOPK,
    initialize_lidu_row,
    fused_li_manage,
    fused_li_manage_mtp,
    scatter_copy,
    sparse_tail_attention,
    sparse_tail_attention_mtp,
    fused_copy_sfa,
    fused_copy_sfa_mtp,
)
from nanovllm.layers.activation import SiluAndMul
from nanovllm.layers.layernorm import RMSNorm
from nanovllm.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from nanovllm.utils.context import get_context


ACL_FORMAT_FRACTAL_NZ = 29
_NPU_MLA_ATTENTION_MASK_CACHE: dict[tuple[str, int], torch.Tensor] = {}
_NPU_MLA_V2_WORKSPACE_CACHE: dict[tuple, torch.Tensor] = {}
_LIDU_MISS_COUNT_SCRATCH_KEY = "lidu_miss_counts_by_layer"
_LiduUpdateResult = tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]
_MtpLiduUpdateResult = tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]


def _synchronize_device(device: torch.device) -> None:
    if device.type == "npu":
        torch.npu.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize(device)


def _hccl_comm_name(group: dist.ProcessGroup, rank: int) -> str:
    backend = group._get_backend(torch.device("npu"))
    try:
        rank = dist.get_global_rank(group, rank)
    except Exception:
        pass
    return backend.get_hccl_comm_name(rank)


def _get_npu_mla_attention_mask(device: torch.device, mask_size: int) -> torch.Tensor:
    key = (str(device), int(mask_size))
    mask = _NPU_MLA_ATTENTION_MASK_CACHE.get(key)
    if mask is None or mask.device != device:
        mask = torch.triu(
            torch.ones(mask_size, mask_size, dtype=torch.int8, device=device),
            diagonal=1,
        ).contiguous()
        _NPU_MLA_ATTENTION_MASK_CACHE[key] = mask
    return mask


def _round_up(value: int, align: int) -> int:
    return ((value + align - 1) // align) * align


def _trans_rope_weight(weight: torch.Tensor, rope_dim: int) -> torch.Tensor:
    if rope_dim == 0:
        return weight.contiguous()
    nope_part = weight[..., :-rope_dim, :]
    rope_part = weight[..., -rope_dim:, :]
    rope_part = torch.cat((rope_part[..., ::2, :], rope_part[..., 1::2, :]), dim=-2)
    return torch.cat((nope_part, rope_part), dim=-2).contiguous()


def _transdata_nz(
    nd_mat: torch.Tensor,
    block_size: tuple[int, int] = (16, 16),
) -> torch.Tensor:
    rows = _round_up(int(nd_mat.shape[0]), block_size[0])
    cols = _round_up(int(nd_mat.shape[1]), block_size[1])
    nd_mat = F.pad(nd_mat, (0, rows - nd_mat.shape[0], 0, cols - nd_mat.shape[1]))
    nz_mat = nd_mat.reshape(
        rows // block_size[0],
        block_size[0],
        cols // block_size[1],
        block_size[1],
    ).permute(2, 0, 1, 3)
    return nz_mat.reshape(nz_mat.shape[0], nz_mat.shape[1] * nz_mat.shape[2], nz_mat.shape[3])


def _to_mlapo_bf16_nz_weight(weight: torch.Tensor) -> torch.Tensor:
    nz_weight = _transdata_nz(weight, block_size=(16, 16)).unsqueeze(0).contiguous()
    return torch_npu.npu_format_cast(nz_weight, ACL_FORMAT_FRACTAL_NZ)


def _rope_interleaved_to_neox(x: torch.Tensor) -> torch.Tensor:
    return torch.cat((x[..., ::2], x[..., 1::2]), dim=-1).contiguous()


def _rotate_half_interleaved(x: torch.Tensor) -> torch.Tensor:
    x = x.view(*x.shape[:-1], -1, 2)
    x1 = x[..., 0]
    x2 = x[..., 1]
    x = torch.stack((-x2, x1), dim=-1)
    return x.flatten(-2)


class GlmRotaryEmbedding(nn.Module):
    def __init__(self, rotary_dim: int, max_position_embeddings: int, rope_parameters: dict) -> None:
        super().__init__()
        self.rotary_dim = rotary_dim
        base = float(rope_parameters.get("rope_theta", 10000.0))
        rope_type = rope_parameters.get("rope_type", "default")
        if rope_type != "default":
            raise ValueError(
                f"GLM-5.1 supports default RoPE only, got {rope_type!r}."
            )
        inv_freq = 1.0 / (
            base
            ** (
                torch.arange(0, rotary_dim, 2, dtype=torch.float32)
                / rotary_dim
            )
        )
        positions = torch.arange(max_position_embeddings, dtype=torch.float32)
        freqs = torch.einsum("i,j->ij", positions, inv_freq)
        self.register_buffer("cos_cache", freqs.cos(), persistent=False)
        self.register_buffer("sin_cache", freqs.sin(), persistent=False)

    def forward(self, positions: torch.Tensor, query: torch.Tensor, key: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        query_dtype = query.dtype
        key_dtype = key.dtype
        positions = positions.to(torch.long)
        cos = self.cos_cache.index_select(0, positions)
        sin = self.sin_cache.index_select(0, positions)
        cos = cos.repeat_interleave(2, dim=-1).unsqueeze(1)
        sin = sin.repeat_interleave(2, dim=-1).unsqueeze(1)
        query = (
            query * cos
            + _rotate_half_interleaved(query.float()).to(query.dtype) * sin
        )
        key = (
            key * cos
            + _rotate_half_interleaved(key.float()).to(key.dtype) * sin
        )
        return query.to(query_dtype), key.to(key_dtype)


class GlmMLP(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int, hidden_act: str, *, disable_tp: bool = False, reduce_results: bool = True) -> None:
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(hidden_size, [intermediate_size, intermediate_size], bias=False, disable_tp=disable_tp)
        self.down_proj = RowParallelLinear(intermediate_size, hidden_size, bias=False, disable_tp=disable_tp, reduce_results=reduce_results)
        if hidden_act != "silu":
            raise ValueError("GLM-5.1 dense/shared MLP requires silu.")
        self.act_fn = SiluAndMul()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.gate_up_proj(hidden_states)
        hidden_states = self.act_fn(hidden_states)
        hidden_states = self.down_proj(hidden_states)
        return hidden_states


class GlmDsaIndexer(nn.Module):
    def __init__(self, config: GlmMoeDsaConfig) -> None:
        super().__init__()
        self.topk_tokens = int(config.index_topk)
        self.n_head = int(config.index_n_heads)
        self.head_dim = int(config.index_head_dim)
        self.rope_dim = int(config.qk_rope_head_dim)
        self.hidden_size = int(config.hidden_size)
        self.q_lora_rank = int(config.q_lora_rank)

        self.wq_b = ReplicatedLinear(self.q_lora_rank, self.n_head * self.head_dim, bias=False)
        self.wk = ReplicatedLinear(self.hidden_size, self.head_dim, bias=False)
        self.k_norm = nn.LayerNorm(self.head_dim, eps=1e-6)
        self.weights_proj = ReplicatedLinear(self.hidden_size, self.n_head, bias=False).to(torch.float32)
        self._output_buffer_key = None
        self._output_buffers = None
        self._weights_proj_bf16_key = None
        self._weights_proj_bf16 = None
        self._wq_b_bmm_t_key = None
        self._wq_b_bmm_t = None

    def post_load_prepare(self) -> None:
        """Materialize decode-only projection weights before graph capture.

        The transformed BMM weight is large and its creation is asynchronous on
        NPU.  Building it lazily in the first decode step lets the custom BMM
        consume a just-created tensor.  Startup latency is irrelevant here, so
        prepare both decode-only caches once and synchronize them with the rest
        of model post-load preparation.
        """

        weight = self.wq_b.weight
        self._query_only_weights_proj_weight(weight.dtype, weight.device)
        self._query_only_wq_b_bmm_t(weight.dtype, weight.device)

    # Output tensors are owned by this layer and reused only in decode. Prefill may have
    # thousands of tokens, so caching those temporary outputs would pin huge per-layer tensors.
    def _get_output_buffers(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        context = get_context()
        if context.is_prefill:
            q_index = torch.empty((hidden_states.shape[0], self.n_head, self.head_dim), dtype=hidden_states.dtype, device=hidden_states.device)
            index_k = torch.empty((hidden_states.shape[0], self.head_dim), dtype=hidden_states.dtype, device=hidden_states.device)
            index_weights = torch.empty((hidden_states.shape[0], self.n_head), dtype=hidden_states.dtype, device=hidden_states.device)
            return q_index, index_k, index_weights
        key = (int(hidden_states.shape[0]), self.n_head, self.head_dim, hidden_states.dtype, hidden_states.device)
        if self._output_buffer_key == key and self._output_buffers is not None:
            return self._output_buffers
        q_index = torch.empty((hidden_states.shape[0], self.n_head, self.head_dim), dtype=hidden_states.dtype, device=hidden_states.device)
        index_k = torch.empty((hidden_states.shape[0], self.head_dim), dtype=hidden_states.dtype, device=hidden_states.device)
        index_weights = torch.empty((hidden_states.shape[0], self.n_head), dtype=hidden_states.dtype, device=hidden_states.device)
        self._output_buffer_key = key
        self._output_buffers = (q_index, index_k, index_weights)
        return self._output_buffers

    # Decode query-only uses a cached low-precision copy of weights_proj. Full
    # indexer and prefill keep the FP32 weight path to stay aligned with the original code.
    def _query_only_weights_proj_weight(self, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        weight = self.weights_proj.weight
        if dtype not in (torch.float16, torch.bfloat16):
            return weight
        key = (dtype, device)
        if self._weights_proj_bf16_key == key and self._weights_proj_bf16 is not None:
            return self._weights_proj_bf16
        self._weights_proj_bf16 = weight.detach().to(device=device, dtype=dtype).contiguous()
        self._weights_proj_bf16_key = key
        return self._weights_proj_bf16

    # query-only q projection can use the same head-major BMM-transpose path as
    # MLAPO. The transformed weight is large, so keep one cached copy per layer.
    def _query_only_wq_b_bmm_t(self, dtype: torch.dtype, device: torch.device) -> torch.Tensor | None:
        if device.type != "npu" or dtype not in (torch.float16, torch.bfloat16):
            return None
        if ascend_ops is None or not hasattr(ascend_ops, "batch_matmul_transpose"):
            return None
        key = (dtype, device)
        if self._wq_b_bmm_t_key == key and self._wq_b_bmm_t is not None:
            return self._wq_b_bmm_t
        weight = self.wq_b.weight
        if weight.dtype != dtype or weight.device != device:
            return None
        self._wq_b_bmm_t = weight.view(self.n_head, self.head_dim, self.q_lora_rank).transpose(1, 2).contiguous()
        self._wq_b_bmm_t_key = key
        return self._wq_b_bmm_t

    # Cache per-forward cos/sin tensors in context.scratch. All layers share the
    # same positions, so this removes repeated tiny H2D/index_select overhead.
    def _rope_cos_sin(self, positions: torch.Tensor, rotary_emb: GlmRotaryEmbedding, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
        context = get_context()
        cache_key = (
            "indexer_rope_cos_sin",
            str(positions.device),
            dtype,
            self.rope_dim,
        )
        cached = context.scratch.get(cache_key)
        if cached is not None:
            return cached

        positions = positions.to(torch.long)
        cos = rotary_emb.cos_cache.index_select(0, positions)
        sin = rotary_emb.sin_cache.index_select(0, positions)
        cos = cos.repeat_interleave(2, dim=-1)
        sin = sin.repeat_interleave(2, dim=-1)
        cos = cos.to(dtype).contiguous()
        sin = sin.to(dtype).contiguous()
        cos = cos.view(cos.shape[0], 1, 1, self.rope_dim)
        sin = sin.view(sin.shape[0], 1, 1, self.rope_dim)
        context.scratch[cache_key] = (cos, sin)
        return cos, sin

    def forward(
        self,
        hidden_states: torch.Tensor,
        q_c: torch.Tensor,
        positions: torch.Tensor,
        rotary_emb: GlmRotaryEmbedding,
        query_only: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]:
        cos, sin = self._rope_cos_sin(positions, rotary_emb, hidden_states.dtype)
        q_index, index_k, index_weights = self._get_output_buffers(hidden_states)
        if query_only:
            # Decode DSA only scores prefill candidates. The decode token key
            # is already in the MLA tail budget, so skip index_k projection.
            weights_proj_weight = self._query_only_weights_proj_weight(hidden_states.dtype, hidden_states.device)
            # The first decode is eager and uses F.linear. Stable decode uses
            # the custom BMM path, including FULL_DECODE_ONLY capture/replay.
            use_query_bmm = not get_context().has_first_decode
            wq_b_bmm_t = (
                None
                if not use_query_bmm
                else self._query_only_wq_b_bmm_t(q_c.dtype, q_c.device)
            )
            dsa_indexer_project_query_only(
                hidden_states,
                q_c,
                cos,
                sin,
                self.wq_b.weight,
                weights_proj_weight,
                q_index,
                index_weights,
                n_head=self.n_head,
                head_dim=self.head_dim,
                rope_dim=self.rope_dim,
                score_scale=1.0,  # LIDU consumes raw weights_proj(x).
                wq_b_bmm_t=wq_b_bmm_t,
                enable_q_bmm=wq_b_bmm_t is not None,
            )
            return q_index, None, index_weights

        # Prefill computes q/k/weights once and writes the cache-facing buffers.
        dsa_indexer_project(
            hidden_states,
            q_c,
            cos,
            sin,
            self.wq_b.weight,
            self.wk.weight,
            self.k_norm.weight,
            self.k_norm.bias,
            self.weights_proj.weight,
            q_index,
            index_k,
            index_weights,
            n_head=self.n_head,
            head_dim=self.head_dim,
            rope_dim=self.rope_dim,
            score_scale=1.0,  # Keep LIDU inputs aligned with GLM DSA.
        )
        return q_index, index_k, index_weights


class GlmMLAAttention(nn.Module):

    def __init__(self, config: GlmMoeDsaConfig, layer_idx: int) -> None:
        super().__init__()
        tp_size = dist.get_world_size()
        if config.num_attention_heads % tp_size != 0:
            raise ValueError("num_attention_heads must be divisible by TP size.")
        self.layer_idx = int(layer_idx)
        self.offload_mode = getattr(
            config, "nanovllm_offload_mode", OFFLOAD_NONE
        )
        self.num_hidden_layers = int(config.num_hidden_layers)
        self.is_mtp_layer = self.layer_idx >= self.num_hidden_layers
        # The 78 target layers are offloaded. The recursively reused MTP layer
        # deliberately keeps a separate dense HBM KV cache and has no DSA
        # indexer weights in the checkpoint.
        self.uses_offload = (
            self.offload_mode != OFFLOAD_NONE and not self.is_mtp_layer
        )
        miss_count_layers = parse_lidu_miss_count_layers(
            os.environ.get("NANOVLLM_LIDU_MISS_COUNT_ON_LAYERS"),
            self.num_hidden_layers,
        )
        tp_rank = dist.get_rank()
        self._lidu_miss_count_layers = miss_count_layers
        self._lidu_miss_count_collect_all = (
            self.uses_offload
            and tp_rank == 0
            and bool(miss_count_layers)
        )
        self._lidu_miss_count_decode_step = 0
        if (
            self.uses_offload
            and tp_rank == 0
            and self.layer_idx == 0
            and miss_count_layers
        ):
            print(
                "LIDU_MISS_COUNT enabled eager-only layers="
                f"{sorted(miss_count_layers)}",
                flush=True,
            )

        self.hidden_size = int(config.hidden_size)
        self.total_num_heads = int(config.num_attention_heads)
        self.num_local_heads = self.total_num_heads // tp_size
        self.q_lora_rank = int(config.q_lora_rank)
        self.kv_lora_rank = int(config.kv_lora_rank)
        self.qk_nope_head_dim = int(config.qk_nope_head_dim)
        self.qk_rope_head_dim = int(config.qk_rope_head_dim)
        self.qk_head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim
        self.v_head_dim = int(config.v_head_dim)
        self.scale = self.qk_head_dim ** -0.5

        self.q_a_proj = ReplicatedLinear(self.hidden_size, self.q_lora_rank, bias=False)
        self.q_a_layernorm = RMSNorm(self.q_lora_rank, eps=float(config.rms_norm_eps))
        self.q_b_proj = ColumnParallelLinear(self.q_lora_rank, self.total_num_heads * self.qk_head_dim, bias=False)
        self.kv_a_proj_with_mqa = ReplicatedLinear(self.hidden_size, self.kv_lora_rank + self.qk_rope_head_dim, bias=False)
        self.kv_a_layernorm = RMSNorm(self.kv_lora_rank, eps=float(config.rms_norm_eps))
        self.kv_b_proj = ColumnParallelLinear(self.kv_lora_rank, self.total_num_heads * (self.qk_nope_head_dim + self.v_head_dim), bias=False)
        self.o_proj = RowParallelLinear(self.total_num_heads * self.v_head_dim, self.hidden_size, bias=False)
        self._tp_hcomm_info = None
        self.rotary_emb = GlmRotaryEmbedding(
            self.qk_rope_head_dim,
            max_position_embeddings=int(config.max_position_embeddings),
            rope_parameters=config.rope_parameters,
        )
        self.indexer_rotary_emb = None
        self.indexer = None
        if self.uses_offload:
            self.indexer_rotary_emb = GlmRotaryEmbedding(
                self.qk_rope_head_dim,
                max_position_embeddings=int(config.max_position_embeddings),
                rope_parameters=config.rope_parameters,
            )
            self.indexer = GlmDsaIndexer(config)
        # Caller-owned custom-op outputs stay alive at fixed graph addresses.
        self._use_sparse_tail_attention = self.uses_offload
        self._use_fused_copy_sfa = self.offload_mode == OFFLOAD_FUSE
        self._fused_li_manage_outputs: dict[
            int, tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        ] = {}
        self._fused_li_manage_mtp_outputs: dict[
            int,
            tuple[
                torch.Tensor,
                torch.Tensor,
                torch.Tensor,
                torch.Tensor,
                torch.Tensor,
            ],
        ] = {}
        self._sparse_tail_outputs: dict[tuple[int, ...], torch.Tensor] = {}

        self.ckv_cache = torch.tensor([])
        self.kpe_cache = torch.tensor([])
        self.index_cache = torch.tensor([])
        self.dram_ckv_cache = torch.tensor([])
        self.dram_kpe_cache = torch.tensor([])
        self.lidu_cache_slots = torch.tensor([])
        self.register_parameter("wd_qkv", None)
        self.w_uk_t = None
        self.w_uv = None
        self.mlapo_wd_qkv = None
        self.mlapo_wu_q = None
        self.mlapo_beta1 = None
        self._decode_mlapo_ql_nope = None
        self._decode_mlapo_q_pe = None
        self._decode_mlapo_inner_out = None
        self._decode_mla_v2_out = None
        self._decode_mla_v2_lse = None
        self._spec_decode_mla_v2_out = None
        self._spec_decode_mla_v2_lse = None

    def can_fuse_o_proj_add_rms_norm(self) -> bool:
        return not self.o_proj.disable_tp and self.o_proj.tp_size > 1 and self.o_proj.reduce_results and self.o_proj.bias is None and hasattr(ascend_ops, "matmul_allreduce_add_rmsnorm")

    def _tp_comm_name(self) -> str:
        if self._tp_hcomm_info is None:
            self._tp_hcomm_info = _hccl_comm_name(dist.group.WORLD, self.o_proj.tp_rank)
        return self._tp_hcomm_info

    # Decode fast path: fuse o_proj + TP all_reduce + residual add + post-attention RMSNorm.
    def o_proj_add_rms_norm(self, attn_output: torch.Tensor, residual: torch.Tensor, norm: RMSNorm) -> tuple[torch.Tensor, torch.Tensor]:
        return ascend_ops.matmul_allreduce_add_rmsnorm(attn_output, self.o_proj.weight, residual, norm.weight, self._tp_comm_name(), self.o_proj.tp_size, self.o_proj.tp_rank, norm.eps, True, True)

    def assign_dsa_cache(
        self,
        ckv_cache: torch.Tensor,
        kpe_cache: torch.Tensor,
        index_cache: torch.Tensor,
        dram_ckv_cache: torch.Tensor,
        dram_kpe_cache: torch.Tensor,
        lidu_cache_slots: torch.Tensor,
    ) -> None:
        self.ckv_cache = ckv_cache
        self.kpe_cache = kpe_cache
        self.index_cache = index_cache
        self.dram_ckv_cache = dram_ckv_cache
        self.dram_kpe_cache = dram_kpe_cache
        self.lidu_cache_slots = lidu_cache_slots

    def assign_mla_cache(
        self,
        ckv_cache: torch.Tensor,
        kpe_cache: torch.Tensor,
    ) -> None:
        self.ckv_cache = ckv_cache
        self.kpe_cache = kpe_cache

    def post_load_prepare(self) -> None:
        if self.wd_qkv is None:
            q_weight = self.q_a_proj.weight.detach().cpu()
            kv_weight = self.kv_a_proj_with_mqa.weight.detach().cpu()
            dtype = self.q_a_proj.weight.dtype
            device = self.q_a_proj.weight.device
            self.q_a_proj._parameters.pop("weight", None)
            self.kv_a_proj_with_mqa._parameters.pop("weight", None)
            gc.collect()
            if device.type == "npu":
                torch.npu.empty_cache()

            wd_qkv = torch.empty((self.q_lora_rank + self.kv_lora_rank + self.qk_rope_head_dim, self.hidden_size), dtype=dtype, device=device)
            wd_qkv[: self.q_lora_rank].copy_(q_weight)
            wd_qkv[self.q_lora_rank :].copy_(kv_weight)
            self.wd_qkv = nn.Parameter(wd_qkv, requires_grad=False)
            del q_weight, kv_weight
            gc.collect()

        if self.w_uk_t is None or self.w_uv is None:
            weight = self.kv_b_proj.weight.data.view(self.num_local_heads, self.qk_nope_head_dim + self.v_head_dim, self.kv_lora_rank)
            self.w_uk_t = weight[:, : self.qk_nope_head_dim, :].contiguous()
            self.w_uv = weight[:, self.qk_nope_head_dim :, :].transpose(1, 2).contiguous()
            self.kv_b_proj._parameters.pop("weight", None)
            gc.collect()
            if self.w_uk_t.device.type == "npu":
                torch.npu.empty_cache()

        self._prepare_decode_mlapo()
        if self.indexer is not None:
            self.indexer.post_load_prepare()

    def _run_dsa_pipeline_with_qc_full_graph(
        self,
        hidden_states: torch.Tensor,
        q_c: torch.Tensor,
        positions: torch.Tensor,
        batch_size: int,
    ) -> _LiduUpdateResult:
        if self.indexer is None or self.indexer_rotary_emb is None:
            raise RuntimeError("LIDU indexer is disabled for this engine.")
        context = get_context()
        if not context.full_decode_graph:
            raise RuntimeError(
                "The graph-visible LIDU pipeline requires FULL_DECODE_ONLY."
            )
        if q_c.shape != (batch_size, self.q_lora_rank):
            raise RuntimeError(
                "DSA FULL_DECODE_ONLY requires MLAPO q_c with shape "
                f"{(batch_size, self.q_lora_rank)}, got {tuple(q_c.shape)}."
            )
        required_context = {
            "candidate_lens": context.candidate_lens,
            "req_pool_entries": context.req_pool_entries,
            "index_block_tables": context.index_block_tables,
            "dram_block_tables": context.dram_block_tables,
        }
        missing = [name for name, value in required_context.items() if value is None]
        if missing:
            raise RuntimeError(
                "DSA FULL_DECODE_ONLY context is missing: " + ", ".join(missing)
            )

        q_index, _, index_weights = self.indexer(
            hidden_states,
            q_c,
            positions,
            self.indexer_rotary_emb,
            query_only=True,
        )
        return self._lidu_update(
            q_index,
            index_weights,
            int(batch_size),
        )

    def _prepare_decode_mlapo(self) -> None:
        if self.mlapo_wd_qkv is not None and self.mlapo_wu_q is not None:
            return

        if self.wd_qkv is None:
            return
        q_weight = self.wd_qkv[: self.q_lora_rank].detach()
        kv_weight = self.wd_qkv[self.q_lora_rank :].detach()

        kv_weight = _trans_rope_weight(kv_weight, self.qk_rope_head_dim)
        wd_qkv = torch.cat((kv_weight, q_weight), dim=0).contiguous()
        self.mlapo_wd_qkv = _to_mlapo_bf16_nz_weight(wd_qkv)

        wu_q = self.q_b_proj.weight.detach().view(self.num_local_heads, self.qk_head_dim, self.q_lora_rank)
        wu_q = _trans_rope_weight(wu_q, self.qk_rope_head_dim)
        wu_q = wu_q.reshape(self.num_local_heads * self.qk_head_dim, self.q_lora_rank).contiguous()
        self.mlapo_wu_q = _to_mlapo_bf16_nz_weight(wu_q)
        self.mlapo_beta1 = torch.zeros_like(self.q_a_layernorm.weight)
        del wd_qkv, wu_q
        gc.collect()
        if q_weight.device.type == "npu":
            torch.npu.empty_cache()

    def _mlapo_cos_sin(self, positions: torch.Tensor, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
        context = get_context()
        cache_key = ("mlapo_cos_sin", str(positions.device), dtype, self.qk_rope_head_dim)
        cached = context.scratch.get(cache_key)
        if cached is not None:
            return cached

        positions = positions.to(torch.long)
        cos = self.rotary_emb.cos_cache.index_select(0, positions)
        sin = self.rotary_emb.sin_cache.index_select(0, positions)
        cos = torch.cat((cos, cos), dim=-1).to(dtype).contiguous()
        sin = torch.cat((sin, sin), dim=-1).to(dtype).contiguous()
        context.scratch[cache_key] = (cos, sin)
        return cos, sin

    def _decode_mlapo_buffers(
        self,
        num_tokens: int,
        dtype: torch.dtype,
        device: torch.device,
        need_inner_out: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        ql_shape = (num_tokens, self.num_local_heads, self.kv_lora_rank)
        qpe_shape = (num_tokens, self.num_local_heads, self.qk_rope_head_dim)
        inner_shape = (num_tokens, self.q_lora_rank) if need_inner_out else (0,)
        if self._decode_mlapo_ql_nope is None or tuple(self._decode_mlapo_ql_nope.shape) != ql_shape or self._decode_mlapo_ql_nope.dtype != dtype or self._decode_mlapo_ql_nope.device != device:
            self._decode_mlapo_ql_nope = torch.empty(ql_shape, dtype=dtype, device=device)
        if self._decode_mlapo_q_pe is None or tuple(self._decode_mlapo_q_pe.shape) != qpe_shape or self._decode_mlapo_q_pe.dtype != dtype or self._decode_mlapo_q_pe.device != device:
            self._decode_mlapo_q_pe = torch.empty(qpe_shape, dtype=dtype, device=device)
        if self._decode_mlapo_inner_out is None or tuple(self._decode_mlapo_inner_out.shape) != inner_shape or self._decode_mlapo_inner_out.dtype != dtype or self._decode_mlapo_inner_out.device != device:
            self._decode_mlapo_inner_out = torch.empty(inner_shape, dtype=dtype, device=device)
        return self._decode_mlapo_ql_nope, self._decode_mlapo_q_pe, self._decode_mlapo_inner_out

    def _decode_mlapo_preprocess(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        need_inner_out: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.mlapo_wd_qkv is None or self.mlapo_wu_q is None:
            self._prepare_decode_mlapo()
        if self.mlapo_wd_qkv is None or self.mlapo_wu_q is None:
            raise RuntimeError("Decode MLAPO weights are not prepared.")

        num_tokens = int(hidden_states.shape[0])
        ql_nope, q_pe, inner_out = self._decode_mlapo_buffers(
            num_tokens,
            dtype=hidden_states.dtype,
            device=hidden_states.device,
            need_inner_out=need_inner_out,
        )
        cos, sin = self._mlapo_cos_sin(positions, hidden_states.dtype)
        context = get_context()
        slotmapping = context.flat_slot_mapping_i32 if context.flat_slot_mapping_i32 is not None else self._flat_slots().to(torch.int32)

        ascend_ops.mla_preprocess(
            hidden_states,
            self.mlapo_wd_qkv,
            None,
            self.q_a_layernorm.weight,
            self.mlapo_beta1,
            self.mlapo_wu_q,
            None,
            self.kv_a_layernorm.weight,
            cos,
            sin,
            self.w_uk_t,
            self.ckv_cache,
            self.kpe_cache,
            slotmapping,
            q_out0=ql_nope,
            kv_cache_out0=self.ckv_cache,
            q_out1=q_pe,
            kv_cache_out1=self.kpe_cache,
            inner_out=inner_out,
            cache_mode="krope_ctkv",
            quant_mode="no_quant",
            enable_inner_out=need_inner_out,
        )
        return ql_nope, q_pe, inner_out

    @property
    def block_size(self) -> int:
        if self.ckv_cache.dim() >= 4 and int(self.ckv_cache.shape[1]) == 1:
            return int(self.ckv_cache.shape[2])
        return int(self.ckv_cache.shape[1])

    def _flat_slots(self) -> torch.Tensor:
        context = get_context()
        if context.flat_slot_mapping is None:
            raise RuntimeError("Explicit cache writes require flat_slot_mapping.")
        return context.flat_slot_mapping

    def finalize_prefill_offload(
        self,
        seq,
        old_hbm_block_table: list[int],
    ) -> None:
        num_full_blocks = int(seq.num_prefill_full_blocks)
        num_sparse_blocks = int(seq.num_sparse_blocks)
        pool_entry = int(seq.offload_pool_entry)
        self.lidu_cache_slots[pool_entry].fill_(-1)

        if num_sparse_blocks >= num_full_blocks:
            # Dense/short requests keep every full prefill block in HBM, so their
            # decode path stays aligned with baseline and no DRAM copy is needed.
            if (
                int(seq.lidu_cache_tokens) > 0
                and num_full_blocks > 0
            ):
                source_tokens = num_full_blocks * self.block_size
                self.lidu_cache_slots[pool_entry, :source_tokens].copy_(
                    torch.arange(
                        source_tokens,
                        dtype=torch.int32,
                        device=self.lidu_cache_slots.device,
                    )
                )
            return

        # Long requests keep only the first prefix block plus suffix blocks in HBM.
        # Persist the full prefill KV to DRAM for future Tx>0 promote copies.
        for logical_block in range(num_full_blocks):
            hbm_block = int(old_hbm_block_table[logical_block])
            dram_block = int(seq.dram_block_table[logical_block])
            self.dram_ckv_cache[dram_block].copy_(self.ckv_cache[hbm_block].to(device=self.dram_ckv_cache.device, non_blocking=False))
            self.dram_kpe_cache[dram_block].copy_(self.kpe_cache[hbm_block].to(device=self.dram_kpe_cache.device, non_blocking=False))
        # Ensure DRAM source KV is ready before later decode promote copies.
        _synchronize_device(self.ckv_cache.device)

    def _store_index_cache(self, index_k: torch.Tensor | None) -> None:
        if index_k is None:
            return
        if self.indexer is None:
            raise RuntimeError("DSA indexer is disabled for this engine.")
        context = get_context()
        index_slots = context.flat_index_slot_mapping if context.flat_index_slot_mapping is not None else self._flat_slots()
        flat_cache = self.index_cache.view(-1, self.indexer.head_dim)
        flat_cache.index_copy_(0, index_slots, index_k)

    def _run_indexer(
        self,
        hidden_states: torch.Tensor,
        q_c: torch.Tensor,
        positions: torch.Tensor,
        query_only: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]:
        if self.indexer is None or self.indexer_rotary_emb is None:
            raise RuntimeError("DSA indexer is disabled for this engine.")
        q_index, index_k, weights = self.indexer(
            hidden_states,
            q_c,
            positions,
            self.indexer_rotary_emb,
            query_only=query_only,
        )
        return q_index, index_k, weights

    def _q_nope_up_proj(self, q_nope: torch.Tensor) -> torch.Tensor:
        num_tokens = q_nope.shape[0]
        if q_nope.dtype in (torch.float16, torch.bfloat16) and num_tokens == 1:
            ql_nope = torch.empty((num_tokens, self.num_local_heads, self.kv_lora_rank), dtype=q_nope.dtype, device=q_nope.device)
            ascend_ops.batch_matmul_transpose(q_nope, self.w_uk_t, ql_nope)
            return ql_nope

        q_nope_by_head = q_nope.transpose(0, 1).contiguous()
        ql_nope = torch.bmm(q_nope_by_head, self.w_uk_t)
        return ql_nope.transpose(0, 1).contiguous()

    def _decode_mla_v2_buffers(self, batch_size: int, dtype: torch.dtype, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        out_shape = (self.num_local_heads, batch_size, 1, self.kv_lora_rank)
        lse_shape = (batch_size,)
        if self._decode_mla_v2_out is None or tuple(self._decode_mla_v2_out.shape) != out_shape or self._decode_mla_v2_out.dtype != dtype or self._decode_mla_v2_out.device != device:
            self._decode_mla_v2_out = torch.empty(out_shape, dtype=dtype, device=device)
        if self._decode_mla_v2_lse is None or tuple(self._decode_mla_v2_lse.shape) != lse_shape or self._decode_mla_v2_lse.dtype != dtype or self._decode_mla_v2_lse.device != device:
            self._decode_mla_v2_lse = torch.empty(lse_shape, dtype=dtype, device=device)
        return self._decode_mla_v2_out, self._decode_mla_v2_lse

    def _decode_mla_v2_workspace_get(self, batch_size: int, query: torch.Tensor, key_cache: torch.Tensor, kwargs: dict) -> torch.Tensor:
        key = (
            batch_size,
            tuple(query.shape),
            str(query.device),
            query.dtype,
            self.block_size,
            self.num_local_heads,
            self.kv_lora_rank,
            kwargs.get("input_layout"),
            kwargs.get("sparse_mode"),
        )
        workspace = _NPU_MLA_V2_WORKSPACE_CACHE.get(key)
        if workspace is None:
            # The FIA v2 workspace depends on operator shape/attrs, not layer
            # weights. Share it across layers instead of keeping 61 copies.
            workspace = torch_npu._npu_fused_infer_attention_score_v2_get_max_workspace(query, key_cache, key_cache, **kwargs)
            _NPU_MLA_V2_WORKSPACE_CACHE[key] = workspace
        return workspace

    @torch.compiler.disable
    def _decode_forward_mla_v2(
        self,
        ql_nope: torch.Tensor,
        q_pe: torch.Tensor,
        block_table: torch.Tensor,
        actual_seq_lengths_key: list[int],
        ckv_cache: torch.Tensor | None = None,
        kpe_cache: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch_size = int(ql_nope.shape[0])
        query = ql_nope.view(batch_size, self.num_local_heads, 1, self.kv_lora_rank).contiguous()
        query_rope = q_pe.view(batch_size, self.num_local_heads, 1, self.qk_rope_head_dim)
        # Nano stores paged MLA cache as [blocks, block_size, kv_heads, dim].
        # FIA v2 with BNSD_NBSD expects [blocks, kv_heads, block_size, dim].
        # MLA has kv_heads=1 here, so this is a metadata-only view,
        # not a big cache copy.
        if ckv_cache is None:
            ckv_cache = self.ckv_cache
        if kpe_cache is None:
            kpe_cache = self.kpe_cache
        key_cache = ckv_cache.view(-1, 1, self.block_size, self.kv_lora_rank)
        key_rope_cache = kpe_cache.view(
            -1,
            1,
            self.block_size,
            self.qk_rope_head_dim,
        )
        kwargs = {
            "query_rope": query_rope,
            "key_rope": key_rope_cache,
            "num_query_heads": self.num_local_heads,
            "num_key_value_heads": 1,
            "input_layout": "BNSD_NBSD",
            "atten_mask": None,
            "sparse_mode": 0,
            "softmax_scale": float(self.scale),
            "block_table": block_table,
            "block_size": self.block_size,
            "actual_seq_qlen": None,
            "actual_seq_kvlen": actual_seq_lengths_key,
        }
        out, lse = self._decode_mla_v2_buffers(batch_size, ql_nope.dtype, ql_nope.device)
        workspace = self._decode_mla_v2_workspace_get(batch_size, query, key_cache, kwargs)
        attention_op = torch_npu.npu_fused_infer_attention_score_v2.out
        if is_full_decode_graph_capturing():
            # FIA-v2 receives actual_seq_kvlen as a host-side list rather than
            # a tensor. Capture it as a refreshable graph task so every replay
            # sees the current sparse-KV length instead of warmup values.
            stream = torch.npu.current_stream()
            event = torch.npu.ExternalEvent()
            event.wait(stream)
            event.reset(stream)
            torch.npu.graph_task_group_begin(stream)
            attention_op(
                query,
                key_cache,
                key_cache,
                **kwargs,
                workspace=workspace,
                out=[out, lse],
            )
            handle = torch.npu.graph_task_group_end(stream)
            record_mla_graph_task(
                MLAGraphTask(
                    handle=handle,
                    event=event,
                    op=attention_op,
                    query=query,
                    key_cache=key_cache,
                    query_rope=query_rope,
                    key_rope_cache=key_rope_cache,
                    block_table=block_table,
                    workspace=workspace,
                    output=out,
                    softmax_lse=lse,
                    num_query_heads=self.num_local_heads,
                    block_size=self.block_size,
                    softmax_scale=float(self.scale),
                )
            )
        else:
            attention_op(
                query,
                key_cache,
                key_cache,
                **kwargs,
                workspace=workspace,
                out=[out, lse],
            )
        # FIA v2 writes NBSD-like output [heads, tokens, 1, dim], matching
        # vllm-ascend's trick to feed v-up without a transpose/copy round trip.
        return out

    def _prefill_forward_npu_mla(
        self,
        ql_nope: torch.Tensor,
        q_pe: torch.Tensor,
    ) -> torch.Tensor:
        context = get_context()
        cu_seqlens = context.cu_seqlens_q
        actual_seq_lengths_query = cu_seqlens[1:]
        actual_seq_lengths_key = context.actual_seq_lengths_kv
        if actual_seq_lengths_key is None:
            actual_seq_lengths_key = (
                cu_seqlens[1:] - cu_seqlens[:-1]
            ).detach().cpu().tolist()
        mla_result = torch_npu.npu_fused_infer_attention_score(
            ql_nope,
            self.ckv_cache.transpose(1, 2),
            self.ckv_cache.transpose(1, 2),
            query_rope=q_pe,
            key_rope=self.kpe_cache.transpose(1, 2),
            num_heads=self.num_local_heads,
            num_key_value_heads=1,
            input_layout="TND",
            atten_mask=_get_npu_mla_attention_mask(ql_nope.device, 2048),
            sparse_mode=3,
            scale=float(self.scale),
            antiquant_mode=0,
            antiquant_scale=None,
            block_table=context.block_tables,
            block_size=self.block_size,
            softmax_lse_flag=False,
            actual_seq_lengths=actual_seq_lengths_query.detach().cpu().tolist(),
            actual_seq_lengths_kv=actual_seq_lengths_key,
        )
        latent = mla_result[0] if isinstance(mla_result, (tuple, list)) else mla_result
        return torch_npu.npu_transpose_batchmatmul(latent.transpose(0, 1).contiguous(), self.w_uv, perm_y=(1, 0, 2)).reshape(latent.shape[0], -1)

    def _spec_decode_forward_npu_mla(
        self,
        ql_nope: torch.Tensor,
        q_pe: torch.Tensor,
    ) -> torch.Tensor:
        """Run the target model's K+1 causal MTP verification."""

        context = get_context()
        if context.cu_seqlens_q is None:
            raise RuntimeError("MTP verification is missing query lengths.")
        if context.actual_seq_lengths_kv is None:
            raise RuntimeError("MTP verification is missing KV lengths.")
        if context.block_tables is None:
            raise RuntimeError("MTP verification is missing block tables.")

        num_tokens = int(ql_nope.shape[0])
        query = ql_nope.view(
            num_tokens, self.num_local_heads, self.kv_lora_rank
        ).contiguous()
        query_rope = q_pe.view(
            num_tokens, self.num_local_heads, self.qk_rope_head_dim
        )
        key_cache = self.ckv_cache.view(
            -1, 1, self.block_size, self.kv_lora_rank
        )
        key_rope_cache = self.kpe_cache.view(
            -1, 1, self.block_size, self.qk_rope_head_dim
        )
        actual_seq_qlen = context.actual_seq_lengths_q
        if actual_seq_qlen is None:
            actual_seq_qlen = (
                context.cu_seqlens_q[1:].detach().cpu().tolist()
            )
        atten_mask = _get_npu_mla_attention_mask(query.device, 2048)
        kwargs = {
            "query_rope": query_rope,
            "key_rope": key_rope_cache,
            "num_query_heads": self.num_local_heads,
            "num_key_value_heads": 1,
            "input_layout": "TND_NTD",
            "atten_mask": atten_mask,
            "sparse_mode": 3,
            "softmax_scale": float(self.scale),
            "block_table": context.block_tables,
            "block_size": self.block_size,
            "actual_seq_qlen": actual_seq_qlen,
            "actual_seq_kvlen": context.actual_seq_lengths_kv,
        }
        if is_full_decode_graph_capturing():
            if (
                self._spec_decode_mla_v2_out is None
                or self._spec_decode_mla_v2_lse is None
            ):
                raise RuntimeError(
                    "MTP target FIA-v2 graph buffers were not warmed before "
                    "capture."
                )
            out = self._spec_decode_mla_v2_out
            lse = self._spec_decode_mla_v2_lse
            workspace = self._decode_mla_v2_workspace_get(
                num_tokens, query, key_cache, kwargs
            )
            attention_op = torch_npu.npu_fused_infer_attention_score_v2.out
            stream = torch.npu.current_stream()
            event = torch.npu.ExternalEvent()
            event.wait(stream)
            event.reset(stream)
            torch.npu.graph_task_group_begin(stream)
            attention_op(
                query,
                key_cache,
                key_cache,
                **kwargs,
                workspace=workspace,
                out=[out, lse],
            )
            handle = torch.npu.graph_task_group_end(stream)
            record_mla_graph_task(
                MLAGraphTask(
                    handle=handle,
                    event=event,
                    op=attention_op,
                    query=query,
                    key_cache=key_cache,
                    query_rope=query_rope,
                    key_rope_cache=key_rope_cache,
                    block_table=context.block_tables,
                    workspace=workspace,
                    output=out,
                    softmax_lse=lse,
                    num_query_heads=self.num_local_heads,
                    block_size=self.block_size,
                    softmax_scale=float(self.scale),
                    input_layout="TND_NTD",
                    atten_mask=atten_mask,
                    sparse_mode=3,
                    actual_seq_qlen=actual_seq_qlen,
                )
            )
            latent = out
        else:
            latent, _ = torch_npu.npu_fused_infer_attention_score_v2(
                query,
                key_cache,
                key_cache,
                **kwargs,
            )
            out_shape = (
                self.num_local_heads,
                num_tokens,
                self.kv_lora_rank,
            )
            if (
                self._spec_decode_mla_v2_out is None
                or tuple(self._spec_decode_mla_v2_out.shape)
                != out_shape
            ):
                self._spec_decode_mla_v2_out = torch.empty(
                    out_shape,
                    dtype=query.dtype,
                    device=query.device,
                )
            if (
                self._spec_decode_mla_v2_lse is None
                or tuple(self._spec_decode_mla_v2_lse.shape) != (num_tokens,)
            ):
                self._spec_decode_mla_v2_lse = torch.empty(
                    num_tokens,
                    dtype=query.dtype,
                    device=query.device,
                )
            self._decode_mla_v2_workspace_get(
                num_tokens, query, key_cache, kwargs
            )
        latent = latent.view(
            self.num_local_heads, num_tokens, self.kv_lora_rank
        )
        return torch_npu.npu_transpose_batchmatmul(
            latent,
            self.w_uv,
            perm_y=(1, 0, 2),
        ).reshape(num_tokens, -1)

    def _lidu_update(
        self,
        q_index: torch.Tensor,
        weights: torch.Tensor,
        batch_size: int,
    ) -> _LiduUpdateResult:
        context = get_context()
        required = {
            "req_pool_entries": context.req_pool_entries,
            "lidu_cache_tokens": context.lidu_cache_tokens,
            "candidate_lens": context.candidate_lens,
            "index_block_tables": context.index_block_tables,
            "block_tables": context.block_tables,
            "dram_block_tables": context.dram_block_tables,
        }
        missing = [name for name, value in required.items() if value is None]
        if missing:
            raise RuntimeError(
                "LIDU decode context is missing: " + ", ".join(missing)
            )

        hbm_kpe = self.kpe_cache.squeeze(2)
        hbm_ckv = self.ckv_cache.squeeze(2)
        dram_kpe = self.dram_kpe_cache.squeeze(2)
        dram_ckv = self.dram_ckv_cache.squeeze(2)

        init_rows = context.lidu_init_rows
        if init_rows is not None and init_rows.numel() > 0:
            if context.full_decode_graph:
                raise RuntimeError("LIDU initialization must remain eager.")
            # Initialization latency is intentionally excluded from the stable
            # decode target.  Process rows independently to cap temporary HBM.
            for row in init_rows.detach().cpu().tolist():
                row = int(row)
                pool_entry = int(context.req_pool_entries[row].item())
                cache_tokens = int(context.lidu_cache_tokens[row].item())
                candidate_len = int(context.candidate_lens[row].item())
                hbm_kpe, hbm_ckv = initialize_lidu_row(
                    query=q_index[row],
                    weights=weights[row],
                    index_cache=self.index_cache,
                    index_block_table=context.index_block_tables[row],
                    candidate_len=candidate_len,
                    cache_tokens=cache_tokens,
                    cache_slots_row=self.lidu_cache_slots[pool_entry],
                    hbm_kpe=hbm_kpe,
                    hbm_ckv=hbm_ckv,
                    dram_kpe=dram_kpe,
                    dram_ckv=dram_ckv,
                    hbm_block_table=context.block_tables[row],
                    dram_block_table=context.dram_block_tables[row],
                    block_size=self.block_size,
                )

        buffers = self._fused_li_manage_outputs.get(batch_size)
        if buffers is None:
            options = dict(dtype=torch.int32, device=q_index.device)
            buffers = (
                torch.zeros((batch_size, 1, LIDU_TOPK), **options),
                torch.zeros((batch_size, 1, LIDU_TOPK), **options),
                torch.zeros((batch_size,), **options),
            )
            self._fused_li_manage_outputs[batch_size] = buffers
        topk_src_ids, topk_dst_slots, miss_counts = buffers
        fused_li_manage(
            q_index[:batch_size],
            weights[:batch_size],
            self.index_cache,
            context.index_block_tables[:batch_size],
            context.candidate_lens[:batch_size],
            context.lidu_cache_tokens[:batch_size],
            context.req_pool_entries[:batch_size],
            self.lidu_cache_slots,
            topk_src_ids,
            topk_dst_slots,
            miss_counts,
        )
        if self._lidu_miss_count_collect_all:
            self._record_lidu_miss_counts(miss_counts, batch_size)
        use_fused_attention_scatter = (
            self._can_use_fused_copy_sfa()
        )
        if not use_fused_attention_scatter:
            scatter_copy(
                topk_src_ids.view(batch_size, -1),
                topk_dst_slots.view(batch_size, -1),
                miss_counts[:batch_size],
                context.block_tables[:batch_size],
                context.dram_block_tables[:batch_size],
                hbm_kpe,
                hbm_ckv,
                dram_kpe,
                dram_ckv,
            )
        return (
            hbm_kpe,
            hbm_ckv,
            topk_dst_slots,
            topk_src_ids,
            miss_counts,
        )

    def _lidu_update_mtp(
        self,
        q_index: torch.Tensor,
        weights: torch.Tensor,
        batch_size: int,
    ) -> _MtpLiduUpdateResult:
        """Update one request cache from the union of its four MTP queries."""

        context = get_context()
        required = {
            "req_pool_entries": context.req_pool_entries,
            "lidu_cache_tokens": context.lidu_cache_tokens,
            "candidate_lens": context.candidate_lens,
            "index_block_tables": context.index_block_tables,
            "block_tables": context.block_tables,
            "dram_block_tables": context.dram_block_tables,
        }
        missing = [name for name, value in required.items() if value is None]
        if missing:
            raise RuntimeError(
                "MTP-LIDU decode context is missing: " + ", ".join(missing)
            )
        if int(q_index.shape[0]) != batch_size * 4:
            raise RuntimeError(
                "MTP-LIDU requires four query rows per request: "
                f"queries={q_index.shape[0]}, batch={batch_size}."
            )

        hbm_kpe = self.kpe_cache.squeeze(2)
        hbm_ckv = self.ckv_cache.squeeze(2)
        dram_kpe = self.dram_kpe_cache.squeeze(2)
        dram_ckv = self.dram_ckv_cache.squeeze(2)

        init_rows = context.lidu_init_rows
        if init_rows is not None and init_rows.numel() > 0:
            if context.full_decode_graph:
                raise RuntimeError("MTP-LIDU initialization must remain eager.")
            for row in init_rows.detach().cpu().tolist():
                row = int(row)
                pool_entry = int(context.req_pool_entries[row].item())
                cache_tokens = int(context.lidu_cache_tokens[row].item())
                candidate_len = int(context.candidate_lens[row].item())
                # Initialization is a deliberately slow first-decode path.
                # Seed top-C from q0, then the fused four-query update below
                # installs the complete protected union before Attention.
                hbm_kpe, hbm_ckv = initialize_lidu_row(
                    query=q_index[row * 4],
                    weights=weights[row * 4],
                    index_cache=self.index_cache,
                    index_block_table=context.index_block_tables[row],
                    candidate_len=candidate_len,
                    cache_tokens=cache_tokens,
                    cache_slots_row=self.lidu_cache_slots[pool_entry],
                    hbm_kpe=hbm_kpe,
                    hbm_ckv=hbm_ckv,
                    dram_kpe=dram_kpe,
                    dram_ckv=dram_ckv,
                    hbm_block_table=context.block_tables[row],
                    dram_block_table=context.dram_block_tables[row],
                    block_size=self.block_size,
                )

        buffers = self._fused_li_manage_mtp_outputs.get(batch_size)
        if buffers is None:
            options = dict(dtype=torch.int32, device=q_index.device)
            buffers = (
                torch.full(
                    (batch_size * 4, 1, LIDU_TOPK), -1, **options
                ),
                torch.zeros(
                    (batch_size * 4, 1, LIDU_TOPK), **options
                ),
                torch.zeros(
                    (batch_size, LIDU_MTP_UNION_CAPACITY), **options
                ),
                torch.zeros(
                    (batch_size, LIDU_MTP_UNION_CAPACITY), **options
                ),
                torch.zeros((batch_size,), **options),
            )
            self._fused_li_manage_mtp_outputs[batch_size] = buffers
        (
            topk_src_ids,
            topk_dst_slots,
            miss_src_ids,
            miss_dst_slots,
            miss_counts,
        ) = buffers
        fused_li_manage_mtp(
            q_index[: batch_size * 4],
            weights[: batch_size * 4],
            self.index_cache,
            context.index_block_tables[:batch_size],
            context.candidate_lens[:batch_size],
            context.lidu_cache_tokens[:batch_size],
            context.req_pool_entries[:batch_size],
            self.lidu_cache_slots,
            topk_src_ids,
            topk_dst_slots,
            miss_src_ids,
            miss_dst_slots,
            miss_counts,
        )
        if self._lidu_miss_count_collect_all:
            self._record_lidu_miss_counts(miss_counts, batch_size)
        if not self._can_use_fused_copy_sfa():
            scatter_copy(
                miss_src_ids,
                miss_dst_slots,
                miss_counts[:batch_size],
                context.block_tables[:batch_size],
                context.dram_block_tables[:batch_size],
                hbm_kpe,
                hbm_ckv,
                dram_kpe,
                dram_ckv,
            )
        return (
            hbm_kpe,
            hbm_ckv,
            topk_dst_slots,
            topk_src_ids,
            miss_src_ids,
            miss_dst_slots,
            miss_counts,
        )

    def _spec_decode_forward_lidu(
        self,
        ql_nope: torch.Tensor,
        q_pe: torch.Tensor,
        q_index: torch.Tensor,
        weights: torch.Tensor,
    ) -> torch.Tensor:
        """Run eager MTP3 target verification through the offload chain."""

        context = get_context()
        if int(ql_nope.shape[0]) % 4:
            raise RuntimeError("MTP3 target query rows must be divisible by four.")
        batch_size = int(ql_nope.shape[0]) // 4
        required = {
            "candidate_query_lens": context.candidate_query_lens,
            "actual_seq_lengths_kv_tensor": context.actual_seq_lengths_kv_tensor,
            "lidu_cache_tokens": context.lidu_cache_tokens,
            "block_tables": context.block_tables,
        }
        missing = [name for name, value in required.items() if value is None]
        if missing:
            raise RuntimeError(
                "MTP sparse-and-tail Attention context is missing: "
                + ", ".join(missing)
            )
        (
            kpe_cache,
            ckv_cache,
            topk_dst_slots,
            topk_src_ids,
            miss_src_ids,
            miss_dst_slots,
            miss_counts,
        ) = self._lidu_update_mtp(q_index, weights, batch_size)
        use_fused_copy_attention = self._can_use_fused_copy_sfa()
        output_key = tuple(ql_nope.shape)
        attention_out = self._sparse_tail_outputs.get(output_key)
        if (
            attention_out is None
            or attention_out.dtype != ql_nope.dtype
            or attention_out.device != ql_nope.device
        ):
            attention_out = torch.empty_like(ql_nope)
            self._sparse_tail_outputs[output_key] = attention_out
        hbm_ckv = ckv_cache.view(
            -1, self.block_size, 1, self.kv_lora_rank
        )
        hbm_kpe = kpe_cache.view(
            -1, self.block_size, 1, self.qk_rope_head_dim
        )
        if use_fused_copy_attention:
            if context.dram_block_tables is None or attention_out is None:
                raise RuntimeError(
                    "Fused MTP copy+Attention requires DRAM block tables "
                    "and a caller-owned output buffer."
                )
            fused_copy_sfa_mtp(
                q_pe,
                ql_nope,
                context.candidate_query_lens[:batch_size],
                context.actual_seq_lengths_kv_tensor[:batch_size],
                context.lidu_cache_tokens[:batch_size],
                topk_dst_slots,
                topk_src_ids,
                miss_src_ids,
                miss_dst_slots,
                miss_counts[:batch_size],
                context.block_tables[:batch_size],
                context.dram_block_tables[:batch_size],
                hbm_kpe,
                hbm_ckv,
                self.dram_kpe_cache.squeeze(2),
                self.dram_ckv_cache.squeeze(2),
                self.scale,
                attention_out,
            )
        else:
            sparse_tail_attention_mtp(
                q_pe,
                ql_nope,
                context.candidate_query_lens[:batch_size],
                context.actual_seq_lengths_kv_tensor[:batch_size],
                context.lidu_cache_tokens[:batch_size],
                topk_dst_slots,
                context.block_tables[:batch_size],
                hbm_kpe,
                hbm_ckv,
                self.scale,
                attention_out,
            )
        latent = attention_out
        return torch_npu.npu_transpose_batchmatmul(
            latent.transpose(0, 1).contiguous(),
            self.w_uv,
            perm_y=(1, 0, 2),
        ).reshape(batch_size * 4, -1)

    def _can_use_fused_copy_sfa(self) -> bool:
        context = get_context()
        return (
            self._use_fused_copy_sfa
            and not context.has_first_decode
            and context.lidu_init_rows is None
        )

    @torch.compiler.disable
    def _record_lidu_miss_counts(
        self,
        miss_counts: torch.Tensor,
        batch_size: int,
    ) -> None:
        if not self._lidu_miss_count_collect_all:
            return
        context = get_context()
        by_layer = context.scratch.setdefault(
            _LIDU_MISS_COUNT_SCRATCH_KEY,
            [],
        )
        by_layer.append(
            (self.layer_idx, miss_counts.reshape(-1)[:batch_size])
        )
        if self.layer_idx != self.num_hidden_layers - 1:
            return
        context.scratch.pop(_LIDU_MISS_COUNT_SCRATCH_KEY, None)
        if [layer for layer, _ in by_layer] != list(
            range(self.num_hidden_layers)
        ):
            raise RuntimeError(
                "LIDU miss-count aggregation did not receive every layer."
            )
        per_layer_values = (
            torch.stack([values for _, values in by_layer])
            .detach()
            .cpu()
            .tolist()
        )
        self._lidu_miss_count_decode_step += 1
        for line in format_lidu_miss_count_report(
            per_layer_values,
            self._lidu_miss_count_layers,
            self._lidu_miss_count_decode_step,
        ):
            print(line, flush=True)

    def _decode_forward_mla(
        self,
        ql_nope: torch.Tensor,
        q_pe: torch.Tensor,
        q_index: torch.Tensor | None,
        weights: torch.Tensor | None,
        dsa_updated: bool = False,
        cache_aliases: _LiduUpdateResult | None = None,
    ) -> torch.Tensor:
        context = get_context()
        batch_size = int(ql_nope.shape[0])
        assert context.actual_seq_lengths_kv is not None
        if context.needs_dsa_update and not dsa_updated:
            if q_index is None or weights is None:
                raise RuntimeError("LIDU decode requires indexer outputs.")
            cache_aliases = self._lidu_update(q_index, weights, batch_size)
        kpe_cache = ckv_cache = topk_dst_slots = None
        topk_src_ids = miss_counts = None
        if cache_aliases is not None:
            (
                kpe_cache,
                ckv_cache,
                topk_dst_slots,
                topk_src_ids,
                miss_counts,
            ) = cache_aliases
        if (
            self._use_sparse_tail_attention
            and not context.has_first_decode
            and topk_dst_slots is not None
        ):
            required = {
                "candidate_query_lens": context.candidate_query_lens,
                "actual_seq_lengths_kv_tensor": (
                    context.actual_seq_lengths_kv_tensor
                ),
                "lidu_cache_tokens": context.lidu_cache_tokens,
                "block_tables": context.block_tables,
            }
            missing = [
                name for name, value in required.items() if value is None
            ]
            if missing:
                raise RuntimeError(
                    "Sparse-and-tail Attention context is missing: "
                    + ", ".join(missing)
                )
            latent_kv_cache = ckv_cache.view(
                -1, self.block_size, 1, self.kv_lora_rank
            )
            hbm_kpe = kpe_cache.view(
                -1, self.block_size, 1, self.qk_rope_head_dim
            )
            output_key = tuple(ql_nope.shape)
            attention_out = self._sparse_tail_outputs.get(output_key)
            if (
                attention_out is None
                or attention_out.dtype != ql_nope.dtype
                or attention_out.device != ql_nope.device
            ):
                attention_out = torch.empty_like(ql_nope)
                self._sparse_tail_outputs[output_key] = attention_out
            if (
                self._can_use_fused_copy_sfa()
                and topk_src_ids is not None
                and miss_counts is not None
            ):
                if context.dram_block_tables is None:
                    raise RuntimeError(
                        "Fused Attention+SCATTER requires DRAM block tables."
                    )
                fused_copy_sfa(
                    q_pe,
                    ql_nope,
                    context.candidate_query_lens[:batch_size],
                    context.actual_seq_lengths_kv_tensor[:batch_size],
                    context.lidu_cache_tokens[:batch_size],
                    topk_dst_slots[:batch_size],
                    topk_src_ids.view(batch_size, -1),
                    miss_counts[:batch_size],
                    context.block_tables[:batch_size],
                    context.dram_block_tables[:batch_size],
                    hbm_kpe,
                    latent_kv_cache,
                    self.dram_kpe_cache.squeeze(2),
                    self.dram_ckv_cache.squeeze(2),
                    self.scale,
                    attention_out,
                )
            else:
                sparse_tail_attention(
                    q_pe,
                    ql_nope,
                    context.candidate_query_lens[:batch_size],
                    context.actual_seq_lengths_kv_tensor[:batch_size],
                    context.lidu_cache_tokens[:batch_size],
                    topk_dst_slots[:batch_size],
                    context.block_tables[:batch_size],
                    hbm_kpe,
                    latent_kv_cache,
                    self.scale,
                    attention_out,
                )
            latent = attention_out
            latent_for_v_up = latent.transpose(0, 1).contiguous()
        else:
            latent = self._decode_forward_mla_v2(
                ql_nope,
                q_pe,
                context.block_tables[:batch_size],
                context.actual_seq_lengths_kv,
                ckv_cache=ckv_cache,
                kpe_cache=kpe_cache,
            )
            latent_for_v_up = latent.view(
                self.num_local_heads,
                batch_size,
                self.kv_lora_rank,
            )
        output = torch_npu.npu_transpose_batchmatmul(
            latent_for_v_up,
            self.w_uv,
            perm_y=(1, 0, 2),
        ).reshape(batch_size, -1)
        return output

    def forward(self, positions: torch.Tensor, hidden_states: torch.Tensor, skip_o_proj: bool = False) -> torch.Tensor:
        context = get_context()

        if self.w_uk_t is None or self.w_uv is None:
            self.post_load_prepare()

        if self.wd_qkv is None:
            self.post_load_prepare()
        if self.wd_qkv is None:
            raise RuntimeError("Fused qkv_a weight is not prepared.")

        use_decode_mlapo = (
            not context.is_prefill
            and not context.is_spec_decode
            and not context.has_first_decode
        )
        needs_decode_dsa_update = bool(context.needs_dsa_update)
        if use_decode_mlapo:
            ql_nope, q_pe, q_c = self._decode_mlapo_preprocess(
                positions,
                hidden_states,
                need_inner_out=needs_decode_dsa_update,
            )
            q_index = index_k = weights = None
            dsa_updated = False
            cache_aliases = None
            if needs_decode_dsa_update:
                if context.full_decode_graph:
                    cache_aliases = self._run_dsa_pipeline_with_qc_full_graph(
                        hidden_states,
                        q_c,
                        positions,
                        int(hidden_states.shape[0]),
                    )
                    dsa_updated = True
                else:
                    q_index, index_k, weights = self._run_indexer(
                        hidden_states,
                        q_c,
                        positions,
                        query_only=True,
                    )
            if index_k is not None:
                self._store_index_cache(index_k)

            attn_output = self._decode_forward_mla(
                ql_nope,
                q_pe,
                q_index,
                weights,
                dsa_updated=dsa_updated,
                cache_aliases=cache_aliases,
            )
            return attn_output if skip_o_proj else self.o_proj(attn_output)

        qkv_a = F.linear(hidden_states, self.wd_qkv)
        q_c, kv = torch.split(qkv_a, [self.q_lora_rank, self.kv_lora_rank + self.qk_rope_head_dim], dim=-1)

        q_c = self.q_a_layernorm(q_c)

        q = self.q_b_proj(q_c).view(-1, self.num_local_heads, self.qk_head_dim)
        q_nope, q_pe = torch.split(q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)

        ckv, k_pe = torch.split(kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)

        ckv = self.kv_a_layernorm(ckv)

        q_pe, k_pe = self.rotary_emb(positions, q_pe, k_pe.unsqueeze(1))

        k_pe = k_pe.squeeze(1)

        # RoPE dot products are unchanged if both q/k use the same basis.
        # Keep MLA RoPE cache in neox order so MLAPO decode can use
        # mla_preprocess output directly.
        q_pe = _rope_interleaved_to_neox(q_pe)
        k_pe = _rope_interleaved_to_neox(k_pe)

        q_index = index_k = weights = None
        if context.needs_dsa_update:
            q_index, index_k, weights = self._run_indexer(
                hidden_states,
                q_c,
                positions,
                query_only=not context.is_prefill,
            )

        flat_slots = self._flat_slots()
        self.ckv_cache.view(-1, self.kv_lora_rank).index_copy_(0, flat_slots, ckv)
        self.kpe_cache.view(-1, self.qk_rope_head_dim).index_copy_(0, flat_slots, k_pe)

        if index_k is not None:
            self._store_index_cache(index_k)

        ql_nope = self._q_nope_up_proj(q_nope)

        if context.is_spec_decode:
            if self.uses_offload and context.needs_dsa_update:
                if q_index is None or weights is None:
                    raise RuntimeError(
                        "MTP-LIDU verification requires indexer outputs."
                    )
                attn_output = self._spec_decode_forward_lidu(
                    ql_nope, q_pe, q_index, weights
                )
            else:
                attn_output = self._spec_decode_forward_npu_mla(ql_nope, q_pe)
        elif context.is_prefill:
            attn_output = self._prefill_forward_npu_mla(ql_nope, q_pe)
        else:
            attn_output = self._decode_forward_mla(ql_nope, q_pe, q_index, weights)

        return attn_output if skip_o_proj else self.o_proj(attn_output)
