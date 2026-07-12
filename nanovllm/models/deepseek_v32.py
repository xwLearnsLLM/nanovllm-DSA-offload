from __future__ import annotations

import json
import math
import os
import gc
from time import perf_counter

import torch
import torch_npu  # type: ignore
import torch.distributed as dist
from torch import nn
import torch.nn.functional as F
from transformers import PretrainedConfig

import nanovllm.ops as ascend_ops
from nanovllm.engine.full_decode_graph import (
    MLAGraphTask,
    is_full_decode_graph_capturing,
    record_mla_graph_task,
)
from nanovllm.models.dsa_indexer_project import (
    dsa_indexer_project,
    dsa_indexer_project_query_only,
    dsa_indexer_pipeline_with_qc_full_graph,
)
from nanovllm.layers.activation import SiluAndMul
from nanovllm.layers.embed_head import ParallelLMHead, VocabParallelEmbedding
from nanovllm.layers.layernorm import RMSNorm
from nanovllm.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from nanovllm.utils.context import get_context
from nanovllm.utils.logger import init_logger


logger = init_logger(__name__)
ACL_FORMAT_FRACTAL_NZ = 29
_NPU_MLA_ATTENTION_MASK_CACHE: dict[tuple[str, int], torch.Tensor] = {}
_NPU_MLA_V2_WORKSPACE_CACHE: dict[tuple, torch.Tensor] = {}
_DSA_GATHER_TOPK = 2048
_NPU_MOE_SHARED_STREAM = None


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.lower() in ("1", "true", "yes", "on")


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _rank_id() -> int:
    try:
        return dist.get_rank() if dist.is_initialized() else 0
    except Exception:
        return 0


def _profile_sync(device: torch.device) -> None:
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


def _moe_shared_stream():
    global _NPU_MOE_SHARED_STREAM
    if _NPU_MOE_SHARED_STREAM is None:
        _NPU_MOE_SHARED_STREAM = torch_npu.npu.Stream()
    return _NPU_MOE_SHARED_STREAM


def _profile_layer_selected(layer_idx: int, num_layers: int) -> bool:
    spec = os.environ.get("NANOVLLM_PROFILE_LAYER_IDS", "0,mid,last").strip()
    if spec.lower() in ("all", "*"):
        return True
    for raw_token in spec.split(","):
        token = raw_token.strip().lower()
        if not token:
            continue
        if token == "mid":
            selected = num_layers // 2
        elif token == "last":
            selected = num_layers - 1
        else:
            try:
                selected = int(token)
            except ValueError:
                continue
            if selected < 0:
                selected += num_layers
        if selected == layer_idx:
            return True
    return False


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


def yarn_get_mscale(scale: float = 1.0, mscale: float = 1.0) -> float:
    if scale <= 1:
        return 1.0
    return 0.1 * mscale * math.log(scale) + 1.0


def _rotate_half_neox(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    x1 = x[..., :half]
    x2 = x[..., half:]
    return torch.cat((-x2, x1), dim=-1)


def _rotate_half_interleaved(x: torch.Tensor) -> torch.Tensor:
    x = x.view(*x.shape[:-1], -1, 2)
    x1 = x[..., 0]
    x2 = x[..., 1]
    x = torch.stack((-x2, x1), dim=-1)
    return x.flatten(-2)


class DeepseekV32Config(PretrainedConfig):
    model_type = "deepseek_v32"

    def __init__(self, **kwargs):
        # Newer transformers standardize RoPE fields during PretrainedConfig
        # construction, so expose these attributes before calling super().
        self.max_position_embeddings = kwargs.get("max_position_embeddings")
        self.rope_scaling = kwargs.get("rope_scaling")
        super().__init__(**kwargs)
        for key, value in kwargs.items():
            setattr(self, key, value)
        self.architectures = kwargs.get("architectures", ["DeepseekV32ForCausalLM"])
        self.nanovllm_pruned_shared_only = kwargs.get("nanovllm_pruned_shared_only", False)
        routed_experts = int(kwargs.get("n_routed_experts", 0) or 0)
        inferred_keep_routed = routed_experts > 0 and not self.nanovllm_pruned_shared_only
        keep_routed_flag = kwargs.get("nanovllm_pruned_keep_routed_experts")
        if keep_routed_flag is None:
            keep_routed_flag = inferred_keep_routed
        self.nanovllm_pruned_keep_routed_experts = bool(keep_routed_flag)
        self.nanovllm_export_format = kwargs.get("nanovllm_export_format", "")
        if getattr(self, "torch_dtype", None) is None and "dtype" in kwargs:
            self.torch_dtype = kwargs["dtype"]
        self._normalize_rope_parameters()

    @classmethod
    def from_pretrained(cls, model_path: str) -> "DeepseekV32Config":
        config_path = os.path.join(model_path, "config.json")
        with open(config_path, "r", encoding="utf-8") as file:
            return cls(**json.load(file))

    def _normalize_rope_parameters(self) -> None:
        rope_scaling = dict(getattr(self, "rope_scaling", None) or {})
        rope_theta = getattr(self, "rope_theta", 10000.0)
        rope_parameters = {
            "rope_theta": rope_theta,
            "rope_type": "default",
        }
        if rope_scaling:
            rope_type = rope_scaling.pop("type", "default")
            for key in ("factor", "beta_fast", "beta_slow", "mscale", "mscale_all_dim", "original_max_position_embeddings"):
                if key in rope_scaling:
                    rope_scaling[key] = float(rope_scaling[key])
            rope_parameters.update(rope_scaling)
            rope_parameters["rope_theta"] = rope_theta
            if rope_type == "yarn":
                rope_parameters["rope_type"] = "deepseek_yarn"
            else:
                rope_parameters["rope_type"] = rope_type
        self.rope_parameters = rope_parameters


class DeepseekScalingRotaryEmbedding(nn.Module):
    def __init__(self, rotary_dim: int, max_position_embeddings: int, rope_parameters: dict, *, is_neox_style: bool) -> None:
        super().__init__()
        self.rotary_dim = rotary_dim
        self.is_neox_style = is_neox_style
        base = float(rope_parameters.get("rope_theta", 10000.0))
        rope_type = rope_parameters.get("rope_type", "default")
        cache_len = max_position_embeddings
        mscale = 1.0

        if rope_type == "deepseek_yarn":
            scaling_factor = float(rope_parameters["factor"])
            original_max_position = int(rope_parameters["original_max_position_embeddings"])
            beta_fast = int(rope_parameters.get("beta_fast", 32))
            beta_slow = int(rope_parameters.get("beta_slow", 1))
            inv_freq = self._compute_deepseek_yarn_inv_freq(
                rotary_dim=rotary_dim,
                base=base,
                original_max_position=original_max_position,
                scaling_factor=scaling_factor,
                beta_fast=beta_fast,
                beta_slow=beta_slow,
            )
            mscale = yarn_get_mscale(scaling_factor, float(rope_parameters.get("mscale_all_dim", 0.0)))
            cache_len = int(original_max_position * scaling_factor)
        else:
            inv_freq = 1.0 / (base ** (torch.arange(0, rotary_dim, 2, dtype=torch.float32) / rotary_dim))

        positions = torch.arange(cache_len, dtype=torch.float32)
        freqs = torch.einsum("i,j->ij", positions, inv_freq)
        self.register_buffer("cos_cache", freqs.cos() * mscale, persistent=False)
        self.register_buffer("sin_cache", freqs.sin() * mscale, persistent=False)

    @staticmethod
    def _yarn_linear_ramp_mask(low: float, high: float, dim: int) -> torch.Tensor:
        if low == high:
            high += 1e-3
        positions = torch.arange(dim, dtype=torch.float32)
        mask = (positions - low) / (high - low)
        return mask.clamp_(0.0, 1.0)

    @staticmethod
    def _yarn_find_correction_dim(num_rotations: float, dim: int, base: float, max_position_embeddings: int) -> float:
        return dim * math.log(max_position_embeddings / (num_rotations * 2 * math.pi)) / (2 * math.log(base))

    @classmethod
    def _compute_deepseek_yarn_inv_freq(
        cls,
        *,
        rotary_dim: int,
        base: float,
        original_max_position: int,
        scaling_factor: float,
        beta_fast: int,
        beta_slow: int,
    ) -> torch.Tensor:
        pos_freqs = base ** (torch.arange(0, rotary_dim, 2, dtype=torch.float32) / rotary_dim)
        inv_freq_extrapolation = 1.0 / pos_freqs
        inv_freq_interpolation = 1.0 / (scaling_factor * pos_freqs)

        low = math.floor(cls._yarn_find_correction_dim(beta_fast, rotary_dim, base, original_max_position))
        high = math.ceil(cls._yarn_find_correction_dim(beta_slow, rotary_dim, base, original_max_position))
        low = max(low, 0)
        high = min(high, rotary_dim // 2 - 1)
        inv_freq_mask = 1.0 - cls._yarn_linear_ramp_mask(low, high, rotary_dim // 2)
        return inv_freq_interpolation * (1.0 - inv_freq_mask) + inv_freq_extrapolation * inv_freq_mask

    def forward(self, positions: torch.Tensor, query: torch.Tensor, key: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        query_dtype = query.dtype
        key_dtype = key.dtype
        positions = positions.to(torch.long)
        cos = self.cos_cache.index_select(0, positions)
        sin = self.sin_cache.index_select(0, positions)
        if self.is_neox_style:
            cos = torch.cat((cos, cos), dim=-1).unsqueeze(1)
            sin = torch.cat((sin, sin), dim=-1).unsqueeze(1)
            rotate_fn = _rotate_half_neox
        else:
            cos = cos.repeat_interleave(2, dim=-1).unsqueeze(1)
            sin = sin.repeat_interleave(2, dim=-1).unsqueeze(1)
            rotate_fn = _rotate_half_interleaved
        query = query * cos + rotate_fn(query.float()).to(query.dtype) * sin
        key = key * cos + rotate_fn(key.float()).to(key.dtype) * sin
        return query.to(query_dtype), key.to(key_dtype)


def _resolve_export_mode(config) -> tuple[bool, bool]:
    is_shared_only = bool(getattr(config, "nanovllm_pruned_shared_only", False))
    keep_routed_flag = getattr(config, "nanovllm_pruned_keep_routed_experts", None)
    routed_experts = int(getattr(config, "n_routed_experts", 0) or 0)

    if keep_routed_flag is None:
        keep_routed_experts = routed_experts > 0 and not is_shared_only
        if routed_experts == 0:
            is_shared_only = True
    else:
        keep_routed_experts = bool(keep_routed_flag)

    return is_shared_only, keep_routed_experts


class DeepseekV32MLP(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int, hidden_act: str, *, disable_tp: bool = False, reduce_results: bool = True) -> None:
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(hidden_size, [intermediate_size, intermediate_size], bias=False, disable_tp=disable_tp)
        self.down_proj = RowParallelLinear(intermediate_size, hidden_size, bias=False, disable_tp=disable_tp, reduce_results=reduce_results)
        if hidden_act != "silu":
            raise ValueError("Only silu is supported for DeepSeek-V3.2.")
        self.act_fn = SiluAndMul()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.gate_up_proj(hidden_states)
        hidden_states = self.act_fn(hidden_states)
        hidden_states = self.down_proj(hidden_states)
        return hidden_states


class DeepseekV32SparseMoeBlock(nn.Module):
    def __init__(self, config: DeepseekV32Config, layer_idx: int) -> None:
        super().__init__()
        self.layer_idx = int(layer_idx)
        self.hidden_size = int(config.hidden_size)
        self.moe_intermediate_size = int(config.moe_intermediate_size)
        self.hidden_act = str(config.hidden_act)
        self.num_experts = int(config.n_routed_experts)
        self.top_k = max(1, min(int(config.num_experts_per_tok), self.num_experts))
        self.renormalize = bool(getattr(config, "norm_topk_prob", True))
        # DeepSeek-V3/V3.2 routed expert gating uses sigmoid scores when the
        # config does not explicitly override the scoring function.
        self.scoring_func = str(getattr(config, "scoring_func", "sigmoid"))
        self.routed_scaling_factor = float(getattr(config, "routed_scaling_factor", 1.0))
        self.num_expert_group = max(1, int(getattr(config, "n_group", 1) or 1))
        self.topk_group = max(1, int(getattr(config, "topk_group", 1) or 1))
        self.num_shared_experts = int(getattr(config, "n_shared_experts", 1) or 1)
        self.enable_expert_parallel = bool(getattr(config, "nanovllm_enable_expert_parallel", False))
        self.ep_size = dist.get_world_size() if self.enable_expert_parallel else 1
        self.ep_rank = dist.get_rank() if self.enable_expert_parallel else 0
        if self.enable_expert_parallel and self.num_experts % self.ep_size != 0:
            raise ValueError("DeepSeek-V3.2 expert_parallel requires n_routed_experts to be divisible by the EP world size.")
        self.num_local_experts = self.num_experts // self.ep_size if self.enable_expert_parallel else self.num_experts
        self.local_expert_start = self.ep_rank * self.num_local_experts
        self.local_expert_end = self.local_expert_start + self.num_local_experts
        self.local_expert_ids = tuple(range(self.local_expert_start, self.local_expert_end))
        self.local_expert_id_set = set(self.local_expert_ids)

        self.gate = ReplicatedLinear(self.hidden_size, self.num_experts, bias=False)
        if getattr(config, "topk_method", None) == "noaux_tc":
            self.gate.e_score_correction_bias = nn.Parameter(torch.empty(self.num_experts, dtype=torch.float32))
        else:
            self.gate.register_parameter("e_score_correction_bias", None)

        self.shared_experts = DeepseekV32MLP(hidden_size=self.hidden_size, intermediate_size=self.moe_intermediate_size * self.num_shared_experts, hidden_act=self.hidden_act, reduce_results=not (self.enable_expert_parallel and self.ep_size > 1))
        self.experts = nn.ModuleDict(
            {
                str(expert_idx): DeepseekV32MLP(hidden_size=self.hidden_size, intermediate_size=self.moe_intermediate_size, hidden_act=self.hidden_act, disable_tp=self.enable_expert_parallel)
                for expert_idx in self.local_expert_ids
            }
        )
        self.local_expert_layers = tuple(self.experts[str(expert_idx)] for expert_idx in self.local_expert_ids)
        self.register_parameter("grouped_w13_weight", None)
        self.register_parameter("grouped_w2_weight", None)

    def post_load_prepare(self) -> None:
        if self.grouped_w13_weight is not None:
            return
        if not self.local_expert_layers:
            return

        first_weight = self.local_expert_layers[0].gate_up_proj.weight
        if first_weight.device.type != "npu":
            return
        dtype = first_weight.dtype
        device = first_weight.device
        del first_weight

        cpu_w13_parts: list[torch.Tensor] = []
        cpu_w2_parts: list[torch.Tensor] = []
        for expert_layer in self.local_expert_layers:
            cpu_w13_parts.append(expert_layer.gate_up_proj.weight.detach().cpu())
            cpu_w2_parts.append(expert_layer.down_proj.weight.detach().cpu())
            expert_layer.gate_up_proj._parameters.pop("weight", None)
            expert_layer.down_proj._parameters.pop("weight", None)

        self.experts = nn.ModuleDict()
        self.local_expert_layers = ()
        gc.collect()
        torch.npu.empty_cache()

        w13 = torch.empty((self.num_local_experts, self.hidden_size, 2 * self.moe_intermediate_size), dtype=dtype, device=device)
        w2 = torch.empty((self.num_local_experts, self.moe_intermediate_size, self.hidden_size), dtype=dtype, device=device)
        for local_idx, (w13_part, w2_part) in enumerate(zip(cpu_w13_parts, cpu_w2_parts)):
            w13[local_idx].copy_(w13_part.transpose(0, 1))
            w2[local_idx].copy_(w2_part.transpose(0, 1))

        self.grouped_w13_weight = nn.Parameter(w13, requires_grad=False)
        self.grouped_w2_weight = nn.Parameter(w2, requires_grad=False)
        del cpu_w13_parts, cpu_w2_parts
        gc.collect()

    def _grouped_topk(self, router_logits: torch.Tensor, weight_dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
        # Decode router logits are already BF16. Keeping that dtype avoids one
        # hot-path FP32 allocation before the fused NPU top-k op.
        router_logits = router_logits.contiguous()
        bias = getattr(self.gate, "e_score_correction_bias", None)
        if bias is not None and bias.dtype != router_logits.dtype:
            bias = bias.to(router_logits.dtype)
        norm_type = 1 if self.scoring_func == "sigmoid" else 0
        if self.scoring_func not in ("softmax", "sigmoid"):
            raise ValueError(f"Unsupported scoring function: {self.scoring_func}")

        topk_weights, topk_ids, _ = ascend_ops.moe_gating_top_k(
            router_logits,
            k=self.top_k,
            k_group=self.topk_group,
            group_count=self.num_expert_group,
            group_select_mode=1,
            renorm=1 if self.renormalize else 0,
            norm_type=norm_type,
            out_flag=False,
            routed_scaling_factor=self.routed_scaling_factor,
            eps=1e-20,
            bias_opt=bias,
        )
        topk_weights = topk_weights.to(weight_dtype)
        return topk_weights, topk_ids

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        sequence_length, hidden_dim = hidden_states.shape
        overlap_shared = hidden_states.device.type == "npu"
        if overlap_shared:
            default_stream = torch.npu.current_stream()
            shared_stream = _moe_shared_stream()
            shared_stream.wait_stream(default_stream)
            with torch.npu.stream(shared_stream):
                shared_output = self.shared_experts(hidden_states)        # Keep shared expert as one overlapped unit; finer split regressed TPOT.
        else:
            shared_output = self.shared_experts(hidden_states)
        router_logits = self.gate(hidden_states)
        routing_weights, selected_experts = self._grouped_topk(router_logits, hidden_states.dtype)
        if self.grouped_w13_weight is None or hidden_states.device.type != "npu":
            raise RuntimeError("Grouped MoE weights are not prepared. Call post_load_prepare() after loading weights on NPU.")
        # The loop expert backend was removed; all routed experts use grouped matmul now.
        routed_hidden_states = self._grouped_experts_forward(hidden_states, selected_experts, routing_weights)

        if overlap_shared:
            torch.npu.current_stream().wait_stream(shared_stream)          # shared_output must be ready before routed + shared accumulation.
        final_hidden_states = routed_hidden_states + shared_output
        if self.enable_expert_parallel and self.ep_size > 1:
            dist.all_reduce(final_hidden_states)
        return final_hidden_states.view(sequence_length, hidden_dim)

    def _grouped_experts_forward(
        self,
        hidden_states: torch.Tensor,
        selected_experts: torch.Tensor,
        routing_weights: torch.Tensor,
    ) -> torch.Tensor:
        if selected_experts.dtype != torch.int32:
            selected_experts = selected_experts.to(torch.int32)
        selected_experts = selected_experts.contiguous()
        local_mask = (selected_experts >= self.local_expert_start) & (selected_experts < self.local_expert_end)
        routing_weights = (routing_weights * local_mask.to(routing_weights.dtype)).contiguous()

        sorted_hidden, expanded_row_idx, expert_tokens, _ = (
            torch_npu.npu_moe_init_routing_v2(
                hidden_states,
                selected_experts,
                scale=None,
                active_num=hidden_states.shape[0] * self.top_k,
                expert_num=self.num_experts,
                expert_tokens_num_type=1,
                expert_tokens_num_flag=True,
                active_expert_range=[self.local_expert_start, self.local_expert_end],
                quant_mode=-1,
            )
        )
        gate_up = torch_npu.npu_grouped_matmul(
            x=[sorted_hidden],
            weight=[self.grouped_w13_weight],
            split_item=2,
            group_list_type=1,
            group_type=0,
            group_list=expert_tokens,
        )[0]
        hidden_states = torch_npu.npu_swiglu(gate_up)
        hidden_states = torch_npu.npu_grouped_matmul(
            x=[hidden_states],
            weight=[self.grouped_w2_weight],
            split_item=2,
            group_list_type=1,
            group_type=0,
            group_list=expert_tokens,
        )[0]
        return torch_npu.npu_moe_token_unpermute(
            permuted_tokens=hidden_states,
            sorted_indices=torch.abs(expanded_row_idx),
            probs=routing_weights,
        )


class DeepseekV32Indexer(nn.Module):
    def __init__(self, config: DeepseekV32Config) -> None:
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
    def _rope_cos_sin(self, positions: torch.Tensor, rotary_emb: DeepseekScalingRotaryEmbedding, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
        context = get_context()
        cache_key = ("indexer_rope_cos_sin", str(positions.device), dtype, self.rope_dim)
        cached = context.scratch.get(cache_key)
        if cached is not None:
            return cached

        positions = positions.to(torch.long)
        cos = rotary_emb.cos_cache.index_select(0, positions)
        sin = rotary_emb.sin_cache.index_select(0, positions)
        cos = torch.cat((cos, cos), dim=-1).to(dtype).contiguous()
        sin = torch.cat((sin, sin), dim=-1).to(dtype).contiguous()
        cos = cos.view(cos.shape[0], 1, 1, self.rope_dim)
        sin = sin.view(sin.shape[0], 1, 1, self.rope_dim)
        context.scratch[cache_key] = (cos, sin)
        return cos, sin

    def forward(self, hidden_states: torch.Tensor, q_c: torch.Tensor, positions: torch.Tensor, rotary_emb: DeepseekScalingRotaryEmbedding, detail: dict[str, float] | None = None, sync_detail: bool = False, query_only: bool = False) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]:
        cos, sin = self._rope_cos_sin(positions, rotary_emb, hidden_states.dtype)
        q_index, index_k, index_weights = self._get_output_buffers(hidden_states)
        if query_only:
            # Decode DSA only scores prefill candidates. The decode token key is already in the MLA tail budget, so skip index_k projection/cache.
            weights_proj_weight = self._query_only_weights_proj_weight(hidden_states.dtype, hidden_states.device)
            wq_b_bmm_t = self._query_only_wq_b_bmm_t(q_c.dtype, q_c.device)
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
                score_scale=1.0,  # vllm-ascend BF16 lightning_indexer consumes raw weights_proj(x).
                wq_b_bmm_t=wq_b_bmm_t,
                enable_q_bmm=wq_b_bmm_t is not None,
                detail=detail,
                sync_detail=sync_detail,
            )
            return q_index, None, index_weights

        # Final dsa_indexer_project interface writes these outputs explicitly; B-stage internals still use framework GEMMs plus AscendC post.
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
            score_scale=1.0,  # Keep lightning_indexer inputs aligned with vllm-ascend BF16 SFA.
            detail=detail,
            sync_detail=sync_detail,
        )
        return q_index, index_k, index_weights


class DeepseekV32DSAAttention(nn.Module):

    def __init__(self, config: DeepseekV32Config, layer_idx: int) -> None:
        super().__init__()
        tp_size = dist.get_world_size()
        if config.num_attention_heads % tp_size != 0:
            raise ValueError("num_attention_heads must be divisible by TP size.")
        self.hidden_size = int(config.hidden_size)
        self.total_num_heads = int(config.num_attention_heads)
        self.num_local_heads = self.total_num_heads // tp_size
        self.q_lora_rank = int(config.q_lora_rank)
        self.kv_lora_rank = int(config.kv_lora_rank)
        self.qk_nope_head_dim = int(config.qk_nope_head_dim)
        self.qk_rope_head_dim = int(config.qk_rope_head_dim)
        self.qk_head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim
        self.v_head_dim = int(config.v_head_dim)
        self.layer_id = layer_idx
        self.num_layers = int(config.num_hidden_layers)
        self.scale = self.qk_head_dim ** -0.5
        if config.rope_parameters.get("rope_type") == "deepseek_yarn":
            mscale = yarn_get_mscale(float(config.rope_parameters["factor"]), float(config.rope_parameters.get("mscale_all_dim", 0.0)))
            self.scale = self.scale * mscale * mscale

        self.q_a_proj = ReplicatedLinear(self.hidden_size, self.q_lora_rank, bias=False)
        self.q_a_layernorm = RMSNorm(self.q_lora_rank, eps=float(config.rms_norm_eps))
        self.q_b_proj = ColumnParallelLinear(self.q_lora_rank, self.total_num_heads * self.qk_head_dim, bias=False)
        self.kv_a_proj_with_mqa = ReplicatedLinear(self.hidden_size, self.kv_lora_rank + self.qk_rope_head_dim, bias=False)
        self.kv_a_layernorm = RMSNorm(self.kv_lora_rank, eps=float(config.rms_norm_eps))
        self.kv_b_proj = ColumnParallelLinear(self.kv_lora_rank, self.total_num_heads * (self.qk_nope_head_dim + self.v_head_dim), bias=False)
        self.o_proj = RowParallelLinear(self.total_num_heads * self.v_head_dim, self.hidden_size, bias=False)
        self._tp_hcomm_info = None
        self.rotary_emb = DeepseekScalingRotaryEmbedding(self.qk_rope_head_dim, max_position_embeddings=int(config.max_position_embeddings), rope_parameters=config.rope_parameters, is_neox_style=False)
        self.indexer_rotary_emb = DeepseekScalingRotaryEmbedding(self.qk_rope_head_dim, max_position_embeddings=int(config.max_position_embeddings), rope_parameters=config.rope_parameters, is_neox_style=not getattr(config, "indexer_rope_interleave", False))
        self.indexer = DeepseekV32Indexer(config)

        self.ckv_cache = torch.tensor([])
        self.kpe_cache = torch.tensor([])
        self.index_cache = torch.tensor([])
        self.dram_ckv_cache = torch.tensor([])
        self.dram_kpe_cache = torch.tensor([])
        self.gather_selection_status = torch.tensor([])
        self.gather_selection_topk = _DSA_GATHER_TOPK
        self.register_parameter("wd_qkv", None)
        self.w_uk_t = None
        self.w_uv = None
        self.mlapo_wd_qkv = None
        self.mlapo_wu_q = None
        self.mlapo_beta1 = None
        self.enable_decode_mlapo = _env_flag("NANOVLLM_ENABLE_DECODE_MLAPO", True)
        self.q_up_bmm_trans_max_tokens = _env_int("NANOVLLM_Q_UP_BMM_TRANS_MAX_TOKENS", 1)
        self.decode_timing_sync = _env_flag("NANOVLLM_DECODE_LAYER_TIMING_SYNC", True)
        self.log_decode_layer_timing = _env_flag("NANOVLLM_LOG_DECODE_LAYER_TIMING", False) and _profile_layer_selected(self.layer_id, self.num_layers)
        self.last_decode_attention_detail: dict[str, float] = {}
        self._decode_mlapo_ql_nope = None
        self._decode_mlapo_q_pe = None
        self._decode_mlapo_inner_out = None
        self._decode_mla_v2_out = None
        self._decode_mla_v2_lse = None
        self._reset_decode_attention_detail()

    def _reset_decode_attention_detail(self) -> None:
        self.last_decode_attention_detail = {
            "mlapo": 0.0,
            "indexer_project": 0.0,
            "dsa_pipeline": 0.0,
            "dsa_lightning_indexer": 0.0,
            "dsa_gather_selection": 0.0,
            "decode_attention_op": 0.0,
            "v_up": 0.0,
            "o_proj": 0.0,
        }

    def _decode_timer_start(self, profile_decode: bool, device) -> float | None:
        if not profile_decode:
            return None
        if self.decode_timing_sync:
            _profile_sync(device)
        return perf_counter()

    def _decode_timer_end(self, profile_decode: bool, name: str, start: float | None, device) -> None:
        if not profile_decode or start is None:
            return
        if self.decode_timing_sync:
            _profile_sync(device)
        self.last_decode_attention_detail[name] += perf_counter() - start

    def _o_proj_forward(self, attn_output: torch.Tensor, profile_decode: bool) -> torch.Tensor:
        if not profile_decode:
            return self.o_proj(attn_output)

        start = self._decode_timer_start(True, attn_output.device)
        if self.o_proj.disable_tp:
            output = F.linear(attn_output, self.o_proj.weight, self.o_proj.bias)
            self._decode_timer_end(True, "o_proj", start, output.device)
            return output

        output = F.linear(attn_output, self.o_proj.weight, self.o_proj.bias if self.o_proj.tp_rank == 0 else None)
        self._decode_timer_end(True, "o_proj", start, output.device)

        if self.o_proj.tp_size > 1 and self.o_proj.reduce_results:
            start = self._decode_timer_start(True, output.device)
            dist.all_reduce(output)
            self._decode_timer_end(True, "o_proj", start, output.device)
        return output

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
        gather_selection_status: torch.Tensor,
    ) -> None:
        self.ckv_cache = ckv_cache
        self.kpe_cache = kpe_cache
        self.index_cache = index_cache
        self.dram_ckv_cache = dram_ckv_cache
        self.dram_kpe_cache = dram_kpe_cache
        self.gather_selection_status = gather_selection_status
        self.gather_selection_topk = min(_DSA_GATHER_TOPK, int(gather_selection_status.shape[-1]) - 1)

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

        if self.enable_decode_mlapo:
            self._prepare_decode_mlapo()

    def _run_dsa_pipeline_with_qc_full_graph(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        batch_size: int,
    ) -> None:
        context = get_context()
        if not context.full_decode_graph or not context.dsa_offload_all_rows:
            raise RuntimeError(
                "The graph-visible DSA pipeline only supports an exact-size "
                "FULL_DECODE_ONLY batch in which every row is offloaded."
            )
        q_a_weight = None if self.wd_qkv is None else self.wd_qkv[: self.q_lora_rank]
        if q_a_weight is None:
            raise RuntimeError(
                "DSA FULL_DECODE_ONLY requires fused q_a weight, but wd_qkv "
                "is not prepared."
            )
        required_context = {
            "candidate_lens": context.candidate_lens,
            "req_pool_entries": context.req_pool_entries,
            "index_block_tables": context.index_block_tables,
            "candidate_query_lens": context.candidate_query_lens,
            "dram_block_tables": context.dram_block_tables,
            "selection_block_tables": context.selection_block_tables,
        }
        missing = [name for name, value in required_context.items() if value is None]
        if missing:
            raise RuntimeError(
                "DSA FULL_DECODE_ONLY context is missing: " + ", ".join(missing)
            )

        active_batch = int(batch_size)
        cos, sin = self.indexer._rope_cos_sin(
            positions,
            self.indexer_rotary_emb,
            hidden_states.dtype,
        )
        q_index, _, index_weights = self.indexer._get_output_buffers(hidden_states)
        dsa_indexer_pipeline_with_qc_full_graph(
            hidden_states,
            cos,
            sin,
            q_a_weight,
            self.q_a_layernorm.weight,
            self.indexer.wq_b.weight,
            self.indexer._query_only_weights_proj_weight(hidden_states.dtype, hidden_states.device),
            q_index,
            index_weights,
            self.index_cache,
            context.candidate_query_lens[:active_batch],
            context.candidate_lens[:active_batch],
            context.index_block_tables[:active_batch],
            self.kpe_cache.squeeze(2),
            self.ckv_cache.squeeze(2),
            context.selection_block_tables[:active_batch],
            self.gather_selection_status,
            context.req_pool_entries[:active_batch],
            self.dram_kpe_cache.squeeze(2),
            self.dram_ckv_cache.squeeze(2),
            context.dram_block_tables[:active_batch],
            q_norm_eps=float(self.q_a_layernorm.eps),
            n_head=self.indexer.n_head,
            head_dim=self.indexer.head_dim,
            rope_dim=self.indexer.rope_dim,
            score_scale=1.0,
            sparse_count=self.gather_selection_topk,
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

    def _decode_mlapo_buffers(self, num_tokens: int, dtype: torch.dtype, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        ql_shape = (num_tokens, self.num_local_heads, self.kv_lora_rank)
        qpe_shape = (num_tokens, self.num_local_heads, self.qk_rope_head_dim)
        if self._decode_mlapo_ql_nope is None or tuple(self._decode_mlapo_ql_nope.shape) != ql_shape or self._decode_mlapo_ql_nope.dtype != dtype or self._decode_mlapo_ql_nope.device != device:
            self._decode_mlapo_ql_nope = torch.empty(ql_shape, dtype=dtype, device=device)
        if self._decode_mlapo_q_pe is None or tuple(self._decode_mlapo_q_pe.shape) != qpe_shape or self._decode_mlapo_q_pe.dtype != dtype or self._decode_mlapo_q_pe.device != device:
            self._decode_mlapo_q_pe = torch.empty(qpe_shape, dtype=dtype, device=device)
        if self._decode_mlapo_inner_out is None or tuple(self._decode_mlapo_inner_out.shape) != (0,) or self._decode_mlapo_inner_out.dtype != dtype or self._decode_mlapo_inner_out.device != device:
            # The CANN enable_inner_out branch is not numerically aligned yet
            # (probe_mla_preprocess sees NaNs / wrong cache writes). Keep MLAPO
            # on the stable non-inner branch and recompute q_c separately when
            # DSA needs it.
            self._decode_mlapo_inner_out = torch.empty(0, dtype=dtype, device=device)
        return self._decode_mlapo_ql_nope, self._decode_mlapo_q_pe, self._decode_mlapo_inner_out

    # Temporary q_c fallback for DSA decode. Remove this once mla_preprocess
    # enable_inner_out=True is fixed and matches the torch reference.
    def _decode_mlapo_q_c_fallback(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.wd_qkv is None:
            raise RuntimeError("Fused qkv_a weight is not prepared.")
        q_c = F.linear(hidden_states, self.wd_qkv[: self.q_lora_rank])
        return self.q_a_layernorm(q_c)

    def _decode_mlapo_preprocess(self, positions: torch.Tensor, hidden_states: torch.Tensor, profile_decode: bool, need_inner_out: bool) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.mlapo_wd_qkv is None or self.mlapo_wu_q is None:
            self._prepare_decode_mlapo()
        if self.mlapo_wd_qkv is None or self.mlapo_wu_q is None:
            raise RuntimeError("Decode MLAPO weights are not prepared.")

        num_tokens = int(hidden_states.shape[0])
        ql_nope, q_pe, inner_out = self._decode_mlapo_buffers(num_tokens, dtype=hidden_states.dtype, device=hidden_states.device)
        cos, sin = self._mlapo_cos_sin(positions, hidden_states.dtype)
        context = get_context()
        slotmapping = context.flat_slot_mapping_i32 if context.flat_slot_mapping_i32 is not None else self._flat_slots().to(torch.int32)

        start = self._decode_timer_start(profile_decode, hidden_states.device)
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
            enable_inner_out=False,
        )
        if need_inner_out:
            inner_out = self._decode_mlapo_q_c_fallback(hidden_states)
        self._decode_timer_end(profile_decode, "mlapo", start, ql_nope.device)
        return ql_nope, q_pe, inner_out

    @property
    def block_size(self) -> int:
        if self.ckv_cache.dim() >= 4 and int(self.ckv_cache.shape[1]) == 1:
            return int(self.ckv_cache.shape[2])
        return int(self.ckv_cache.shape[1])

    def _flat_slots(self) -> torch.Tensor:
        context = get_context()
        if context.flat_slot_mapping is not None:
            return context.flat_slot_mapping
        slot_mapping = context.slot_mapping
        if slot_mapping.dim() == 2:
            return (
                slot_mapping[:, 0].to(torch.long) * self.block_size
                + slot_mapping[:, 1].to(torch.long)
            )
        return slot_mapping.to(torch.long)

    def finalize_prefill_offload(
        self,
        seq,
        old_hbm_block_table: list[int],
    ) -> None:
        num_full_blocks = int(seq.num_prefill_full_blocks)
        num_sparse_blocks = int(seq.num_sparse_blocks)
        self.gather_selection_status[int(seq.hbm_cached_tokens_pool_entry)].fill_(-1)  # Reset per-request selection state before decode.

        if num_sparse_blocks >= num_full_blocks:
            # Dense/short requests keep every full prefill block in HBM, so their
            # decode path stays aligned with baseline and no DRAM copy is needed.
            return

        # Long requests keep only the first prefix block plus suffix blocks in HBM.
        # Persist the full prefill KV to DRAM for future Tx>0 promote copies.
        for logical_block in range(num_full_blocks):
            hbm_block = int(old_hbm_block_table[logical_block])
            dram_block = int(seq.dram_block_table[logical_block])
            self.dram_ckv_cache[dram_block].copy_(self.ckv_cache[hbm_block].to(device=self.dram_ckv_cache.device, non_blocking=False))
            self.dram_kpe_cache[dram_block].copy_(self.kpe_cache[hbm_block].to(device=self.dram_kpe_cache.device, non_blocking=False))
        _profile_sync(self.ckv_cache.device)                    # Ensure DRAM source KV is ready before later decode promote copies.

    def _store_index_cache(self, index_k: torch.Tensor | None) -> None:
        if index_k is None:
            return
        context = get_context()
        index_slots = context.flat_index_slot_mapping if context.flat_index_slot_mapping is not None else self._flat_slots()
        self.index_cache.view(-1, self.indexer.head_dim).index_copy_(0, index_slots, index_k)

    def _run_indexer(
        self,
        hidden_states: torch.Tensor,
        q_c: torch.Tensor,
        positions: torch.Tensor,
        profile_decode: bool,
        query_only: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]:
        start = self._decode_timer_start(profile_decode, hidden_states.device)
        q_index, index_k, weights = self.indexer(
            hidden_states,
            q_c,
            positions,
            self.indexer_rotary_emb,
            None,
            False,
            query_only=query_only,
        )
        self._decode_timer_end(
            profile_decode,
            "indexer_project",
            start,
            hidden_states.device,
        )
        return q_index, index_k, weights

    def _q_nope_up_proj(self, q_nope: torch.Tensor) -> torch.Tensor:
        num_tokens = q_nope.shape[0]
        if q_nope.dtype in (torch.float16, torch.bfloat16) and self.q_up_bmm_trans_max_tokens > 0 and num_tokens == 1 and num_tokens <= self.q_up_bmm_trans_max_tokens:
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
        key = (batch_size, str(query.device), query.dtype, self.block_size, self.num_local_heads, self.kv_lora_rank)
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
    ) -> torch.Tensor:
        batch_size = int(ql_nope.shape[0])
        query = ql_nope.view(batch_size, self.num_local_heads, 1, self.kv_lora_rank).contiguous()
        query_rope = q_pe.view(batch_size, self.num_local_heads, 1, self.qk_rope_head_dim)
        # Nano stores paged MLA cache as [blocks, block_size, kv_heads, dim].
        # FIA v2 with BNSD_NBSD expects [blocks, kv_heads, block_size, dim].
        # DeepSeek MLA has kv_heads=1 here, so this is a metadata-only view,
        # not a big cache copy.
        key_cache = self.ckv_cache.view(-1, 1, self.block_size, self.kv_lora_rank)
        key_rope_cache = self.kpe_cache.view(
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
        actual_seq_lengths_key = cu_seqlens[1:] - cu_seqlens[:-1]
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
            actual_seq_lengths_kv=actual_seq_lengths_key.detach().cpu().tolist(),
        )
        latent = mla_result[0] if isinstance(mla_result, (tuple, list)) else mla_result
        return torch_npu.npu_transpose_batchmatmul(latent.transpose(0, 1).contiguous(), self.w_uv, perm_y=(1, 0, 2)).reshape(latent.shape[0], -1)

    def _dsa_offload_update(self, q_index: torch.Tensor, weights: torch.Tensor, batch_size: int) -> None:
        context = get_context()
        if not context.needs_dsa_update:
            return
        required_context = {
            "candidate_lens": context.candidate_lens,
            "req_pool_entries": context.req_pool_entries,
            "index_block_tables": context.index_block_tables,
            "candidate_query_lens": context.candidate_query_lens,
            "dram_block_tables": context.dram_block_tables,
            "selection_block_tables": context.selection_block_tables,
        }
        missing = [name for name, value in required_context.items() if value is None]
        if missing:
            raise RuntimeError("DSA offload context is missing: " + ", ".join(missing))

        if context.dsa_offload_all_rows:
            active_batch = batch_size
            q_index_active = q_index[:batch_size]
            weights_active = weights[:batch_size]
            index_tables = context.index_block_tables[:batch_size]
            dram_tables = context.dram_block_tables[:batch_size]
            selection_block_table = context.selection_block_tables[:batch_size]
            candidate_lens = context.candidate_lens[:batch_size]
            req_pool_entries = context.req_pool_entries[:batch_size]
        else:
            rows = context.dsa_offload_rows
            if rows is None or rows.numel() == 0:
                return
            active_batch = int(rows.numel())
            q_index_active = q_index.index_select(0, rows)
            weights_active = weights.index_select(0, rows)
            index_tables = context.index_block_tables.index_select(0, rows)
            dram_tables = context.dram_block_tables.index_select(0, rows)
            selection_block_table = context.selection_block_tables.index_select(0, rows)
            candidate_lens = context.candidate_lens.index_select(0, rows)
            req_pool_entries = context.req_pool_entries.index_select(0, rows)
        candidate_query_lens = context.candidate_query_lens[:active_batch]

        profile_decode = self.log_decode_layer_timing
        start = self._decode_timer_start(profile_decode, q_index.device)
        topk_indices = ascend_ops.npu_lightning_indexer(
            query=q_index_active,
            key=self.index_cache,
            weights=weights_active,
            actual_seq_lengths_query=candidate_query_lens,
            actual_seq_lengths_key=candidate_lens,
            block_table=index_tables,
            layout_query="TND",
            layout_key="PA_BSND",
            sparse_count=self.gather_selection_topk,
            sparse_mode=3,
        )
        self._decode_timer_end(profile_decode, "dsa_lightning_indexer", start, topk_indices.device)

        start = self._decode_timer_start(profile_decode, self.ckv_cache.device)
        # gather_selection_status is a request pool; req_pool_entries maps active rows to persistent status rows.
        # selection_block_table is per-batch because it is rebuilt in prepare_decode like hbm_block_tables.
        ascend_ops.npu_gather_selection_kv_cache(
            self.kpe_cache.squeeze(2),
            self.ckv_cache.squeeze(2),
            selection_block_table,
            self.gather_selection_status,
            req_pool_entries,
            topk_indices.view(active_batch, 1, 1, self.gather_selection_topk),
            self.dram_kpe_cache.squeeze(2),
            self.dram_ckv_cache.squeeze(2),
            dram_tables,
            candidate_lens,
        )
        self._decode_timer_end(profile_decode, "dsa_gather_selection", start, self.ckv_cache.device)

    def _decode_forward_mla(
        self,
        ql_nope: torch.Tensor,
        q_pe: torch.Tensor,
        q_index: torch.Tensor | None,
        weights: torch.Tensor | None,
        dsa_updated: bool = False,
    ) -> torch.Tensor:
        context = get_context()
        batch_size = int(ql_nope.shape[0])
        assert context.actual_seq_lengths_kv is not None
        profile_decode = self.log_decode_layer_timing
        if context.needs_dsa_update and not dsa_updated:
            if q_index is None or weights is None:
                raise RuntimeError("DSA offload decode requires indexer outputs.")
            self._dsa_offload_update(q_index, weights, batch_size)
        start = self._decode_timer_start(profile_decode, ql_nope.device)
        latent = self._decode_forward_mla_v2(
            ql_nope,
            q_pe,
            context.block_tables[:batch_size],
            context.actual_seq_lengths_kv,
        )
        self._decode_timer_end(profile_decode, "decode_attention_op", start, ql_nope.device)
        start = self._decode_timer_start(profile_decode, latent.device)
        output = torch_npu.npu_transpose_batchmatmul(latent.view(self.num_local_heads, batch_size, self.kv_lora_rank), self.w_uv, perm_y=(1, 0, 2)).reshape(batch_size, -1)
        self._decode_timer_end(profile_decode, "v_up", start, output.device)
        return output

    def forward(self, positions: torch.Tensor, hidden_states: torch.Tensor, skip_o_proj: bool = False) -> torch.Tensor:
        context = get_context()
        profile_decode = self.log_decode_layer_timing and not context.is_prefill
        if profile_decode:
            self._reset_decode_attention_detail()

        if self.w_uk_t is None or self.w_uv is None:
            self.post_load_prepare()

        if self.wd_qkv is None:
            self.post_load_prepare()
        if self.wd_qkv is None:
            raise RuntimeError("Fused qkv_a weight is not prepared.")

        use_decode_mlapo = self.enable_decode_mlapo and not context.is_prefill and not context.has_first_decode
        needs_decode_dsa_update = bool(context.needs_dsa_update)
        if use_decode_mlapo:
            use_full_graph_dsa = needs_decode_dsa_update and context.full_decode_graph
            ql_nope, q_pe, q_c = self._decode_mlapo_preprocess(
                positions,
                hidden_states,
                profile_decode,
                need_inner_out=needs_decode_dsa_update and not use_full_graph_dsa,
            )
            q_index = index_k = weights = None
            dsa_updated = False
            if needs_decode_dsa_update:
                if use_full_graph_dsa:
                    self._run_dsa_pipeline_with_qc_full_graph(
                        hidden_states,
                        positions,
                        int(hidden_states.shape[0]),
                    )
                    dsa_updated = True
                else:
                    if q_c is None or q_c.numel() == 0:
                        q_c = self._decode_mlapo_q_c_fallback(hidden_states)
                    q_index, index_k, weights = self._run_indexer(hidden_states, q_c, positions, profile_decode, query_only=True)
            if index_k is not None:
                self._store_index_cache(index_k)

            attn_output = self._decode_forward_mla(
                ql_nope,
                q_pe,
                q_index,
                weights,
                dsa_updated=dsa_updated,
            )
            return attn_output if skip_o_proj else self._o_proj_forward(attn_output, profile_decode)

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
            q_index, index_k, weights = self._run_indexer(hidden_states, q_c, positions, profile_decode, query_only=not context.is_prefill)

        flat_slots = self._flat_slots()
        self.ckv_cache.view(-1, self.kv_lora_rank).index_copy_(0, flat_slots, ckv)
        self.kpe_cache.view(-1, self.qk_rope_head_dim).index_copy_(0, flat_slots, k_pe)

        if index_k is not None:
            self._store_index_cache(index_k)

        ql_nope = self._q_nope_up_proj(q_nope)

        if context.is_prefill:
            attn_output = self._prefill_forward_npu_mla(ql_nope, q_pe)
        else:
            attn_output = self._decode_forward_mla(ql_nope, q_pe, q_index, weights)

        return attn_output if skip_o_proj else self._o_proj_forward(attn_output, profile_decode)


class DeepseekV32DecoderLayer(nn.Module):
    def __init__(self, config: DeepseekV32Config, layer_idx: int) -> None:
        super().__init__()
        self.layer_idx = int(layer_idx)
        self.self_attn = DeepseekV32DSAAttention(config, layer_idx)
        is_shared_only, keep_routed_experts = _resolve_export_mode(config)
        if layer_idx >= int(config.first_k_dense_replace) and is_shared_only:
            self.mlp = DeepseekV32MLP(hidden_size=int(config.hidden_size), intermediate_size=int(config.moe_intermediate_size) * int(getattr(config, "n_shared_experts", 1) or 1), hidden_act=str(config.hidden_act))
        elif layer_idx < int(config.first_k_dense_replace):
            self.mlp = DeepseekV32MLP(hidden_size=int(config.hidden_size), intermediate_size=int(config.intermediate_size), hidden_act=str(config.hidden_act))
        elif keep_routed_experts:
            self.mlp = DeepseekV32SparseMoeBlock(config, layer_idx)
        else:
            raise ValueError("DeepSeek-V3.2 in nano-vllm-ascend currently expects either the shared-only export or the keep-routed-experts export.")
        self.input_layernorm = RMSNorm(int(config.hidden_size), eps=float(config.rms_norm_eps))
        self.post_attention_layernorm = RMSNorm(int(config.hidden_size), eps=float(config.rms_norm_eps))
        self.log_decode_layer_timing = _env_flag("NANOVLLM_LOG_DECODE_LAYER_TIMING", False) and _profile_layer_selected(self.layer_idx, int(config.num_hidden_layers))
        self.decode_timing_sync = _env_flag("NANOVLLM_DECODE_LAYER_TIMING_SYNC", True)

    def forward(self, positions: torch.Tensor, hidden_states: torch.Tensor, residual: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        context = get_context()
        profile_decode = self.log_decode_layer_timing and not context.is_prefill
        sync_decode = profile_decode and self.decode_timing_sync
        fuse_o_proj_norm = (not context.is_prefill) and self.self_attn.can_fuse_o_proj_add_rms_norm()
        if profile_decode:
            if sync_decode:
                _profile_sync(hidden_states.device)
            attention_start = perf_counter()
        hidden_states = self.self_attn(positions, hidden_states, skip_o_proj=fuse_o_proj_norm)
        if fuse_o_proj_norm:
            start = self.self_attn._decode_timer_start(profile_decode, hidden_states.device)
            hidden_states, residual = self.self_attn.o_proj_add_rms_norm(hidden_states, residual, self.post_attention_layernorm)
            self.self_attn._decode_timer_end(profile_decode, "o_proj", start, hidden_states.device)
        else:
            if profile_decode and sync_decode:
                _profile_sync(hidden_states.device)
            if profile_decode:
                attention_total = perf_counter() - attention_start
            hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        if profile_decode:
            if sync_decode:
                _profile_sync(hidden_states.device)
            if fuse_o_proj_norm:
                attention_total = perf_counter() - attention_start
            mlp_start = perf_counter()
        hidden_states = self.mlp(hidden_states)
        if profile_decode:
            if sync_decode:
                _profile_sync(hidden_states.device)
            attention_detail = self.self_attn.last_decode_attention_detail
            dsa_total = attention_detail["dsa_pipeline"] + attention_detail["dsa_lightning_indexer"] + attention_detail["dsa_gather_selection"]
            logger.info(
                "Decode layer timing: rank=%d layer=%d tokens=%d "
                "attention_total=%.6fs mlapo=%.6fs indexer_project=%.6fs "
                "dsa_total=%.6fs dsa_pipeline=%.6fs dsa_lightning_indexer=%.6fs dsa_gather_selection=%.6fs "
                "decode_attention_op=%.6fs v_up=%.6fs o_proj=%.6fs moe_total=%.6fs",
                _rank_id(),
                self.layer_idx,
                int(hidden_states.shape[0]),
                attention_total,
                attention_detail["mlapo"],
                attention_detail["indexer_project"],
                dsa_total,
                attention_detail["dsa_pipeline"],
                attention_detail["dsa_lightning_indexer"],
                attention_detail["dsa_gather_selection"],
                attention_detail["decode_attention_op"],
                attention_detail["v_up"],
                attention_detail["o_proj"],
                perf_counter() - mlp_start,
            )
        return hidden_states, residual


class DeepseekV32Model(nn.Module):
    def __init__(self, config: DeepseekV32Config) -> None:
        super().__init__()
        self.config = config
        self.embed_tokens = VocabParallelEmbedding(
            int(config.vocab_size),
            int(config.hidden_size),
        )
        self.layers = nn.ModuleList(
            [
                DeepseekV32DecoderLayer(config, layer_idx)
                for layer_idx in range(int(config.num_hidden_layers))
            ]
        )
        self.norm = RMSNorm(
            int(config.hidden_size),
            eps=float(config.rms_norm_eps),
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        hidden_states = self.embed_tokens(input_ids)
        residual = None
        for layer in self.layers:
            hidden_states, residual = layer(positions, hidden_states, residual)
        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states

    def post_load_prepare(self) -> None:
        for layer in self.layers:
            layer.self_attn.post_load_prepare()
            if isinstance(layer.mlp, DeepseekV32SparseMoeBlock):
                layer.mlp.post_load_prepare()


class DeepseekV32ForCausalLM(nn.Module):
    packed_modules_mapping = {
        "gate_proj": ("gate_up_proj", 0),
        "up_proj": ("gate_up_proj", 1),
    }

    def __init__(self, config: DeepseekV32Config) -> None:
        super().__init__()
        is_shared_only, keep_routed_experts = _resolve_export_mode(config)
        if not (is_shared_only or keep_routed_experts):
            raise ValueError(
                "DeepSeek-V3.2 support in nano-vllm-ascend currently expects "
                "either a shared-only export or a routed-expert BF16 model "
                "whose config keeps n_routed_experts > 0."
            )
        self.model = DeepseekV32Model(config)
        self.lm_head = ParallelLMHead(
            int(config.vocab_size),
            int(config.hidden_size),
        )
        if getattr(config, "tie_word_embeddings", False):
            self.lm_head.weight.data = self.model.embed_tokens.weight.data

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        return self.model(input_ids, positions)

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        return self.lm_head(hidden_states)

    def post_load_prepare(self) -> None:
        self.model.post_load_prepare()

    def weight_name_mapping(self, weight_name: str) -> str | None:
        if ".mlp.experts." not in weight_name:
            return weight_name
        parts = weight_name.split(".")
        try:
            layer_idx = int(parts[2])
            expert_idx = int(parts[5])
        except (IndexError, ValueError):
            return weight_name
        layer = self.model.layers[layer_idx].mlp
        if not isinstance(layer, DeepseekV32SparseMoeBlock):
            return weight_name
        if expert_idx in layer.local_expert_id_set:
            return weight_name
        return None
