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
from nanovllm.models.dsa_offload_ops import (
    dsa_index_update,
    dsa_indexer_score,
    dsa_scatter_h2d,
)
from nanovllm.models.dsa_indexer_project import dsa_indexer_project, dsa_indexer_project_post_available, dsa_indexer_project_query_only
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
_DSA_OFFLOAD_BUFFER_CACHE = {}
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


def _first_tensor(value):
    if isinstance(value, (tuple, list)):
        return value[0]
    return value


def _rank_id() -> int:
    try:
        return dist.get_rank() if dist.is_initialized() else 0
    except Exception:
        return 0


def _ranked_debug_dump_path(path: str, rank: int) -> str:
    path = path.strip()
    if not path:
        return ""
    root, ext = os.path.splitext(path)
    actual_path = f"{root}_rank{rank}{ext}" if ext else f"{path}_rank{rank}.txt"
    parent = os.path.dirname(actual_path)
    if parent and not os.path.isdir(parent):
        return ""
    return actual_path


def _profile_sync(device: torch.device) -> None:
    if device.type == "npu":
        torch.npu.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize(device)


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
        self.architectures = kwargs.get(
            "architectures", ["DeepseekV32ForCausalLM"]
        )
        self.nanovllm_pruned_shared_only = kwargs.get(
            "nanovllm_pruned_shared_only", False
        )
        routed_experts = int(kwargs.get("n_routed_experts", 0) or 0)
        inferred_keep_routed = (
            routed_experts > 0 and not self.nanovllm_pruned_shared_only
        )
        keep_routed_flag = kwargs.get("nanovllm_pruned_keep_routed_experts")
        if keep_routed_flag is None:
            keep_routed_flag = inferred_keep_routed
        self.nanovllm_pruned_keep_routed_experts = bool(keep_routed_flag)
        self.nanovllm_export_format = kwargs.get(
            "nanovllm_export_format", ""
        )
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
            for key in ("factor", "beta_fast", "beta_slow", "mscale",
                        "mscale_all_dim", "original_max_position_embeddings"):
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
    def __init__(
        self,
        rotary_dim: int,
        max_position_embeddings: int,
        rope_parameters: dict,
        *,
        is_neox_style: bool,
    ) -> None:
        super().__init__()
        self.rotary_dim = rotary_dim
        self.is_neox_style = is_neox_style
        base = float(rope_parameters.get("rope_theta", 10000.0))
        rope_type = rope_parameters.get("rope_type", "default")
        cache_len = max_position_embeddings
        mscale = 1.0

        if rope_type == "deepseek_yarn":
            scaling_factor = float(rope_parameters["factor"])
            original_max_position = int(
                rope_parameters["original_max_position_embeddings"]
            )
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
            mscale = yarn_get_mscale(
                scaling_factor,
                float(rope_parameters.get("mscale_all_dim", 0.0)),
            )
            cache_len = int(original_max_position * scaling_factor)
        else:
            inv_freq = 1.0 / (
                base
                ** (
                    torch.arange(0, rotary_dim, 2, dtype=torch.float32)
                    / rotary_dim
                )
            )

        positions = torch.arange(cache_len, dtype=torch.float32)
        freqs = torch.einsum("i,j->ij", positions, inv_freq)
        self.register_buffer(
            "cos_cache",
            freqs.cos() * mscale,
            persistent=False,
        )
        self.register_buffer(
            "sin_cache",
            freqs.sin() * mscale,
            persistent=False,
        )

    @staticmethod
    def _yarn_linear_ramp_mask(
        low: float,
        high: float,
        dim: int,
    ) -> torch.Tensor:
        if low == high:
            high += 1e-3
        positions = torch.arange(dim, dtype=torch.float32)
        mask = (positions - low) / (high - low)
        return mask.clamp_(0.0, 1.0)

    @staticmethod
    def _yarn_find_correction_dim(
        num_rotations: float,
        dim: int,
        base: float,
        max_position_embeddings: int,
    ) -> float:
        return (
            dim
            * math.log(max_position_embeddings / (num_rotations * 2 * math.pi))
            / (2 * math.log(base))
        )

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
        pos_freqs = base ** (
            torch.arange(0, rotary_dim, 2, dtype=torch.float32) / rotary_dim
        )
        inv_freq_extrapolation = 1.0 / pos_freqs
        inv_freq_interpolation = 1.0 / (scaling_factor * pos_freqs)

        low = math.floor(
            cls._yarn_find_correction_dim(
                beta_fast, rotary_dim, base, original_max_position
            )
        )
        high = math.ceil(
            cls._yarn_find_correction_dim(
                beta_slow, rotary_dim, base, original_max_position
            )
        )
        low = max(low, 0)
        high = min(high, rotary_dim // 2 - 1)
        inv_freq_mask = 1.0 - cls._yarn_linear_ramp_mask(
            low, high, rotary_dim // 2
        )
        return (
            inv_freq_interpolation * (1.0 - inv_freq_mask)
            + inv_freq_extrapolation * inv_freq_mask
        )

    def forward(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
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
    is_shared_only = bool(
        getattr(config, "nanovllm_pruned_shared_only", False)
    )
    keep_routed_flag = getattr(
        config,
        "nanovllm_pruned_keep_routed_experts",
        None,
    )
    routed_experts = int(getattr(config, "n_routed_experts", 0) or 0)

    if keep_routed_flag is None:
        keep_routed_experts = routed_experts > 0 and not is_shared_only
        if routed_experts == 0:
            is_shared_only = True
    else:
        keep_routed_experts = bool(keep_routed_flag)

    return is_shared_only, keep_routed_experts


class DeepseekV32MLP(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        *,
        disable_tp: bool = False,
        reduce_results: bool = True,
    ) -> None:
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size,
            [intermediate_size, intermediate_size],
            bias=False,
            disable_tp=disable_tp,
        )
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=False,
            disable_tp=disable_tp,
            reduce_results=reduce_results,
        )
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
        self.routed_scaling_factor = float(
            getattr(config, "routed_scaling_factor", 1.0)
        )
        self.num_expert_group = max(1, int(getattr(config, "n_group", 1) or 1))
        self.topk_group = max(1, int(getattr(config, "topk_group", 1) or 1))
        self.num_shared_experts = int(getattr(config, "n_shared_experts", 1) or 1)
        self.enable_expert_parallel = bool(
            getattr(config, "nanovllm_enable_expert_parallel", False)
        )
        self.ep_size = dist.get_world_size() if self.enable_expert_parallel else 1
        self.ep_rank = dist.get_rank() if self.enable_expert_parallel else 0
        if self.enable_expert_parallel and self.num_experts % self.ep_size != 0:
            raise ValueError(
                "DeepSeek-V3.2 expert_parallel requires n_routed_experts to "
                "be divisible by the EP world size."
            )
        self.num_local_experts = (
            self.num_experts // self.ep_size
            if self.enable_expert_parallel
            else self.num_experts
        )
        self.local_expert_start = self.ep_rank * self.num_local_experts
        self.local_expert_end = self.local_expert_start + self.num_local_experts
        self.local_expert_ids = tuple(
            range(self.local_expert_start, self.local_expert_end)
        )
        self.local_expert_id_set = set(self.local_expert_ids)

        self.gate = ReplicatedLinear(self.hidden_size, self.num_experts, bias=False)
        if getattr(config, "topk_method", None) == "noaux_tc":
            self.gate.e_score_correction_bias = nn.Parameter(
                torch.empty(self.num_experts, dtype=torch.float32)
            )
        else:
            self.gate.register_parameter("e_score_correction_bias", None)

        self.shared_experts = DeepseekV32MLP(
            hidden_size=self.hidden_size,
            intermediate_size=self.moe_intermediate_size * self.num_shared_experts,
            hidden_act=self.hidden_act,
            reduce_results=not (self.enable_expert_parallel and self.ep_size > 1),
        )
        self.experts = nn.ModuleDict(
            {
                str(expert_idx): DeepseekV32MLP(
                    hidden_size=self.hidden_size,
                    intermediate_size=self.moe_intermediate_size,
                    hidden_act=self.hidden_act,
                    disable_tp=self.enable_expert_parallel,
                )
                for expert_idx in self.local_expert_ids
            }
        )
        self.local_expert_layers = tuple(
            self.experts[str(expert_idx)] for expert_idx in self.local_expert_ids
        )
        self.register_parameter("grouped_w13_weight", None)
        self.register_parameter("grouped_w2_weight", None)
        self.moe_backend = os.environ.get("NANOVLLM_MOE_BACKEND", "grouped").strip().lower()
        if self.moe_backend not in ("grouped", "loop"):
            raise ValueError("NANOVLLM_MOE_BACKEND must be 'grouped' or 'loop'.")

    def post_load_prepare(self) -> None:
        if self.grouped_w13_weight is not None:
            return
        if self.moe_backend != "grouped":
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
            cpu_w13_parts.append(
                expert_layer.gate_up_proj.weight.detach().cpu()
            )
            cpu_w2_parts.append(
                expert_layer.down_proj.weight.detach().cpu()
            )
            expert_layer.gate_up_proj._parameters.pop("weight", None)
            expert_layer.down_proj._parameters.pop("weight", None)

        self.experts = nn.ModuleDict()
        self.local_expert_layers = ()
        gc.collect()
        torch.npu.empty_cache()

        w13 = torch.empty(
            (
                self.num_local_experts,
                self.hidden_size,
                2 * self.moe_intermediate_size,
            ),
            dtype=dtype,
            device=device,
        )
        w2 = torch.empty(
            (
                self.num_local_experts,
                self.moe_intermediate_size,
                self.hidden_size,
            ),
            dtype=dtype,
            device=device,
        )
        for local_idx, (w13_part, w2_part) in enumerate(
            zip(cpu_w13_parts, cpu_w2_parts)
        ):
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
        if (
            self.moe_backend == "grouped"
            and self.grouped_w13_weight is not None
            and hidden_states.device.type == "npu"
        ):
            routed_hidden_states = self._grouped_experts_forward(
                hidden_states,
                selected_experts,
                routing_weights,
            )
        else:
            routed_hidden_states = self._loop_experts_forward(
                hidden_states,
                selected_experts,
                routing_weights,
            )

        if overlap_shared:
            torch.npu.current_stream().wait_stream(shared_stream)          # shared_output must be ready before routed + shared accumulation.
        final_hidden_states = routed_hidden_states + shared_output
        if self.enable_expert_parallel and self.ep_size > 1:
            dist.all_reduce(final_hidden_states)
        return final_hidden_states.view(sequence_length, hidden_dim)

    def _loop_experts_forward(
        self,
        hidden_states: torch.Tensor,
        selected_experts: torch.Tensor,
        routing_weights: torch.Tensor,
    ) -> torch.Tensor:
        hidden_dim = hidden_states.shape[-1]
        routing_weights = routing_weights.to(hidden_states.dtype)
        routed_hidden_states = torch.zeros(
            hidden_states.shape,
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )

        for expert_idx, expert_layer in zip(
            self.local_expert_ids,
            self.local_expert_layers,
        ):
            top_x, idx = torch.where(selected_experts == expert_idx)
            if top_x.numel() == 0:
                continue
            current_state = hidden_states[top_x].reshape(-1, hidden_dim)
            current_hidden_states = (
                expert_layer(current_state)
                * routing_weights[top_x, idx, None]
            )
            routed_hidden_states.index_add_(
                0,
                top_x,
                current_hidden_states.to(hidden_states.dtype),
            )
        return routed_hidden_states

    def _grouped_experts_forward(
        self,
        hidden_states: torch.Tensor,
        selected_experts: torch.Tensor,
        routing_weights: torch.Tensor,
    ) -> torch.Tensor:
        if selected_experts.dtype != torch.int32:
            selected_experts = selected_experts.to(torch.int32)
        selected_experts = selected_experts.contiguous()
        local_mask = (
            (selected_experts >= self.local_expert_start)
            & (selected_experts < self.local_expert_end)
        )
        routing_weights = (
            routing_weights * local_mask.to(routing_weights.dtype)
        ).contiguous()

        sorted_hidden, expanded_row_idx, expert_tokens, _ = (
            torch_npu.npu_moe_init_routing_v2(
                hidden_states,
                selected_experts,
                scale=None,
                active_num=hidden_states.shape[0] * self.top_k,
                expert_num=self.num_experts,
                expert_tokens_num_type=1,
                expert_tokens_num_flag=True,
                active_expert_range=[
                    self.local_expert_start,
                    self.local_expert_end,
                ],
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
        self.softmax_scale = self.head_dim ** -0.5

        self.wq_b = ReplicatedLinear(self.q_lora_rank, self.n_head * self.head_dim, bias=False)
        self.wk = ReplicatedLinear(self.hidden_size, self.head_dim, bias=False)
        self.k_norm = nn.LayerNorm(self.head_dim, eps=1e-6)
        self.weights_proj = ReplicatedLinear(self.hidden_size, self.n_head, bias=False).to(torch.float32)
        self.last_q_project_path = "linear"
        self._output_buffer_key = None
        self._output_buffers = None

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

    # Cache per-forward cos/sin tensors in context.scratch. All layers share the
    # same positions, so this removes repeated tiny H2D/index_select overhead.
    def _rope_cos_sin(
        self,
        positions: torch.Tensor,
        rotary_emb: DeepseekScalingRotaryEmbedding,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
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

    def forward(
        self,
        hidden_states: torch.Tensor,
        q_c: torch.Tensor,
        positions: torch.Tensor,
        rotary_emb: DeepseekScalingRotaryEmbedding,
        detail: dict[str, float] | None = None,
        sync_detail: bool = False,
        query_only: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]:
        cos, sin = self._rope_cos_sin(positions, rotary_emb, hidden_states.dtype)
        q_index, index_k, index_weights = self._get_output_buffers(hidden_states)
        if query_only:
            self.last_q_project_path = "query_only"
            # Decode DSA only scores prefill candidates. The decode token key is
            # already in the MLA tail budget, so skip index_k projection/cache.
            dsa_indexer_project_query_only(
                hidden_states,
                q_c,
                cos,
                sin,
                self.wq_b.weight,
                self.weights_proj.weight,
                q_index,
                index_weights,
                n_head=self.n_head,
                head_dim=self.head_dim,
                rope_dim=self.rope_dim,
                score_scale=self.softmax_scale * (self.n_head ** -0.5),
                detail=detail,
                sync_detail=sync_detail,
            )
            return q_index, None, index_weights

        self.last_q_project_path = "linear+ascendc_post" if hidden_states.device.type == "npu" and dsa_indexer_project_post_available() else "linear"
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
            score_scale=self.softmax_scale * (self.n_head ** -0.5),
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
            mscale = yarn_get_mscale(
                float(config.rope_parameters["factor"]),
                float(config.rope_parameters.get("mscale_all_dim", 0.0)),
            )
            self.scale = self.scale * mscale * mscale

        self.q_a_proj = ReplicatedLinear(
            self.hidden_size,
            self.q_lora_rank,
            bias=False,
        )
        self.q_a_layernorm = RMSNorm(
            self.q_lora_rank,
            eps=float(config.rms_norm_eps),
        )
        self.q_b_proj = ColumnParallelLinear(
            self.q_lora_rank,
            self.total_num_heads * self.qk_head_dim,
            bias=False,
        )
        self.kv_a_proj_with_mqa = ReplicatedLinear(
            self.hidden_size,
            self.kv_lora_rank + self.qk_rope_head_dim,
            bias=False,
        )
        self.kv_a_layernorm = RMSNorm(
            self.kv_lora_rank,
            eps=float(config.rms_norm_eps),
        )
        self.kv_b_proj = ColumnParallelLinear(
            self.kv_lora_rank,
            self.total_num_heads * (self.qk_nope_head_dim + self.v_head_dim),
            bias=False,
        )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.v_head_dim,
            self.hidden_size,
            bias=False,
        )
        self.rotary_emb = DeepseekScalingRotaryEmbedding(
            self.qk_rope_head_dim,
            max_position_embeddings=int(config.max_position_embeddings),
            rope_parameters=config.rope_parameters,
            is_neox_style=False,
        )
        self.indexer_rotary_emb = DeepseekScalingRotaryEmbedding(
            self.qk_rope_head_dim,
            max_position_embeddings=int(config.max_position_embeddings),
            rope_parameters=config.rope_parameters,
            is_neox_style=not getattr(
                config, "indexer_rope_interleave", False
            ),
        )
        self.indexer = DeepseekV32Indexer(config)

        self.ckv_cache = torch.tensor([])
        self.kpe_cache = torch.tensor([])
        self.index_cache = torch.tensor([])
        self.dram_ckv_cache = torch.tensor([])
        self.dram_kpe_cache = torch.tensor([])
        self.hbm_cached_tokens_pool = torch.tensor([])
        self.dsa_offload_max_copy_tokens = 2048
        self.dsa_offload_fixed_tx = 64
        self.register_parameter("wd_qkv", None)
        self.w_uk_t = None
        self.w_uv = None
        self.mlapo_wd_qkv = None
        self.mlapo_wu_q = None
        self.mlapo_beta1 = None
        self.fuse_qkv_a = _env_flag("NANOVLLM_FUSE_QKV_A", True)
        self.free_kv_b_proj = _env_flag("NANOVLLM_FREE_KV_B_PROJ", True)
        self.enable_decode_mlapo = _env_flag(
            "NANOVLLM_ENABLE_DECODE_MLAPO",
            True,
        )
        self.q_up_bmm_trans_max_tokens = _env_int(
            "NANOVLLM_Q_UP_BMM_TRANS_MAX_TOKENS",
            1,
        )
        self.decode_timing_sync = _env_flag(
            "NANOVLLM_DECODE_LAYER_TIMING_SYNC",
            True,
        )
        self.debug_decode_mlapo_compare = _env_flag("NANOVLLM_DEBUG_DECODE_MLAPO_COMPARE", False)
        self.debug_decode_mlapo_compare_rank = _env_int("NANOVLLM_DEBUG_DECODE_MLAPO_COMPARE_RANK", 0)
        self.debug_decode_mlapo_compare_limit = _env_int("NANOVLLM_DEBUG_DECODE_MLAPO_COMPARE_LIMIT", 2)
        self._debug_decode_mlapo_compare_count = 0
        self.debug_decode_metadata = _env_flag("NANOVLLM_DSA_DEBUG_DECODE_METADATA", False)
        self.debug_decode_metadata_rank = _env_int("NANOVLLM_DSA_DEBUG_DECODE_METADATA_RANK", 0)
        self.debug_decode_metadata_sample = max(1, _env_int("NANOVLLM_DSA_DEBUG_DECODE_METADATA_SAMPLE", 8))
        self.debug_index_update_dump_file = _ranked_debug_dump_path(
            os.environ.get("NANOVLLM_DSA_DEBUG_INDEX_UPDATE_DUMP_PATH", ""),
            _rank_id(),
        )
        self._debug_index_update_dump_count = 0
        self.log_decode_layer_timing = _env_flag(
            "NANOVLLM_LOG_DECODE_LAYER_TIMING",
            False,
        ) and _profile_layer_selected(self.layer_id, self.num_layers)
        self.last_decode_attention_op_time = 0.0
        self.last_decode_attention_detail: dict[str, float] = {}
        self.last_decode_indexer_detail: dict[str, float] = {}
        self.last_decode_indexer_q_path = "unprepared"
        self._decode_mlapo_ql_nope = None
        self._decode_mlapo_q_pe = None
        self._decode_mlapo_inner_out = None
        self._decode_mla_v2_out = None
        self._decode_mla_v2_lse = None
        self._reset_decode_attention_detail()

    def _reset_decode_attention_detail(self) -> None:
        self.last_decode_attention_detail = {
            "qkv_a": 0.0,
            "q_norm": 0.0,
            "q_b": 0.0,
            "kv_split": 0.0,
            "kv_norm": 0.0,
            "rotary": 0.0,
            "k_squeeze": 0.0,
            "mlapo": 0.0,
            "indexer_project": 0.0,
            "cache": 0.0,
            "index_cache": 0.0,
            "q_up": 0.0,
            "dsa_indexer_score": 0.0,
            "dsa_index_update": 0.0,
            "dsa_scatter_h2d": 0.0,
            "decode_attention_op": 0.0,
            "v_up": 0.0,
            "o_linear": 0.0,
            "o_all_reduce": 0.0,
        }
        self.last_decode_indexer_detail = {
            "q_proj": 0.0,
            "k_proj": 0.0,
            "k_norm": 0.0,
            "rope": 0.0,
            "weights_proj": 0.0,
        }

    def _decode_timer_start(self, profile_decode: bool, device) -> float | None:
        if not profile_decode:
            return None
        if self.decode_timing_sync:
            _profile_sync(device)
        return perf_counter()

    def _decode_timer_end(
        self,
        profile_decode: bool,
        name: str,
        start: float | None,
        device,
    ) -> None:
        if not profile_decode or start is None:
            return
        if self.decode_timing_sync:
            _profile_sync(device)
        self.last_decode_attention_detail[name] += perf_counter() - start

    def _o_proj_forward(
        self,
        attn_output: torch.Tensor,
        profile_decode: bool,
    ) -> torch.Tensor:
        if not profile_decode:
            return self.o_proj(attn_output)

        start = self._decode_timer_start(True, attn_output.device)
        if self.o_proj.disable_tp:
            output = F.linear(
                attn_output,
                self.o_proj.weight,
                self.o_proj.bias,
            )
            self._decode_timer_end(True, "o_linear", start, output.device)
            return output

        output = F.linear(
            attn_output,
            self.o_proj.weight,
            self.o_proj.bias if self.o_proj.tp_rank == 0 else None,
        )
        self._decode_timer_end(True, "o_linear", start, output.device)

        if self.o_proj.tp_size > 1 and self.o_proj.reduce_results:
            start = self._decode_timer_start(True, output.device)
            dist.all_reduce(output)
            self._decode_timer_end(True, "o_all_reduce", start, output.device)
        return output

    def assign_dsa_cache(
        self,
        ckv_cache: torch.Tensor,
        kpe_cache: torch.Tensor,
        index_cache: torch.Tensor,
        dram_ckv_cache: torch.Tensor,
        dram_kpe_cache: torch.Tensor,
        hbm_cached_tokens_pool: torch.Tensor,
        dsa_offload_max_copy_tokens: int,
        dsa_offload_fixed_tx: int,
    ) -> None:
        self.ckv_cache = ckv_cache
        self.kpe_cache = kpe_cache
        self.index_cache = index_cache
        self.dram_ckv_cache = dram_ckv_cache
        self.dram_kpe_cache = dram_kpe_cache
        self.hbm_cached_tokens_pool = hbm_cached_tokens_pool
        self.dsa_offload_max_copy_tokens = int(dsa_offload_max_copy_tokens)
        self.dsa_offload_fixed_tx = int(dsa_offload_fixed_tx)

    def post_load_prepare(self) -> None:
        if self.fuse_qkv_a and self.wd_qkv is None:
            q_weight = self.q_a_proj.weight.detach().cpu()
            kv_weight = self.kv_a_proj_with_mqa.weight.detach().cpu()
            dtype = self.q_a_proj.weight.dtype
            device = self.q_a_proj.weight.device
            self.q_a_proj._parameters.pop("weight", None)
            self.kv_a_proj_with_mqa._parameters.pop("weight", None)
            gc.collect()
            if device.type == "npu":
                torch.npu.empty_cache()

            wd_qkv = torch.empty(
                (
                    self.q_lora_rank + self.kv_lora_rank + self.qk_rope_head_dim,
                    self.hidden_size,
                ),
                dtype=dtype,
                device=device,
            )
            wd_qkv[: self.q_lora_rank].copy_(q_weight)
            wd_qkv[self.q_lora_rank :].copy_(kv_weight)
            self.wd_qkv = nn.Parameter(wd_qkv, requires_grad=False)
            del q_weight, kv_weight
            gc.collect()

        if self.w_uk_t is None or self.w_uv is None:
            weight = self.kv_b_proj.weight.data.view(
                self.num_local_heads,
                self.qk_nope_head_dim + self.v_head_dim,
                self.kv_lora_rank,
            )
            self.w_uk_t = weight[:, : self.qk_nope_head_dim, :].contiguous()
            self.w_uv = (
                weight[:, self.qk_nope_head_dim :, :]
                .transpose(1, 2)
                .contiguous()
            )
            if self.free_kv_b_proj:
                self.kv_b_proj._parameters.pop("weight", None)
                gc.collect()
                if self.w_uk_t.device.type == "npu":
                    torch.npu.empty_cache()

        if self.enable_decode_mlapo:
            self._prepare_decode_mlapo()

    def _prepare_decode_mlapo(self) -> None:
        if self.mlapo_wd_qkv is not None and self.mlapo_wu_q is not None:
            return

        if self.fuse_qkv_a:
            if self.wd_qkv is None:
                return
            q_weight = self.wd_qkv[: self.q_lora_rank].detach()
            kv_weight = self.wd_qkv[self.q_lora_rank :].detach()
        else:
            q_weight = self.q_a_proj.weight.detach()
            kv_weight = self.kv_a_proj_with_mqa.weight.detach()

        kv_weight = _trans_rope_weight(kv_weight, self.qk_rope_head_dim)
        wd_qkv = torch.cat((kv_weight, q_weight), dim=0).contiguous()
        self.mlapo_wd_qkv = _to_mlapo_bf16_nz_weight(wd_qkv)

        wu_q = self.q_b_proj.weight.detach().view(
            self.num_local_heads,
            self.qk_head_dim,
            self.q_lora_rank,
        )
        wu_q = _trans_rope_weight(wu_q, self.qk_rope_head_dim)
        wu_q = wu_q.reshape(
            self.num_local_heads * self.qk_head_dim,
            self.q_lora_rank,
        ).contiguous()
        self.mlapo_wu_q = _to_mlapo_bf16_nz_weight(wu_q)
        self.mlapo_beta1 = torch.zeros_like(self.q_a_layernorm.weight)
        del wd_qkv, wu_q
        gc.collect()
        if q_weight.device.type == "npu":
            torch.npu.empty_cache()

    def _mlapo_cos_sin(
        self,
        positions: torch.Tensor,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        context = get_context()
        cache_key = (
            "mlapo_cos_sin",
            str(positions.device),
            dtype,
            self.qk_rope_head_dim,
        )
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
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        ql_shape = (num_tokens, self.num_local_heads, self.kv_lora_rank)
        qpe_shape = (num_tokens, self.num_local_heads, self.qk_rope_head_dim)
        if (
            self._decode_mlapo_ql_nope is None
            or tuple(self._decode_mlapo_ql_nope.shape) != ql_shape
            or self._decode_mlapo_ql_nope.dtype != dtype
            or self._decode_mlapo_ql_nope.device != device
        ):
            self._decode_mlapo_ql_nope = torch.empty(
                ql_shape,
                dtype=dtype,
                device=device,
            )
        if (
            self._decode_mlapo_q_pe is None
            or tuple(self._decode_mlapo_q_pe.shape) != qpe_shape
            or self._decode_mlapo_q_pe.dtype != dtype
            or self._decode_mlapo_q_pe.device != device
        ):
            self._decode_mlapo_q_pe = torch.empty(
                qpe_shape,
                dtype=dtype,
                device=device,
            )
        if (
            self._decode_mlapo_inner_out is None
            or tuple(self._decode_mlapo_inner_out.shape) != (0,)
            or self._decode_mlapo_inner_out.dtype != dtype
            or self._decode_mlapo_inner_out.device != device
        ):
            # The CANN enable_inner_out branch is not numerically aligned yet
            # (probe_mla_preprocess sees NaNs / wrong cache writes). Keep MLAPO
            # on the stable non-inner branch and recompute q_c separately when
            # DSA needs it.
            self._decode_mlapo_inner_out = torch.empty(0, dtype=dtype, device=device)
        return self._decode_mlapo_ql_nope, self._decode_mlapo_q_pe, self._decode_mlapo_inner_out

    # Temporary q_c fallback for DSA decode. Remove this once mla_preprocess
    # enable_inner_out=True is fixed and matches the torch reference.
    def _decode_mlapo_q_c_fallback(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.fuse_qkv_a:
            q_c = F.linear(hidden_states, self.wd_qkv[: self.q_lora_rank])
        else:
            q_c = self.q_a_proj(hidden_states)
        return self.q_a_layernorm(q_c)

    def _debug_tensor_diff(self, name: str, actual: torch.Tensor, expected: torch.Tensor) -> None:
        actual_f = actual.float()
        expected_f = expected.float()
        diff = (actual_f - expected_f).abs()
        finite = torch.isfinite(diff) & torch.isfinite(actual_f) & torch.isfinite(expected_f)
        nonfinite_count = int((~finite).sum().item()) if diff.numel() else 0
        finite_diff = diff[finite]
        max_abs = float(finite_diff.max().item()) if finite_diff.numel() else float("nan")
        mean_abs = float(finite_diff.mean().item()) if finite_diff.numel() else float("nan")
        bad_count = int(((diff > 0.03125 + 0.01 * expected_f.abs()) | (~finite)).sum().item()) if diff.numel() else 0
        logger.info(
            "MLAPO live diff: rank=%d layer=%d %s shape=%s max_abs=%.6g mean_abs=%.6g bad_count=%d nonfinite_count=%d",
            _rank_id(),
            self.layer_id,
            name,
            tuple(actual.shape),
            max_abs,
            mean_abs,
            bad_count,
            nonfinite_count,
        )

    # Debug-only exact compare against the torch decode path using real model
    # hidden states. This catches MLAPO mismatches that random standalone probes
    # may miss.
    def _debug_compare_decode_mlapo(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        ql_nope: torch.Tensor,
        q_pe: torch.Tensor,
    ) -> None:
        if not self.debug_decode_mlapo_compare:
            return
        rank = _rank_id()
        if self.debug_decode_mlapo_compare_rank >= 0 and rank != self.debug_decode_mlapo_compare_rank:
            return
        if self._debug_decode_mlapo_compare_count >= self.debug_decode_mlapo_compare_limit:
            return
        if not _profile_layer_selected(self.layer_id, self.num_layers):
            return

        with torch.inference_mode():
            if self.fuse_qkv_a:
                qkv_a = F.linear(hidden_states, self.wd_qkv)
                q_c, kv = torch.split(qkv_a, [self.q_lora_rank, self.kv_lora_rank + self.qk_rope_head_dim], dim=-1)
            else:
                q_c = self.q_a_proj(hidden_states)
                kv = self.kv_a_proj_with_mqa(hidden_states)
            q_c = self.q_a_layernorm(q_c)
            q = self.q_b_proj(q_c).view(-1, self.num_local_heads, self.qk_head_dim)
            q_nope, q_pe_ref = torch.split(q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)
            ckv_ref, k_pe_ref = torch.split(kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
            ckv_ref = self.kv_a_layernorm(ckv_ref)
            q_pe_ref, k_pe_ref = self.rotary_emb(positions, q_pe_ref, k_pe_ref.unsqueeze(1))
            k_pe_ref = k_pe_ref.squeeze(1)
            q_pe_ref = _rope_interleaved_to_neox(q_pe_ref)
            k_pe_ref = _rope_interleaved_to_neox(k_pe_ref)
            ql_nope_ref = self._q_nope_up_proj(q_nope)
            slots = self._flat_slots()
            ckv_actual = self.ckv_cache.view(-1, self.kv_lora_rank).index_select(0, slots)
            kpe_actual = self.kpe_cache.view(-1, self.qk_rope_head_dim).index_select(0, slots)

            logger.info(
                "MLAPO live compare: rank=%d layer=%d positions=%s slots=%s needs_dsa_update=%s",
                rank,
                self.layer_id,
                positions.detach().cpu().tolist(),
                slots.detach().cpu().tolist(),
                get_context().needs_dsa_update,
            )
            self._debug_tensor_diff("ql_nope", ql_nope, ql_nope_ref)
            self._debug_tensor_diff("q_pe", q_pe, q_pe_ref)
            self._debug_tensor_diff("ckv_current_slots", ckv_actual, ckv_ref)
            self._debug_tensor_diff("kpe_current_slots", kpe_actual, k_pe_ref)
            self._debug_decode_mlapo_compare_count += 1

    def _decode_mlapo_preprocess(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        profile_decode: bool,
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
        )
        cos, sin = self._mlapo_cos_sin(positions, hidden_states.dtype)
        slotmapping = self._flat_slots_i32()

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
        self._debug_compare_decode_mlapo(positions, hidden_states, ql_nope, q_pe)
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

    def _flat_slots_i32(self) -> torch.Tensor:
        context = get_context()
        if context.flat_slot_mapping_i32 is not None:
            return context.flat_slot_mapping_i32
        return self._flat_slots().to(torch.int32)

    def _flat_index_slots(self) -> torch.Tensor:
        context = get_context()
        if context.flat_index_slot_mapping is not None:
            return context.flat_index_slot_mapping
        return self._flat_slots()

    def _debug_decode_metadata_enabled(self) -> bool:
        if not self.debug_decode_metadata:
            return False
        rank = _rank_id()
        if self.debug_decode_metadata_rank >= 0 and rank != self.debug_decode_metadata_rank:
            return False
        context = get_context()
        return bool(context.has_first_decode) and _profile_layer_selected(self.layer_id, self.num_layers)

    # Debug-only: verify the sparse decode metadata seen by MLA after prefill
    # materialization. This is intentionally D2H-heavy and must stay behind the
    # NANOVLLM_DSA_DEBUG_DECODE_METADATA gate.
    def _debug_log_decode_metadata(
        self,
        *,
        batch_size: int,
        block_table: torch.Tensor,
        actual_seq_lengths_key: list[int],
    ) -> None:
        if not self._debug_decode_metadata_enabled():
            return
        context = get_context()
        required = {
            "slot_mapping": context.slot_mapping,
            "flat_slot_mapping": context.flat_slot_mapping,
            "candidate_lens": context.candidate_lens,
            "sparse_selected_lens": context.sparse_selected_lens,
            "prefill_tail_lens": context.prefill_tail_lens,
            "decode_lens": context.decode_lens,
            "req_pool_entries": context.req_pool_entries,
            "hbm_block_tables": context.hbm_block_tables,
        }
        missing = [name for name, value in required.items() if value is None]
        if missing:
            logger.info("DSA decode metadata: rank=%d layer=%d missing=%s", _rank_id(), self.layer_id, missing)
            return

        sample = self.debug_decode_metadata_sample
        slot_cpu = context.slot_mapping[:batch_size].detach().to("cpu", dtype=torch.long)
        flat_slot_cpu = context.flat_slot_mapping[:batch_size].detach().to("cpu", dtype=torch.long)
        candidate_cpu = context.candidate_lens[:batch_size].detach().to("cpu", dtype=torch.long).tolist()
        selected_cpu = context.sparse_selected_lens[:batch_size].detach().to("cpu", dtype=torch.long).tolist()
        tail_cpu = context.prefill_tail_lens[:batch_size].detach().to("cpu", dtype=torch.long).tolist()
        decode_cpu = context.decode_lens[:batch_size].detach().to("cpu", dtype=torch.long).tolist()
        entries_cpu = context.req_pool_entries[:batch_size].detach().to("cpu", dtype=torch.long).tolist()
        tables_cpu = block_table[:batch_size].detach().to("cpu", dtype=torch.long)

        for b in range(batch_size):
            candidate_len = int(candidate_cpu[b])
            selected_len = int(selected_cpu[b])
            tail_len = int(tail_cpu[b])
            decode_len = int(decode_cpu[b])
            actual_len = int(actual_seq_lengths_key[b]) if b < len(actual_seq_lengths_key) else -1
            used_blocks = max(1, (max(actual_len, 0) + self.block_size - 1) // self.block_size)
            selected_blocks = (selected_len + self.block_size - 1) // self.block_size if selected_len > 0 else 0
            tail_decode_offset = tail_len + max(decode_len, 1) - 1              # logical offset inside tail+decode region
            expected_logical = selected_blocks + tail_decode_offset // self.block_size
            expected_offset = tail_decode_offset % self.block_size
            inspect_blocks = max(used_blocks, expected_logical + 1, 1)
            hbm_table = [int(v) for v in tables_cpu[b, :inspect_blocks].tolist()]
            expected_block = hbm_table[expected_logical] if 0 <= expected_logical < len(hbm_table) else -1
            expected_flat = expected_block * self.block_size + expected_offset if expected_block >= 0 else -1
            slot_block = int(slot_cpu[b, 0].item())
            slot_offset = int(slot_cpu[b, 1].item())
            flat_slot = int(flat_slot_cpu[b].item())
            entry = int(entries_cpu[b])
            selected_first: list[int] = []
            selected_last: list[int] = []
            if selected_len > 0 and self.hbm_cached_tokens_pool.numel() and 0 <= entry < self.hbm_cached_tokens_pool.shape[1]:
                pool_cpu = self.hbm_cached_tokens_pool[self.layer_id, entry, :selected_len].detach().to("cpu", dtype=torch.long)
                selected_first = pool_cpu[:sample].tolist()
                selected_last = pool_cpu[-sample:].tolist()
            logger.info(
                "DSA decode metadata: rank=%d layer=%d batch=%d entry=%d candidate_len=%d selected_len=%d "
                "tail_len=%d decode_len=%d actual_kv_len=%d selected_blocks=%d used_blocks=%d "
                "slot=(%d,%d) flat_slot=%d expected_logical=%d expected_slot=(%d,%d) expected_flat=%d "
                "slot_match=%s hbm_table_first=%s hbm_table_last=%s selected_first=%s selected_last=%s",
                _rank_id(),
                self.layer_id,
                b,
                entry,
                candidate_len,
                selected_len,
                tail_len,
                decode_len,
                actual_len,
                selected_blocks,
                used_blocks,
                slot_block,
                slot_offset,
                flat_slot,
                expected_logical,
                expected_block,
                expected_offset,
                expected_flat,
                str(flat_slot == expected_flat),
                hbm_table[:sample],
                hbm_table[-sample:],
                selected_first,
                selected_last,
            )

    def _debug_index_update_enabled(self) -> bool:
        return bool(self.debug_index_update_dump_file) and _profile_layer_selected(self.layer_id, self.num_layers)

    def _debug_write_index_update_record(self, record: dict) -> None:
        if not self.debug_index_update_dump_file:
            return
        try:
            with open(self.debug_index_update_dump_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        except OSError as exc:
            logger.warning(
                "Disable NANOVLLM_DSA_DEBUG_INDEX_UPDATE_DUMP_PATH for rank=%d layer=%d because write failed: %s",
                _rank_id(),
                self.layer_id,
                exc,
            )
            self.debug_index_update_dump_file = ""

    def _debug_index_update_pool_snapshot(
        self,
        pool_slice: torch.Tensor,
        selected_lens: torch.Tensor,
        req_pool_entries: torch.Tensor,
        batch_size: int,
    ) -> list[dict] | None:
        if not self._debug_index_update_enabled():
            return None
        selected_cpu = selected_lens[:batch_size].detach().to("cpu", dtype=torch.long).tolist()
        entries_cpu = req_pool_entries[:batch_size].detach().to("cpu", dtype=torch.long).tolist()
        snapshots: list[dict] = []
        for b in range(batch_size):
            selected_len = max(0, int(selected_cpu[b]))
            entry = int(entries_cpu[b])
            active_pool: list[int] = []
            if 0 <= entry < int(pool_slice.shape[0]) and selected_len > 0:
                active_pool = pool_slice[entry, :selected_len].detach().to("cpu", dtype=torch.long).tolist()
            snapshots.append({"batch": b, "entry": entry, "selected_len": selected_len, "pool_active": active_pool})
        return snapshots

    def _debug_dump_index_update(
        self,
        *,
        update_id: int,
        pool_before: list[dict] | None,
        pool_slice: torch.Tensor,
        promote_idx: torch.Tensor,
        demote_idx: torch.Tensor,
        copy_counts: torch.Tensor,
        candidate_lens: torch.Tensor,
        selected_lens: torch.Tensor,
        req_pool_entries: torch.Tensor,
        batch_size: int,
    ) -> None:
        if pool_before is None or not self._debug_index_update_enabled():
            return
        candidate_cpu = candidate_lens[:batch_size].detach().to("cpu", dtype=torch.long).tolist()
        selected_cpu = selected_lens[:batch_size].detach().to("cpu", dtype=torch.long).tolist()
        entries_cpu = req_pool_entries[:batch_size].detach().to("cpu", dtype=torch.long).tolist()
        copy_counts_cpu = copy_counts[:batch_size].detach().to("cpu", dtype=torch.long).tolist()
        promote_cpu = promote_idx[:batch_size].detach().to("cpu", dtype=torch.long).tolist()
        demote_cpu = demote_idx[:batch_size].detach().to("cpu", dtype=torch.long).tolist()
        pool_after = self._debug_index_update_pool_snapshot(pool_slice, selected_lens, req_pool_entries, batch_size) or []
        for b in range(batch_size):
            copy_count = max(0, int(copy_counts_cpu[b]))
            record = {
                "event": "dsa_index_update",
                "rank": _rank_id(),
                "layer": self.layer_id,
                "update_id": update_id,
                "batch": b,
                "entry": int(entries_cpu[b]),
                "candidate_len": int(candidate_cpu[b]),
                "selected_len": int(selected_cpu[b]),
                "copy_count": copy_count,
                "promote_idx_full": promote_cpu[b],
                "demote_idx_full": demote_cpu[b],
                "promote_idx_valid": promote_cpu[b][:copy_count],
                "demote_idx_valid": demote_cpu[b][:copy_count],
                "pool_before_active": pool_before[b]["pool_active"] if b < len(pool_before) else [],
                "pool_after_active": pool_after[b]["pool_active"] if b < len(pool_after) else [],
            }
            self._debug_write_index_update_record(record)

    def _debug_validate_scatter_h2d(
        self,
        *,
        update_id: int,
        pool_slice: torch.Tensor,
        copy_counts: torch.Tensor,
        candidate_lens: torch.Tensor,
        selected_lens: torch.Tensor,
        req_pool_entries: torch.Tensor,
        hbm_block_tables: torch.Tensor,
        dram_block_tables: torch.Tensor,
        batch_size: int,
    ) -> None:
        if not self._debug_index_update_enabled():
            return
        _profile_sync(self.ckv_cache.device)
        candidate_cpu = candidate_lens[:batch_size].detach().to("cpu", dtype=torch.long).tolist()
        selected_cpu = selected_lens[:batch_size].detach().to("cpu", dtype=torch.long).tolist()
        entries_cpu = req_pool_entries[:batch_size].detach().to("cpu", dtype=torch.long).tolist()
        copy_counts_cpu = copy_counts[:batch_size].detach().to("cpu", dtype=torch.long).tolist()
        block_size = self.block_size
        for b in range(batch_size):
            candidate_len = int(candidate_cpu[b])
            selected_len = int(selected_cpu[b])
            entry = int(entries_cpu[b])
            base_record = {
                "event": "dsa_scatter_h2d_validate",
                "rank": _rank_id(),
                "layer": self.layer_id,
                "update_id": update_id,
                "batch": b,
                "entry": entry,
                "candidate_len": candidate_len,
                "selected_len": selected_len,
                "copy_count": int(copy_counts_cpu[b]),
            }
            if candidate_len <= selected_len:
                base_record["status"] = "skipped_dense_request_no_dram_copy"
                self._debug_write_index_update_record(base_record)
                continue
            if selected_len <= 0 or not (0 <= entry < int(pool_slice.shape[0])):
                base_record["status"] = "skipped_invalid_pool_entry_or_selected_len"
                self._debug_write_index_update_record(base_record)
                continue

            token_ids = pool_slice[entry, :selected_len].to(torch.long)
            valid_mask = (token_ids >= 0) & (token_ids < candidate_len)
            valid_count = int(valid_mask.sum().item())
            token_ids_cpu = token_ids.detach().to("cpu", dtype=torch.long).tolist()
            if valid_count != selected_len:
                base_record.update({
                    "status": "invalid_selected_token_id",
                    "valid_count": valid_count,
                    "selected_tokens": token_ids_cpu,
                })
                self._debug_write_index_update_record(base_record)
                continue

            logical_slots = torch.arange(selected_len, dtype=torch.long, device=token_ids.device)
            hbm_logical_blocks = logical_slots // block_size
            hbm_offsets = logical_slots % block_size
            dram_logical_blocks = token_ids // block_size
            dram_offsets = token_ids % block_size
            hbm_table = hbm_block_tables[b].to(torch.long)
            dram_table = dram_block_tables[b].to(torch.long)
            if int(hbm_logical_blocks.max().item()) >= int(hbm_table.shape[0]) or int(dram_logical_blocks.max().item()) >= int(dram_table.shape[0]):
                base_record.update({
                    "status": "block_table_too_short",
                    "max_hbm_logical_block": int(hbm_logical_blocks.max().item()),
                    "hbm_table_len": int(hbm_table.shape[0]),
                    "max_dram_logical_block": int(dram_logical_blocks.max().item()),
                    "dram_table_len": int(dram_table.shape[0]),
                    "selected_tokens": token_ids_cpu,
                })
                self._debug_write_index_update_record(base_record)
                continue

            hbm_blocks = hbm_table.index_select(0, hbm_logical_blocks)
            dram_blocks = dram_table.index_select(0, dram_logical_blocks)
            hbm_ckv = self.ckv_cache[hbm_blocks, hbm_offsets]
            hbm_kpe = self.kpe_cache[hbm_blocks, hbm_offsets]
            dram_ckv = self.dram_ckv_cache[dram_blocks, dram_offsets]
            dram_kpe = self.dram_kpe_cache[dram_blocks, dram_offsets]
            ckv_bad_slots = (hbm_ckv != dram_ckv).reshape(selected_len, -1).any(dim=1)
            kpe_bad_slots = (hbm_kpe != dram_kpe).reshape(selected_len, -1).any(dim=1)
            bad_slots = ckv_bad_slots | kpe_bad_slots
            bad_slot_ids = torch.nonzero(bad_slots, as_tuple=False).flatten()
            first_bad_slot = int(bad_slot_ids[0].item()) if bad_slot_ids.numel() else -1
            ckv_diff = (hbm_ckv.float() - dram_ckv.float()).abs()
            kpe_diff = (hbm_kpe.float() - dram_kpe.float()).abs()
            base_record.update({
                "status": "ok" if first_bad_slot < 0 else "mismatch",
                "selected_tokens": token_ids_cpu,
                "selected_first": token_ids_cpu[: self.debug_decode_metadata_sample],
                "selected_last": token_ids_cpu[-self.debug_decode_metadata_sample :],
                "mismatch_slots": int(bad_slots.sum().item()),
                "first_bad_slot": first_bad_slot,
                "first_bad_token": int(token_ids[first_bad_slot].item()) if first_bad_slot >= 0 else -1,
                "ckv_max_abs": float(ckv_diff.max().item()) if ckv_diff.numel() else 0.0,
                "kpe_max_abs": float(kpe_diff.max().item()) if kpe_diff.numel() else 0.0,
                "ckv_nan_count": int(torch.isnan(hbm_ckv.float()).sum().item() + torch.isnan(dram_ckv.float()).sum().item()),
                "kpe_nan_count": int(torch.isnan(hbm_kpe.float()).sum().item() + torch.isnan(dram_kpe.float()).sum().item()),
            })
            self._debug_write_index_update_record(base_record)

    def _init_prefill_sparse_token_pool(self, seq) -> None:
        entry = int(seq.hbm_cached_tokens_pool_entry)
        num_full_blocks = int(seq.num_prefill_full_blocks)
        num_sparse_blocks = int(seq.num_sparse_blocks)
        pool = self.hbm_cached_tokens_pool[self.layer_id, entry]
        pool.fill_(-1)
        if num_full_blocks <= 0 or num_sparse_blocks <= 0:
            return

        if num_sparse_blocks >= num_full_blocks:
            dense_tokens = num_full_blocks * self.block_size
            if dense_tokens > int(pool.shape[0]):
                raise RuntimeError("DSA sparse token pool is too small for dense prefill bookkeeping.")
            pool[:dense_tokens] = torch.arange(dense_tokens, dtype=torch.int32, device=pool.device)
            return

        prefix_tokens = self.block_size                         # Keep the first physical prefix block.
        suffix_blocks = max(0, num_sparse_blocks - 1)
        suffix_tokens = suffix_blocks * self.block_size
        selected_tokens = prefix_tokens + suffix_tokens
        if selected_tokens > int(pool.shape[0]):
            raise RuntimeError("DSA sparse token pool is too small for prefix+suffix bookkeeping.")

        pool[:prefix_tokens] = torch.arange(prefix_tokens, dtype=torch.int32, device=pool.device)
        if suffix_tokens > 0:
            candidate_tokens = num_full_blocks * self.block_size
            suffix_start = candidate_tokens - suffix_tokens
            pool[prefix_tokens:selected_tokens] = torch.arange(suffix_start, candidate_tokens, dtype=torch.int32, device=pool.device)

    def finalize_prefill_offload(
        self,
        seq,
        old_hbm_block_table: list[int],
    ) -> None:
        num_full_blocks = int(seq.num_prefill_full_blocks)
        num_sparse_blocks = int(seq.num_sparse_blocks)
        self._init_prefill_sparse_token_pool(seq)

        if num_sparse_blocks >= num_full_blocks:
            # Dense/short requests keep every full prefill block in HBM, so their
            # decode path stays aligned with baseline and no DRAM copy is needed.
            return

        # Long requests keep only the first prefix block plus suffix blocks in HBM.
        # Persist the full prefill KV to DRAM for future Tx>0 promote copies.
        for logical_block in range(num_full_blocks):
            hbm_block = int(old_hbm_block_table[logical_block])
            dram_block = int(seq.dram_block_table[logical_block])
            self.dram_ckv_cache[dram_block].copy_(
                self.ckv_cache[hbm_block].to(
                    device=self.dram_ckv_cache.device,
                    non_blocking=True,
                ),
            )
            self.dram_kpe_cache[dram_block].copy_(
                self.kpe_cache[hbm_block].to(
                    device=self.dram_kpe_cache.device,
                    non_blocking=True,
                ),
            )

    def _store_mla_cache(
        self,
        ckv: torch.Tensor,
        kpe: torch.Tensor,
    ) -> None:
        flat_slots = self._flat_slots()
        self.ckv_cache.view(-1, self.kv_lora_rank).index_copy_(
            0,
            flat_slots,
            ckv,
        )
        self.kpe_cache.view(-1, self.qk_rope_head_dim).index_copy_(
            0,
            flat_slots,
            kpe,
        )

    def _store_index_cache(self, index_k: torch.Tensor | None) -> None:
        if index_k is None:
            return
        index_slots = self._flat_index_slots()
        self.index_cache.view(-1, self.indexer.head_dim).index_copy_(
            0,
            index_slots,
            index_k,
        )

    def _run_indexer(
        self,
        hidden_states: torch.Tensor,
        q_c: torch.Tensor,
        positions: torch.Tensor,
        profile_decode: bool,
        query_only: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]:
        start = self._decode_timer_start(profile_decode, hidden_states.device)
        indexer_detail = (
            {key: 0.0 for key in self.last_decode_indexer_detail}
            if profile_decode
            else None
        )
        q_index, index_k, weights = self.indexer(
            hidden_states,
            q_c,
            positions,
            self.indexer_rotary_emb,
            indexer_detail,
            self.decode_timing_sync,
            query_only=query_only,
        )
        if indexer_detail is not None:
            self.last_decode_indexer_detail = indexer_detail
            self.last_decode_indexer_q_path = self.indexer.last_q_project_path
        self._decode_timer_end(
            profile_decode,
            "indexer_project",
            start,
            hidden_states.device,
        )
        return q_index, index_k, weights

    def _q_nope_up_proj(self, q_nope: torch.Tensor) -> torch.Tensor:
        num_tokens = q_nope.shape[0]
        if (
            q_nope.dtype in (torch.float16, torch.bfloat16)
            and self.q_up_bmm_trans_max_tokens > 0
            and num_tokens == 1
            and num_tokens <= self.q_up_bmm_trans_max_tokens
        ):
            ql_nope = torch.empty(
                (
                    num_tokens,
                    self.num_local_heads,
                    self.kv_lora_rank,
                ),
                dtype=q_nope.dtype,
                device=q_nope.device,
            )
            ascend_ops.batch_matmul_transpose(
                q_nope,
                self.w_uk_t,
                ql_nope,
            )
            return ql_nope

        q_nope_by_head = q_nope.transpose(0, 1).contiguous()
        ql_nope = torch.bmm(q_nope_by_head, self.w_uk_t)
        return ql_nope.transpose(0, 1).contiguous()

    def _v_up_proj(self, latent: torch.Tensor) -> torch.Tensor:
        num_tokens = latent.shape[0]
        latent_by_head = latent.transpose(0, 1).contiguous()
        output = torch_npu.npu_transpose_batchmatmul(
            latent_by_head,
            self.w_uv,
            perm_y=(1, 0, 2),
        )
        return output.reshape(num_tokens, -1)

    def _v_up_proj_head_major(
        self,
        latent_by_head: torch.Tensor,
        num_tokens: int,
    ) -> torch.Tensor:
        latent_by_head = latent_by_head.view(
            self.num_local_heads,
            num_tokens,
            self.kv_lora_rank,
        )
        output = torch_npu.npu_transpose_batchmatmul(
            latent_by_head,
            self.w_uv,
            perm_y=(1, 0, 2),
        )
        return output.reshape(num_tokens, -1)

    def _decode_mla_v2_buffers(
        self,
        batch_size: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        out_shape = (
            self.num_local_heads,
            batch_size,
            1,
            self.kv_lora_rank,
        )
        lse_shape = (batch_size,)
        if (
            self._decode_mla_v2_out is None
            or tuple(self._decode_mla_v2_out.shape) != out_shape
            or self._decode_mla_v2_out.dtype != dtype
            or self._decode_mla_v2_out.device != device
        ):
            self._decode_mla_v2_out = torch.empty(
                out_shape,
                dtype=dtype,
                device=device,
            )
        if (
            self._decode_mla_v2_lse is None
            or tuple(self._decode_mla_v2_lse.shape) != lse_shape
            or self._decode_mla_v2_lse.dtype != dtype
            or self._decode_mla_v2_lse.device != device
        ):
            self._decode_mla_v2_lse = torch.empty(
                lse_shape,
                dtype=dtype,
                device=device,
            )
        return self._decode_mla_v2_out, self._decode_mla_v2_lse

    def _decode_mla_v2_workspace_get(
        self,
        batch_size: int,
        query: torch.Tensor,
        key_cache: torch.Tensor,
        kwargs: dict,
    ) -> torch.Tensor:
        key = (
            batch_size,
            str(query.device),
            query.dtype,
            self.block_size,
            self.num_local_heads,
            self.kv_lora_rank,
        )
        workspace = _NPU_MLA_V2_WORKSPACE_CACHE.get(key)
        if workspace is None:
            # The FIA v2 workspace depends on operator shape/attrs, not layer
            # weights. Share it across layers instead of keeping 61 copies.
            workspace = torch_npu._npu_fused_infer_attention_score_v2_get_max_workspace(
                query,
                key_cache,
                key_cache,
                **kwargs,
            )
            _NPU_MLA_V2_WORKSPACE_CACHE[key] = workspace
        return workspace

    def _decode_forward_mla_v2(
        self,
        ql_nope: torch.Tensor,
        q_pe: torch.Tensor,
        block_table: torch.Tensor,
        actual_seq_lengths_key: list[int],
    ) -> torch.Tensor:
        batch_size = int(ql_nope.shape[0])
        query = ql_nope.view(
            batch_size,
            self.num_local_heads,
            1,
            self.kv_lora_rank,
        ).contiguous()
        query_rope = q_pe.view(
            batch_size,
            self.num_local_heads,
            1,
            self.qk_rope_head_dim,
        )
        # Nano stores paged MLA cache as [blocks, block_size, kv_heads, dim].
        # FIA v2 with BNSD_NBSD expects [blocks, kv_heads, block_size, dim].
        # DeepSeek MLA has kv_heads=1 here, so this is a metadata-only view,
        # not a big cache copy.
        key_cache = self.ckv_cache.view(
            -1,
            1,
            self.block_size,
            self.kv_lora_rank,
        )
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
        out, lse = self._decode_mla_v2_buffers(
            batch_size,
            ql_nope.dtype,
            ql_nope.device,
        )
        workspace = self._decode_mla_v2_workspace_get(
            batch_size,
            query,
            key_cache,
            kwargs,
        )
        torch_npu.npu_fused_infer_attention_score_v2.out(
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

    def _prefill_forward(
        self,
        ql_nope: torch.Tensor,
        q_pe: torch.Tensor,
    ) -> torch.Tensor:
        return self._prefill_forward_npu_mla(ql_nope, q_pe)

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
        latent = _first_tensor(mla_result)
        output = self._v_up_proj(latent)
        return output

    def _dsa_offload_buffers(
        self,
        batch_size: int,
        candidate_capacity: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        global _DSA_OFFLOAD_BUFFER_CACHE
        copy_capacity = int(self.dsa_offload_max_copy_tokens)
        key = (str(device), dtype)
        cached = _DSA_OFFLOAD_BUFFER_CACHE.get(key)
        if cached is None:
            cached_batch = cached_candidates = cached_copy = 0
            buffers = None
        else:
            cached_batch, cached_candidates, cached_copy, buffers = cached

        if (
            buffers is None
            or cached_batch < batch_size
            or cached_candidates < candidate_capacity
            or cached_copy < copy_capacity
        ):
            cached_batch = max(cached_batch, batch_size)
            cached_candidates = max(cached_candidates, candidate_capacity)
            cached_copy = max(cached_copy, copy_capacity)
            score_out = torch.empty(
                (cached_batch, cached_candidates),
                dtype=dtype,
                device=device,
            )
            promote_idx = torch.empty(
                (cached_batch, cached_copy),
                dtype=torch.int32,
                device=device,
            )
            demote_idx = torch.empty(
                (cached_batch, cached_copy),
                dtype=torch.int32,
                device=device,
            )
            copy_counts = torch.empty(
                (cached_batch,),
                dtype=torch.int32,
                device=device,
            )
            buffers = (score_out, promote_idx, demote_idx, copy_counts)
            _DSA_OFFLOAD_BUFFER_CACHE[key] = (
                cached_batch,
                cached_candidates,
                cached_copy,
                buffers,
            )

        score_out, promote_idx, demote_idx, copy_counts = buffers
        return (
            score_out[:batch_size, :cached_candidates],
            promote_idx[:batch_size, :copy_capacity],
            demote_idx[:batch_size, :copy_capacity],
            copy_counts[:batch_size],
        )

    def _dsa_offload_update(
        self,
        q_index: torch.Tensor,
        weights: torch.Tensor,
        batch_size: int,
    ) -> None:
        context = get_context()
        if not context.needs_dsa_update or self.dsa_offload_fixed_tx <= 0:
            return
        required_context = {
            "candidate_lens": context.candidate_lens,
            "sparse_selected_lens": context.sparse_selected_lens,
            "req_pool_entries": context.req_pool_entries,
            "index_block_tables": context.index_block_tables,
            "candidate_query_lens": context.candidate_query_lens,
            "hbm_block_tables": context.hbm_block_tables,
            "dram_block_tables": context.dram_block_tables,
        }
        missing = [name for name, value in required_context.items() if value is None]
        if missing:
            raise RuntimeError(
                "DSA offload context is missing: " + ", ".join(missing)
            )
        candidate_lens = context.candidate_lens[:batch_size]
        candidate_query_lens = context.candidate_query_lens[:batch_size]
        selected_lens = context.sparse_selected_lens[:batch_size]
        req_pool_entries = context.req_pool_entries[:batch_size]
        max_candidate = max(int(context.max_candidate_len), 1)

        (
            score_out,
            promote_idx,
            demote_idx,
            copy_counts,
        ) = self._dsa_offload_buffers(
            batch_size,
            max_candidate,
            q_index.dtype,
            q_index.device,
        )

        profile_decode = self.log_decode_layer_timing
        start = self._decode_timer_start(profile_decode, q_index.device)
        dsa_indexer_score(
            q_index[:batch_size],
            self.index_cache,
            weights[:batch_size],
            context.index_block_tables[:batch_size],
            candidate_lens,
            score_out,
            actual_seq_lengths_query=candidate_query_lens,
        )
        self._decode_timer_end(
            profile_decode,
            "dsa_indexer_score",
            start,
            score_out.device,
        )

        pool_slice = self.hbm_cached_tokens_pool[self.layer_id]
        update_id = self._debug_index_update_dump_count
        self._debug_index_update_dump_count += 1
        pool_before = self._debug_index_update_pool_snapshot(
            pool_slice,
            selected_lens,
            req_pool_entries,
            batch_size,
        )
        start = self._decode_timer_start(profile_decode, score_out.device)
        dsa_index_update(
            score_out,
            pool_slice,
            promote_idx,
            demote_idx,
            copy_counts,
            candidate_lens,
            selected_lens,
            req_pool_entries,
            min(self.dsa_offload_fixed_tx, self.dsa_offload_max_copy_tokens),
        )
        self._decode_timer_end(
            profile_decode,
            "dsa_index_update",
            start,
            score_out.device,
        )
        self._debug_dump_index_update(
            update_id=update_id,
            pool_before=pool_before,
            pool_slice=pool_slice,
            promote_idx=promote_idx,
            demote_idx=demote_idx,
            copy_counts=copy_counts,
            candidate_lens=candidate_lens,
            selected_lens=selected_lens,
            req_pool_entries=req_pool_entries,
            batch_size=batch_size,
        )

        start = self._decode_timer_start(profile_decode, self.ckv_cache.device)
        dsa_scatter_h2d(
            promote_idx,
            demote_idx,
            copy_counts,
            context.hbm_block_tables[:batch_size],
            context.dram_block_tables[:batch_size],
            self.ckv_cache,
            self.kpe_cache,
            self.dram_ckv_cache,
            self.dram_kpe_cache,
        )
        self._decode_timer_end(
            profile_decode,
            "dsa_scatter_h2d",
            start,
            self.ckv_cache.device,
        )
        self._debug_validate_scatter_h2d(
            update_id=update_id,
            pool_slice=pool_slice,
            copy_counts=copy_counts,
            candidate_lens=candidate_lens,
            selected_lens=selected_lens,
            req_pool_entries=req_pool_entries,
            hbm_block_tables=context.hbm_block_tables[:batch_size],
            dram_block_tables=context.dram_block_tables[:batch_size],
            batch_size=batch_size,
        )

    def _decode_forward(
        self,
        ql_nope: torch.Tensor,
        q_pe: torch.Tensor,
        q_index: torch.Tensor | None,
        weights: torch.Tensor | None,
    ) -> torch.Tensor:
        self.last_decode_attention_op_time = 0.0
        if not get_context().needs_dsa_update or self.dsa_offload_fixed_tx <= 0:
            return self._decode_forward_mla(ql_nope, q_pe, q_index, weights)
        if q_index is None or weights is None:
            raise RuntimeError("DSA offload decode requires indexer outputs.")
        return self._decode_forward_mla(ql_nope, q_pe, q_index, weights)

    def _decode_forward_mla(
        self,
        ql_nope: torch.Tensor,
        q_pe: torch.Tensor,
        q_index: torch.Tensor | None,
        weights: torch.Tensor | None,
    ) -> torch.Tensor:
        context = get_context()
        batch_size = int(ql_nope.shape[0])
        actual_seq_lengths_key = context.actual_seq_lengths_kv
        assert actual_seq_lengths_key is not None
        block_table = context.block_tables[:batch_size]
        profile_decode = self.log_decode_layer_timing
        self._debug_log_decode_metadata(batch_size=batch_size, block_table=block_table, actual_seq_lengths_key=actual_seq_lengths_key)
        if context.needs_dsa_update and self.dsa_offload_fixed_tx > 0:
            if q_index is None or weights is None:
                raise RuntimeError("DSA offload decode requires indexer outputs.")
            self._dsa_offload_update(q_index, weights, batch_size)
        start = self._decode_timer_start(profile_decode, ql_nope.device)
        latent = self._decode_forward_mla_v2(
            ql_nope,
            q_pe,
            block_table,
            actual_seq_lengths_key,
        )
        self._decode_timer_end(
            profile_decode,
            "decode_attention_op",
            start,
            ql_nope.device,
        )
        self.last_decode_attention_op_time = self.last_decode_attention_detail[
            "decode_attention_op"
        ]
        start = self._decode_timer_start(profile_decode, latent.device)
        output = self._v_up_proj_head_major(latent, batch_size)
        self._decode_timer_end(profile_decode, "v_up", start, output.device)
        return output

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        context = get_context()
        profile_decode = self.log_decode_layer_timing and not context.is_prefill
        if profile_decode:
            self._reset_decode_attention_detail()

        if self.w_uk_t is None or self.w_uv is None:
            self.post_load_prepare()

        if self.fuse_qkv_a and self.wd_qkv is None:
            self.post_load_prepare()

        use_decode_mlapo = (
            self.enable_decode_mlapo
            and not context.is_prefill
            and not context.has_first_decode
        )
        needs_decode_dsa_update = bool(context.needs_dsa_update and self.dsa_offload_fixed_tx > 0)
        if use_decode_mlapo:
            ql_nope, q_pe, q_c = self._decode_mlapo_preprocess(
                positions,
                hidden_states,
                profile_decode,
                need_inner_out=needs_decode_dsa_update,
            )
            q_index = index_k = weights = None
            if needs_decode_dsa_update:
                q_index, index_k, weights = self._run_indexer(
                    hidden_states,
                    q_c,
                    positions,
                    profile_decode,
                    query_only=True,
                )
            if index_k is not None:
                start = self._decode_timer_start(profile_decode, index_k.device)
                self._store_index_cache(index_k)
                self._decode_timer_end(profile_decode, "index_cache", start, index_k.device)

            attn_output = self._decode_forward(ql_nope, q_pe, q_index, weights)
            return self._o_proj_forward(attn_output, profile_decode)

        start = self._decode_timer_start(profile_decode, hidden_states.device)
        if self.fuse_qkv_a:
            qkv_a = F.linear(hidden_states, self.wd_qkv)
            q_c, kv = torch.split(
                qkv_a,
                [self.q_lora_rank, self.kv_lora_rank + self.qk_rope_head_dim],
                dim=-1,
            )
        else:
            q_c = self.q_a_proj(hidden_states)
            kv = self.kv_a_proj_with_mqa(hidden_states)
        self._decode_timer_end(profile_decode, "qkv_a", start, hidden_states.device)

        start = self._decode_timer_start(profile_decode, q_c.device)
        q_c = self.q_a_layernorm(q_c)
        self._decode_timer_end(profile_decode, "q_norm", start, q_c.device)

        start = self._decode_timer_start(profile_decode, q_c.device)
        q = self.q_b_proj(q_c).view(
            -1, self.num_local_heads, self.qk_head_dim
        )
        q_nope, q_pe = torch.split(
            q,
            [self.qk_nope_head_dim, self.qk_rope_head_dim],
            dim=-1,
        )
        self._decode_timer_end(profile_decode, "q_b", start, q.device)

        start = self._decode_timer_start(profile_decode, kv.device)
        ckv, k_pe = torch.split(
            kv,
            [self.kv_lora_rank, self.qk_rope_head_dim],
            dim=-1,
        )
        self._decode_timer_end(profile_decode, "kv_split", start, kv.device)

        start = self._decode_timer_start(profile_decode, ckv.device)
        ckv = self.kv_a_layernorm(ckv)
        self._decode_timer_end(profile_decode, "kv_norm", start, ckv.device)

        start = self._decode_timer_start(profile_decode, q_pe.device)
        q_pe, k_pe = self.rotary_emb(positions, q_pe, k_pe.unsqueeze(1))
        self._decode_timer_end(profile_decode, "rotary", start, q_pe.device)

        start = self._decode_timer_start(profile_decode, k_pe.device)
        k_pe = k_pe.squeeze(1)
        self._decode_timer_end(profile_decode, "k_squeeze", start, k_pe.device)

        # RoPE dot products are unchanged if both q/k use the same basis.
        # Keep MLA RoPE cache in neox order so MLAPO decode can use
        # mla_preprocess output directly.
        q_pe = _rope_interleaved_to_neox(q_pe)
        k_pe = _rope_interleaved_to_neox(k_pe)

        q_index = index_k = weights = None
        if context.needs_dsa_update and (context.is_prefill or self.dsa_offload_fixed_tx > 0):
            q_index, index_k, weights = self._run_indexer(
                hidden_states,
                q_c,
                positions,
                profile_decode,
                query_only=not context.is_prefill,
            )

        start = self._decode_timer_start(profile_decode, ckv.device)
        self._store_mla_cache(ckv, k_pe)
        self._decode_timer_end(profile_decode, "cache", start, ckv.device)

        if index_k is not None:
            start = self._decode_timer_start(profile_decode, index_k.device)
            self._store_index_cache(index_k)
            self._decode_timer_end(profile_decode, "index_cache", start, index_k.device)

        start = self._decode_timer_start(profile_decode, q_nope.device)
        ql_nope = self._q_nope_up_proj(q_nope)
        self._decode_timer_end(profile_decode, "q_up", start, ql_nope.device)

        if context.is_prefill:
            attn_output = self._prefill_forward(ql_nope, q_pe)
        else:
            attn_output = self._decode_forward(ql_nope, q_pe, q_index, weights)

        return self._o_proj_forward(attn_output, profile_decode)


class DeepseekV32DecoderLayer(nn.Module):
    def __init__(self, config: DeepseekV32Config, layer_idx: int) -> None:
        super().__init__()
        self.layer_idx = int(layer_idx)
        self.self_attn = DeepseekV32DSAAttention(config, layer_idx)
        is_shared_only, keep_routed_experts = _resolve_export_mode(config)
        if (
            layer_idx >= int(config.first_k_dense_replace)
            and is_shared_only
        ):
            intermediate_size = int(config.moe_intermediate_size) * int(
                getattr(config, "n_shared_experts", 1) or 1
            )
            self.mlp = DeepseekV32MLP(
                hidden_size=int(config.hidden_size),
                intermediate_size=intermediate_size,
                hidden_act=str(config.hidden_act),
            )
        elif layer_idx < int(config.first_k_dense_replace):
            intermediate_size = int(config.intermediate_size)
            self.mlp = DeepseekV32MLP(
                hidden_size=int(config.hidden_size),
                intermediate_size=intermediate_size,
                hidden_act=str(config.hidden_act),
            )
        elif keep_routed_experts:
                self.mlp = DeepseekV32SparseMoeBlock(config, layer_idx)
        else:
            raise ValueError(
                "DeepSeek-V3.2 in nano-vllm-ascend currently expects either "
                "the shared-only export or the keep-routed-experts export."
            )
        self.input_layernorm = RMSNorm(
            int(config.hidden_size),
            eps=float(config.rms_norm_eps),
        )
        self.post_attention_layernorm = RMSNorm(
            int(config.hidden_size),
            eps=float(config.rms_norm_eps),
        )
        self.log_decode_layer_timing = _env_flag(
            "NANOVLLM_LOG_DECODE_LAYER_TIMING",
            False,
        ) and _profile_layer_selected(self.layer_idx, int(config.num_hidden_layers))
        self.decode_timing_sync = _env_flag(
            "NANOVLLM_DECODE_LAYER_TIMING_SYNC",
            True,
        )
        self.mlp_kind = (
            "moe" if isinstance(self.mlp, DeepseekV32SparseMoeBlock) else "dense_mlp"
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        context = get_context()
        profile_decode = self.log_decode_layer_timing and not context.is_prefill
        if profile_decode and self.decode_timing_sync:
            _profile_sync(hidden_states.device)
            attention_start = perf_counter()
        elif profile_decode:
            attention_start = perf_counter()
        hidden_states = self.self_attn(positions, hidden_states)
        if profile_decode and self.decode_timing_sync:
            _profile_sync(hidden_states.device)
            attention_total = perf_counter() - attention_start
        elif profile_decode:
            attention_total = perf_counter() - attention_start
        hidden_states, residual = self.post_attention_layernorm(
            hidden_states, residual
        )
        if profile_decode and self.decode_timing_sync:
            _profile_sync(hidden_states.device)
            mlp_start = perf_counter()
        elif profile_decode:
            mlp_start = perf_counter()
        hidden_states = self.mlp(hidden_states)
        if profile_decode and self.decode_timing_sync:
            _profile_sync(hidden_states.device)
        if profile_decode:
            attention_detail = self.self_attn.last_decode_attention_detail
            indexer_detail = self.self_attn.last_decode_indexer_detail
            attention_gap = attention_total - sum(attention_detail.values())
            kv_rope_total = (
                attention_detail["kv_split"]
                + attention_detail["kv_norm"]
                + attention_detail["rotary"]
                + attention_detail["k_squeeze"]
            )
            o_proj_total = (
                attention_detail["o_linear"]
                + attention_detail["o_all_reduce"]
            )
            dsa_total = (
                attention_detail["dsa_indexer_score"]
                + attention_detail["dsa_index_update"]
                + attention_detail["dsa_scatter_h2d"]
            )
            logger.info(
                "Decode layer timing: rank=%d layer=%d tokens=%d "
                "attention_total=%.6fs qkv_a=%.6fs q_norm=%.6fs "
                "q_b=%.6fs kv_rope=%.6fs kv_split=%.6fs kv_norm=%.6fs "
                "rotary=%.6fs k_squeeze=%.6fs mlapo=%.6fs indexer_project=%.6fs "
                "indexer_q_proj=%.6fs indexer_k_proj=%.6fs "
                "indexer_k_norm=%.6fs indexer_rope=%.6fs "
                "indexer_weights=%.6fs "
                "indexer_q_path=%s "
                "cache=%.6fs index_cache=%.6fs "
                "q_up=%.6fs dsa_total=%.6fs dsa_indexer_score=%.6fs "
                "dsa_index_update=%.6fs dsa_scatter_h2d=%.6fs "
                "decode_attention_op=%.6fs v_up=%.6fs "
                "o_proj=%.6fs o_linear=%.6fs o_all_reduce=%.6fs "
                "attention_gap=%.6fs "
                "moe_total=%.6fs mlp_kind=%s moe_backend=%s",
                _rank_id(),
                self.layer_idx,
                int(hidden_states.shape[0]),
                attention_total,
                attention_detail["qkv_a"],
                attention_detail["q_norm"],
                attention_detail["q_b"],
                kv_rope_total,
                attention_detail["kv_split"],
                attention_detail["kv_norm"],
                attention_detail["rotary"],
                attention_detail["k_squeeze"],
                attention_detail["mlapo"],
                attention_detail["indexer_project"],
                indexer_detail["q_proj"],
                indexer_detail["k_proj"],
                indexer_detail["k_norm"],
                indexer_detail["rope"],
                indexer_detail["weights_proj"],
                self.self_attn.last_decode_indexer_q_path,
                attention_detail["cache"],
                attention_detail["index_cache"],
                attention_detail["q_up"],
                dsa_total,
                attention_detail["dsa_indexer_score"],
                attention_detail["dsa_index_update"],
                attention_detail["dsa_scatter_h2d"],
                attention_detail["decode_attention_op"],
                attention_detail["v_up"],
                o_proj_total,
                attention_detail["o_linear"],
                attention_detail["o_all_reduce"],
                attention_gap,
                perf_counter() - mlp_start,
                self.mlp_kind,
                getattr(self.mlp, "moe_backend", "dense"),
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
