from __future__ import annotations

import json
import math
import os
import gc
from pathlib import Path
from time import perf_counter

import torch
import torch_npu  # type: ignore
import torch.distributed as dist
from torch import nn
import torch.nn.functional as F
from transformers import PretrainedConfig

import nanovllm.ops as ascend_ops
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
SFA_SPARSE_COUNT = 2048
_NPU_MLA_ATTENTION_MASK_CACHE: dict[tuple[str, int], torch.Tensor] = {}


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


def _is_rank0() -> bool:
    try:
        return (not dist.is_initialized()) or dist.get_rank() == 0
    except Exception:
        return True


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


def _hadamard_transform(x: torch.Tensor) -> torch.Tensor:
    dim = x.shape[-1]
    if dim == 0 or dim & (dim - 1):
        raise ValueError(
            "Hadamard transform expects the last dimension to be a power of 2."
        )
    y = x.float().reshape(-1, dim)
    block = 1
    while block < dim:
        y = y.view(-1, dim // (block * 2), 2, block)
        left = y[:, :, 0, :]
        right = y[:, :, 1, :]
        y = torch.cat((left + right, left - right), dim=-1).reshape(-1, dim)
        block *= 2
    return y.reshape_as(x.float())


def _rotate_activation(x: torch.Tensor) -> torch.Tensor:
    return _hadamard_transform(x) * (x.shape[-1] ** -0.5)


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

    def _grouped_topk(
        self,
        router_logits: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        router_logits = router_logits.float()
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
        topk_weights = topk_weights.float()
        return topk_weights, topk_ids

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        sequence_length, hidden_dim = hidden_states.shape
        shared_output = self.shared_experts(hidden_states)
        router_logits = self.gate(hidden_states)
        routing_weights, selected_experts = self._grouped_topk(router_logits)
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

        self.wq_b = ReplicatedLinear(
            self.q_lora_rank,
            self.n_head * self.head_dim,
            bias=False,
        )
        self.wk = ReplicatedLinear(
            self.hidden_size,
            self.head_dim,
            bias=False,
        )
        self.k_norm = nn.LayerNorm(self.head_dim, eps=1e-6)
        self.weights_proj = ReplicatedLinear(
            self.hidden_size,
            self.n_head,
            bias=False,
        ).to(torch.float32)

    def forward(
        self,
        hidden_states: torch.Tensor,
        q_c: torch.Tensor,
        positions: torch.Tensor,
        rotary_emb: DeepseekScalingRotaryEmbedding,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        q = self.wq_b(q_c).view(-1, self.n_head, self.head_dim)
        q_pe, q_nope = torch.split(
            q,
            [self.rope_dim, self.head_dim - self.rope_dim],
            dim=-1,
        )

        k = self.k_norm(self.wk(hidden_states))
        k_pe, k_nope = torch.split(
            k,
            [self.rope_dim, self.head_dim - self.rope_dim],
            dim=-1,
        )
        q_pe, k_pe = rotary_emb(positions, q_pe, k_pe.unsqueeze(1))
        q = torch.cat((q_pe, q_nope), dim=-1)
        k = torch.cat((k_pe.squeeze(1), k_nope), dim=-1)
        q = _rotate_activation(q).to(q.dtype)
        k = _rotate_activation(k).to(k.dtype)

        weights = self.weights_proj(hidden_states.float())
        weights = weights * self.softmax_scale * (self.n_head ** -0.5)
        return q, k, weights.to(hidden_states.dtype)


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
        self.register_parameter("wd_qkv", None)
        self.w_uk_t = None
        self.w_uv = None
        self.fuse_qkv_a = _env_flag("NANOVLLM_FUSE_QKV_A", True)
        self.free_kv_b_proj = _env_flag("NANOVLLM_FREE_KV_B_PROJ", True)
        self.q_up_bmm_trans_max_tokens = _env_int(
            "NANOVLLM_Q_UP_BMM_TRANS_MAX_TOKENS",
            1,
        )
        self.decode_timing_sync = _env_flag(
            "NANOVLLM_DECODE_LAYER_TIMING_SYNC",
            True,
        )
        self.use_npu_sfa_decode = _env_flag(
            "NANOVLLM_ENABLE_NPU_SFA_DECODE",
            False,
        )
        self.decode_attention_backend = os.environ.get(
            "NANOVLLM_DECODE_ATTENTION_BACKEND",
            "mla",
        ).strip().lower()
        if self.use_npu_sfa_decode:
            self.decode_attention_backend = "sfa"
        if self.decode_attention_backend not in ("mla", "sfa", "torch"):
            raise ValueError(
                "NANOVLLM_DECODE_ATTENTION_BACKEND must be one of "
                "'mla', 'sfa', or 'torch'."
            )
        self.compare_npu_sfa_decode = _env_flag(
            "NANOVLLM_COMPARE_NPU_SFA_DECODE",
            False,
        )
        self.sfa_dump_dir = os.environ.get("NANOVLLM_DUMP_NPU_SFA_INPUTS")
        self.sfa_dump_max_calls = _env_int("NANOVLLM_DUMP_NPU_SFA_MAX_CALLS", 1)
        self.sfa_dump_count = 0
        self.log_npu_sfa_inputs = _env_flag("NANOVLLM_LOG_NPU_SFA_INPUTS", False)
        self.log_npu_sfa_timing = _env_flag(
            "NANOVLLM_LOG_NPU_SFA_TIMING",
            False,
        ) and _profile_layer_selected(self.layer_id, self.num_layers)
        self._npu_sfa_logged = {"prefill": False, "decode": False}
        self.log_decode_layer_timing = _env_flag(
            "NANOVLLM_LOG_DECODE_LAYER_TIMING",
            False,
        ) and _profile_layer_selected(self.layer_id, self.num_layers)
        self.last_decode_attention_op_time = 0.0
        self.last_decode_attention_detail: dict[str, float] = {}
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
            "indexer": 0.0,
            "cache": 0.0,
            "q_up": 0.0,
            "decode_attention_op": 0.0,
            "v_up": 0.0,
            "o_linear": 0.0,
            "o_all_reduce": 0.0,
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
    ) -> None:
        self.ckv_cache = ckv_cache
        self.kpe_cache = kpe_cache
        self.index_cache = index_cache

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

    def _store_cache(
        self,
        ckv: torch.Tensor,
        kpe: torch.Tensor,
        index_k: torch.Tensor | None,
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
        if index_k is None:
            return
        self.index_cache.view(-1, self.indexer.head_dim).index_copy_(
            0,
            flat_slots,
            index_k,
        )

    def _get_sequence_slots_from_block_table(
        self,
        block_table_row: torch.Tensor,
        seq_len: int,
    ) -> torch.Tensor:
        num_blocks = (seq_len + self.block_size - 1) // self.block_size
        blocks = block_table_row[:num_blocks].to(torch.long).tolist()
        slots: list[int] = []
        remaining = seq_len
        for block in blocks:
            take = min(self.block_size, remaining)
            start = block * self.block_size
            slots.extend(range(start, start + take))
            remaining -= take
        return torch.tensor(
            slots,
            device=self.ckv_cache.device,
            dtype=torch.long,
        )

    @staticmethod
    def _tensor_desc(name: str, tensor: torch.Tensor) -> str:
        return (
            f"{name}=shape={tuple(tensor.shape)} dtype={tensor.dtype} "
            f"device={tensor.device} contiguous={tensor.is_contiguous()} "
            f"stride={tuple(tensor.stride())} storage_offset={tensor.storage_offset()}"
        )

    @staticmethod
    def _tensor_range_desc(name: str, tensor: torch.Tensor) -> str:
        desc = DeepseekV32DSAAttention._tensor_desc(name, tensor)
        if tensor.numel() == 0:
            return f"{desc} empty=True"
        try:
            min_value = tensor.min().item()
            max_value = tensor.max().item()
        except Exception as exc:
            return f"{desc} range_error={exc!r}"
        return f"{desc} min={min_value:.6g} max={max_value:.6g}"

    def _maybe_log_npu_sfa_inputs(
        self,
        phase: str,
        query: torch.Tensor,
        key: torch.Tensor,
        sparse_indices: torch.Tensor,
        block_table: torch.Tensor,
        actual_seq_lengths_query: torch.Tensor,
        actual_seq_lengths_key: torch.Tensor,
        query_rope: torch.Tensor,
        key_rope: torch.Tensor,
    ) -> None:
        if not self.log_npu_sfa_inputs or self._npu_sfa_logged[phase]:
            return
        if not _is_rank0():
            return
        self._npu_sfa_logged[phase] = True
        logger.info(
            "NPU SFA input summary: phase=%s layer=%d %s %s %s %s "
            "%s %s %s %s sparse_count=%d scale=%.8f",
            phase,
            self.layer_id,
            self._tensor_desc("query", query),
            self._tensor_desc("key", key),
            self._tensor_desc("sparse_indices", sparse_indices),
            self._tensor_desc("block_table", block_table),
            self._tensor_desc("actual_seq_lengths_query", actual_seq_lengths_query),
            self._tensor_desc("actual_seq_lengths_key", actual_seq_lengths_key),
            self._tensor_desc("query_rope", query_rope),
            self._tensor_desc("key_rope", key_rope),
            SFA_SPARSE_COUNT,
            self.scale,
        )

    def _maybe_dump_npu_sfa_inputs(
        self,
        phase: str,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        sparse_indices: torch.Tensor,
        block_table: torch.Tensor,
        actual_seq_lengths_query: torch.Tensor,
        actual_seq_lengths_key: torch.Tensor,
        query_rope: torch.Tensor,
        key_rope: torch.Tensor,
    ) -> None:
        if not self.sfa_dump_dir:
            return
        if self.sfa_dump_count >= self.sfa_dump_max_calls:
            return
        if not _profile_layer_selected(self.layer_id, self.num_layers):
            return
        dump_dir = Path(self.sfa_dump_dir)
        dump_dir.mkdir(parents=True, exist_ok=True)
        try:
            rank = dist.get_rank() if dist.is_initialized() else 0
        except Exception:
            rank = 0
        path = (
            dump_dir
            / f"sfa_rank{rank}_layer{self.layer_id:03d}_"
            f"seq{self.sfa_dump_count:03d}_{phase}.pt"
        )
        payload = {
            "rank": rank,
            "layer_id": self.layer_id,
            "phase": phase,
            "scale_value": float(self.scale),
            "sparse_block_size": 1,
            "sparse_mode": 3,
            "sparse_count": SFA_SPARSE_COUNT,
            "layout_query": "TND",
            "layout_kv": "PA_BSND",
            "query": query.detach().cpu(),
            "key": key.detach().cpu(),
            "value": value.detach().cpu(),
            "sparse_indices": sparse_indices.detach().cpu(),
            "block_table": block_table.detach().cpu(),
            "actual_seq_lengths_query": actual_seq_lengths_query.detach().cpu(),
            "actual_seq_lengths_kv": actual_seq_lengths_key.detach().cpu(),
            "query_rope": query_rope.detach().cpu(),
            "key_rope": key_rope.detach().cpu(),
        }
        torch.save(payload, path)
        self.sfa_dump_count += 1
        if _is_rank0():
            logger.info("Dumped NPU SFA inputs to %s", path)

    def _log_npu_sfa_compare(
        self,
        phase: str,
        actual: torch.Tensor,
        expected: torch.Tensor,
    ) -> None:
        if not _is_rank0():
            return
        diff = (actual.float() - expected.float()).abs()
        max_abs = float(diff.max().item()) if diff.numel() else 0.0
        denom = expected.float().abs().max().clamp_min(1e-6)
        max_rel = float((diff.max() / denom).item()) if diff.numel() else 0.0
        logger.info(
            "NPU SFA compare: phase=%s layer=%d max_abs=%.6g max_rel=%.6g "
            "actual_shape=%s expected_shape=%s",
            phase,
            self.layer_id,
            max_abs,
            max_rel,
            tuple(actual.shape),
            tuple(expected.shape),
        )

    def _npu_sfa_forward(
        self,
        ql_nope: torch.Tensor,
        q_pe: torch.Tensor,
        q_index: torch.Tensor,
        weights: torch.Tensor,
        actual_seq_lengths_query: torch.Tensor,
        actual_seq_lengths_key: torch.Tensor,
        block_table: torch.Tensor,
        phase: str,
    ) -> torch.Tensor:
        log_timing = self.log_npu_sfa_timing
        if log_timing:
            _profile_sync(ql_nope.device)
            start = perf_counter()
        ckv_cache = self.ckv_cache.transpose(1, 2).contiguous()
        kpe_cache = self.kpe_cache.transpose(1, 2).contiguous()
        topk_indices = ascend_ops.npu_lightning_indexer(
            query=q_index,
            key=self.index_cache,
            weights=weights,
            actual_seq_lengths_query=actual_seq_lengths_query,
            actual_seq_lengths_key=actual_seq_lengths_key,
            block_table=block_table,
            layout_query="TND",
            layout_key="PA_BSND",
            sparse_count=SFA_SPARSE_COUNT,
            sparse_mode=3,
        )
        if log_timing:
            _profile_sync(ql_nope.device)
            logger.info(
                "NPU SFA timing: rank=%d phase=%s layer=%d op=indexer "
                "elapsed=%.6fs %s actual_seq_lengths_query=%s "
                "actual_seq_lengths_key=%s block_table_shape=%s",
                _rank_id(),
                phase,
                self.layer_id,
                perf_counter() - start,
                self._tensor_range_desc("topk_indices", topk_indices),
                actual_seq_lengths_query.detach().cpu().tolist(),
                actual_seq_lengths_key.detach().cpu().tolist(),
                tuple(block_table.shape),
            )
            start = perf_counter()
        self._maybe_log_npu_sfa_inputs(
            phase,
            ql_nope,
            ckv_cache,
            topk_indices,
            block_table,
            actual_seq_lengths_query,
            actual_seq_lengths_key,
            q_pe,
            kpe_cache,
        )
        self._maybe_dump_npu_sfa_inputs(
            phase,
            ql_nope,
            ckv_cache,
            ckv_cache,
            topk_indices,
            block_table,
            actual_seq_lengths_query,
            actual_seq_lengths_key,
            q_pe,
            kpe_cache,
        )
        if phase == "decode" and self.log_decode_layer_timing:
            _profile_sync(ql_nope.device)
            attention_op_start = perf_counter()
        latent = ascend_ops.npu_sparse_flash_attention(
            query=ql_nope,
            key=ckv_cache,
            value=ckv_cache,
            sparse_indices=topk_indices,
            scale_value=float(self.scale),
            sparse_block_size=1,
            block_table=block_table,
            actual_seq_lengths_query=actual_seq_lengths_query,
            actual_seq_lengths_kv=actual_seq_lengths_key,
            query_rope=q_pe,
            key_rope=kpe_cache,
            layout_query="TND",
            layout_kv="PA_BSND",
            sparse_mode=3,
        )
        if phase == "decode" and self.log_decode_layer_timing:
            _profile_sync(ql_nope.device)
            self.last_decode_attention_op_time = (
                perf_counter() - attention_op_start
            )
            self.last_decode_attention_detail["decode_attention_op"] = (
                self.last_decode_attention_op_time
            )
        if log_timing:
            _profile_sync(ql_nope.device)
            logger.info(
                "NPU SFA timing: rank=%d phase=%s layer=%d op=sfa "
                "elapsed=%.6fs %s",
                _rank_id(),
                phase,
                self.layer_id,
                perf_counter() - start,
                self._tensor_range_desc("latent", latent),
            )
            start = perf_counter()
        if phase == "decode":
            profile_decode = self.log_decode_layer_timing
            v_up_start = self._decode_timer_start(profile_decode, latent.device)
            output = self._v_up_proj(latent)
            self._decode_timer_end(
                profile_decode,
                "v_up",
                v_up_start,
                output.device,
            )
        else:
            output = self._v_up_proj(latent)
        if log_timing:
            _profile_sync(output.device)
            logger.info(
                "NPU SFA timing: rank=%d phase=%s layer=%d op=v_up "
                "elapsed=%.6fs %s",
                _rank_id(),
                phase,
                self.layer_id,
                perf_counter() - start,
                self._tensor_range_desc("output", output),
            )
        return output

    def _compute_npu_indexer_indices(
        self,
        q_index: torch.Tensor,
        weights: torch.Tensor,
        block_table_row: torch.Tensor,
        valid_len: int,
    ) -> torch.Tensor:
        device = q_index.device
        seq_lengths_query = torch.ones((1,), dtype=torch.int32, device=device)
        seq_lengths_key = torch.tensor([valid_len], dtype=torch.int32, device=device)
        topk_indices = ascend_ops.npu_lightning_indexer(
            query=q_index.unsqueeze(0),
            key=self.index_cache,
            weights=weights.unsqueeze(0),
            actual_seq_lengths_query=seq_lengths_query,
            actual_seq_lengths_key=seq_lengths_key,
            block_table=block_table_row.unsqueeze(0),
            layout_query="TND",
            layout_key="PA_BSND",
            sparse_count=SFA_SPARSE_COUNT,
            sparse_mode=3,
        )
        topk = min(SFA_SPARSE_COUNT, valid_len)
        return topk_indices.flatten()[:topk].to(torch.long)

    def _sparse_attention_single(
        self,
        ql_nope: torch.Tensor,
        q_pe: torch.Tensor,
        selected_ckv: torch.Tensor,
        selected_kpe: torch.Tensor,
    ) -> torch.Tensor:
        scores = torch.einsum("hl,sl->hs", ql_nope.float(), selected_ckv.float())
        scores = scores + torch.einsum(
            "hr,sr->hs",
            q_pe.float(),
            selected_kpe.float(),
        )
        scores = scores * self.scale
        probs = torch.softmax(scores, dim=-1).to(selected_ckv.dtype)
        latent = torch.einsum("hs,sl->hl", probs, selected_ckv)
        return self._v_up_proj(latent.unsqueeze(0)).reshape(-1)

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
        output = torch.bmm(latent_by_head, self.w_uv)
        return output.transpose(0, 1).reshape(num_tokens, -1)

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
        log_timing = self.log_npu_sfa_timing
        if log_timing:
            _profile_sync(ql_nope.device)
            start = perf_counter()
        mla_result = torch_npu.npu_fused_infer_attention_score(
            ql_nope,
            self.ckv_cache,
            self.ckv_cache,
            query_rope=q_pe,
            key_rope=self.kpe_cache,
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
        if log_timing:
            _profile_sync(ql_nope.device)
            logger.info(
                "NPU MLA timing: rank=%d phase=prefill layer=%d op=mla "
                "elapsed=%.6fs %s actual_seq_lengths_query=%s "
                "actual_seq_lengths_key=%s block_table_shape=%s",
                _rank_id(),
                self.layer_id,
                perf_counter() - start,
                self._tensor_range_desc("latent", latent),
                actual_seq_lengths_query.detach().cpu().tolist(),
                actual_seq_lengths_key.detach().cpu().tolist(),
                tuple(context.block_tables.shape),
            )
            start = perf_counter()
        output = self._v_up_proj(latent)
        if log_timing:
            _profile_sync(output.device)
            logger.info(
                "NPU MLA timing: rank=%d phase=prefill layer=%d op=v_up "
                "elapsed=%.6fs %s",
                _rank_id(),
                self.layer_id,
                perf_counter() - start,
                self._tensor_range_desc("output", output),
            )
        return output

    def _decode_forward_loop(
        self,
        ql_nope: torch.Tensor,
        q_pe: torch.Tensor,
        q_index: torch.Tensor,
        weights: torch.Tensor,
    ) -> torch.Tensor:
        context = get_context()
        outputs: list[torch.Tensor] = []
        for seq_idx, seq_len in enumerate(context.context_lens.to(torch.long).tolist()):
            seq_slots = self._get_sequence_slots_from_block_table(
                context.block_tables[seq_idx],
                seq_len,
            )
            seq_ckv = self.ckv_cache.view(-1, self.kv_lora_rank).index_select(
                0, seq_slots
            )
            seq_kpe = self.kpe_cache.view(-1, self.qk_rope_head_dim).index_select(
                0, seq_slots
            )
            selected = self._compute_npu_indexer_indices(
                q_index[seq_idx],
                weights[seq_idx],
                context.block_tables[seq_idx],
                seq_len,
            )
            outputs.append(
                self._sparse_attention_single(
                    ql_nope[seq_idx],
                    q_pe[seq_idx],
                    seq_ckv.index_select(0, selected),
                    seq_kpe.index_select(0, selected),
                )
            )
        return torch.stack(outputs, dim=0)

    def _decode_forward(
        self,
        ql_nope: torch.Tensor,
        q_pe: torch.Tensor,
        q_index: torch.Tensor | None,
        weights: torch.Tensor | None,
    ) -> torch.Tensor:
        self.last_decode_attention_op_time = 0.0
        if self.decode_attention_backend == "mla":
            return self._decode_forward_mla(ql_nope, q_pe)
        if q_index is None or weights is None:
            raise RuntimeError(
                "Sparse decode backends require indexer outputs, but they were "
                "not computed for this forward pass."
            )
        if self.decode_attention_backend == "sfa":
            sfa_output = self._decode_forward_sfa(
                ql_nope,
                q_pe,
                q_index,
                weights,
            )
            if self.compare_npu_sfa_decode:
                reference = self._decode_forward_torch(
                    ql_nope,
                    q_pe,
                    q_index,
                    weights,
                )
                self._log_npu_sfa_compare("decode", sfa_output, reference)
            return sfa_output
        return self._decode_forward_torch(ql_nope, q_pe, q_index, weights)

    def _decode_forward_mla(
        self,
        ql_nope: torch.Tensor,
        q_pe: torch.Tensor,
    ) -> torch.Tensor:
        context = get_context()
        batch_size = int(ql_nope.shape[0])
        actual_seq_lengths_query = context.actual_seq_lengths_query
        actual_seq_lengths_key = context.actual_seq_lengths_kv
        assert actual_seq_lengths_query is not None
        assert actual_seq_lengths_key is not None
        profile_decode = self.log_decode_layer_timing
        start = self._decode_timer_start(profile_decode, ql_nope.device)
        mla_result = torch_npu.npu_fused_infer_attention_score(
            ql_nope,
            self.ckv_cache,
            self.ckv_cache,
            query_rope=q_pe,
            key_rope=self.kpe_cache,
            num_heads=self.num_local_heads,
            num_key_value_heads=1,
            input_layout="TND",
            atten_mask=None,
            sparse_mode=0,
            scale=float(self.scale),
            antiquant_mode=0,
            antiquant_scale=None,
            block_table=context.block_tables[:batch_size],
            block_size=self.block_size,
            softmax_lse_flag=False,
            actual_seq_lengths=actual_seq_lengths_query,
            actual_seq_lengths_kv=actual_seq_lengths_key,
        )
        latent = _first_tensor(mla_result)
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
        output = self._v_up_proj(latent)
        self._decode_timer_end(profile_decode, "v_up", start, output.device)
        return output

    def _decode_forward_sfa(
        self,
        ql_nope: torch.Tensor,
        q_pe: torch.Tensor,
        q_index: torch.Tensor,
        weights: torch.Tensor,
    ) -> torch.Tensor:
        context = get_context()
        batch_size = int(ql_nope.shape[0])
        actual_seq_lengths_query = torch.arange(
            1,
            batch_size + 1,
            dtype=torch.int32,
            device=ql_nope.device,
        )
        actual_seq_lengths_key = context.context_lens[:batch_size]
        return self._npu_sfa_forward(
            ql_nope[:batch_size],
            q_pe[:batch_size],
            q_index[:batch_size],
            weights[:batch_size],
            actual_seq_lengths_query,
            actual_seq_lengths_key,
            context.block_tables[:batch_size],
            "decode",
        )

    def _decode_forward_torch(
        self,
        ql_nope: torch.Tensor,
        q_pe: torch.Tensor,
        q_index: torch.Tensor,
        weights: torch.Tensor,
    ) -> torch.Tensor:
        profile_decode = self.log_decode_layer_timing
        start = self._decode_timer_start(profile_decode, ql_nope.device)
        output = self._decode_forward_loop(ql_nope, q_pe, q_index, weights)
        self._decode_timer_end(
            profile_decode,
            "decode_attention_op",
            start,
            output.device,
        )
        self.last_decode_attention_op_time = self.last_decode_attention_detail[
            "decode_attention_op"
        ]
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

        use_sparse_decode = self.decode_attention_backend in ("sfa", "torch")
        if use_sparse_decode:
            start = self._decode_timer_start(profile_decode, hidden_states.device)
            q_index, index_k, weights = self.indexer(
                hidden_states,
                q_c,
                positions,
                self.indexer_rotary_emb,
            )
            self._decode_timer_end(
                profile_decode,
                "indexer",
                start,
                hidden_states.device,
            )
        else:
            q_index = None
            index_k = None
            weights = None

        start = self._decode_timer_start(profile_decode, ckv.device)
        self._store_cache(ckv, k_pe, index_k)
        self._decode_timer_end(profile_decode, "cache", start, ckv.device)

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
            logger.info(
                "Decode layer timing: rank=%d layer=%d tokens=%d "
                "attention_total=%.6fs qkv_a=%.6fs q_norm=%.6fs "
                "q_b=%.6fs kv_rope=%.6fs kv_split=%.6fs kv_norm=%.6fs "
                "rotary=%.6fs k_squeeze=%.6fs indexer=%.6fs cache=%.6fs "
                "q_up=%.6fs decode_attention_op=%.6fs v_up=%.6fs "
                "o_proj=%.6fs o_linear=%.6fs o_all_reduce=%.6fs "
                "attention_gap=%.6fs "
                "moe_total=%.6fs mlp_kind=%s moe_backend=%s backend=%s",
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
                attention_detail["indexer"],
                attention_detail["cache"],
                attention_detail["q_up"],
                attention_detail["decode_attention_op"],
                attention_detail["v_up"],
                o_proj_total,
                attention_detail["o_linear"],
                attention_detail["o_all_reduce"],
                attention_gap,
                perf_counter() - mlp_start,
                self.mlp_kind,
                getattr(self.mlp, "moe_backend", "dense"),
                self.self_attn.decode_attention_backend,
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
