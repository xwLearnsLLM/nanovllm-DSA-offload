import json
import math
import os
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn
from transformers import PretrainedConfig

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
        return query, key


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
        self.scoring_func = str(getattr(config, "scoring_func", "softmax"))
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
        trace_dir = os.environ.get("NANOVLLM_MOE_TRACE_DIR")
        self.trace_dir = Path(trace_dir) if trace_dir else None
        if self.trace_dir is not None:
            self.trace_dir.mkdir(parents=True, exist_ok=True)
        self.trace_prefill_only = (
            os.environ.get("NANOVLLM_MOE_TRACE_PREFILL_ONLY", "1").lower()
            in ("1", "true", "yes", "on")
        )
        self.trace_max_tokens_per_call = int(
            os.environ.get("NANOVLLM_MOE_TRACE_MAX_TOKENS_PER_CALL", "0")
        )
        self.trace_counter = 0

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

    def _maybe_trace_inputs(self, hidden_states: torch.Tensor) -> None:
        if self.trace_dir is None:
            return
        if dist.is_initialized() and dist.get_rank() != 0:
            return
        context = get_context()
        if self.trace_prefill_only and not context.is_prefill:
            return
        traced_states = hidden_states.detach()
        if self.trace_max_tokens_per_call > 0:
            traced_states = traced_states[: self.trace_max_tokens_per_call]
        payload = {
            "hidden_states": traced_states.to(torch.bfloat16).cpu(),
            "is_prefill": bool(context.is_prefill),
            "layer_idx": self.layer_idx,
        }
        file_path = (
            self.trace_dir
            / f"layer_{self.layer_idx:03d}_call_{self.trace_counter:06d}.pt"
        )
        torch.save(payload, file_path)
        self.trace_counter += 1

    def _grouped_topk(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del hidden_states
        grouped_topk = self._grouped_topk_npu(router_logits)
        if grouped_topk is not None:
            return grouped_topk

        router_logits = router_logits.float()
        if self.scoring_func == "softmax":
            scores = torch.softmax(router_logits, dim=-1)
        elif self.scoring_func == "sigmoid":
            scores = router_logits.sigmoid()
        else:
            raise ValueError(f"Unsupported scoring function: {self.scoring_func}")

        bias = getattr(self.gate, "e_score_correction_bias", None)
        if self.num_expert_group > 1:
            num_tokens = scores.shape[0]
            experts_per_group = scores.shape[-1] // self.num_expert_group
            if experts_per_group * self.num_expert_group != scores.shape[-1]:
                raise ValueError(
                    "n_group must divide the number of routed experts."
                )
            if bias is not None:
                original_scores = scores
                scores = scores + bias.unsqueeze(0)
                group_take = min(2, experts_per_group)
                group_scores = (
                    scores.view(num_tokens, self.num_expert_group, experts_per_group)
                    .topk(group_take, dim=-1)[0]
                    .sum(dim=-1)
                )
            else:
                group_scores = scores.view(
                    num_tokens, self.num_expert_group, experts_per_group
                ).max(dim=-1).values
            topk_group = min(self.topk_group, self.num_expert_group)
            top_k = min(self.top_k, topk_group * experts_per_group)
            group_idx = torch.topk(
                group_scores, k=topk_group, dim=-1, sorted=False
            ).indices
            group_mask = torch.zeros_like(group_scores)
            group_mask.scatter_(1, group_idx, 1)
            score_mask = (
                group_mask.unsqueeze(-1)
                .expand(num_tokens, self.num_expert_group, experts_per_group)
                .reshape(num_tokens, -1)
            )
            tmp_scores = scores.masked_fill(~score_mask.bool(), float("-inf"))
            if bias is not None:
                topk_ids = torch.topk(
                    tmp_scores, k=top_k, dim=-1, sorted=False
                ).indices
                topk_weights = original_scores.gather(1, topk_ids)
            else:
                topk_weights, topk_ids = torch.topk(
                    tmp_scores, k=top_k, dim=-1, sorted=False
                )
        else:
            if bias is not None:
                original_scores = scores
                topk_ids = torch.topk(
                    scores + bias.unsqueeze(0),
                    k=self.top_k,
                    dim=-1,
                    sorted=False,
                ).indices
                topk_weights = original_scores.gather(1, topk_ids)
            else:
                topk_weights, topk_ids = torch.topk(
                    scores, k=self.top_k, dim=-1, sorted=False
                )

        if self.renormalize:
            topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
        if self.routed_scaling_factor != 1.0:
            topk_weights = topk_weights * self.routed_scaling_factor
        return topk_weights, topk_ids

    def _grouped_topk_npu(
        self,
        router_logits: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        if router_logits.device.type != "npu":
            return None

        try:
            import torch_npu  # type: ignore
        except Exception:
            return None

        gating_op = getattr(torch_npu, "npu_moe_gating_top_k", None)
        if gating_op is None:
            return None

        bias = getattr(self.gate, "e_score_correction_bias", None)
        group_select_mode = 1 if bias is not None else 0
        norm_type = 1 if self.scoring_func == "sigmoid" else 0
        if self.scoring_func not in ("softmax", "sigmoid"):
            return None

        try:
            topk_weights, topk_ids, _ = gating_op(
                router_logits.float(),
                self.top_k,
                bias=bias,
                k_group=self.topk_group,
                group_count=self.num_expert_group,
                group_select_mode=group_select_mode,
                renorm=1 if self.renormalize else 0,
                norm_type=norm_type,
                out_flag=False,
                routed_scaling_factor=self.routed_scaling_factor,
            )
        except Exception:
            return None

        return topk_weights.float(), topk_ids.long()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        sequence_length, hidden_dim = hidden_states.shape
        self._maybe_trace_inputs(hidden_states)
        shared_output = self.shared_experts(hidden_states)
        router_logits = self.gate(hidden_states)
        routing_weights, selected_experts = self._grouped_topk(
            hidden_states,
            router_logits,
        )
        routing_weights = routing_weights.to(hidden_states.dtype)

        routed_hidden_states = torch.zeros(
            hidden_states.shape,
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )
        expert_mask = torch.nn.functional.one_hot(
            selected_experts, num_classes=self.num_experts
        ).permute(2, 1, 0)
        expert_hitted = torch.greater(
            expert_mask.sum(dim=(-1, -2)), 0
        ).nonzero(as_tuple=False).flatten()

        for expert_idx in expert_hitted.tolist():
            if expert_idx not in self.local_expert_id_set:
                continue
            expert_layer = self.experts[str(expert_idx)]
            idx, top_x = torch.where(expert_mask[expert_idx])
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

        if self.enable_expert_parallel and self.ep_size > 1:
            dist.all_reduce(routed_hidden_states)

        final_hidden_states = routed_hidden_states + shared_output
        return final_hidden_states.view(sequence_length, hidden_dim)


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
        )

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

        weights = self.weights_proj(hidden_states)
        weights = weights * self.softmax_scale * (self.n_head ** -0.5)
        return q, k, weights


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
        self.index_topk = int(config.index_topk)
        self.layer_id = layer_idx
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
        self.w_uk_t = None
        self.w_uv = None

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
        weight = self.kv_b_proj.weight.data.view(
            self.num_local_heads,
            self.qk_nope_head_dim + self.v_head_dim,
            self.kv_lora_rank,
        )
        self.w_uk_t = weight[:, : self.qk_nope_head_dim, :].contiguous()
        self.w_uv = weight[:, self.qk_nope_head_dim :, :].transpose(1, 2).contiguous()

    @property
    def block_size(self) -> int:
        return int(self.ckv_cache.shape[1])

    def _flat_slots(self) -> torch.Tensor:
        slot_mapping = get_context().slot_mapping
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
        index_k: torch.Tensor,
    ) -> None:
        flat_slots = self._flat_slots()
        self.ckv_cache.view(-1, self.kv_lora_rank).index_copy_(
            0,
            flat_slots,
            ckv.to(self.ckv_cache.dtype),
        )
        self.kpe_cache.view(-1, self.qk_rope_head_dim).index_copy_(
            0,
            flat_slots,
            kpe.to(self.kpe_cache.dtype),
        )
        self.index_cache.view(-1, self.indexer.head_dim).index_copy_(
            0,
            flat_slots,
            index_k.to(self.index_cache.dtype),
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

    def _compute_topk_indices(
        self,
        q_index: torch.Tensor,
        weights: torch.Tensor,
        key_cache: torch.Tensor,
        valid_len: int,
    ) -> torch.Tensor:
        key_cache = key_cache[:valid_len].float()
        q_index = q_index.float()
        weights = weights.float()
        # Upstream DeepSeek-V3.2 indexer applies ReLU to the per-head
        # retrieval logits before weighting heads together.
        scores = torch.einsum("hd,sd->hs", q_index, key_cache)
        scores = (scores.relu() * weights.unsqueeze(-1)).sum(dim=0)
        topk = min(self.index_topk, valid_len)
        return torch.topk(scores, k=topk, dim=-1).indices

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
        out = torch.einsum("hl,hlv->hv", latent, self.w_uv)
        return out.reshape(-1)

    def _prefill_forward(
        self,
        ql_nope: torch.Tensor,
        q_pe: torch.Tensor,
        q_index: torch.Tensor,
        weights: torch.Tensor,
    ) -> torch.Tensor:
        context = get_context()
        cu_seqlens = context.cu_seqlens_q.to(torch.long).tolist()
        slot_mapping = context.slot_mapping.to(torch.long)
        outputs: list[torch.Tensor] = []
        for seq_idx in range(len(cu_seqlens) - 1):
            seq_start = cu_seqlens[seq_idx]
            seq_end = cu_seqlens[seq_idx + 1]
            seq_slots = slot_mapping[seq_start:seq_end]
            seq_ckv = self.ckv_cache.view(-1, self.kv_lora_rank).index_select(
                0, seq_slots
            )
            seq_kpe = self.kpe_cache.view(-1, self.qk_rope_head_dim).index_select(
                0, seq_slots
            )
            seq_index = self.index_cache.view(-1, self.indexer.head_dim).index_select(
                0, seq_slots
            )
            for token_idx in range(seq_end - seq_start):
                selected = self._compute_topk_indices(
                    q_index[seq_start + token_idx],
                    weights[seq_start + token_idx],
                    seq_index,
                    token_idx + 1,
                )
                outputs.append(
                    self._sparse_attention_single(
                        ql_nope[seq_start + token_idx],
                        q_pe[seq_start + token_idx],
                        seq_ckv.index_select(0, selected),
                        seq_kpe.index_select(0, selected),
                    )
                )
        return torch.stack(outputs, dim=0)

    def _decode_forward(
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
            seq_index = self.index_cache.view(-1, self.indexer.head_dim).index_select(
                0, seq_slots
            )
            selected = self._compute_topk_indices(
                q_index[seq_idx],
                weights[seq_idx],
                seq_index,
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

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        if self.w_uk_t is None or self.w_uv is None:
            self.post_load_prepare()

        q_c = self.q_a_layernorm(self.q_a_proj(hidden_states))
        q = self.q_b_proj(q_c).view(
            -1, self.num_local_heads, self.qk_head_dim
        )
        q_nope, q_pe = torch.split(
            q,
            [self.qk_nope_head_dim, self.qk_rope_head_dim],
            dim=-1,
        )

        kv = self.kv_a_proj_with_mqa(hidden_states)
        ckv, k_pe = torch.split(
            kv,
            [self.kv_lora_rank, self.qk_rope_head_dim],
            dim=-1,
        )
        ckv = self.kv_a_layernorm(ckv)
        q_pe, k_pe = self.rotary_emb(positions, q_pe, k_pe.unsqueeze(1))
        k_pe = k_pe.squeeze(1)

        q_index, index_k, weights = self.indexer(
            hidden_states,
            q_c,
            positions,
            self.indexer_rotary_emb,
        )
        self._store_cache(ckv, k_pe, index_k)

        ql_nope = torch.einsum("thp,hpl->thl", q_nope, self.w_uk_t)
        if get_context().is_prefill:
            attn_output = self._prefill_forward(
                ql_nope, q_pe, q_index, weights
            )
        else:
            attn_output = self._decode_forward(
                ql_nope, q_pe, q_index, weights
            )
        return self.o_proj(attn_output)


class DeepseekV32DecoderLayer(nn.Module):
    def __init__(self, config: DeepseekV32Config, layer_idx: int) -> None:
        super().__init__()
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
        hidden_states = self.self_attn(positions, hidden_states)
        hidden_states, residual = self.post_attention_layernorm(
            hidden_states, residual
        )
        hidden_states = self.mlp(hidden_states)
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
