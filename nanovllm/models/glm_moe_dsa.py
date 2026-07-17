from __future__ import annotations

from collections.abc import Callable
import re

import torch
import torch.distributed as dist
import torch_npu  # type: ignore
from torch import nn

import nanovllm.ops as ascend_ops
from nanovllm.layers.embed_head import ParallelLMHead, VocabParallelEmbedding
from nanovllm.layers.layernorm import RMSNorm
from nanovllm.layers.linear import ReplicatedLinear
from nanovllm.models.deepseek_v32 import (
    DeepseekV32DSAAttention,
    DeepseekV32MLP,
)
from nanovllm.models.glm_moe_dsa_config import GlmMoeDsaConfig
from nanovllm.utils.context import get_context
from nanovllm.utils.glm_quant import (
    GLM_BALANCED_MOE_EXPERT_IDS_KEY,
    balanced_moe_expert_ids,
    float32_scale_to_int64_bits,
    should_skip_glm_checkpoint_weight,
)
from nanovllm.utils.loader import WeightTarget


_EXPERT_WEIGHT_RE = re.compile(
    r"^model\.layers\.(?P<layer>\d+)\.mlp\.experts\."
    r"(?P<expert>\d+)\.(?P<projection>gate_proj|up_proj|down_proj)\."
    r"(?P<field>weight|weight_scale|weight_offset|scale_bias)$"
)
ACL_FORMAT_FRACTAL_NZ = 29


class GlmW4A8SparseMoeBlock(nn.Module):
    """GLM routed experts backed directly by packed ModelSlim W4A8 weights."""

    def __init__(self, config: GlmMoeDsaConfig, layer_idx: int) -> None:
        super().__init__()
        self.layer_idx = int(layer_idx)
        self.hidden_size = int(config.hidden_size)
        self.moe_intermediate_size = int(config.moe_intermediate_size)
        self.hidden_act = str(config.hidden_act)
        self.num_experts = int(config.n_routed_experts)
        self.top_k = int(config.num_experts_per_tok)
        self.renormalize = bool(getattr(config, "norm_topk_prob", True))
        self.scoring_func = str(getattr(config, "scoring_func", "sigmoid"))
        self.routed_scaling_factor = float(
            getattr(config, "routed_scaling_factor", 1.0)
        )
        self.num_expert_group = int(getattr(config, "n_group", 1) or 1)
        self.topk_group = int(getattr(config, "topk_group", 1) or 1)
        self.num_shared_experts = int(
            getattr(config, "n_shared_experts", 1) or 1
        )
        self.enable_expert_parallel = bool(
            getattr(config, "nanovllm_enable_expert_parallel", False)
        )
        if not self.enable_expert_parallel:
            raise ValueError("GLM W4A8 routed experts require expert parallel.")
        self.ep_size = dist.get_world_size()
        self.ep_rank = dist.get_rank()
        if self.num_experts % self.ep_size:
            raise ValueError(
                "n_routed_experts must be divisible by the EP world size."
            )
        self.num_local_experts = self.num_experts // self.ep_size
        self.local_expert_start = self.ep_rank * self.num_local_experts
        self.local_expert_end = self.local_expert_start + self.num_local_experts
        self.local_expert_ids = tuple(
            range(self.local_expert_start, self.local_expert_end)
        )
        self.local_expert_id_set = set(self.local_expert_ids)

        # GLM routes in FP32. Only this small projection stays FP32; the model
        # hidden state and shared expert path remain BF16.
        self.gate = ReplicatedLinear(
            self.hidden_size, self.num_experts, bias=False
        )
        self.gate.weight.data = self.gate.weight.data.to(torch.float32)
        if getattr(config, "topk_method", None) == "noaux_tc":
            self.gate.e_score_correction_bias = nn.Parameter(
                torch.empty(self.num_experts, dtype=torch.float32)
            )
        else:
            self.gate.register_parameter("e_score_correction_bias", None)

        self.shared_experts = DeepseekV32MLP(
            hidden_size=self.hidden_size,
            intermediate_size=(
                self.moe_intermediate_size * self.num_shared_experts
            ),
            hidden_act=self.hidden_act,
            # Routed + shared results are reduced together below.
            reduce_results=False,
        )

        # version=1.0.0 stores two INT4 values in every INT8 value along the
        # logical output dimension. Allocate the checkpoint layout directly;
        # allocating temporary BF16 experts would exceed one card's HBM.
        self.w13_weight = nn.Parameter(
            torch.empty(
                self.num_local_experts,
                self.moe_intermediate_size,
                self.hidden_size,
                dtype=torch.int8,
            ),
            requires_grad=False,
        )
        self.w2_weight = nn.Parameter(
            torch.empty(
                self.num_local_experts,
                self.hidden_size // 2,
                self.moe_intermediate_size,
                dtype=torch.int8,
            ),
            requires_grad=False,
        )
        self.w13_weight_scale = nn.Parameter(
            torch.empty(
                self.num_local_experts,
                2 * self.moe_intermediate_size,
                1,
                dtype=torch.float32,
            ),
            requires_grad=False,
        )
        self.w2_weight_scale = nn.Parameter(
            torch.empty(
                self.num_local_experts,
                self.hidden_size,
                1,
                dtype=torch.float32,
            ),
            requires_grad=False,
        )
        self.w13_scale_bias = nn.Parameter(
            torch.empty(
                self.num_local_experts,
                2 * self.moe_intermediate_size,
                1,
                dtype=torch.float32,
            ),
            requires_grad=False,
        )
        self.w2_scale_bias = nn.Parameter(
            torch.empty(
                self.num_local_experts,
                self.hidden_size,
                16,
                dtype=torch.float32,
            ),
            requires_grad=False,
        )

        self.w13_weight.weight_loader = self._load_w13_weight
        self.w2_weight.weight_loader = self._load_w2_weight
        self.w13_weight_scale.weight_loader = self._load_w13_scale
        self.w2_weight_scale.weight_loader = self._load_w2_scale
        self.w13_scale_bias.weight_loader = self._load_w13_scale_bias
        self.w2_scale_bias.weight_loader = self._load_w2_scale_bias
        self._loaded_components: set[tuple[int, str, str]] = set()
        self._weights_processed = False

    def _check_local_index(self, local_idx: int) -> None:
        if not 0 <= local_idx < self.num_local_experts:
            raise IndexError(
                f"Local expert index {local_idx} is outside "
                f"[0, {self.num_local_experts})."
            )

    def _load_w13_weight(
        self,
        param: nn.Parameter,
        tensor: torch.Tensor,
        local_idx: int,
        projection: str,
    ) -> None:
        self._check_local_index(local_idx)
        shard_size = self.moe_intermediate_size // 2
        expected = (shard_size, self.hidden_size)
        if tuple(tensor.shape) != expected:
            raise ValueError(
                f"Unexpected {projection} packed weight shape "
                f"{tuple(tensor.shape)}; expected {expected}."
            )
        shard = 0 if projection == "gate_proj" else 1
        param.data[local_idx].narrow(
            0, shard * shard_size, shard_size
        ).copy_(tensor)
        self._loaded_components.add((local_idx, projection, "weight"))

    def _load_w2_weight(
        self,
        param: nn.Parameter,
        tensor: torch.Tensor,
        local_idx: int,
        projection: str,
    ) -> None:
        self._check_local_index(local_idx)
        expected = (self.hidden_size // 2, self.moe_intermediate_size)
        if projection != "down_proj" or tuple(tensor.shape) != expected:
            raise ValueError(
                f"Unexpected down_proj packed weight shape "
                f"{tuple(tensor.shape)}; expected {expected}."
            )
        param.data[local_idx].copy_(tensor)
        self._loaded_components.add((local_idx, projection, "weight"))

    def _load_w13_aux(
        self,
        param: nn.Parameter,
        tensor: torch.Tensor,
        local_idx: int,
        projection: str,
        field: str,
    ) -> None:
        self._check_local_index(local_idx)
        expected = (self.moe_intermediate_size, 1)
        if tuple(tensor.shape) != expected:
            raise ValueError(
                f"Unexpected {projection}.{field} shape {tuple(tensor.shape)}; "
                f"expected {expected}."
            )
        shard = 0 if projection == "gate_proj" else 1
        param.data[local_idx].narrow(
            0,
            shard * self.moe_intermediate_size,
            self.moe_intermediate_size,
        ).copy_(tensor)
        self._loaded_components.add((local_idx, projection, field))

    def _load_w13_scale(
        self, param, tensor, local_idx: int, projection: str
    ) -> None:
        self._load_w13_aux(
            param, tensor, local_idx, projection, "weight_scale"
        )

    def _load_w13_scale_bias(
        self, param, tensor, local_idx: int, projection: str
    ) -> None:
        self._load_w13_aux(
            param, tensor, local_idx, projection, "scale_bias"
        )

    def _load_w2_scale(
        self,
        param: nn.Parameter,
        tensor: torch.Tensor,
        local_idx: int,
        projection: str,
    ) -> None:
        self._check_local_index(local_idx)
        expected = (self.hidden_size, 1)
        if projection != "down_proj" or tuple(tensor.shape) != expected:
            raise ValueError(
                f"Unexpected down_proj.weight_scale shape "
                f"{tuple(tensor.shape)}; expected {expected}."
            )
        param.data[local_idx].copy_(tensor)
        self._loaded_components.add(
            (local_idx, projection, "weight_scale")
        )

    def _load_w2_scale_bias(
        self,
        param: nn.Parameter,
        tensor: torch.Tensor,
        local_idx: int,
        projection: str,
    ) -> None:
        self._check_local_index(local_idx)
        expected = (self.hidden_size, 16)
        if projection != "down_proj" or tuple(tensor.shape) != expected:
            raise ValueError(
                f"Unexpected down_proj.scale_bias shape "
                f"{tuple(tensor.shape)}; expected {expected}."
            )
        param.data[local_idx].copy_(tensor)
        self._loaded_components.add((local_idx, projection, "scale_bias"))

    def post_load_prepare(self) -> None:
        if self._weights_processed:
            return
        expected_count = self.num_local_experts * 3 * 3
        if len(self._loaded_components) != expected_count:
            raise RuntimeError(
                f"GLM layer {self.layer_idx} loaded "
                f"{len(self._loaded_components)}/{expected_count} W4 expert "
                "components. Check the quant checkpoint and EP mapping."
            )

        # Mirror vLLM-Ascend's default ModelSlim layout. The on-disk values are
        # already two-INT4-per-INT8 packed; NZ is a layout conversion only.
        w13 = self.w13_weight.data.transpose(1, 2).contiguous()
        w2 = self.w2_weight.data.transpose(1, 2).contiguous()
        if w13.shape[-1] % 4 or w2.shape[-1] % 4:
            raise ValueError("Packed W4 weights require a last dim divisible by 4.")
        w13 = torch_npu.npu_format_cast(w13, ACL_FORMAT_FRACTAL_NZ)
        w2 = torch_npu.npu_format_cast(w2, ACL_FORMAT_FRACTAL_NZ)
        self.w13_weight.data = w13.view(torch.int32).contiguous()
        self.w2_weight.data = w2.view(torch.int32).contiguous()

        w13_scale = self.w13_weight_scale.data.transpose(1, 2).contiguous()
        w2_scale = self.w2_weight_scale.data.transpose(1, 2).contiguous()
        self.w13_weight_scale.data = float32_scale_to_int64_bits(w13_scale)
        self.w2_weight_scale.data = float32_scale_to_int64_bits(w2_scale)
        self.w13_scale_bias.data = (
            self.w13_scale_bias.data.transpose(1, 2)
            .contiguous()
            .sum(dim=1)
        )
        self.w2_scale_bias.data = (
            self.w2_scale_bias.data.transpose(1, 2)
            .contiguous()
            .sum(dim=1)
        )
        self._weights_processed = True

    def _grouped_topk(
        self, router_logits: torch.Tensor, output_dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.scoring_func not in ("softmax", "sigmoid"):
            raise ValueError(
                f"Unsupported GLM scoring function {self.scoring_func!r}."
            )
        bias = getattr(self.gate, "e_score_correction_bias", None)
        topk_weights, topk_ids, _ = ascend_ops.moe_gating_top_k(
            router_logits.contiguous(),
            k=self.top_k,
            k_group=self.topk_group,
            group_count=self.num_expert_group,
            group_select_mode=1,
            renorm=1 if self.renormalize else 0,
            norm_type=1 if self.scoring_func == "sigmoid" else 0,
            out_flag=False,
            routed_scaling_factor=self.routed_scaling_factor,
            eps=1e-20,
            bias_opt=bias,
        )
        balanced_ids = get_context().scratch.get(
            GLM_BALANCED_MOE_EXPERT_IDS_KEY
        )
        if balanced_ids is not None:
            if balanced_ids.shape != topk_ids.shape:
                raise RuntimeError(
                    "GLM balanced MoE warmup route shape changed: "
                    f"expected={tuple(topk_ids.shape)}, "
                    f"actual={tuple(balanced_ids.shape)}."
                )
            topk_ids = balanced_ids
        return topk_weights.to(output_dtype), topk_ids

    def _grouped_experts_forward(
        self,
        hidden_states: torch.Tensor,
        selected_experts: torch.Tensor,
        routing_weights: torch.Tensor,
        stats_callback: Callable[[str, torch.Tensor], None] | None = None,
    ) -> torch.Tensor:
        selected_experts = selected_experts.to(torch.int32).contiguous()
        local_mask = (selected_experts >= self.local_expert_start) & (
            selected_experts < self.local_expert_end
        )
        routing_weights = (
            routing_weights * local_mask.to(routing_weights.dtype)
        ).contiguous()

        sorted_hidden, expanded_row_idx, expert_tokens, per_token_scale = (
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
                quant_mode=1,
            )
        )
        expert_tokens = expert_tokens.to(torch.int64)
        if stats_callback is not None:
            stats_callback("moe_sorted_hidden", sorted_hidden)
            stats_callback("moe_per_token_scale", per_token_scale)
            stats_callback("moe_expert_tokens", expert_tokens)

        gate_up = torch_npu.npu_grouped_matmul(
            x=[sorted_hidden],
            weight=[self.w13_weight],
            scale=[self.w13_weight_scale],
            bias=[self.w13_scale_bias],
            per_token_scale=[per_token_scale],
            split_item=2,
            group_list_type=1,
            group_type=0,
            group_list=expert_tokens,
            output_dtype=torch.bfloat16,
        )[0]
        if stats_callback is not None:
            stats_callback("moe_gate_up", gate_up)
        activated = torch_npu.npu_swiglu(gate_up)
        activated, activated_scale = torch_npu.npu_dynamic_quant(activated)
        if stats_callback is not None:
            stats_callback("moe_activated_quant", activated)
            stats_callback("moe_activated_scale", activated_scale)
        routed_output = torch_npu.npu_grouped_matmul(
            x=[activated],
            weight=[self.w2_weight],
            scale=[self.w2_weight_scale],
            bias=[self.w2_scale_bias],
            per_token_scale=[activated_scale],
            split_item=2,
            group_list_type=1,
            group_type=0,
            group_list=expert_tokens,
            output_dtype=torch.bfloat16,
        )[0]
        if stats_callback is not None:
            stats_callback("moe_routed_gmm_output", routed_output)
        output = torch_npu.npu_moe_token_unpermute(
            permuted_tokens=routed_output,
            sorted_indices=torch.abs(expanded_row_idx),
            probs=routing_weights,
        )
        if stats_callback is not None:
            stats_callback("moe_routed_unpermuted", output)
        return output

    def forward(
        self,
        hidden_states: torch.Tensor,
        stats_callback: Callable[[str, torch.Tensor], None] | None = None,
    ) -> torch.Tensor:
        if not self._weights_processed or hidden_states.device.type != "npu":
            raise RuntimeError(
                "GLM W4A8 MoE requires post-load packed weights on NPU."
            )
        shared_output = self.shared_experts(hidden_states)
        router_logits = self.gate(hidden_states.to(torch.float32))
        routing_weights, selected_experts = self._grouped_topk(
            router_logits, hidden_states.dtype
        )
        if stats_callback is not None:
            stats_callback("moe_shared_output", shared_output)
            stats_callback("moe_router_logits", router_logits)
            stats_callback("moe_routing_weights", routing_weights)
            stats_callback("moe_selected_experts", selected_experts)
        output = self._grouped_experts_forward(
            hidden_states,
            selected_experts,
            routing_weights,
            stats_callback=stats_callback,
        )
        output = output + shared_output
        if stats_callback is not None:
            stats_callback("moe_pre_allreduce", output)
        if self.ep_size > 1:
            dist.all_reduce(output)
        if stats_callback is not None:
            stats_callback("moe_post_allreduce", output)
        return output.view_as(hidden_states)


class GlmMoeDsaDecoderLayer(nn.Module):
    def __init__(self, config: GlmMoeDsaConfig, layer_idx: int) -> None:
        super().__init__()
        self.self_attn = DeepseekV32DSAAttention(config, layer_idx)
        if layer_idx < int(config.first_k_dense_replace):
            self.mlp = DeepseekV32MLP(
                hidden_size=int(config.hidden_size),
                intermediate_size=int(config.intermediate_size),
                hidden_act=str(config.hidden_act),
            )
        else:
            self.mlp = GlmW4A8SparseMoeBlock(config, layer_idx)
        self.input_layernorm = RMSNorm(
            int(config.hidden_size), eps=float(config.rms_norm_eps)
        )
        self.post_attention_layernorm = RMSNorm(
            int(config.hidden_size), eps=float(config.rms_norm_eps)
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
            hidden_states, residual = self.input_layernorm(
                hidden_states, residual
            )
        pipeline_stats = self.self_attn._dsa_native_pipeline_enabled
        if pipeline_stats:
            self.self_attn._print_dsa_native_pipeline_tensor(
                "decoder_attn_input", hidden_states
            )
            self.self_attn._print_dsa_native_pipeline_tensor(
                "decoder_residual_input", residual
            )
        context = get_context()
        fuse_o_proj_norm = (
            not context.is_prefill
            and self.self_attn.can_fuse_o_proj_add_rms_norm()
        )
        hidden_states = self.self_attn(
            positions, hidden_states, skip_o_proj=fuse_o_proj_norm
        )
        if pipeline_stats:
            self.self_attn._print_dsa_native_pipeline_tensor(
                "decoder_attn_output", hidden_states
            )
        if fuse_o_proj_norm:
            hidden_states, residual = self.self_attn.o_proj_add_rms_norm(
                hidden_states, residual, self.post_attention_layernorm
            )
        else:
            hidden_states, residual = self.post_attention_layernorm(
                hidden_states, residual
            )
        if pipeline_stats:
            self.self_attn._print_dsa_native_pipeline_tensor(
                "decoder_post_attn_norm", hidden_states
            )
            self.self_attn._print_dsa_native_pipeline_tensor(
                "decoder_residual_after_attn", residual
            )
        if pipeline_stats and isinstance(self.mlp, GlmW4A8SparseMoeBlock):
            mlp_output = self.mlp(
                hidden_states,
                stats_callback=self.self_attn._print_dsa_native_pipeline_tensor,
            )
        else:
            mlp_output = self.mlp(hidden_states)
        if pipeline_stats:
            self.self_attn._print_dsa_native_pipeline_tensor(
                "decoder_mlp_output", mlp_output
            )
        return mlp_output, residual


class GlmMoeDsaModel(nn.Module):
    def __init__(self, config: GlmMoeDsaConfig) -> None:
        super().__init__()
        self.config = config
        self.embed_tokens = VocabParallelEmbedding(
            int(config.vocab_size), int(config.hidden_size)
        )
        self.layers = nn.ModuleList(
            GlmMoeDsaDecoderLayer(config, layer_idx)
            for layer_idx in range(int(config.num_hidden_layers))
        )
        self.norm = RMSNorm(
            int(config.hidden_size), eps=float(config.rms_norm_eps)
        )

    def forward(
        self, input_ids: torch.Tensor, positions: torch.Tensor
    ) -> torch.Tensor:
        hidden_states = self.embed_tokens(input_ids)
        residual = None
        for layer in self.layers:
            hidden_states, residual = layer(
                positions, hidden_states, residual
            )
        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states

    def post_load_prepare(self) -> None:
        for layer in self.layers:
            layer.self_attn.post_load_prepare()
            if isinstance(layer.mlp, GlmW4A8SparseMoeBlock):
                layer.mlp.post_load_prepare()


class GlmMoeDsaForCausalLM(nn.Module):
    packed_modules_mapping = {
        "gate_proj": ("gate_up_proj", 0),
        "up_proj": ("gate_up_proj", 1),
    }

    def __init__(self, config: GlmMoeDsaConfig) -> None:
        super().__init__()
        if not bool(getattr(config, "rope_interleave", True)):
            raise ValueError("GLM stage 1 expects rope_interleave=true.")
        self.config = config
        self.model = GlmMoeDsaModel(config)
        self.lm_head = ParallelLMHead(
            int(config.vocab_size), int(config.hidden_size)
        )
        # The checkpoint stores lm_head in FP32; retain it for logits accuracy.
        self.lm_head.weight.data = self.lm_head.weight.data.to(torch.float32)

    def forward(
        self, input_ids: torch.Tensor, positions: torch.Tensor
    ) -> torch.Tensor:
        return self.model(input_ids, positions)

    def full_decode_graph_eager_warmup(
        self, input_ids: torch.Tensor, positions: torch.Tensor
    ) -> int:
        """Warm GLM W4A8 kernels while ensuring every EP rank receives work."""

        sparse_moe = next(
            (
                layer.mlp
                for layer in self.model.layers
                if isinstance(layer.mlp, GlmW4A8SparseMoeBlock)
            ),
            None,
        )
        if sparse_moe is None:
            return 0

        # Use distinct, valid vocabulary rows for GLM's normal warmup/capture
        # instead of routing every dummy row from the same zero embedding.
        # Runtime decode overwrites this fixed-address input buffer.
        input_ids.copy_(
            torch.arange(
                input_ids.numel(),
                dtype=input_ids.dtype,
                device=input_ids.device,
            ).reshape_as(input_ids)
        )
        routes_per_pass = int(input_ids.numel()) * sparse_moe.top_k
        warmup_passes = max(
            1,
            (sparse_moe.ep_size + routes_per_pass - 1) // routes_per_pass,
        )
        context = get_context()
        try:
            for pass_index in range(warmup_passes):
                context.scratch.clear()
                context.scratch[GLM_BALANCED_MOE_EXPERT_IDS_KEY] = (
                    balanced_moe_expert_ids(
                        rows=int(input_ids.numel()),
                        top_k=sparse_moe.top_k,
                        num_experts=sparse_moe.num_experts,
                        ep_size=sparse_moe.ep_size,
                        route_offset=pass_index * routes_per_pass,
                        device=input_ids.device,
                        dtype=torch.int32,
                    )
                )
                self(input_ids, positions)
                torch.npu.synchronize()
        finally:
            # Never let dummy routes leak into the normal warmup, graph capture,
            # or runtime replay.
            context.scratch.clear()
        return warmup_passes

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.lm_head(hidden_states.to(torch.float32))

    def post_load_prepare(self) -> None:
        self.model.post_load_prepare()

    def weight_name_mapping(self, weight_name: str) -> str | WeightTarget | None:
        # The offload branch loads the native GLM DSA indexer.  It excludes
        # only the layer-78 MTP draft model/rotation, which nano-vLLM does not
        # execute.
        if should_skip_glm_checkpoint_weight(weight_name):
            return None

        match = _EXPERT_WEIGHT_RE.match(weight_name)
        if match is None:
            return weight_name
        layer_idx = int(match.group("layer"))
        expert_idx = int(match.group("expert"))
        projection = match.group("projection")
        field = match.group("field")
        if layer_idx >= len(self.model.layers):
            return None
        moe = self.model.layers[layer_idx].mlp
        if not isinstance(moe, GlmW4A8SparseMoeBlock):
            return None
        if expert_idx not in moe.local_expert_id_set:
            return None
        if field == "weight_offset":
            # The generic quant loader reads local offsets only to assert that
            # this checkpoint is symmetric, then skips parameter lookup.
            return weight_name

        local_idx = expert_idx - moe.local_expert_start
        prefix = f"model.layers.{layer_idx}.mlp"
        if projection in ("gate_proj", "up_proj"):
            target = {
                "weight": "w13_weight",
                "weight_scale": "w13_weight_scale",
                "scale_bias": "w13_scale_bias",
            }[field]
        else:
            target = {
                "weight": "w2_weight",
                "weight_scale": "w2_weight_scale",
                "scale_bias": "w2_scale_bias",
            }[field]
        return WeightTarget(
            f"{prefix}.{target}", (local_idx, projection)
        )
