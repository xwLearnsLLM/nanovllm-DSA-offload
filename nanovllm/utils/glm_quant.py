from __future__ import annotations

import numpy as np
import torch


GLM_BALANCED_MOE_EXPERT_IDS_KEY = "glm_balanced_moe_expert_ids"


def should_skip_glm_checkpoint_weight(weight_name: str) -> bool:
    """Return whether a GLM checkpoint tensor belongs to the unused MTP path."""

    return weight_name.startswith("model.layers.78.") or weight_name == "rot.weight"


def balanced_moe_expert_ids(
    rows: int,
    top_k: int,
    num_experts: int,
    ep_size: int,
    *,
    route_offset: int = 0,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.int32,
) -> torch.Tensor:
    """Build deterministic dummy routes that rotate work across EP ranks.

    This is used only by GLM's pre-capture eager warmup. Normal graph capture
    and every replay retain the model's real top-k routing result.
    """

    rows = int(rows)
    top_k = int(top_k)
    num_experts = int(num_experts)
    ep_size = int(ep_size)
    route_offset = int(route_offset)
    if rows <= 0 or top_k <= 0:
        raise ValueError(f"rows and top_k must be positive, got {rows}, {top_k}.")
    if ep_size <= 0 or num_experts <= 0 or num_experts % ep_size:
        raise ValueError(
            "num_experts must be positive and divisible by ep_size, got "
            f"num_experts={num_experts}, ep_size={ep_size}."
        )
    if top_k > num_experts:
        raise ValueError(
            f"top_k must not exceed num_experts, got {top_k} > {num_experts}."
        )
    if route_offset < 0:
        raise ValueError(f"route_offset must be non-negative, got {route_offset}.")

    num_local_experts = num_experts // ep_size
    slots = torch.arange(
        rows * top_k,
        dtype=torch.int64,
        device=device,
    ) + route_offset
    expert_ranks = torch.remainder(slots, ep_size)
    local_experts = torch.remainder(
        torch.div(slots, ep_size, rounding_mode="floor"),
        num_local_experts,
    )
    expert_ids = expert_ranks * num_local_experts + local_experts
    return expert_ids.reshape(rows, top_k).to(dtype=dtype)


def float32_scale_to_int64_bits(scale: torch.Tensor) -> torch.Tensor:
    """Encode FP32 dequant scales in the INT64 format expected by NPU GMM."""

    device = scale.device
    scale_cpu = scale.detach().to(device="cpu", dtype=torch.float32).contiguous()
    bits = scale_cpu.numpy().view(np.uint32).astype(np.int64, copy=True)
    return torch.from_numpy(bits).to(device=device)
