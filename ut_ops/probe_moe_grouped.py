from __future__ import annotations

import argparse
from time import perf_counter

import torch
import torch.nn.functional as F


def _describe(name: str, tensor: torch.Tensor) -> str:
    return (
        f"{name}=shape={tuple(tensor.shape)} dtype={tensor.dtype} "
        f"device={tensor.device} contiguous={tensor.is_contiguous()} "
        f"stride={tuple(tensor.stride())}"
    )


def _sync(device: torch.device) -> None:
    if device.type == "npu":
        torch.npu.synchronize()


def _make_balanced_topk(
    num_tokens: int,
    topk: int,
    num_experts: int,
    local_start: int,
    num_local_experts: int,
    device: torch.device,
) -> torch.Tensor:
    ids = torch.empty((num_tokens, topk), dtype=torch.int64, device=device)
    local_end = local_start + num_local_experts
    for t in range(num_tokens):
        ids[t, 0] = local_start + (t % num_local_experts)
        for k in range(1, topk):
            ids[t, k] = (t + k * 17) % num_experts
            if ids[t, k] == ids[t, 0]:
                ids[t, k] = (ids[t, k] + 1) % num_experts
    if topk > 1 and local_end < num_experts:
        ids[0::2, -1] = local_end + (torch.arange((num_tokens + 1) // 2, device=device) % (num_experts - local_end))
    return ids.contiguous()


def _random_topk_weights(
    num_tokens: int,
    topk: int,
    device: torch.device,
) -> torch.Tensor:
    weights = torch.rand((num_tokens, topk), dtype=torch.float32, device=device)
    return (weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-20)).contiguous()


def _reference_local_moe(
    hidden_states: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    w13: torch.Tensor,
    w2: torch.Tensor,
    local_start: int,
) -> torch.Tensor:
    num_tokens, hidden_size = hidden_states.shape
    num_local_experts = w13.shape[0]
    out = torch.zeros(
        (num_tokens, hidden_size),
        dtype=hidden_states.dtype,
        device=hidden_states.device,
    )
    for local_idx in range(num_local_experts):
        expert_idx = local_start + local_idx
        token_idx, route_idx = torch.where(topk_ids == expert_idx)
        if token_idx.numel() == 0:
            continue
        gate_up = F.linear(hidden_states[token_idx], w13[local_idx])
        gate, up = gate_up.chunk(2, dim=-1)
        routed = F.linear(F.silu(gate) * up, w2[local_idx])
        routed = routed * topk_weights[token_idx, route_idx, None].to(routed.dtype)
        out.index_add_(0, token_idx, routed)
    return out


def _grouped_moe(
    hidden_states: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    w13: torch.Tensor,
    w2: torch.Tensor,
    num_experts: int,
    local_start: int,
    num_local_experts: int,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    import torch_npu

    sorted_hidden, expanded_row_idx, expert_tokens, pertoken_scale = (
        torch_npu.npu_moe_init_routing_v2(
            hidden_states,
            topk_ids,
            scale=None,
            active_num=hidden_states.shape[0] * topk_ids.shape[1],
            expert_num=num_experts,
            expert_tokens_num_type=1,
            expert_tokens_num_flag=True,
            active_expert_range=[local_start, local_start + num_local_experts],
            quant_mode=-1,
        )
    )
    expert_tokens = expert_tokens.to(torch.int64)

    gate_up = torch_npu.npu_grouped_matmul(
        x=[sorted_hidden],
        weight=[w13.transpose(1, 2).contiguous()],
        split_item=2,
        group_list_type=1,
        group_type=0,
        group_list=expert_tokens,
    )[0]
    activated = torch_npu.npu_swiglu(gate_up)
    routed = torch_npu.npu_grouped_matmul(
        x=[activated],
        weight=[w2.transpose(1, 2).contiguous()],
        split_item=2,
        group_list_type=1,
        group_type=0,
        group_list=expert_tokens,
    )[0]
    out = torch_npu.npu_moe_token_unpermute(
        permuted_tokens=routed,
        sorted_indices=torch.abs(expanded_row_idx),
        probs=topk_weights,
    )
    metadata = {
        "sorted_hidden": sorted_hidden,
        "expanded_row_idx": expanded_row_idx,
        "expert_tokens": expert_tokens,
    }
    if pertoken_scale is not None:
        metadata["pertoken_scale"] = pertoken_scale
    return out, metadata


def _diff(left: torch.Tensor, right: torch.Tensor) -> str:
    left_f = left.float()
    right_f = right.float()
    abs_diff = (left_f - right_f).abs()
    max_abs = float(abs_diff.max().item())
    mean_abs = float(abs_diff.mean().item())
    rms = float(torch.sqrt(torch.mean(abs_diff * abs_diff)).item())
    denom = torch.linalg.vector_norm(left_f).clamp_min(1e-20)
    rel_l2 = float((torch.linalg.vector_norm(left_f - right_f) / denom).item())
    cosine = float(F.cosine_similarity(left_f.flatten(), right_f.flatten(), dim=0).item())
    value_range = float((left_f.max() - left_f.min()).clamp_min(1e-20).item())
    return (
        f"max_abs={max_abs:.8g} mean_abs={mean_abs:.8g} rms={rms:.8g} "
        f"rel_l2={rel_l2:.8g} cosine={cosine:.8g} "
        f"value_range={value_range:.8g} "
        f"relative_max_error={max_abs / value_range:.8g} "
        f"relative_mean_abs_error={mean_abs / value_range:.8g}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--tokens", type=int, default=128)
    parser.add_argument("--hidden-size", type=int, default=512)
    parser.add_argument("--intermediate-size", type=int, default=256)
    parser.add_argument("--num-experts", type=int, default=32)
    parser.add_argument("--num-local-experts", type=int, default=8)
    parser.add_argument("--local-start", type=int, default=0)
    parser.add_argument("--topk", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    import torch_npu  # noqa: F401

    device = torch.device(args.device)
    torch.npu.set_device(device)
    torch.manual_seed(args.seed)

    if args.local_start < 0:
        raise ValueError("--local-start must be non-negative")
    if args.num_local_experts <= 0:
        raise ValueError("--num-local-experts must be positive")
    if args.local_start + args.num_local_experts > args.num_experts:
        raise ValueError("local expert range exceeds --num-experts")

    hidden_states = torch.randn(
        (args.tokens, args.hidden_size),
        dtype=torch.bfloat16,
        device=device,
    )
    w13 = torch.randn(
        (args.num_local_experts, 2 * args.intermediate_size, args.hidden_size),
        dtype=torch.bfloat16,
        device=device,
    ) / (args.hidden_size**0.5)
    w2 = torch.randn(
        (args.num_local_experts, args.hidden_size, args.intermediate_size),
        dtype=torch.bfloat16,
        device=device,
    ) / (args.intermediate_size**0.5)
    topk_ids = _make_balanced_topk(
        args.tokens,
        args.topk,
        args.num_experts,
        args.local_start,
        args.num_local_experts,
        device,
    )
    topk_weights = _random_topk_weights(args.tokens, args.topk, device)

    print(
        "MOE_PROBE config "
        f"tokens={args.tokens} hidden={args.hidden_size} "
        f"intermediate={args.intermediate_size} topk={args.topk} "
        f"num_experts={args.num_experts} "
        f"local_range=[{args.local_start},{args.local_start + args.num_local_experts})"
    )
    print("MOE_PROBE " + _describe("hidden_states", hidden_states))
    print("MOE_PROBE " + _describe("topk_ids", topk_ids))
    print("MOE_PROBE " + _describe("topk_weights", topk_weights))
    print("MOE_PROBE " + _describe("w13", w13))
    print("MOE_PROBE " + _describe("w2", w2))

    _sync(device)
    ref_start = perf_counter()
    ref = _reference_local_moe(
        hidden_states,
        topk_ids,
        topk_weights,
        w13,
        w2,
        args.local_start,
    )
    _sync(device)
    print(f"MOE_PROBE after_reference elapsed={perf_counter() - ref_start:.6f}s")

    _sync(device)
    grouped_start = perf_counter()
    grouped, metadata = _grouped_moe(
        hidden_states,
        topk_ids,
        topk_weights,
        w13,
        w2,
        args.num_experts,
        args.local_start,
        args.num_local_experts,
    )
    _sync(device)
    print(f"MOE_PROBE after_grouped elapsed={perf_counter() - grouped_start:.6f}s")
    print("MOE_PROBE " + _describe("grouped_out", grouped))
    for name, tensor in metadata.items():
        print("MOE_PROBE " + _describe(name, tensor))
        if name == "expert_tokens":
            print(f"MOE_PROBE expert_tokens_values={tensor.detach().cpu().tolist()}")

    print("MOE_PROBE diff reference_vs_grouped " + _diff(ref, grouped))

    grouped_times = []
    for _ in range(args.warmup):
        _grouped_moe(
            hidden_states,
            topk_ids,
            topk_weights,
            w13,
            w2,
            args.num_experts,
            args.local_start,
            args.num_local_experts,
        )
    _sync(device)
    for _ in range(args.iters):
        start = perf_counter()
        _grouped_moe(
            hidden_states,
            topk_ids,
            topk_weights,
            w13,
            w2,
            args.num_experts,
            args.local_start,
            args.num_local_experts,
        )
        _sync(device)
        grouped_times.append((perf_counter() - start) * 1000.0)
    if grouped_times:
        times = torch.tensor(grouped_times, dtype=torch.float32)
        print(
            "MOE_BENCH grouped "
            f"avg_ms={float(times.mean()):.6f} "
            f"min_ms={float(times.min()):.6f} "
            f"max_ms={float(times.max()):.6f} "
            f"p99_ms={float(torch.quantile(times, 0.99)):.6f}"
        )


if __name__ == "__main__":
    main()
