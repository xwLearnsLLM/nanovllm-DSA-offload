from __future__ import annotations

import argparse
import time

import torch
import torch.nn.functional as F

try:
    import torch_npu  # type: ignore
except Exception:  # pragma: no cover - local non-Ascend syntax checks
    torch_npu = None

try:
    import nanovllm.ops as ascend_ops
except Exception:  # pragma: no cover
    ascend_ops = None


def sync(device: torch.device) -> None:
    if device.type == "npu" and torch_npu is not None:
        torch.npu.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize(device)


def desc(name: str, tensor: torch.Tensor) -> str:
    return (
        f"{name}=shape={tuple(tensor.shape)} dtype={tensor.dtype} "
        f"device={tensor.device} contiguous={tensor.is_contiguous()} "
        f"stride={tuple(tensor.stride())}"
    )


def diff_report(name: str, actual: torch.Tensor, expected: torch.Tensor) -> str:
    actual_f = actual.float()
    expected_f = expected.float()
    diff = (actual_f - expected_f).abs()
    denom = expected_f.abs().max().clamp_min(1e-6)
    return (
        f"{name}: max_abs={float(diff.max().item()):.6g} "
        f"mean_abs={float(diff.mean().item()):.6g} "
        f"max_rel={float((diff.max() / denom).item()):.6g}"
    )


def rotate_half_neox(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    x1 = x[..., :half]
    x2 = x[..., half:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rope_neox(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    view_shape = (cos.shape[0],) + (1,) * (x.dim() - 2) + (cos.shape[-1],)
    cos = cos.view(view_shape)
    sin = sin.view(view_shape)
    return (x * cos) + (rotate_half_neox(x) * sin)


def make_cos_sin(
    tokens: int,
    rope_dim: int,
    dtype: torch.dtype,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    base = torch.arange(tokens * rope_dim, dtype=torch.float32, device=device)
    base = base.view(tokens, rope_dim) / max(rope_dim, 1)
    return base.cos().to(dtype), base.sin().to(dtype)


def indexer_current(
    hidden_states: torch.Tensor,
    q_c: torch.Tensor,
    weights: dict[str, torch.Tensor],
    cos: torch.Tensor,
    sin: torch.Tensor,
    *,
    n_head: int,
    head_dim: int,
    rope_dim: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    q = F.linear(q_c, weights["wq_b"]).view(-1, n_head, head_dim)
    q_pe, q_nope = torch.split(q, [rope_dim, head_dim - rope_dim], dim=-1)

    k = F.linear(hidden_states, weights["wk"])
    k = F.layer_norm(
        k,
        (head_dim,),
        weights["k_norm_weight"],
        weights["k_norm_bias"],
        eps=1e-6,
    )
    k_pe, k_nope = torch.split(k, [rope_dim, head_dim - rope_dim], dim=-1)

    q_pe = apply_rope_neox(q_pe, cos, sin)
    k_pe = apply_rope_neox(k_pe.unsqueeze(1), cos, sin).squeeze(1)
    q = torch.cat((q_pe, q_nope), dim=-1)
    k = torch.cat((k_pe, k_nope), dim=-1)

    score_weights = F.linear(hidden_states.float(), weights["weights_proj"].float())
    score_weights = score_weights * (head_dim**-0.5) * (n_head**-0.5)
    return q, k, score_weights.to(hidden_states.dtype)


def indexer_fused_wk_weights_bf16(
    hidden_states: torch.Tensor,
    q_c: torch.Tensor,
    weights: dict[str, torch.Tensor],
    cos: torch.Tensor,
    sin: torch.Tensor,
    *,
    n_head: int,
    head_dim: int,
    rope_dim: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    q = F.linear(q_c, weights["wq_b"]).view(-1, n_head, head_dim)
    q_pe, q_nope = torch.split(q, [rope_dim, head_dim - rope_dim], dim=-1)

    wk_weights_proj = torch.cat(
        (
            weights["wk"],
            weights["weights_proj"].to(hidden_states.dtype),
        ),
        dim=0,
    )
    kw = F.linear(hidden_states, wk_weights_proj)
    k_raw, score_weights = torch.split(kw, [head_dim, n_head], dim=-1)
    k = F.layer_norm(
        k_raw,
        (head_dim,),
        weights["k_norm_weight"],
        weights["k_norm_bias"],
        eps=1e-6,
    )
    k_pe, k_nope = torch.split(k, [rope_dim, head_dim - rope_dim], dim=-1)

    q_pe = apply_rope_neox(q_pe, cos, sin)
    k_pe = apply_rope_neox(k_pe.unsqueeze(1), cos, sin).squeeze(1)
    q = torch.cat((q_pe, q_nope), dim=-1)
    k = torch.cat((k_pe, k_nope), dim=-1)

    score_weights = score_weights * (head_dim**-0.5) * (n_head**-0.5)
    return q, k, score_weights


def q_project_bmm_transpose(
    q_c: torch.Tensor,
    wq_b: torch.Tensor,
    *,
    n_head: int,
    head_dim: int,
) -> torch.Tensor:
    if ascend_ops is None:
        raise RuntimeError("nanovllm.ops is not available")
    q_c_by_head = q_c.unsqueeze(1).expand(-1, n_head, -1).contiguous()
    wq_by_head = (
        wq_b.view(n_head, head_dim, q_c.shape[-1])
        .transpose(1, 2)
        .contiguous()
    )
    out = torch.empty(
        (q_c.shape[0], n_head, head_dim),
        dtype=q_c.dtype,
        device=q_c.device,
    )
    ascend_ops.batch_matmul_transpose(q_c_by_head, wq_by_head, out)
    return out


def bench(fn, warmup: int, iters: int, device: torch.device) -> tuple[object, float]:
    result = None
    for _ in range(warmup):
        result = fn()
    sync(device)
    start = time.perf_counter()
    for _ in range(iters):
        result = fn()
    sync(device)
    elapsed_ms = (time.perf_counter() - start) * 1000.0 / max(iters, 1)
    return result, elapsed_ms


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--tokens", type=int, default=4)
    parser.add_argument("--hidden-size", type=int, default=7168)
    parser.add_argument("--q-lora-rank", type=int, default=1536)
    parser.add_argument("--heads", type=int, default=64)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--rope-dim", type=int, default=64)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--skip-bmm-transpose", action="store_true")
    args = parser.parse_args()

    device = torch.device(args.device)
    if device.type == "npu":
        if torch_npu is None:
            raise RuntimeError("torch_npu is required for NPU runs")
        torch.npu.set_device(device)
    torch.manual_seed(args.seed)

    dtype = torch.bfloat16
    hidden_states = torch.randn(
        args.tokens,
        args.hidden_size,
        dtype=dtype,
        device=device,
    )
    q_c = torch.randn(
        args.tokens,
        args.q_lora_rank,
        dtype=dtype,
        device=device,
    )
    weights = {
        "wq_b": torch.randn(
            args.heads * args.head_dim,
            args.q_lora_rank,
            dtype=dtype,
            device=device,
        ),
        "wk": torch.randn(
            args.head_dim,
            args.hidden_size,
            dtype=dtype,
            device=device,
        ),
        "weights_proj": torch.randn(
            args.heads,
            args.hidden_size,
            dtype=torch.float32,
            device=device,
        ),
        "k_norm_weight": torch.randn(args.head_dim, dtype=dtype, device=device),
        "k_norm_bias": torch.randn(args.head_dim, dtype=dtype, device=device),
    }
    cos, sin = make_cos_sin(args.tokens, args.rope_dim, dtype, device)

    print("INDEXER_PROBE config " + " ".join(f"{k}={v}" for k, v in vars(args).items()))
    print("INDEXER_PROBE " + desc("hidden_states", hidden_states))
    print("INDEXER_PROBE " + desc("q_c", q_c))

    common_kwargs = {
        "n_head": args.heads,
        "head_dim": args.head_dim,
        "rope_dim": args.rope_dim,
    }

    ref, ref_ms = bench(
        lambda: indexer_current(hidden_states, q_c, weights, cos, sin, **common_kwargs),
        args.warmup,
        args.iters,
        device,
    )
    fused, fused_ms = bench(
        lambda: indexer_fused_wk_weights_bf16(
            hidden_states,
            q_c,
            weights,
            cos,
            sin,
            **common_kwargs,
        ),
        args.warmup,
        args.iters,
        device,
    )

    assert ref is not None and fused is not None
    ref_q, ref_k, ref_weights = ref
    fused_q, fused_k, fused_weights = fused
    print("INDEXER_DIFF fused_wk_weights_bf16 " + diff_report("q", fused_q, ref_q))
    print("INDEXER_DIFF fused_wk_weights_bf16 " + diff_report("k", fused_k, ref_k))
    print(
        "INDEXER_DIFF fused_wk_weights_bf16 "
        + diff_report("weights", fused_weights, ref_weights)
    )
    print(
        f"INDEXER_BENCH current_avg_ms={ref_ms:.6f} "
        f"fused_wk_weights_bf16_avg_ms={fused_ms:.6f}"
    )

    if (
        not args.skip_bmm_transpose
        and device.type == "npu"
        and ascend_ops is not None
        and hasattr(ascend_ops, "batch_matmul_transpose")
    ):
        q_linear = F.linear(q_c, weights["wq_b"]).view(
            -1,
            args.heads,
            args.head_dim,
        )
        q_bmm, q_bmm_ms = bench(
            lambda: q_project_bmm_transpose(
                q_c,
                weights["wq_b"],
                n_head=args.heads,
                head_dim=args.head_dim,
            ),
            args.warmup,
            args.iters,
            device,
        )
        assert isinstance(q_bmm, torch.Tensor)
        print("INDEXER_DIFF q_bmm_transpose " + diff_report("q", q_bmm, q_linear))
        print(f"INDEXER_BENCH q_bmm_transpose_avg_ms={q_bmm_ms:.6f}")
    else:
        print("INDEXER_BENCH q_bmm_transpose skipped")


if __name__ == "__main__":
    main()
