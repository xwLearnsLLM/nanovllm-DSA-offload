from __future__ import annotations

import argparse
from time import perf_counter

import torch

try:
    import torch_npu  # type: ignore
except Exception:
    torch_npu = None

from nanovllm.models import dsa_indexer_project_real
from nanovllm.models.dsa_indexer_project import _apply_rope_neox_reference


def sync(device: torch.device) -> None:
    if device.type == "npu" and torch_npu is not None:
        torch.npu.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize(device)


def diff_report(name: str, actual: torch.Tensor, expected: torch.Tensor) -> str:
    a = actual.float()
    e = expected.float()
    diff = (a - e).abs()
    rel = diff / e.abs().clamp_min(1e-6)
    return f"{name}: max_abs={diff.max().item():.6g} mean_abs={diff.mean().item():.6g} max_rel={rel.max().item():.6g}"


def assert_close(name: str, actual: torch.Tensor, expected: torch.Tensor, atol: float, rtol: float) -> None:
    a = actual.float()
    e = expected.float()
    diff = (a - e).abs()
    max_abs = diff.max().item()
    max_allowed = (atol + rtol * e.abs()).max().item()
    if max_abs > max_allowed:
        raise AssertionError(f"{name} mismatch: max_abs={max_abs:.6g} max_allowed={max_allowed:.6g}")


def reference_post(
    q: torch.Tensor,
    k: torch.Tensor,
    weights: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    score_scale: float,
    rope_dim: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    q_pe, q_nope = torch.split(q, [rope_dim, q.shape[-1] - rope_dim], dim=-1)
    k_pe, k_nope = torch.split(k, [rope_dim, k.shape[-1] - rope_dim], dim=-1)
    if q.device.type == "npu" and torch_npu is not None and q.dtype in (torch.float16, torch.bfloat16):
        q_pe = torch_npu.npu_rotary_mul(q_pe.unsqueeze(2), cos, sin).squeeze(2)
        k_pe = torch_npu.npu_rotary_mul(k_pe.unsqueeze(1).unsqueeze(2), cos, sin).squeeze(2).squeeze(1)
    else:
        q_pe = _apply_rope_neox_reference(q_pe, cos, sin, rope_dim)
        k_pe = _apply_rope_neox_reference(k_pe.unsqueeze(1), cos, sin, rope_dim).squeeze(1)
    return torch.cat((q_pe, q_nope), dim=-1), torch.cat((k_pe, k_nope), dim=-1), (weights * score_scale).to(q.dtype)


def bench(fn, device: torch.device, warmup: int, iters: int) -> float:
    for _ in range(warmup):
        fn()
    sync(device)
    start = perf_counter()
    for _ in range(iters):
        fn()
    sync(device)
    return (perf_counter() - start) * 1000.0 / max(iters, 1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--tokens", type=int, default=4)
    parser.add_argument("--heads", type=int, default=64)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--rope-dim", type=int, default=64)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--atol", type=float, default=0.03125)
    parser.add_argument("--rtol", type=float, default=0.01)
    args = parser.parse_args()

    if not dsa_indexer_project_real.is_available():
        raise RuntimeError("dsa_indexer_project_post is not available. Rebuild with: bash scripts/build_nanovllm_ops.sh")

    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    q = torch.randn(args.tokens, args.heads, args.head_dim, device=device, dtype=torch.bfloat16).contiguous()
    k = torch.randn(args.tokens, args.head_dim, device=device, dtype=torch.bfloat16).contiguous()
    weights = torch.randn(args.tokens, args.heads, device=device, dtype=torch.float32).contiguous()
    cos = torch.randn(args.tokens, 1, 1, args.rope_dim, device=device, dtype=torch.bfloat16).contiguous()
    sin = torch.randn(args.tokens, 1, 1, args.rope_dim, device=device, dtype=torch.bfloat16).contiguous()
    score_scale = 0.125

    ref_q, ref_k, ref_w = reference_post(q, k, weights, cos, sin, score_scale, args.rope_dim)
    op_q = torch.empty_like(q)
    op_k = torch.empty_like(k)
    op_w = torch.empty(weights.shape, dtype=q.dtype, device=device)
    dsa_indexer_project_real.dsa_indexer_project_post_real_out(q, k, weights, cos, sin, op_q, op_k, op_w, score_scale, args.rope_dim)
    sync(device)

    print(
        "INDEXER_POST_PROBE "
        f"device={args.device} tokens={args.tokens} heads={args.heads} "
        f"head_dim={args.head_dim} rope_dim={args.rope_dim} warmup={args.warmup} iters={args.iters} "
        f"binding={dsa_indexer_project_real.binding_version()}"
    )
    print("INDEXER_POST_DIFF " + diff_report("q", op_q, ref_q))
    print("INDEXER_POST_DIFF " + diff_report("k", op_k, ref_k))
    print("INDEXER_POST_DIFF " + diff_report("weights", op_w, ref_w))
    assert_close("q", op_q, ref_q, args.atol, args.rtol)
    assert_close("k", op_k, ref_k, args.atol, args.rtol)
    assert_close("weights", op_w, ref_w, args.atol, args.rtol)

    ref_ms = bench(lambda: reference_post(q, k, weights, cos, sin, score_scale, args.rope_dim), device, args.warmup, args.iters)
    def run_post_out():
        q_out = torch.empty_like(q)
        k_out = torch.empty_like(k)
        w_out = torch.empty(weights.shape, dtype=q.dtype, device=device)
        return dsa_indexer_project_real.dsa_indexer_project_post_real_out(q, k, weights, cos, sin, q_out, k_out, w_out, score_scale, args.rope_dim)

    op_ms = bench(run_post_out, device, args.warmup, args.iters)
    print(f"INDEXER_POST_BENCH reference_avg_ms={ref_ms:.6f} ascendc_avg_ms={op_ms:.6f}")


if __name__ == "__main__":
    main()
