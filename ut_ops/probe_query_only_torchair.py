from __future__ import annotations

import argparse
from time import perf_counter

import torch
import torch.nn.functional as F

try:
    import torch_npu  # type: ignore
except Exception:
    torch_npu = None

from nanovllm.models.dsa_indexer_project import (
    dsa_indexer_project_query_only,
    dsa_indexer_project_query_only_torchair,
    dsa_indexer_project_query_only_with_qc_torchair,
    warmup_dsa_query_only_torchair,
    warmup_dsa_query_only_with_qc_torchair,
)


def sync(device: torch.device) -> None:
    if device.type == "npu" and torch_npu is not None:
        torch.npu.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize(device)


def parse_ints(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    orig_dtype = x.dtype
    x_float = x.float()
    var = x_float.pow(2).mean(dim=-1, keepdim=True)
    return (x_float * torch.rsqrt(var + float(eps))).to(orig_dtype) * weight


def make_cos_sin(tokens: int, rope_dim: int, dtype: torch.dtype, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    base = torch.arange(tokens * rope_dim, dtype=torch.float32, device=device).view(tokens, rope_dim)
    base = base / max(rope_dim, 1)
    cos = base.cos().to(dtype).view(tokens, 1, 1, rope_dim).contiguous()
    sin = base.sin().to(dtype).view(tokens, 1, 1, rope_dim).contiguous()
    return cos, sin


def diff_report(name: str, actual: torch.Tensor, expected: torch.Tensor) -> tuple[float, float, float]:
    diff = (actual.float() - expected.float()).abs()
    max_abs = float(diff.max().item())
    mean_abs = float(diff.mean().item())
    denom = expected.float().abs().max().clamp_min(1e-6)
    max_rel = float((diff.max() / denom).item())
    print(f"QUERY_ONLY_DIFF {name}: max_abs={max_abs:.6g} mean_abs={mean_abs:.6g} max_rel={max_rel:.6g}")
    return max_abs, mean_abs, max_rel


def assert_close(name: str, actual: torch.Tensor, expected: torch.Tensor, atol: float, rtol: float) -> None:
    max_abs, _, max_rel = diff_report(name, actual, expected)
    if max_abs <= float(atol) or max_rel <= float(rtol):
        return
    raise AssertionError(f"{name} mismatch: max_abs={max_abs:.6g} max_rel={max_rel:.6g} atol={atol} rtol={rtol}")


def bench(fn, device: torch.device, warmup: int, iters: int) -> float:
    with torch.inference_mode():
        for _ in range(max(warmup, 0)):
            fn()
        sync(device)
        start = perf_counter()
        for _ in range(max(iters, 1)):
            fn()
        sync(device)
    return (perf_counter() - start) * 1000.0 / max(iters, 1)


def run_one(args, tokens: int, device: torch.device) -> None:
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    torch.manual_seed(args.seed + int(tokens))
    hidden_states = torch.randn(tokens, args.hidden_size, dtype=dtype, device=device)
    q_a_weight = torch.randn(args.q_lora_rank, args.hidden_size, dtype=dtype, device=device)
    q_norm_weight = torch.randn(args.q_lora_rank, dtype=dtype, device=device)
    wq_b_weight = torch.randn(args.n_head * args.head_dim, args.q_lora_rank, dtype=dtype, device=device)
    weights_proj_weight = torch.randn(args.n_head, args.hidden_size, dtype=torch.float32, device=device)
    cos, sin = make_cos_sin(tokens, args.rope_dim, dtype, device)
    score_scale = float(args.score_scale)
    q_c = rms_norm(F.linear(hidden_states, q_a_weight), q_norm_weight, args.q_norm_eps)

    q_ref = torch.empty((tokens, args.n_head, args.head_dim), dtype=dtype, device=device)
    w_ref = torch.empty((tokens, args.n_head), dtype=dtype, device=device)
    dsa_indexer_project_query_only(
        hidden_states,
        q_c,
        cos,
        sin,
        wq_b_weight,
        weights_proj_weight,
        q_ref,
        w_ref,
        n_head=args.n_head,
        head_dim=args.head_dim,
        rope_dim=args.rope_dim,
        score_scale=score_scale,
    )

    warmup_q = warmup_dsa_query_only_torchair(
        tokens_list=[tokens],
        hidden_size=args.hidden_size,
        q_lora_rank=args.q_lora_rank,
        n_head=args.n_head,
        head_dim=args.head_dim,
        rope_dim=args.rope_dim,
        dtype=dtype,
        device=device,
        wq_b_weight=wq_b_weight,
        weights_proj_weight=weights_proj_weight,
        score_scale=score_scale,
    )
    warmup_with_qc = warmup_dsa_query_only_with_qc_torchair(
        tokens_list=[tokens],
        hidden_size=args.hidden_size,
        q_lora_rank=args.q_lora_rank,
        n_head=args.n_head,
        head_dim=args.head_dim,
        rope_dim=args.rope_dim,
        dtype=dtype,
        device=device,
        q_a_weight=q_a_weight,
        q_norm_weight=q_norm_weight,
        wq_b_weight=wq_b_weight,
        weights_proj_weight=weights_proj_weight,
        q_norm_eps=args.q_norm_eps,
        score_scale=score_scale,
    )
    print(f"QUERY_ONLY_WARMUP tokens={tokens} q_only={warmup_q.get(tokens)} with_qc={warmup_with_qc.get(tokens)}")

    q_ta = torch.empty_like(q_ref)
    w_ta = torch.empty_like(w_ref)
    q_ta, w_ta, used_q, reason_q = dsa_indexer_project_query_only_torchair(
        hidden_states,
        q_c,
        cos,
        sin,
        wq_b_weight,
        weights_proj_weight,
        q_ta,
        w_ta,
        n_head=args.n_head,
        head_dim=args.head_dim,
        rope_dim=args.rope_dim,
        score_scale=score_scale,
        allow_compile=True,
    )
    print(f"QUERY_ONLY_PATH tokens={tokens} q_only_used={used_q} reason={reason_q}")

    q_with_qc = torch.empty_like(q_ref)
    w_with_qc = torch.empty_like(w_ref)
    q_with_qc, w_with_qc, used_with_qc, reason_with_qc = dsa_indexer_project_query_only_with_qc_torchair(
        hidden_states,
        cos,
        sin,
        q_a_weight,
        q_norm_weight,
        wq_b_weight,
        weights_proj_weight,
        q_with_qc,
        w_with_qc,
        q_norm_eps=args.q_norm_eps,
        n_head=args.n_head,
        head_dim=args.head_dim,
        rope_dim=args.rope_dim,
        score_scale=score_scale,
        allow_compile=True,
    )
    print(f"QUERY_ONLY_PATH tokens={tokens} with_qc_used={used_with_qc} reason={reason_with_qc}")

    assert_close(f"tokens={tokens} q_only q", q_ta, q_ref, args.atol, args.rtol)
    assert_close(f"tokens={tokens} q_only weights", w_ta, w_ref, args.atol, args.rtol)
    assert_close(f"tokens={tokens} with_qc q", q_with_qc, q_ref, args.atol, args.rtol)
    assert_close(f"tokens={tokens} with_qc weights", w_with_qc, w_ref, args.atol, args.rtol)

    def current_fn():
        q_out = torch.empty_like(q_ref)
        w_out = torch.empty_like(w_ref)
        return dsa_indexer_project_query_only(
            hidden_states,
            q_c,
            cos,
            sin,
            wq_b_weight,
            weights_proj_weight,
            q_out,
            w_out,
            n_head=args.n_head,
            head_dim=args.head_dim,
            rope_dim=args.rope_dim,
            score_scale=score_scale,
        )

    def q_only_fn():
        q_out = torch.empty_like(q_ref)
        w_out = torch.empty_like(w_ref)
        return dsa_indexer_project_query_only_torchair(
            hidden_states,
            q_c,
            cos,
            sin,
            wq_b_weight,
            weights_proj_weight,
            q_out,
            w_out,
            n_head=args.n_head,
            head_dim=args.head_dim,
            rope_dim=args.rope_dim,
            score_scale=score_scale,
            allow_compile=False,
        )

    def with_qc_fn():
        q_out = torch.empty_like(q_ref)
        w_out = torch.empty_like(w_ref)
        return dsa_indexer_project_query_only_with_qc_torchair(
            hidden_states,
            cos,
            sin,
            q_a_weight,
            q_norm_weight,
            wq_b_weight,
            weights_proj_weight,
            q_out,
            w_out,
            q_norm_eps=args.q_norm_eps,
            n_head=args.n_head,
            head_dim=args.head_dim,
            rope_dim=args.rope_dim,
            score_scale=score_scale,
            allow_compile=False,
        )

    current_ms = bench(current_fn, device, args.warmup, args.iters)
    q_only_ms = bench(q_only_fn, device, args.warmup, args.iters) if used_q else float("nan")
    with_qc_ms = bench(with_qc_fn, device, args.warmup, args.iters) if used_with_qc else float("nan")
    print(f"QUERY_ONLY_BENCH tokens={tokens} current_avg_ms={current_ms:.6f} q_only_torchair_avg_ms={q_only_ms:.6f} with_qc_torchair_avg_ms={with_qc_ms:.6f}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--tokens", default="4")
    parser.add_argument("--dtype", choices=("bf16", "fp16"), default="bf16")
    parser.add_argument("--hidden-size", type=int, default=7168)
    parser.add_argument("--q-lora-rank", type=int, default=1536)
    parser.add_argument("--n-head", type=int, default=64)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--rope-dim", type=int, default=64)
    parser.add_argument("--q-norm-eps", type=float, default=1e-6)
    parser.add_argument("--score-scale", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--atol", type=float, default=1.0)
    parser.add_argument("--rtol", type=float, default=0.01)
    args = parser.parse_args()

    device = torch.device(args.device)
    print(
        "QUERY_ONLY_PROBE "
        f"device={device} tokens={args.tokens} dtype={args.dtype} hidden_size={args.hidden_size} "
        f"q_lora_rank={args.q_lora_rank} n_head={args.n_head} head_dim={args.head_dim} "
        f"rope_dim={args.rope_dim} score_scale={args.score_scale} warmup={args.warmup} iters={args.iters}"
    )
    for tokens in parse_ints(args.tokens):
        run_one(args, int(tokens), device)


if __name__ == "__main__":
    main()
