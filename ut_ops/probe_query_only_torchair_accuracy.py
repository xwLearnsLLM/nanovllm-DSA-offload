from __future__ import annotations

import argparse
from time import perf_counter

import torch
import torch.nn.functional as F

try:
    import torch_npu  # type: ignore
except Exception:  # pragma: no cover - local syntax checks
    torch_npu = None

try:
    import nanovllm.ops as ascend_ops
except Exception as exc:  # pragma: no cover - Ascend ops are built on board
    ascend_ops = None
    ascend_ops_import_error = repr(exc)
else:
    ascend_ops_import_error = None

from nanovllm.models.dsa_indexer_project import (
    _apply_rope_neox_reference,
    _dsa_indexer_project_query_only_out_functional,
    _q_project,
    dsa_indexer_project_q_path,
    dsa_indexer_project_query_only,
    dsa_indexer_project_query_only_torchair,
)


def sync(device: torch.device) -> None:
    if device.type == "npu" and torch_npu is not None:
        torch.npu.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize(device)


def parse_ints(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def desc(name: str, tensor: torch.Tensor) -> str:
    return (
        f"{name}=shape={tuple(tensor.shape)} dtype={tensor.dtype} "
        f"device={tensor.device} contiguous={tensor.is_contiguous()} stride={tuple(tensor.stride())}"
    )


def diff_stats(actual: torch.Tensor, expected: torch.Tensor) -> tuple[float, float, float, int, int]:
    actual_f = actual.float()
    expected_f = expected.float()
    diff = (actual_f - expected_f).abs()
    max_abs = float(diff.max().item()) if diff.numel() else 0.0
    mean_abs = float(diff.mean().item()) if diff.numel() else 0.0
    denom = expected_f.abs().max().clamp_min(1e-6)
    max_rel = float((diff.max() / denom).item()) if diff.numel() else 0.0
    actual_bad = int((~torch.isfinite(actual_f)).sum().item()) if actual_f.numel() else 0
    expected_bad = int((~torch.isfinite(expected_f)).sum().item()) if expected_f.numel() else 0
    return max_abs, mean_abs, max_rel, actual_bad, expected_bad


def print_diff(name: str, actual: torch.Tensor, expected: torch.Tensor) -> tuple[float, float, float]:
    max_abs, mean_abs, max_rel, actual_bad, expected_bad = diff_stats(actual, expected)
    print(
        f"QOTA_DIFF {name}: max_abs={max_abs:.6g} mean_abs={mean_abs:.6g} "
        f"max_rel={max_rel:.6g} actual_bad={actual_bad} expected_bad={expected_bad}"
    )
    return max_abs, mean_abs, max_rel


def assert_close(name: str, actual: torch.Tensor, expected: torch.Tensor, atol: float, rtol: float) -> None:
    max_abs, _, max_rel = print_diff(name, actual, expected)
    if max_abs <= float(atol) or max_rel <= float(rtol):
        return
    raise AssertionError(f"{name} mismatch: max_abs={max_abs:.6g} max_rel={max_rel:.6g} atol={atol} rtol={rtol}")


def sentinel_report(name: str, tensor: torch.Tensor, sentinel: float) -> None:
    count = int((tensor == tensor.new_tensor(float(sentinel))).sum().item()) if tensor.numel() else 0
    print(f"QOTA_SENTINEL {name}: remaining={count} numel={tensor.numel()}")


def topk_overlap_report(name: str, actual_score: torch.Tensor, expected_score: torch.Tensor, k: int) -> None:
    k = max(1, min(int(k), int(actual_score.shape[-1]), int(expected_score.shape[-1])))
    actual_topk = torch.topk(actual_score.float(), k=k, dim=-1).indices.detach().cpu()
    expected_topk = torch.topk(expected_score.float(), k=k, dim=-1).indices.detach().cpu()
    overlaps: list[int] = []
    flat_actual = actual_topk.reshape(-1, k)
    flat_expected = expected_topk.reshape(-1, k)
    for row in range(flat_actual.shape[0]):
        overlaps.append(len(set(flat_actual[row].tolist()) & set(flat_expected[row].tolist())))
    min_overlap = min(overlaps) if overlaps else 0
    max_overlap = max(overlaps) if overlaps else 0
    mean_overlap = sum(overlaps) / max(len(overlaps), 1)
    print(
        f"QOTA_TOPK {name}: topk={k} min_overlap={min_overlap}/{k} "
        f"mean_overlap={mean_overlap:.3f}/{k} max_overlap={max_overlap}/{k}"
    )


def make_cos_sin(tokens: int, rope_dim: int, dtype: torch.dtype, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    base = torch.arange(tokens * rope_dim, dtype=torch.float32, device=device).view(tokens, rope_dim)
    base = base / max(rope_dim, 1)
    cos = base.cos().to(dtype).view(tokens, 1, 1, rope_dim).contiguous()
    sin = base.sin().to(dtype).view(tokens, 1, 1, rope_dim).contiguous()
    return cos, sin


def make_wq_b_bmm_t(wq_b_weight: torch.Tensor, n_head: int, head_dim: int, q_lora_rank: int) -> torch.Tensor | None:
    if wq_b_weight.device.type != "npu" or ascend_ops is None or not hasattr(ascend_ops, "batch_matmul_transpose"):
        return None
    return wq_b_weight.view(n_head, head_dim, q_lora_rank).transpose(1, 2).contiguous()


def current_q_rope(
    q: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    *,
    rope_dim: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    q_pe, q_nope = torch.split(q, [int(rope_dim), q.shape[-1] - int(rope_dim)], dim=-1)
    if q.device.type == "npu" and torch_npu is not None and q.dtype in (torch.float16, torch.bfloat16):
        q_pe_rot = torch_npu.npu_rotary_mul(q_pe.unsqueeze(2), cos, sin).squeeze(2)
    else:
        q_pe_rot = _apply_rope_neox_reference(q_pe, cos, sin, int(rope_dim))
    return q_pe, q_pe_rot, torch.cat((q_pe_rot, q_nope), dim=-1)


def functional_q_rope(
    q: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    *,
    rope_dim: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    q_pe, q_nope = torch.split(q, [int(rope_dim), q.shape[-1] - int(rope_dim)], dim=-1)
    q_pe_rot = _apply_rope_neox_reference(q_pe, cos, sin, int(rope_dim))
    return q_pe, q_pe_rot, torch.cat((q_pe_rot, q_nope), dim=-1)


def score_proxy(q_index: torch.Tensor, weights: torch.Tensor, candidate_k: torch.Tensor) -> torch.Tensor:
    per_head = torch.einsum("bhd,cd->bhc", q_index.float(), candidate_k.float())
    return (per_head * weights.float().unsqueeze(-1)).sum(dim=1)


def bench(fn, device: torch.device, warmup: int, iters: int) -> float:
    with torch.inference_mode():
        for _ in range(max(int(warmup), 0)):
            fn()
        sync(device)
        start = perf_counter()
        for _ in range(max(int(iters), 1)):
            fn()
        sync(device)
    return (perf_counter() - start) * 1000.0 / max(int(iters), 1)


def run_one(args: argparse.Namespace, tokens: int, device: torch.device) -> None:
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    torch.manual_seed(int(args.seed) + int(tokens))
    hidden_states = torch.randn((tokens, args.hidden_size), dtype=dtype, device=device)
    q_c = torch.randn((tokens, args.q_lora_rank), dtype=dtype, device=device)
    wq_b_weight = torch.randn((args.n_head * args.head_dim, args.q_lora_rank), dtype=dtype, device=device)
    weights_proj_weight = torch.randn((args.n_head, args.hidden_size), dtype=dtype if args.weights_dtype == "bf16" else torch.float32, device=device)
    cos, sin = make_cos_sin(tokens, args.rope_dim, dtype, device)
    wq_b_bmm_t = make_wq_b_bmm_t(wq_b_weight, args.n_head, args.head_dim, args.q_lora_rank) if args.use_bmm_transpose else None
    enable_q_bmm = wq_b_bmm_t is not None

    print(
        "QOTA_CASE "
        f"tokens={tokens} dtype={dtype} weights_dtype={weights_proj_weight.dtype} "
        f"q_path={dsa_indexer_project_q_path(q_c, wq_b_bmm_t, enable_q_bmm)} "
        f"ascend_ops_available={ascend_ops is not None} ascend_ops_import_error={ascend_ops_import_error}"
    )
    print("QOTA_TENSOR " + desc("hidden_states", hidden_states))
    print("QOTA_TENSOR " + desc("q_c", q_c))
    print("QOTA_TENSOR " + desc("cos", cos))
    print("QOTA_TENSOR " + desc("sin", sin))

    q_linear = F.linear(q_c, wq_b_weight).view(tokens, args.n_head, args.head_dim)
    q_project = _q_project(q_c, wq_b_weight, wq_b_bmm_t, args.n_head, args.head_dim, enable_q_bmm)
    assert_close(f"tokens={tokens} q_project_vs_linear", q_project, q_linear, args.atol, args.rtol)

    q_pe_current, q_pe_current_rot, q_current_manual = current_q_rope(q_project, cos, sin, rope_dim=args.rope_dim)
    q_pe_func, q_pe_func_rot, q_func_manual = functional_q_rope(q_linear, cos, sin, rope_dim=args.rope_dim)
    assert_close(f"tokens={tokens} q_pe_before_rope_current_vs_func", q_pe_current, q_pe_func, args.atol, args.rtol)
    print_diff(f"tokens={tokens} q_pe_after_rope_current_vs_func", q_pe_current_rot, q_pe_func_rot)
    print_diff(f"tokens={tokens} q_index_manual_current_vs_func", q_current_manual, q_func_manual)

    q_current = torch.empty((tokens, args.n_head, args.head_dim), dtype=dtype, device=device)
    w_current = torch.empty((tokens, args.n_head), dtype=dtype, device=device)
    q_func = torch.empty_like(q_current)
    w_func = torch.empty_like(w_current)
    q_ta = torch.empty_like(q_current)
    w_ta = torch.empty_like(w_current)

    dsa_indexer_project_query_only(
        hidden_states,
        q_c,
        cos,
        sin,
        wq_b_weight,
        weights_proj_weight,
        q_current,
        w_current,
        n_head=args.n_head,
        head_dim=args.head_dim,
        rope_dim=args.rope_dim,
        score_scale=args.score_scale,
        wq_b_bmm_t=wq_b_bmm_t,
        enable_q_bmm=enable_q_bmm,
    )
    _dsa_indexer_project_query_only_out_functional(
        hidden_states,
        q_c,
        cos,
        sin,
        wq_b_weight,
        weights_proj_weight,
        q_func,
        w_func,
        n_head=args.n_head,
        head_dim=args.head_dim,
        rope_dim=args.rope_dim,
        score_scale=args.score_scale,
    )

    print_diff(f"tokens={tokens} current_api_vs_manual q", q_current, q_current_manual)
    print_diff(f"tokens={tokens} functional_api_vs_manual q", q_func, q_func_manual)
    print_diff(f"tokens={tokens} current_vs_functional q", q_current, q_func)
    print_diff(f"tokens={tokens} current_vs_functional weights", w_current, w_func)

    for repeat in range(int(args.repeats)):
        q_ta.fill_(float(args.sentinel))
        w_ta.fill_(float(args.sentinel))
        q_ta, w_ta, used_torchair, reason = dsa_indexer_project_query_only_torchair(
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
            score_scale=args.score_scale,
            allow_compile=repeat == 0,
        )
        sync(device)
        print(f"QOTA_TORCHAIR tokens={tokens} repeat={repeat} used={used_torchair} reason={reason}")
        sentinel_report(f"tokens={tokens} repeat={repeat} q", q_ta, args.sentinel)
        sentinel_report(f"tokens={tokens} repeat={repeat} weights", w_ta, args.sentinel)
        print_diff(f"tokens={tokens} repeat={repeat} torchair_vs_functional q", q_ta, q_func)
        print_diff(f"tokens={tokens} repeat={repeat} torchair_vs_functional weights", w_ta, w_func)
        print_diff(f"tokens={tokens} repeat={repeat} torchair_vs_current q", q_ta, q_current)
        print_diff(f"tokens={tokens} repeat={repeat} torchair_vs_current weights", w_ta, w_current)

        if int(args.score_proxy_candidates) > 0:
            candidate_k = torch.randn((int(args.score_proxy_candidates), args.head_dim), dtype=dtype, device=device)
            score_current = score_proxy(q_current, w_current, candidate_k)
            score_torchair = score_proxy(q_ta, w_ta, candidate_k)
            print_diff(f"tokens={tokens} repeat={repeat} score_proxy_torchair_vs_current", score_torchair, score_current)
            topk_overlap_report(f"tokens={tokens} repeat={repeat} score_proxy_torchair_vs_current", score_torchair, score_current, args.score_proxy_topk)

    def current_fn():
        q_out = torch.empty_like(q_current)
        w_out = torch.empty_like(w_current)
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
            score_scale=args.score_scale,
            wq_b_bmm_t=wq_b_bmm_t,
            enable_q_bmm=enable_q_bmm,
        )

    def torchair_fn():
        q_out = torch.empty_like(q_current)
        w_out = torch.empty_like(w_current)
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
            score_scale=args.score_scale,
            allow_compile=False,
        )

    current_ms = bench(current_fn, device, args.warmup, args.iters)
    torchair_ms = bench(torchair_fn, device, args.warmup, args.iters)
    print(f"QOTA_BENCH tokens={tokens} current_avg_ms={current_ms:.6f} torchair_avg_ms={torchair_ms:.6f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose decode query-only TorchAir accuracy.")
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--tokens", default="10")
    parser.add_argument("--dtype", choices=("bf16", "fp16"), default="bf16")
    parser.add_argument("--weights-dtype", choices=("bf16", "fp32"), default="bf16")
    parser.add_argument("--hidden-size", type=int, default=7168)
    parser.add_argument("--q-lora-rank", type=int, default=1536)
    parser.add_argument("--n-head", type=int, default=64)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--rope-dim", type=int, default=64)
    parser.add_argument("--score-scale", type=float, default=1.0)
    parser.add_argument("--use-bmm-transpose", action="store_true")
    parser.add_argument("--score-proxy-candidates", type=int, default=512)
    parser.add_argument("--score-proxy-topk", type=int, default=128)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--sentinel", type=float, default=-123.0)
    parser.add_argument("--atol", type=float, default=1.0)
    parser.add_argument("--rtol", type=float, default=0.01)
    args = parser.parse_args()

    device = torch.device(args.device)
    print(
        "QOTA_CONFIG "
        f"device={device} tokens={args.tokens} dtype={args.dtype} weights_dtype={args.weights_dtype} "
        f"use_bmm_transpose={args.use_bmm_transpose} repeats={args.repeats} warmup={args.warmup} iters={args.iters}"
    )
    for tokens in parse_ints(args.tokens):
        run_one(args, int(tokens), device)


if __name__ == "__main__":
    main()
