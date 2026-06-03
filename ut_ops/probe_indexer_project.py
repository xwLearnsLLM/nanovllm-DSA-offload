from __future__ import annotations

import argparse
import time

import torch
import torch.nn.functional as F

from nanovllm.models.dsa_indexer_project import dsa_indexer_project, dsa_indexer_project_post_available, dsa_indexer_project_q_path
from nanovllm.models import dsa_indexer_project_real

try:
    import torch_npu  # type: ignore
except Exception:  # pragma: no cover - local non-Ascend syntax checks
    torch_npu = None

try:
    import nanovllm.ops as ascend_ops
    ascend_ops_import_error = None
except Exception as exc:  # pragma: no cover
    ascend_ops = None
    ascend_ops_import_error = repr(exc)


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


def assert_close(
    name: str,
    actual: torch.Tensor,
    expected: torch.Tensor,
    *,
    atol: float,
    rtol: float,
) -> None:
    actual_f = actual.float()
    expected_f = expected.float()
    diff = (actual_f - expected_f).abs()
    max_abs_tensor = diff.max()
    denom = expected_f.abs().max().clamp_min(1e-6)
    max_rel_tensor = max_abs_tensor / denom

    # The NPU bf16 q path may use a different GEMM kernel from the reference
    # F.linear path. For projection outputs we care about global numerical
    # agreement; torch.allclose-style per-element tolerance is too strict near
    # zero and can fail even when max_rel is already within budget.
    max_abs = float(max_abs_tensor.item())
    max_rel = float(max_rel_tensor.item())
    if max_abs <= atol or max_rel <= rtol:
        return

    flat_idx = int(diff.reshape(-1).argmax().item())
    actual_bad = float(actual_f.reshape(-1)[flat_idx].item())
    expected_bad = float(expected_f.reshape(-1)[flat_idx].item())
    raise AssertionError(
        f"{name} mismatch: max_abs={max_abs:.6g} max_rel={max_rel:.6g} "
        f"atol={atol} rtol={rtol} worst_flat_idx={flat_idx} "
        f"actual={actual_bad:.6g} expected={expected_bad:.6g}"
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


def apply_rope_model_path(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, *, is_k: bool) -> torch.Tensor:
    if x.device.type == "npu" and torch_npu is not None and x.dtype in (torch.float16, torch.bfloat16):
        cos4 = cos.view(cos.shape[0], 1, 1, cos.shape[-1]).contiguous()
        sin4 = sin.view(sin.shape[0], 1, 1, sin.shape[-1]).contiguous()
        if is_k:
            return torch_npu.npu_rotary_mul(x.unsqueeze(1).unsqueeze(2), cos4, sin4).squeeze(2).squeeze(1)
        return torch_npu.npu_rotary_mul(x.unsqueeze(2), cos4, sin4).squeeze(2)
    if is_k:
        return apply_rope_neox(x.unsqueeze(1), cos, sin).squeeze(1)
    return apply_rope_neox(x, cos, sin)


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
    return q, k, score_weights.to(hidden_states.dtype)


def indexer_model_reference(
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
    k = F.layer_norm(k, (head_dim,), weights["k_norm_weight"], weights["k_norm_bias"], eps=1e-6)
    k_pe, k_nope = torch.split(k, [rope_dim, head_dim - rope_dim], dim=-1)

    q_pe = apply_rope_model_path(q_pe, cos, sin, is_k=False)
    k_pe = apply_rope_model_path(k_pe, cos, sin, is_k=True)
    q = torch.cat((q_pe, q_nope), dim=-1)
    k = torch.cat((k_pe, k_nope), dim=-1)

    score_weights = F.linear(hidden_states.float(), weights["weights_proj"].float())
    return q, k, score_weights.to(hidden_states.dtype)


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


def q_project_bmm_transpose_cached(
    q_c: torch.Tensor,
    wq_b_bmm_t: torch.Tensor,
    *,
    n_head: int,
    head_dim: int,
) -> torch.Tensor:
    if ascend_ops is None:
        raise RuntimeError("nanovllm.ops is not available")
    q_c_by_head = q_c.unsqueeze(1).expand(-1, n_head, -1).contiguous()
    out = torch.empty(
        (q_c.shape[0], n_head, head_dim),
        dtype=q_c.dtype,
        device=q_c.device,
    )
    ascend_ops.batch_matmul_transpose(q_c_by_head, wq_b_bmm_t, out)
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


def format_detail(prefix: str, detail: dict[str, float], iters: int) -> str:
    fields = " ".join(
        f"{name}={detail.get(name, 0.0) * 1000.0 / max(iters, 1):.6f}ms"
        for name in ("q_proj", "k_proj", "k_norm", "rope", "weights_proj")
    )
    return f"{prefix} {fields}"


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
    parser.add_argument("--atol", type=float, default=0.03125)
    parser.add_argument("--rtol", type=float, default=0.01)
    parser.add_argument("--skip-bmm-transpose", action="store_true")
    parser.add_argument("--use-bmm-transpose", action="store_true")
    parser.add_argument("--reuse-output-buffers", action="store_true")
    parser.add_argument("--profile-detail", action="store_true")
    parser.add_argument("--detail-iters", type=int, default=20)
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
    print(
        "INDEXER_PROBE "
        f"ascend_ops_available={ascend_ops is not None} "
        f"batch_matmul_transpose_available={ascend_ops is not None and hasattr(ascend_ops, 'batch_matmul_transpose')} "
        f"ascend_ops_import_error={ascend_ops_import_error} "
        f"dsa_indexer_project_post_available={dsa_indexer_project_post_available()} "
        f"dsa_indexer_project_post_binding={dsa_indexer_project_real.binding_version()}"
    )

    common_kwargs = {
        "n_head": args.heads,
        "head_dim": args.head_dim,
        "rope_dim": args.rope_dim,
    }
    score_scale = 1.0
    enable_q_bmm = bool(args.use_bmm_transpose and not args.skip_bmm_transpose)
    cos_op = cos.view(args.tokens, 1, 1, args.rope_dim).contiguous()
    sin_op = sin.view(args.tokens, 1, 1, args.rope_dim).contiguous()
    wq_b_bmm_t = None
    if device.type == "npu" and not args.skip_bmm_transpose:
        wq_b_bmm_t = (
            weights["wq_b"]
            .view(args.heads, args.head_dim, args.q_lora_rank)
            .transpose(1, 2)
            .contiguous()
        )
    reusable_outputs = None
    if args.reuse_output_buffers:
        reusable_outputs = (
            torch.empty((args.tokens, args.heads, args.head_dim), dtype=dtype, device=device),
            torch.empty((args.tokens, args.head_dim), dtype=dtype, device=device),
            torch.empty((args.tokens, args.heads), dtype=dtype, device=device),
        )

    def run_dsa_indexer_project(detail: dict[str, float] | None = None, sync_detail: bool = False):
        if reusable_outputs is None:
            q_out = torch.empty((args.tokens, args.heads, args.head_dim), dtype=dtype, device=device)
            k_out = torch.empty((args.tokens, args.head_dim), dtype=dtype, device=device)
            weights_out = torch.empty((args.tokens, args.heads), dtype=dtype, device=device)
        else:
            q_out, k_out, weights_out = reusable_outputs
        return dsa_indexer_project(
            hidden_states,
            q_c,
            cos_op,
            sin_op,
            weights["wq_b"],
            weights["wk"],
            weights["k_norm_weight"],
            weights["k_norm_bias"],
            weights["weights_proj"],
            q_out,
            k_out,
            weights_out,
            n_head=args.heads,
            head_dim=args.head_dim,
            rope_dim=args.rope_dim,
            score_scale=score_scale,
            wq_b_bmm_t=wq_b_bmm_t,
            enable_q_bmm=enable_q_bmm,
            detail=detail,
            sync_detail=sync_detail,
        )

    ref, ref_ms = bench(
        lambda: indexer_current(hidden_states, q_c, weights, cos, sin, **common_kwargs),
        args.warmup,
        args.iters,
        device,
    )
    model_ref, model_ref_ms = bench(
        lambda: indexer_model_reference(hidden_states, q_c, weights, cos, sin, **common_kwargs),
        args.warmup,
        args.iters,
        device,
    )
    op, op_ms = bench(
        lambda: run_dsa_indexer_project(),
        args.warmup,
        args.iters,
        device,
    )
    assert ref is not None and model_ref is not None and op is not None
    ref_q, ref_k, _ = ref
    model_ref_q, model_ref_k, model_ref_weights = model_ref
    op_q, op_k, op_weights = op

    print(
        "INDEXER_PROBE "
        f"dsa_indexer_project_q_path={dsa_indexer_project_q_path(q_c, wq_b_bmm_t, enable_q_bmm)} "
        f"reuse_output_buffers={args.reuse_output_buffers}"
    )
    print("INDEXER_DIFF dsa_indexer_project_vs_model_ref " + diff_report("q", op_q, model_ref_q))
    print("INDEXER_DIFF dsa_indexer_project_vs_model_ref " + diff_report("k", op_k, model_ref_k))
    print(
        "INDEXER_DIFF dsa_indexer_project_vs_model_ref "
        + diff_report("weights", op_weights, model_ref_weights)
    )
    print("INDEXER_DIFF model_ref_vs_manual_rope_ref " + diff_report("q", model_ref_q, ref_q))
    print("INDEXER_DIFF model_ref_vs_manual_rope_ref " + diff_report("k", model_ref_k, ref_k))
    assert_close("dsa_indexer_project q", op_q, model_ref_q, atol=args.atol, rtol=args.rtol)
    assert_close("dsa_indexer_project k", op_k, model_ref_k, atol=args.atol, rtol=args.rtol)
    assert_close(
        "dsa_indexer_project weights",
        op_weights,
        model_ref_weights,
        atol=args.atol,
        rtol=args.rtol,
    )
    print(
        f"INDEXER_BENCH manual_ref_avg_ms={ref_ms:.6f} "
        f"model_ref_avg_ms={model_ref_ms:.6f} "
        f"dsa_indexer_project_avg_ms={op_ms:.6f}"
    )

    if args.profile_detail:
        detail: dict[str, float] = {}
        for _ in range(args.detail_iters):
            run_dsa_indexer_project(detail=detail, sync_detail=True)
        print(format_detail("INDEXER_DETAIL dsa_indexer_project", detail, args.detail_iters))

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

        assert wq_b_bmm_t is not None
        q_bmm_cached, q_bmm_cached_ms = bench(
            lambda: q_project_bmm_transpose_cached(
                q_c,
                wq_b_bmm_t,
                n_head=args.heads,
                head_dim=args.head_dim,
            ),
            args.warmup,
            args.iters,
            device,
        )
        assert isinstance(q_bmm_cached, torch.Tensor)
        print(
            "INDEXER_DIFF q_bmm_transpose_cached "
            + diff_report("q", q_bmm_cached, q_linear)
        )
        print(
            "INDEXER_BENCH "
            f"q_bmm_transpose_cached_avg_ms={q_bmm_cached_ms:.6f}"
        )
    else:
        print("INDEXER_BENCH q_bmm_transpose skipped")


if __name__ == "__main__":
    main()
