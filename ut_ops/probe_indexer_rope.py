from __future__ import annotations

import argparse
import time

import torch

try:
    import torch_npu  # type: ignore
except Exception:  # pragma: no cover - local non-Ascend syntax checks
    torch_npu = None


def sync(device: torch.device) -> None:
    if device.type == "npu" and torch_npu is not None:
        torch.npu.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize(device)


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


def desc(name: str, tensor: torch.Tensor) -> str:
    return (
        f"{name}=shape={tuple(tensor.shape)} dtype={tensor.dtype} "
        f"device={tensor.device} contiguous={tensor.is_contiguous()} "
        f"stride={tuple(tensor.stride())}"
    )


def rotate_half_neox(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    x1 = x[..., :half]
    x2 = x[..., half:]
    return torch.cat((-x2, x1), dim=-1)


def manual_rope_neox(
    q_pe: torch.Tensor,
    k_pe: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    q_dtype = q_pe.dtype
    k_dtype = k_pe.dtype
    cos_q = cos.view(cos.shape[0], 1, cos.shape[-1])
    sin_q = sin.view(sin.shape[0], 1, sin.shape[-1])
    q_out = q_pe * cos_q + rotate_half_neox(q_pe.float()).to(q_pe.dtype) * sin_q
    k_out = k_pe * cos_q + rotate_half_neox(k_pe.float()).to(k_pe.dtype) * sin_q
    return q_out.to(q_dtype), k_out.to(k_dtype)


def npu_rotary_mul_rope(
    q_pe: torch.Tensor,
    k_pe: torch.Tensor,
    cos4: torch.Tensor,
    sin4: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if torch_npu is None:
        raise RuntimeError("torch_npu is not available")
    q_out = torch_npu.npu_rotary_mul(q_pe.unsqueeze(2), cos4, sin4).squeeze(2)
    k_out = torch_npu.npu_rotary_mul(k_pe.unsqueeze(2), cos4, sin4).squeeze(2)
    return q_out, k_out


def make_cos_sin_cache(
    max_positions: int,
    rope_dim: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    inv_freq = 1.0 / (
        10000.0
        ** (torch.arange(0, rope_dim, 2, dtype=torch.float32, device=device) / rope_dim)
    )
    positions = torch.arange(max_positions, dtype=torch.float32, device=device)
    freqs = torch.einsum("i,j->ij", positions, inv_freq)
    return freqs.cos(), freqs.sin()


def build_full_cos_sin(
    cos_cache: torch.Tensor,
    sin_cache: torch.Tensor,
    positions: torch.Tensor,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    cos = cos_cache.index_select(0, positions)
    sin = sin_cache.index_select(0, positions)
    return (
        torch.cat((cos, cos), dim=-1).to(dtype).contiguous(),
        torch.cat((sin, sin), dim=-1).to(dtype).contiguous(),
    )


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


def parse_dtype(name: str) -> torch.dtype:
    if name == "bf16":
        return torch.bfloat16
    if name == "fp16":
        return torch.float16
    if name == "fp32":
        return torch.float32
    raise ValueError(f"unsupported dtype: {name}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--tokens", type=int, default=4)
    parser.add_argument("--heads", type=int, default=64)
    parser.add_argument("--rope-dim", type=int, default=64)
    parser.add_argument("--max-position", type=int, default=18016)
    parser.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    device = torch.device(args.device)
    if device.type == "npu":
        if torch_npu is None:
            raise RuntimeError("torch_npu is required for --device npu:*")
        torch.npu.set_device(device)
    dtype = parse_dtype(args.dtype)
    torch.manual_seed(args.seed)

    q_pe = torch.randn(
        args.tokens,
        args.heads,
        args.rope_dim,
        dtype=dtype,
        device=device,
    )
    k_pe = torch.randn(
        args.tokens,
        1,
        args.rope_dim,
        dtype=dtype,
        device=device,
    )
    positions = torch.arange(args.tokens, dtype=torch.long, device=device)
    cos_cache, sin_cache = make_cos_sin_cache(
        max(args.max_position, args.tokens),
        args.rope_dim,
        device,
    )
    cos, sin = build_full_cos_sin(cos_cache, sin_cache, positions, dtype)
    cos4 = cos.view(args.tokens, 1, 1, args.rope_dim)
    sin4 = sin.view(args.tokens, 1, 1, args.rope_dim)

    print(
        "INDEXER_ROPE_PROBE config "
        f"device={device} tokens={args.tokens} heads={args.heads} "
        f"rope_dim={args.rope_dim} dtype={dtype} warmup={args.warmup} "
        f"iters={args.iters} seed={args.seed}"
    )
    print("INDEXER_ROPE_PROBE " + desc("q_pe", q_pe))
    print("INDEXER_ROPE_PROBE " + desc("k_pe", k_pe))
    print("INDEXER_ROPE_PROBE " + desc("cos4", cos4))

    ref, manual_ms = bench(
        lambda: manual_rope_neox(q_pe, k_pe, cos, sin),
        args.warmup,
        args.iters,
        device,
    )
    print(f"INDEXER_ROPE_BENCH manual_cached_avg_ms={manual_ms:.6f}")

    if torch_npu is not None and device.type == "npu":
        fast, npu_cached_ms = bench(
            lambda: npu_rotary_mul_rope(q_pe, k_pe, cos4, sin4),
            args.warmup,
            args.iters,
            device,
        )
        print("INDEXER_ROPE_DIFF npu_cached q " + diff_report("", fast[0], ref[0]).lstrip(": "))
        print("INDEXER_ROPE_DIFF npu_cached k " + diff_report("", fast[1], ref[1]).lstrip(": "))
        print(f"INDEXER_ROPE_BENCH npu_cached_avg_ms={npu_cached_ms:.6f}")

        def npu_with_index_select():
            local_cos, local_sin = build_full_cos_sin(
                cos_cache,
                sin_cache,
                positions,
                dtype,
            )
            return npu_rotary_mul_rope(
                q_pe,
                k_pe,
                local_cos.view(args.tokens, 1, 1, args.rope_dim),
                local_sin.view(args.tokens, 1, 1, args.rope_dim),
            )

        fast_select, npu_select_ms = bench(
            npu_with_index_select,
            args.warmup,
            args.iters,
            device,
        )
        print(
            "INDEXER_ROPE_DIFF npu_with_index_select q "
            + diff_report("", fast_select[0], ref[0]).lstrip(": ")
        )
        print(
            "INDEXER_ROPE_DIFF npu_with_index_select k "
            + diff_report("", fast_select[1], ref[1]).lstrip(": ")
        )
        print(f"INDEXER_ROPE_BENCH npu_with_index_select_avg_ms={npu_select_ms:.6f}")


if __name__ == "__main__":
    main()
