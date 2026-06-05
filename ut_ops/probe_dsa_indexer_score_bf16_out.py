from __future__ import annotations

import argparse
import time

import torch

try:
    import torch_npu  # type: ignore  # noqa: F401
except Exception:  # pragma: no cover - local non-Ascend syntax checks
    torch_npu = None

import nanovllm.ops as ascend_ops


def sync(device: torch.device) -> None:
    if device.type == "npu" and torch_npu is not None:
        torch.npu.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize(device)


def desc(name: str, tensor: torch.Tensor) -> str:
    return (
        f"{name}=shape={tuple(tensor.shape)} dtype={tensor.dtype} "
        f"device={tensor.device} contiguous={tensor.is_contiguous()} stride={tuple(tensor.stride())}"
    )


def make_block_table(batch_size: int, table_cols: int, num_blocks: int, device: torch.device) -> torch.Tensor:
    rows = []
    base = 0
    for _ in range(batch_size):
        row = torch.arange(base, base + table_cols, dtype=torch.int32, device=device) % num_blocks
        rows.append(row)
        base += table_cols
    return torch.stack(rows, dim=0).contiguous()


def assert_close(name: str, actual: torch.Tensor, expected: torch.Tensor, atol: float, rtol: float) -> None:
    diff = (actual.float() - expected.float()).abs()
    max_abs = float(diff.max().item()) if diff.numel() else 0.0
    mean_abs = float(diff.mean().item()) if diff.numel() else 0.0
    allowed = atol + rtol * expected.float().abs()
    bad = diff > allowed
    bad_count = int(bad.sum().item()) if bad.numel() else 0
    print(f"QK_BF16_OUT_DIFF {name}: max_abs={max_abs:.6g} mean_abs={mean_abs:.6g} bad_count={bad_count}")
    if bad_count:
        raise AssertionError(f"{name} mismatch: max_abs={max_abs:.6g} bad_count={bad_count}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Probe npu_dsa_indexer_score_bf16_out against npu_dsa_indexer_score.")
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--heads", type=int, default=64)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument("--block-count", type=int, default=64)
    parser.add_argument("--extra-block-cols", type=int, default=4)
    parser.add_argument("--extra-output-cols", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--atol", type=float, default=0.0)
    parser.add_argument("--rtol", type=float, default=0.0)
    args = parser.parse_args()

    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    batch_size = int(args.batch_size)
    heads = int(args.heads)
    head_dim = int(args.head_dim)
    block_size = int(args.block_size)
    block_count = int(args.block_count)
    table_cols = block_count + int(args.extra_block_cols)
    score_count = block_count * block_size
    output_stride = score_count + int(args.extra_output_cols)
    num_blocks = max(batch_size * table_cols + 8, table_cols + 1)

    query = torch.randn((batch_size, heads, head_dim), dtype=torch.bfloat16, device=device)
    key = torch.randn((num_blocks, block_size, 1, head_dim), dtype=torch.bfloat16, device=device)
    weights = torch.randn((batch_size, heads), dtype=torch.bfloat16, device=device)
    actual_seq_q = torch.arange(1, batch_size + 1, dtype=torch.int32, device=device)
    actual_seq_k = torch.full((batch_size,), score_count, dtype=torch.int32, device=device)
    full_block_table = make_block_table(batch_size, table_cols, num_blocks, device)
    narrow_block_table = full_block_table[:, :block_count].contiguous()

    print(
        "QK_BF16_OUT config "
        f"device={device} batch_size={batch_size} heads={heads} head_dim={head_dim} "
        f"block_size={block_size} block_count={block_count} table_cols={table_cols} "
        f"score_count={score_count} output_stride={output_stride} warmup={args.warmup} iters={args.iters}"
    )
    print(desc("query", query))
    print(desc("key", key))
    print(desc("weights", weights))
    print(desc("full_block_table", full_block_table))

    ref = ascend_ops.npu_dsa_indexer_score(
        query,
        key,
        weights,
        actual_seq_q,
        actual_seq_k,
        narrow_block_table,
        "TND",
        "PA_BSND",
    )
    out = torch.empty((batch_size, output_stride), dtype=torch.bfloat16, device=device)
    tail_sentinel = torch.full((batch_size, int(args.extra_output_cols)), 7.0, dtype=torch.bfloat16, device=device)
    out[:, score_count:].copy_(tail_sentinel)
    ascend_ops.npu_dsa_indexer_score_bf16_out(
        query,
        key,
        weights,
        actual_seq_q,
        actual_seq_k,
        full_block_table,
        block_count,
        out,
        "TND",
        "PA_BSND",
    )
    sync(device)

    expected = ref[:, 0, :score_count].to(torch.bfloat16)
    assert_close("new_vs_old_bf16", out[:, :score_count], expected, args.atol, args.rtol)
    if int(args.extra_output_cols) > 0:
        tail = out[:, score_count:]
        assert_close("tail_unchanged", tail, tail_sentinel, 0.0, 0.0)

    for _ in range(args.warmup):
        _ = ascend_ops.npu_dsa_indexer_score(
            query,
            key,
            weights,
            actual_seq_q,
            actual_seq_k,
            narrow_block_table,
            "TND",
            "PA_BSND",
        )
    sync(device)
    t0 = time.perf_counter()
    for _ in range(args.iters):
        _ = ascend_ops.npu_dsa_indexer_score(
            query,
            key,
            weights,
            actual_seq_q,
            actual_seq_k,
            narrow_block_table,
            "TND",
            "PA_BSND",
        )
    sync(device)
    old_ms = (time.perf_counter() - t0) * 1000.0 / max(args.iters, 1)

    for _ in range(args.warmup):
        ascend_ops.npu_dsa_indexer_score_bf16_out(
            query,
            key,
            weights,
            actual_seq_q,
            actual_seq_k,
            full_block_table,
            block_count,
            out,
            "TND",
            "PA_BSND",
        )
    sync(device)
    t0 = time.perf_counter()
    for _ in range(args.iters):
        ascend_ops.npu_dsa_indexer_score_bf16_out(
            query,
            key,
            weights,
            actual_seq_q,
            actual_seq_k,
            full_block_table,
            block_count,
            out,
            "TND",
            "PA_BSND",
        )
    sync(device)
    new_ms = (time.perf_counter() - t0) * 1000.0 / max(args.iters, 1)
    print(f"QK_BF16_OUT_BENCH old_float_return_avg_ms={old_ms:.6f} new_bf16_out_avg_ms={new_ms:.6f}")


if __name__ == "__main__":
    main()
