from __future__ import annotations

import argparse
import sys
from pathlib import Path
from time import perf_counter
from typing import Any


ROOT = Path(__file__).resolve().parents[2]


def _make_topk_pair(
    *, batch: int, topk: int, full_len: int, overlap: float, seed: int
) -> tuple[Any, Any]:
    import torch

    if not 0.0 <= overlap <= 1.0:
        raise ValueError("--overlap must be in [0, 1]")
    keep = int(round(topk * overlap))
    replace = topk - keep
    candidate_len = full_len - 1  # the kernel treats the newest token as non-reusable
    if candidate_len - topk < replace:
        raise ValueError("--full-len does not contain enough replacement token ids")

    generator = torch.Generator().manual_seed(seed)
    old = torch.empty((batch, 1, 1, topk), dtype=torch.int32)
    new = torch.empty_like(old)
    for row in range(batch):
        permutation = torch.randperm(candidate_len, generator=generator)
        old_row = permutation[:topk]
        kept_positions = torch.randperm(topk, generator=generator)[:keep]
        replacement = permutation[topk : topk + replace]
        new_row = torch.cat((old_row[kept_positions], replacement))
        new_row = new_row[torch.randperm(topk, generator=generator)]
        old[row, 0, 0].copy_(old_row.to(torch.int32))
        new[row, 0, 0].copy_(new_row.to(torch.int32))
    return old, new


def _make_block_table(
    *, batch: int, blocks_per_row: int, seed: int, device: Any
) -> tuple[Any, Any]:
    import torch

    generator = torch.Generator().manual_seed(seed)
    rows = []
    for row in range(batch):
        rows.append(
            torch.randperm(blocks_per_row, generator=generator, dtype=torch.int64)
            + row * blocks_per_row
        )
    cpu = torch.stack(rows).to(torch.int32).contiguous()
    return cpu, cpu.to(device)


def _swapped_from_cpu(cpu_tensor: Any, device: Any) -> Any:
    import torch_npu

    swapped = torch_npu.empty_with_swapped_memory(
        cpu_tensor.shape, dtype=cpu_tensor.dtype, device=device
    )
    swapped.fill_(0)
    swapped.add_(cpu_tensor.to(device))
    return swapped


def _make_inputs(args: argparse.Namespace, device: Any) -> dict[str, Any]:
    import torch

    generator = torch.Generator().manual_seed(args.seed)
    full_blocks_per_row = (args.full_len + args.block_size - 1) // args.block_size
    selection_blocks_per_row = (args.topk + args.block_size - 1) // args.block_size
    full_block_count = args.batch_size * full_blocks_per_row
    selection_block_count = args.batch_size * selection_blocks_per_row

    full_k_rope_cpu = torch.randn(
        (full_block_count, args.block_size, args.rope_dim),
        generator=generator,
        dtype=torch.float32,
    ).to(torch.bfloat16).contiguous()
    full_kv_cache_cpu = torch.randn(
        (full_block_count, args.block_size, args.kv_dim),
        generator=generator,
        dtype=torch.float32,
    ).to(torch.bfloat16).contiguous()
    full_table_cpu, full_table = _make_block_table(
        batch=args.batch_size,
        blocks_per_row=full_blocks_per_row,
        seed=args.seed + 1,
        device=device,
    )
    selection_table_cpu, selection_table = _make_block_table(
        batch=args.batch_size,
        blocks_per_row=selection_blocks_per_row,
        seed=args.seed + 2,
        device=device,
    )
    old_topk_cpu, new_topk_cpu = _make_topk_pair(
        batch=args.batch_size,
        topk=args.topk,
        full_len=args.full_len,
        overlap=args.overlap,
        seed=args.seed + 3,
    )

    pool_capacity = args.batch_size + 3
    req_pool_entries_cpu = torch.arange(
        args.batch_size, 0, -1, dtype=torch.int32
    )
    return {
        "full_k_rope_cpu": full_k_rope_cpu,
        "full_kv_cache_cpu": full_kv_cache_cpu,
        "full_k_rope": _swapped_from_cpu(full_k_rope_cpu, device),
        "full_kv_cache": _swapped_from_cpu(full_kv_cache_cpu, device),
        "full_kv_block_table_cpu": full_table_cpu,
        "full_kv_block_table": full_table,
        "full_kv_actual_seq": torch.full(
            (args.batch_size,), args.full_len, dtype=torch.int32, device=device
        ),
        "selection_k_rope": torch.zeros(
            (selection_block_count, args.block_size, args.rope_dim),
            dtype=torch.bfloat16,
            device=device,
        ),
        "selection_kv_cache": torch.zeros(
            (selection_block_count, args.block_size, args.kv_dim),
            dtype=torch.bfloat16,
            device=device,
        ),
        "selection_kv_block_table_cpu": selection_table_cpu,
        "selection_kv_block_table": selection_table,
        "selection_kv_block_status_pool": torch.full(
            (pool_capacity, 1, 1, args.topk + 1),
            -1,
            dtype=torch.int32,
            device=device,
        ),
        "req_pool_entries_cpu": req_pool_entries_cpu,
        "req_pool_entries": req_pool_entries_cpu.to(device),
        "old_topk_cpu": old_topk_cpu,
        "new_topk_cpu": new_topk_cpu,
        "old_topk": old_topk_cpu.to(device),
        "new_topk": new_topk_cpu.to(device),
    }


def _call_op(inputs: dict[str, Any], topk_indices: Any) -> None:
    import nanovllm.ops as ascend_ops

    ascend_ops.npu_gather_selection_kv_cache(
        inputs["selection_k_rope"],
        inputs["selection_kv_cache"],
        inputs["selection_kv_block_table"],
        inputs["selection_kv_block_status_pool"],
        inputs["req_pool_entries"],
        topk_indices,
        inputs["full_k_rope"],
        inputs["full_kv_cache"],
        inputs["full_kv_block_table"],
        inputs["full_kv_actual_seq"],
    )


def _validate_step(
    args: argparse.Namespace,
    inputs: dict[str, Any],
    expected_topk_cpu: Any,
    step: int,
) -> None:
    import torch

    torch.npu.synchronize()
    status_pool = inputs["selection_kv_block_status_pool"].detach().cpu()
    selection_rope = inputs["selection_k_rope"].detach().cpu()
    selection_kv = inputs["selection_kv_cache"].detach().cpu()
    req_entries = inputs["req_pool_entries_cpu"].to(torch.int64)
    used_entries = set(int(value) for value in req_entries.tolist())

    for pool_row in range(status_pool.shape[0]):
        if pool_row not in used_entries and not bool((status_pool[pool_row] == -1).all()):
            raise AssertionError(f"unused pool row {pool_row} was modified")

    for row in range(args.batch_size):
        status = status_pool[int(req_entries[row]), 0, 0]
        selected = status[: args.topk].to(torch.int64)
        expected = expected_topk_cpu[row, 0, 0].to(torch.int64)
        if torch.unique(selected).numel() != args.topk:
            raise AssertionError(f"step={step} row={row} status contains duplicates")
        if not torch.equal(torch.sort(selected).values, torch.sort(expected).values):
            raise AssertionError(f"step={step} row={row} selected token set mismatch")
        if int(status[args.topk]) != args.topk:
            raise AssertionError(f"step={step} row={row} actual selection length mismatch")

        positions = torch.arange(args.topk, dtype=torch.int64)
        dst_blocks = inputs["selection_kv_block_table_cpu"][row].to(torch.int64)[
            positions // args.block_size
        ]
        dst_offsets = positions % args.block_size
        src_blocks = inputs["full_kv_block_table_cpu"][row].to(torch.int64)[
            selected // args.block_size
        ]
        src_offsets = selected % args.block_size
        actual_rope = selection_rope[dst_blocks, dst_offsets]
        actual_kv = selection_kv[dst_blocks, dst_offsets]
        expected_rope = inputs["full_k_rope_cpu"][src_blocks, src_offsets]
        expected_kv = inputs["full_kv_cache_cpu"][src_blocks, src_offsets]
        if not torch.equal(actual_rope, expected_rope):
            diff = float((actual_rope.float() - expected_rope.float()).abs().max())
            raise AssertionError(f"step={step} row={row} K-RoPE mismatch max_diff={diff:g}")
        if not torch.equal(actual_kv, expected_kv):
            diff = float((actual_kv.float() - expected_kv.float()).abs().max())
            raise AssertionError(f"step={step} row={row} latent-KV mismatch max_diff={diff:g}")

    print(
        "GSKV_PARALLEL_CHECK "
        f"step={step} rows={args.batch_size} ok=1"
    )


def _validate_short_row_skip(args: argparse.Namespace, inputs: dict[str, Any]) -> None:
    import torch

    skip_row = args.batch_size - 1
    pool_entry = int(inputs["req_pool_entries_cpu"][skip_row])
    selection_blocks = inputs["selection_kv_block_table"][skip_row].to(torch.int64)
    rope_before = inputs["selection_k_rope"][selection_blocks].clone()
    kv_before = inputs["selection_kv_cache"][selection_blocks].clone()
    status_before = inputs["selection_kv_block_status_pool"][pool_entry].clone()

    inputs["full_kv_actual_seq"][skip_row] = args.topk
    inputs["req_pool_entries"][skip_row] = -1
    _call_op(inputs, inputs["old_topk"])
    torch.npu.synchronize()
    if not torch.equal(inputs["selection_k_rope"][selection_blocks], rope_before):
        raise AssertionError("short-row K-RoPE cache was modified")
    if not torch.equal(inputs["selection_kv_cache"][selection_blocks], kv_before):
        raise AssertionError("short-row latent-KV cache was modified")
    if not torch.equal(inputs["selection_kv_block_status_pool"][pool_entry], status_before):
        raise AssertionError("short-row pool status was modified")

    inputs["full_kv_actual_seq"][skip_row] = args.full_len
    inputs["req_pool_entries"][skip_row] = pool_entry
    print(
        "GSKV_PARALLEL_SHORT_ROW_CHECK "
        f"row={skip_row} ok=1"
    )


def _validate_long_row_zero_miss(
    args: argparse.Namespace,
    inputs: dict[str, Any],
    topk_indices: Any,
    expected_topk_cpu: Any,
    previous_topk_cpu: Any | None,
    step: int,
) -> None:
    """Poison a prior miss destination and prove a zero-miss step does not copy it."""
    import torch

    row = 0
    pool_entry = int(inputs["req_pool_entries_cpu"][row])
    status_before = inputs["selection_kv_block_status_pool"][pool_entry].clone()
    status_cpu = status_before.detach().cpu()[0, 0, : args.topk].to(torch.int64)

    slot = 0
    if previous_topk_cpu is not None:
        previous = set(
            int(token)
            for token in previous_topk_cpu[row, 0, 0].to(torch.int64).tolist()
        )
        slot = next(
            (
                index
                for index, token in enumerate(status_cpu.tolist())
                if int(token) not in previous
            ),
            -1,
        )
        if slot < 0:
            raise AssertionError("zero-miss check could not find a previous miss destination")

    dst_block = int(
        inputs["selection_kv_block_table_cpu"][row, slot // args.block_size]
    )
    dst_offset = slot % args.block_size
    rope_view = inputs["selection_k_rope"][dst_block, dst_offset]
    kv_view = inputs["selection_kv_cache"][dst_block, dst_offset]
    rope_before = rope_view.clone()
    kv_before = kv_view.clone()
    rope_view.fill_(17.0)
    kv_view.fill_(-23.0)
    poisoned_rope = rope_view.clone()
    poisoned_kv = kv_view.clone()

    _call_op(inputs, topk_indices)
    torch.npu.synchronize()
    if not torch.equal(rope_view, poisoned_rope):
        raise AssertionError(
            f"step={step} row={row} zero-miss step replayed a stale K-RoPE copy"
        )
    if not torch.equal(kv_view, poisoned_kv):
        raise AssertionError(
            f"step={step} row={row} zero-miss step replayed a stale latent-KV copy"
        )
    if not torch.equal(
        inputs["selection_kv_block_status_pool"][pool_entry], status_before
    ):
        raise AssertionError(f"step={step} row={row} zero-miss step changed cache status")

    rope_view.copy_(rope_before)
    kv_view.copy_(kv_before)
    _validate_step(args, inputs, expected_topk_cpu, step)
    print(
        "GSKV_PARALLEL_ZERO_MISS_CHECK "
        f"step={step} row={row} slot={slot} ok=1"
    )


def _run_test(args: argparse.Namespace) -> None:
    if (
        args.topk != 2048
        or args.block_size != 128
        or args.rope_dim != 64
        or args.kv_dim != 512
    ):
        raise ValueError(
            "this UT intentionally targets topk=2048, block=128, rope=64, kv=512"
        )
    if args.full_len <= args.topk:
        raise ValueError("--full-len must be greater than --topk")

    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    import torch
    __import__("torch_npu")  # Registers torch.npu and the PrivateUse1 backend.

    try:
        import nanovllm.ops as ascend_ops
    except Exception as exc:
        raise RuntimeError(
            "nanovllm Ascend ops are unavailable; rebuild with scripts/build_nanovllm_ops.sh"
        ) from exc
    if not hasattr(ascend_ops, "npu_gather_selection_kv_cache"):
        raise RuntimeError("npu_gather_selection_kv_cache is unavailable; rebuild the custom ops")

    device = torch.device(args.device)
    torch.npu.set_device(device)
    print(
        "GSKV_PARALLEL_CONFIG "
        f"batch={args.batch_size} full_len={args.full_len} overlap={args.overlap:g}"
    )
    inputs = _make_inputs(args, device)

    _call_op(inputs, inputs["old_topk"])
    _validate_step(args, inputs, inputs["old_topk_cpu"], 0)
    _validate_long_row_zero_miss(
        args,
        inputs,
        inputs["old_topk"],
        inputs["old_topk_cpu"],
        None,
        1,
    )
    _call_op(inputs, inputs["new_topk"])
    _validate_step(args, inputs, inputs["new_topk_cpu"], 2)
    _validate_long_row_zero_miss(
        args,
        inputs,
        inputs["new_topk"],
        inputs["new_topk_cpu"],
        inputs["old_topk_cpu"],
        3,
    )
    _call_op(inputs, inputs["old_topk"])
    _validate_step(args, inputs, inputs["old_topk_cpu"], 4)
    _validate_short_row_skip(args, inputs)

    for _ in range(args.warmup):
        _call_op(inputs, inputs["old_topk"])
        _call_op(inputs, inputs["new_topk"])
    torch.npu.synchronize()
    start = perf_counter()
    for _ in range(args.iters):
        _call_op(inputs, inputs["old_topk"])
        _call_op(inputs, inputs["new_topk"])
    torch.npu.synchronize()
    avg_ms = (perf_counter() - start) * 1000.0 / max(args.iters * 2, 1)
    miss_count = args.topk - int(round(args.topk * args.overlap))
    print(
        "GSKV_PARALLEL_RESULT "
        f"avg_ms={avg_ms:.6f} "
        f"misses_per_row={miss_count} warmup={args.warmup} iters={args.iters}"
    )
    print("GSKV_PARALLEL_UT_OK")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate and benchmark the all-core GatherSelection copy tiling."
    )
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--batch-size", type=int, default=6)
    parser.add_argument("--full-len", type=int, default=32768)
    parser.add_argument("--topk", type=int, default=2048)
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument("--rope-dim", type=int, default=64)
    parser.add_argument("--kv-dim", type=int, default=512)
    parser.add_argument("--overlap", type=float, default=0.6)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--seed", type=int, default=1234)
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    _run_test(args)


if __name__ == "__main__":
    main()
