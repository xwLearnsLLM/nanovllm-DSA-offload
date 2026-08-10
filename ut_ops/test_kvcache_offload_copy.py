#!/usr/bin/env python3
"""INT8 KVCache HBM-to-DRAM offload correctness and latency test."""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import torch
import torch_npu  # type: ignore

import nanovllm.ops  # noqa: F401  # Load repository-local custom operators.
from ut_ops._op_utils import require_local_opapi


MAX_COPY_CAP = 65536
DRAM_POISON = -113


@dataclass
class Case:
    device: torch.device
    device_name: str
    hbm_cache_cpu: torch.Tensor
    hbm_cache: torch.Tensor
    dram_cache: torch.Tensor
    hbm_table_cpu: torch.Tensor
    dram_table_cpu: torch.Tensor
    copy_counts_cpu: torch.Tensor
    hbm_table: torch.Tensor
    dram_table: torch.Tensor
    copy_counts: torch.Tensor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--copy-cap", type=int, default=32)
    parser.add_argument("--copy-min", type=int, default=0)
    parser.add_argument("--copy-max", type=int, default=16)
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument("--cache-dim", type=int, default=512)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive.")
    if not 0 < args.copy_cap <= MAX_COPY_CAP:
        raise ValueError(f"--copy-cap must be in [1, {MAX_COPY_CAP}].")
    if not 0 <= args.copy_min <= args.copy_max <= args.copy_cap:
        raise ValueError(
            "copy counts must satisfy 0 <= min <= max <= copy-cap."
        )
    if args.block_size <= 0 or args.cache_dim <= 0:
        raise ValueError("--block-size and --cache-dim must be positive.")
    if args.warmup < 0 or args.iters <= 0:
        raise ValueError("--warmup must be non-negative and --iters positive.")


def random_block_table_with_guard(
    batch_size: int,
    copy_cap: int,
    generator: torch.Generator,
) -> tuple[torch.Tensor, int]:
    table_entries = batch_size * copy_cap
    physical_blocks = table_entries + 1
    table = torch.randperm(
        physical_blocks,
        generator=generator,
        dtype=torch.int64,
    )[:table_entries].reshape(batch_size, copy_cap)
    return table.to(torch.int32).contiguous(), physical_blocks


def make_case(args: argparse.Namespace) -> Case:
    if not hasattr(torch_npu, "empty_with_swapped_memory"):
        raise RuntimeError(
            "torch_npu.empty_with_swapped_memory is unavailable. "
            "This test refuses to replace the DRAM destination with HBM."
        )

    device = torch.device(args.device)
    device_index = (
        device.index
        if device.index is not None
        else torch.npu.current_device()
    )
    get_device_name = getattr(torch.npu, "get_device_name", None)
    if get_device_name is None:
        get_device_name = torch_npu.npu.get_device_name
    device_name = get_device_name(device_index)
    generator = torch.Generator().manual_seed(args.seed)

    hbm_table_cpu, hbm_blocks = random_block_table_with_guard(
        args.batch_size, args.copy_cap, generator
    )
    dram_table_cpu, dram_blocks = random_block_table_with_guard(
        args.batch_size, args.copy_cap, generator
    )
    hbm_cache_cpu = torch.randint(
        -128,
        128,
        (hbm_blocks, args.block_size, args.cache_dim),
        generator=generator,
        dtype=torch.int8,
    )
    dram_cache = torch_npu.empty_with_swapped_memory(
        (dram_blocks, args.block_size, args.cache_dim),
        dtype=torch.int8,
        device=device,
    )

    copy_counts_cpu = torch.randint(
        args.copy_min,
        args.copy_max + 1,
        (args.batch_size,),
        generator=generator,
        dtype=torch.int32,
    )
    if args.batch_size > 1 and args.copy_min == 0:
        copy_counts_cpu[0] = 0
    if args.copy_max > 0:
        copy_counts_cpu[-1] = args.copy_max

    return Case(
        device=device,
        device_name=device_name,
        hbm_cache_cpu=hbm_cache_cpu,
        hbm_cache=hbm_cache_cpu.to(device),
        dram_cache=dram_cache,
        hbm_table_cpu=hbm_table_cpu,
        dram_table_cpu=dram_table_cpu,
        copy_counts_cpu=copy_counts_cpu,
        hbm_table=hbm_table_cpu.to(device),
        dram_table=dram_table_cpu.to(device),
        copy_counts=copy_counts_cpu.to(device),
    )


def call_offload(
    case: Case,
    copy_counts: torch.Tensor | None = None,
) -> torch.Tensor:
    return torch.ops.nanovllm_dsa.kvcache_offload_copy.default(
        case.hbm_cache,
        case.dram_cache,
        case.hbm_table,
        case.dram_table,
        case.copy_counts if copy_counts is None else copy_counts,
    )


def dram_to_cpu(case: Case) -> torch.Tensor:
    # Materialize swapped-memory data in an ordinary HBM tensor before the
    # standard device-to-host transfer.
    staging = torch.empty(
        case.dram_cache.shape,
        dtype=case.dram_cache.dtype,
        device=case.device,
    )
    staging.copy_(case.dram_cache)
    torch.npu.synchronize()
    return staging.cpu()


def assert_all_dram_poisoned(case: Case) -> None:
    actual = dram_to_cpu(case)
    if not torch.equal(actual, torch.full_like(actual, DRAM_POISON)):
        raise AssertionError("copy_count=0 modified the DRAM cache.")


def active_pairs(case: Case) -> tuple[torch.Tensor, torch.Tensor]:
    sources: list[torch.Tensor] = []
    destinations: list[torch.Tensor] = []
    for row, count in enumerate(case.copy_counts_cpu.tolist()):
        sources.append(case.hbm_table_cpu[row, :count].to(torch.int64))
        destinations.append(case.dram_table_cpu[row, :count].to(torch.int64))
    return torch.cat(sources), torch.cat(destinations)


def assert_copied(case: Case) -> None:
    source_blocks, destination_blocks = active_pairs(case)
    actual = dram_to_cpu(case)
    if source_blocks.numel():
        expected_active = case.hbm_cache_cpu.index_select(0, source_blocks)
        actual_active = actual.index_select(0, destination_blocks)
        if not torch.equal(actual_active, expected_active):
            mismatch = int(torch.count_nonzero(actual_active != expected_active))
            raise AssertionError(
                f"HBM-to-DRAM block copy mismatch in {mismatch} bytes."
            )

    active_destinations = set(destination_blocks.tolist())
    guard = next(
        block
        for block in range(case.dram_cache.shape[0])
        if block not in active_destinations
    )
    if not torch.equal(
        actual[guard],
        torch.full_like(actual[guard], DRAM_POISON),
    ):
        raise AssertionError("An inactive DRAM guard block was modified.")


def run(args: argparse.Namespace) -> None:
    opapi_path = require_local_opapi()
    case = make_case(args)
    copied_blocks = int(case.copy_counts_cpu.sum())
    block_bytes = args.block_size * args.cache_dim
    payload_bytes = copied_blocks * block_bytes

    print(
        "KVCACHE_OFFLOAD_COPY_CONFIG "
        f"device={case.device} device_name={case.device_name!r} "
        f"dtype=int8 batch={args.batch_size} copy_cap={args.copy_cap} "
        f"copy_range=[{args.copy_min},{args.copy_max}] "
        f"copy_counts={case.copy_counts_cpu.tolist()} "
        f"block_shape=[{args.block_size},{args.cache_dim}] "
        f"block_bytes={block_bytes} opapi={opapi_path}",
        flush=True,
    )

    case.dram_cache.fill_(DRAM_POISON)
    zero_counts = torch.zeros_like(case.copy_counts)
    call_offload(case, zero_counts)
    torch.npu.synchronize()
    assert_all_dram_poisoned(case)

    output = call_offload(case)
    torch.npu.synchronize()
    if output.data_ptr() != case.dram_cache.data_ptr():
        raise AssertionError("The output did not alias the DRAM input cache.")
    assert_copied(case)
    print(
        "KVCACHE_OFFLOAD_COPY_HBM_TO_DRAM_CHECK "
        "allocator=empty_with_swapped_memory "
        f"copied_blocks={copied_blocks} payload_bytes={payload_bytes} "
        "guard_unchanged=1 zero_count_unchanged=1 ok=1",
        flush=True,
    )

    for _ in range(args.warmup):
        call_offload(case)
    torch.npu.synchronize()

    start = torch.npu.Event(enable_timing=True)
    end = torch.npu.Event(enable_timing=True)
    start.record()
    for _ in range(args.iters):
        call_offload(case)
    end.record()
    end.synchronize()
    avg_us = start.elapsed_time(end) * 1000 / args.iters
    assert_copied(case)
    payload_gbps = (
        payload_bytes / (avg_us * 1000)
        if payload_bytes and avg_us
        else 0.0
    )
    print(
        "KVCACHE_OFFLOAD_COPY_RESULT "
        f"copied_blocks={copied_blocks} avg_us={avg_us:.3f} "
        f"payload_gbps={payload_gbps:.3f} timer=npu_event "
        f"warmup={args.warmup} iters={args.iters}",
        flush=True,
    )
    print("KVCACHE_OFFLOAD_COPY_UT_OK", flush=True)


def main() -> None:
    args = parse_args()
    validate_args(args)
    run(args)


if __name__ == "__main__":
    main()
