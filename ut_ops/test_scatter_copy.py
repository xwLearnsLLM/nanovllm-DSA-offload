#!/usr/bin/env python3
"""Repository-bundled SCATTER correctness and NPU-event latency test."""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import torch
import torch_npu  # type: ignore

import nanovllm.ops  # noqa: F401  # Load repository-local custom operators.
from ut_ops._op_utils import require_local_opapi


BLOCK_SIZE = 128
KPE_DIM = 64
CKV_DIM = 512
MAX_COPY_CAP = 65536
CKV_POISON = 37.0
KPE_POISON = -29.0


@dataclass
class Case:
    device: torch.device
    device_name: str
    dram_kpe_cpu: torch.Tensor
    dram_ckv_cpu: torch.Tensor
    dram_kpe: torch.Tensor
    dram_ckv: torch.Tensor
    hbm_kpe: torch.Tensor
    hbm_ckv: torch.Tensor
    dram_table_cpu: torch.Tensor
    hbm_table_cpu: torch.Tensor
    dram_table: torch.Tensor
    hbm_table: torch.Tensor
    src_ids_cpu: torch.Tensor
    dst_slots_cpu: torch.Tensor
    copy_counts_cpu: torch.Tensor
    src_ids: torch.Tensor
    dst_slots: torch.Tensor
    copy_counts: torch.Tensor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--source-len", type=int, default=20000)
    parser.add_argument("--hbm-slots", type=int, default=6144)
    parser.add_argument("--copy-min", type=int, default=0)
    parser.add_argument("--copy-max", type=int, default=300)
    parser.add_argument("--copy-cap", type=int, default=2048)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive.")
    if args.source_len <= 0 or args.hbm_slots <= 0:
        raise ValueError("--source-len and --hbm-slots must be positive.")
    if not 0 <= args.copy_min <= args.copy_max:
        raise ValueError(
            "--copy-min/--copy-max must satisfy 0 <= min <= max."
        )
    if not 0 < args.copy_cap <= MAX_COPY_CAP:
        raise ValueError(
            f"--copy-cap must be in [1,{MAX_COPY_CAP}]."
        )
    if args.copy_max > args.copy_cap:
        raise ValueError("--copy-max must not exceed --copy-cap.")
    if args.copy_max > args.source_len:
        raise ValueError("--copy-max must not exceed --source-len.")
    if args.copy_max >= args.hbm_slots:
        raise ValueError(
            "--hbm-slots must exceed --copy-max so an untouched "
            "guard token exists."
        )
    if args.warmup < 0 or args.iters <= 0:
        raise ValueError("--warmup must be non-negative and --iters positive.")


def random_block_table(
    batch_size: int,
    blocks_per_row: int,
    generator: torch.Generator,
) -> tuple[torch.Tensor, int]:
    total_blocks = batch_size * blocks_per_row
    table = torch.randperm(
        total_blocks,
        generator=generator,
        dtype=torch.int64,
    ).reshape(batch_size, blocks_per_row)
    return table.to(torch.int32).contiguous(), total_blocks


def swapped_from_cpu(
    cpu: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    if not hasattr(torch_npu, "empty_with_swapped_memory"):
        raise RuntimeError(
            "torch_npu.empty_with_swapped_memory is unavailable. "
            "This test refuses to replace DRAM with an HBM tensor."
        )
    tensor = torch_npu.empty_with_swapped_memory(
        cpu.shape,
        dtype=cpu.dtype,
        device=device,
    )
    tensor.fill_(0)
    staging = cpu.to(device)
    tensor.add_(staging)
    torch.npu.synchronize()
    del staging
    torch.npu.empty_cache()
    return tensor


def make_case(args: argparse.Namespace) -> Case:
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

    source_blocks_per_row = (
        args.source_len + BLOCK_SIZE - 1
    ) // BLOCK_SIZE
    hbm_blocks_per_row = (
        args.hbm_slots + BLOCK_SIZE - 1
    ) // BLOCK_SIZE
    dram_table_cpu, dram_blocks = random_block_table(
        args.batch_size,
        source_blocks_per_row,
        generator,
    )
    hbm_table_cpu, hbm_blocks = random_block_table(
        args.batch_size,
        hbm_blocks_per_row,
        generator,
    )

    dram_kpe_cpu = torch.randn(
        dram_blocks,
        BLOCK_SIZE,
        KPE_DIM,
        generator=generator,
        dtype=torch.float32,
    ).to(torch.bfloat16)
    dram_ckv_cpu = torch.randn(
        dram_blocks,
        BLOCK_SIZE,
        CKV_DIM,
        generator=generator,
        dtype=torch.float32,
    ).to(torch.bfloat16)

    src_ids_cpu = torch.full(
        (args.batch_size, args.copy_cap),
        -1,
        dtype=torch.int32,
    )
    dst_slots_cpu = torch.full_like(src_ids_cpu, -1)
    sampled_counts = torch.randint(
        args.copy_min,
        args.copy_max + 1,
        (args.batch_size,),
        generator=generator,
        dtype=torch.int64,
    )
    counts = sampled_counts.tolist()
    for row, count in enumerate(counts):
        src_ids_cpu[row, :count] = torch.randperm(
            args.source_len,
            generator=generator,
        )[:count].to(torch.int32)
        dst_slots_cpu[row, :count] = torch.randperm(
            args.hbm_slots,
            generator=generator,
        )[:count].to(torch.int32)
    copy_counts_cpu = torch.tensor(counts, dtype=torch.int32)

    return Case(
        device=device,
        device_name=device_name,
        dram_kpe_cpu=dram_kpe_cpu,
        dram_ckv_cpu=dram_ckv_cpu,
        dram_kpe=swapped_from_cpu(dram_kpe_cpu, device),
        dram_ckv=swapped_from_cpu(dram_ckv_cpu, device),
        hbm_kpe=torch.empty(
            hbm_blocks,
            BLOCK_SIZE,
            KPE_DIM,
            dtype=torch.bfloat16,
            device=device,
        ),
        hbm_ckv=torch.empty(
            hbm_blocks,
            BLOCK_SIZE,
            CKV_DIM,
            dtype=torch.bfloat16,
            device=device,
        ),
        dram_table_cpu=dram_table_cpu,
        hbm_table_cpu=hbm_table_cpu,
        dram_table=dram_table_cpu.to(device),
        hbm_table=hbm_table_cpu.to(device),
        src_ids_cpu=src_ids_cpu,
        dst_slots_cpu=dst_slots_cpu,
        copy_counts_cpu=copy_counts_cpu,
        src_ids=src_ids_cpu.to(device),
        dst_slots=dst_slots_cpu.to(device),
        copy_counts=copy_counts_cpu.to(device),
    )


def active_physical_rows(
    case: Case,
) -> tuple[torch.Tensor, torch.Tensor]:
    source_rows: list[torch.Tensor] = []
    destination_rows: list[torch.Tensor] = []
    for row, count in enumerate(case.copy_counts_cpu.tolist()):
        source = case.src_ids_cpu[row, :count].to(torch.int64)
        destination = case.dst_slots_cpu[row, :count].to(torch.int64)
        source_rows.append(
            case.dram_table_cpu[row].to(torch.int64)[
                source // BLOCK_SIZE
            ]
            * BLOCK_SIZE
            + source % BLOCK_SIZE
        )
        destination_rows.append(
            case.hbm_table_cpu[row].to(torch.int64)[
                destination // BLOCK_SIZE
            ]
            * BLOCK_SIZE
            + destination % BLOCK_SIZE
        )
    return torch.cat(source_rows), torch.cat(destination_rows)


def poison_hbm(case: Case) -> None:
    case.hbm_ckv.fill_(CKV_POISON)
    case.hbm_kpe.fill_(KPE_POISON)


def call_scatter(
    case: Case,
    copy_counts: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    return torch.ops.nanovllm_dsa.scatter_copy.default(
        case.hbm_kpe,
        case.hbm_ckv,
        case.dram_kpe,
        case.dram_ckv,
        case.hbm_table,
        case.dram_table,
        case.src_ids,
        case.dst_slots,
        case.copy_counts if copy_counts is None else copy_counts,
    )


def assert_all_hbm_poisoned(case: Case) -> None:
    ckv_ok = bool(torch.all(case.hbm_ckv == CKV_POISON).item())
    kpe_ok = bool(torch.all(case.hbm_kpe == KPE_POISON).item())
    if not ckv_ok or not kpe_ok:
        raise AssertionError("copy_count=0 modified the HBM cache.")


def assert_copied(
    case: Case,
    source_rows: torch.Tensor,
    destination_rows: torch.Tensor,
) -> None:
    expected_ckv = case.dram_ckv_cpu.view(-1, CKV_DIM)[source_rows]
    expected_kpe = case.dram_kpe_cpu.view(-1, KPE_DIM)[source_rows]
    destination_rows_npu = destination_rows.to(case.device)
    actual_ckv = (
        case.hbm_ckv.view(-1, CKV_DIM)[destination_rows_npu].cpu()
    )
    actual_kpe = (
        case.hbm_kpe.view(-1, KPE_DIM)[destination_rows_npu].cpu()
    )
    if not torch.equal(actual_ckv, expected_ckv):
        max_abs = (
            actual_ckv.float() - expected_ckv.float()
        ).abs().max()
        raise AssertionError(
            f"DRAM->HBM CKV mismatch, max_abs={float(max_abs):.9f}"
        )
    if not torch.equal(actual_kpe, expected_kpe):
        max_abs = (
            actual_kpe.float() - expected_kpe.float()
        ).abs().max()
        raise AssertionError(
            f"DRAM->HBM KPE mismatch, max_abs={float(max_abs):.9f}"
        )

    active = set(destination_rows.tolist())
    total_rows = case.hbm_ckv.shape[0] * BLOCK_SIZE
    guard = next(index for index in range(total_rows) if index not in active)
    guard_ckv = case.hbm_ckv.view(-1, CKV_DIM)[guard].cpu()
    guard_kpe = case.hbm_kpe.view(-1, KPE_DIM)[guard].cpu()
    if not torch.equal(
        guard_ckv,
        torch.full_like(guard_ckv, CKV_POISON),
    ) or not torch.equal(
        guard_kpe,
        torch.full_like(guard_kpe, KPE_POISON),
    ):
        raise AssertionError("An inactive HBM guard row was modified.")


def run(args: argparse.Namespace) -> None:
    opapi_path = require_local_opapi()
    case = make_case(args)
    source_rows, destination_rows = active_physical_rows(case)
    copied_tokens = int(case.copy_counts_cpu.sum())
    payload_bytes = copied_tokens * (CKV_DIM + KPE_DIM) * 2

    print(
        "SCATTER_COPY_CONFIG "
        f"device={case.device} device_name={case.device_name!r} "
        f"dtype=bf16 batch={args.batch_size} "
        f"source_len={args.source_len} hbm_slots={args.hbm_slots} "
        f"copy_cap={args.copy_cap} "
        f"copy_range=[{args.copy_min},{args.copy_max}] "
        f"copy_counts={case.copy_counts_cpu.tolist()} "
        f"opapi={opapi_path}",
        flush=True,
    )

    poison_hbm(case)
    torch.npu.synchronize()
    assert_all_hbm_poisoned(case)
    out_kpe, out_ckv = call_scatter(case)
    torch.npu.synchronize()
    if (
        out_kpe.data_ptr() != case.hbm_kpe.data_ptr()
        or out_ckv.data_ptr() != case.hbm_ckv.data_ptr()
    ):
        raise AssertionError("HBM outputs did not alias the input caches.")
    if copied_tokens:
        assert_copied(case, source_rows, destination_rows)
    else:
        assert_all_hbm_poisoned(case)
    print(
        "SCATTER_COPY_DRAM_TO_HBM_CHECK "
        "allocator=empty_with_swapped_memory "
        f"copied_tokens={copied_tokens} "
        f"payload_bytes={payload_bytes} guard_unchanged=1 ok=1",
        flush=True,
    )

    for _ in range(args.warmup):
        call_scatter(case)
    torch.npu.synchronize()

    start = torch.npu.Event(enable_timing=True)
    end = torch.npu.Event(enable_timing=True)
    start.record()
    for _ in range(args.iters):
        call_scatter(case)
    end.record()
    end.synchronize()
    avg_us = start.elapsed_time(end) * 1000 / args.iters

    if copied_tokens:
        assert_copied(case, source_rows, destination_rows)
    else:
        assert_all_hbm_poisoned(case)
    payload_gbps = (
        payload_bytes / (avg_us * 1000)
        if payload_bytes and avg_us
        else 0.0
    )
    print(
        "SCATTER_COPY_RESULT "
        f"copy_min={args.copy_min} copy_max={args.copy_max} "
        f"copy_cap={args.copy_cap} copied_tokens={copied_tokens} "
        f"avg_us={avg_us:.3f} payload_gbps={payload_gbps:.3f} "
        f"timer=npu_event warmup={args.warmup} iters={args.iters}",
        flush=True,
    )
    print("SCATTER_COPY_UT_OK", flush=True)


def main() -> None:
    args = parse_args()
    validate_args(args)
    run(args)


if __name__ == "__main__":
    main()
