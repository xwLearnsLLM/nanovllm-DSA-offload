#!/usr/bin/env python3
"""Validate the A5 C8 LightningIndexer plus request-pool update path."""

from __future__ import annotations

import argparse
import statistics

import torch

import nanovllm_dsa_a5
import torch_npu  # type: ignore  # noqa: E402,F401

from _c8_lidu_case import C8Case, make_case, official_c8_lightning_indexer
from _lidu_utils import (
    MAX_CACHE_TOKENS,
    MAX_SOURCE_CAPACITY,
    TOPK,
    assert_18bit_boundary_selected,
    assert_pool_row,
    assert_request_pool_entries,
    assert_update_row,
    feasible_miss,
    miss_ranges,
)
from _utils import csv_ints, require_a5


BLOCK_SIZE = 128


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--batch-sizes", type=csv_ints, default=csv_ints("24"))
    parser.add_argument("--source-lens", type=csv_ints, default=csv_ints("20096"))
    parser.add_argument("--heads", type=csv_ints, default=csv_ints("32,64"))
    parser.add_argument("--cache-tokens", type=csv_ints, default=csv_ints("6144"))
    parser.add_argument("--miss-ranges", type=miss_ranges, default=miss_ranges("0:300"))
    parser.add_argument("--pool-extra", type=int, default=7)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--allow-non-a5", action="store_true")
    return parser.parse_args()


def check_args(args: argparse.Namespace) -> None:
    if any(batch <= 0 for batch in args.batch_sizes):
        raise ValueError("all batch sizes must be positive")
    if any(length <= 0 or length % BLOCK_SIZE for length in args.source_lens):
        raise ValueError("source lengths must be positive multiples of 128")
    if any(length > MAX_SOURCE_CAPACITY for length in args.source_lens):
        raise ValueError("source length exceeds the 18-bit token-ID capacity")
    if any(heads not in (32, 64) for heads in args.heads):
        raise ValueError("C8 LightningIndexer heads must be 32 or 64")
    valid_budget = lambda value: value == 0 or (
        TOPK <= value <= MAX_CACHE_TOKENS and value % BLOCK_SIZE == 0
    )
    if any(not valid_budget(value) for value in args.cache_tokens):
        raise ValueError("cache tokens must be 0 or block aligned in [2048,16256]")
    if args.pool_extra < 0 or args.warmup < 0 or args.iters < 0:
        raise ValueError("pool-extra/warmup/iters must be non-negative")


def launch(case: C8Case, pool: torch.Tensor):
    return torch.ops.nanovllm_dsa.fused_li_manage_c8.default(
        case.query,
        case.key,
        case.weights,
        case.query_scale,
        case.key_scale,
        case.actual_q,
        case.req_entries,
        pool,
        case.cache_tokens,
        case.candidate_lens,
        case.block_table,
    )


def launch_out(
    case: C8Case,
    pool: torch.Tensor,
    source_ids: torch.Tensor,
    destination_slots: torch.Tensor,
    miss_counts: torch.Tensor,
):
    return torch.ops.nanovllm_dsa.fused_li_manage_c8_out.default(
        case.query,
        case.key,
        case.weights,
        case.query_scale,
        case.key_scale,
        case.actual_q,
        case.req_entries,
        pool,
        case.cache_tokens,
        case.candidate_lens,
        case.block_table,
        source_ids,
        destination_slots,
        miss_counts,
    )


def check_case(case: C8Case) -> None:
    pool = case.initial_pool.clone()
    old_pool = pool.cpu()
    source_ids, destination_slots, miss_counts, alias = launch(case, pool)
    torch.npu.synchronize()
    if alias.data_ptr() != pool.data_ptr():
        raise AssertionError("C8 LIDU cache alias does not alias request state")
    sources = source_ids.reshape(case.query.size(0), TOPK).cpu()
    slots = destination_slots.reshape(case.query.size(0), TOPK).cpu()
    counts = miss_counts.cpu()
    updated = pool.cpu()
    native = case.native_topk.cpu().reshape(case.query.size(0), TOPK)
    assert_request_pool_entries(
        case.req_entries_cpu, case.query.size(0), updated.size(0)
    )
    active_rows = set(case.req_entries_cpu.tolist())
    for pool_row in range(updated.size(0)):
        if pool_row not in active_rows and not torch.equal(
            updated[pool_row], old_pool[pool_row]
        ):
            raise AssertionError(f"inactive request-pool row {pool_row} changed")
    for batch_row in range(case.query.size(0)):
        pool_row = int(case.req_entries_cpu[batch_row])
        budget = int(case.cache_tokens[batch_row].cpu())
        candidate_len = int(case.candidate_lens[batch_row].cpu())
        assert_pool_row(old_pool[pool_row], candidate_len, budget)
        assert_pool_row(updated[pool_row], candidate_len, budget)
        assert_update_row(
            label=f"C8 row {batch_row} first update",
            sources=sources[batch_row],
            slots=slots[batch_row],
            reference=native[batch_row],
            old_row=old_pool[pool_row],
            new_row=updated[pool_row],
            candidate_len=candidate_len,
            budget=budget,
            expected_miss=case.target_misses[batch_row],
            actual_miss=int(counts[batch_row]),
        )

    # Isolate the repository-local update stage from the official C8 LI launch.
    low_pool = case.initial_pool.clone()
    low_outputs = torch.ops.nanovllm_dsa._fused_li_manage_c8_cache_update.default(
        case.native_topk.reshape(case.query.size(0), 1, TOPK),
        case.req_entries,
        low_pool,
        case.cache_tokens,
        case.candidate_lens,
    )
    torch.npu.synchronize()
    if not all(
        torch.equal(left.cpu(), right.cpu())
        for left, right in zip(low_outputs[:3], (source_ids, destination_slots, miss_counts))
    ) or not torch.equal(low_pool.cpu(), pool.cpu()):
        raise AssertionError("C8 high-level LIDU and isolated update stage disagree")

    second_sources, second_slots, second_counts, _ = launch(case, pool)
    torch.npu.synchronize()
    second_sources = second_sources.reshape(case.query.size(0), TOPK).cpu()
    second_slots = second_slots.reshape(case.query.size(0), TOPK).cpu()
    second_counts = second_counts.cpu()
    second_updated = pool.cpu()
    if not torch.equal(second_updated, updated):
        raise AssertionError("repeated zero-miss C8 update changed cache state")
    for batch_row in range(case.query.size(0)):
        pool_row = int(case.req_entries_cpu[batch_row])
        budget = int(case.cache_tokens[batch_row].cpu())
        candidate_len = int(case.candidate_lens[batch_row].cpu())
        assert_pool_row(second_updated[pool_row], candidate_len, budget)
        assert_update_row(
            label=f"C8 row {batch_row} repeated update",
            sources=second_sources[batch_row],
            slots=second_slots[batch_row],
            reference=native[batch_row],
            old_row=updated[pool_row],
            new_row=second_updated[pool_row],
            candidate_len=candidate_len,
            budget=budget,
            expected_miss=0,
            actual_miss=int(second_counts[batch_row]),
        )

    out_pool = case.initial_pool.clone()
    out_sources = torch.full_like(source_ids, -777777)
    out_slots = torch.full_like(destination_slots, -777777)
    out_counts = torch.full_like(miss_counts, -777777)
    outputs = launch_out(
        case, out_pool, out_sources, out_slots, out_counts
    )
    torch.npu.synchronize()
    expected_ptrs = (
        out_sources.data_ptr(),
        out_slots.data_ptr(),
        out_counts.data_ptr(),
        out_pool.data_ptr(),
    )
    if tuple(tensor.data_ptr() for tensor in outputs) != expected_ptrs:
        raise AssertionError("C8 LIDU out path changed caller-owned addresses")
    if not all(
        torch.equal(left.cpu(), right.cpu())
        for left, right in zip(outputs[:3], (source_ids, destination_slots, miss_counts))
    ):
        raise AssertionError("allocating and caller-owned C8 LIDU paths disagree")
    if not torch.equal(out_pool.cpu(), updated):
        raise AssertionError("caller-owned C8 LIDU path produced different cache state")

    print(
        "A5_FUSED_LI_MANAGE_C8_CHECK "
        f"heads={case.query.size(1)} batch={case.query.size(0)} "
        f"source_capacity={case.source_capacity} "
        f"candidates={case.candidate_lens.cpu().tolist()} "
        f"budgets={case.cache_tokens.cpu().tolist()} misses={counts.tolist()} "
        "official_c8_li_topk=1 unordered_unique_pool_entries=1 "
        "topk_unique_range=1 hit_slots_preserved=1 repeat_mapping_preserved=1 "
        "repeat_zero_miss=1 isolated_update_match=1 out_alias=1 ok=1",
        flush=True,
    )


def event_benchmark(case: C8Case, warmup: int, iters: int) -> tuple[float, float]:
    pool = case.initial_pool.clone()
    for _ in range(warmup):
        pool.copy_(case.initial_pool)
        launch(case, pool)
    torch.npu.synchronize()

    starts = [torch.npu.Event(enable_timing=True) for _ in range(iters)]
    ends = [torch.npu.Event(enable_timing=True) for _ in range(iters)]
    retained = []
    for start, end in zip(starts, ends):
        pool.copy_(case.initial_pool)
        torch.npu.synchronize()
        start.record()
        retained.append(launch(case, pool))
        end.record()
    ends[-1].synchronize()
    lidu_us = statistics.mean(
        start.elapsed_time(end) for start, end in zip(starts, ends)
    ) * 1000

    native_starts = [torch.npu.Event(enable_timing=True) for _ in range(iters)]
    native_ends = [torch.npu.Event(enable_timing=True) for _ in range(iters)]
    for start, end in zip(native_starts, native_ends):
        start.record()
        official_c8_lightning_indexer(
            case.query,
            case.key,
            case.weights,
            case.query_scale,
            case.key_scale,
            case.actual_q,
            case.candidate_lens,
            case.block_table,
        )
        end.record()
    native_ends[-1].synchronize()
    native_us = statistics.mean(
        start.elapsed_time(end) for start, end in zip(native_starts, native_ends)
    ) * 1000
    if not retained:
        raise AssertionError("timed C8 outputs were not retained")
    return native_us, lidu_us


def check_meta() -> None:
    query = torch.empty((3, 32, 128), dtype=torch.float8_e4m3fn, device="meta")
    key = torch.empty(
        (96, BLOCK_SIZE, 1, 128), dtype=torch.float8_e4m3fn, device="meta"
    )
    weights = torch.empty((3, 32), dtype=torch.bfloat16, device="meta")
    query_scale = torch.empty((3, 32), dtype=torch.float32, device="meta")
    key_scale = torch.empty(
        (96, BLOCK_SIZE, 1), dtype=torch.float32, device="meta"
    )
    ints = torch.empty((3,), dtype=torch.int32, device="meta")
    req = torch.empty((3,), dtype=torch.int32, device="meta")
    pool = torch.empty((7, 12288), dtype=torch.int32, device="meta")
    table = torch.empty((3, 96), dtype=torch.int32, device="meta")
    outputs = torch.ops.nanovllm_dsa.fused_li_manage_c8.default(
        query,
        key,
        weights,
        query_scale,
        key_scale,
        ints,
        req,
        pool,
        ints,
        ints,
        table,
    )
    expected = [(3, 1, TOPK), (3, 1, TOPK), (3,), (7, 12288)]
    if [tuple(tensor.shape) for tensor in outputs] != expected:
        raise AssertionError("C8 LIDU Meta implementation returned wrong shapes")
    if outputs[3] is not pool and outputs[3]._cdata != pool._cdata:
        raise AssertionError("C8 LIDU Meta cache output does not alias its input")
    print("A5_FUSED_LI_MANAGE_C8_META_CHECK ok=1", flush=True)


def main() -> None:
    args = parse_args()
    check_args(args)
    device = torch.device(args.device)
    if device.type != "npu":
        raise ValueError("--device must select an NPU")
    torch.npu.set_device(device)
    torch.npu.config.allow_internal_format = False
    device_name = require_a5(device, args.allow_non_a5)
    check_meta()
    mixed = make_case(
        device,
        6,
        32768,
        32,
        [0, 3072, 6144, 8192, 12288, MAX_CACHE_TOKENS],
        (0, 300),
        args.pool_extra,
        args.seed + 100,
        [2048, 8192, 12288, 20096, 24576, 32768],
    )
    check_case(mixed)
    miss_edges = make_case(
        device,
        2,
        4096,
        32,
        [2048, 2048],
        (0, TOPK),
        args.pool_extra,
        args.seed + 200,
    )
    check_case(miss_edges)
    boundary = make_case(
        device,
        1,
        MAX_SOURCE_CAPACITY,
        32,
        [6144],
        (0, 300),
        args.pool_extra,
        args.seed + 300,
    )
    assert_18bit_boundary_selected(
        boundary, "A5_FUSED_LI_MANAGE_C8_18BIT_BOUNDARY_CHECK"
    )
    check_case(boundary)

    case_index = 0
    for heads in args.heads:
        for batch in args.batch_sizes:
            for source_len in args.source_lens:
                for budget in args.cache_tokens:
                    for miss_range in args.miss_ranges:
                        if not any(
                            feasible_miss(source_len, budget, miss)
                            for miss in range(miss_range[0], miss_range[1] + 1)
                        ):
                            print(
                                "A5_FUSED_LI_MANAGE_C8_SKIP "
                                f"heads={heads} batch={batch} "
                                f"source_len={source_len} C={budget} "
                                f"miss_range={miss_range} reason=infeasible",
                                flush=True,
                            )
                            continue
                        case = make_case(
                            device,
                            batch,
                            source_len,
                            heads,
                            [budget] * batch,
                            miss_range,
                            args.pool_extra,
                            args.seed + case_index,
                        )
                        case_index += 1
                        check_case(case)
                        if args.iters > 0:
                            native_us, lidu_us = event_benchmark(
                                case, args.warmup, args.iters
                            )
                            print(
                                "A5_FUSED_LI_MANAGE_C8_RESULT "
                                f"device_name={device_name!r} heads={heads} "
                                f"batch={batch} source_len={source_len} C={budget} "
                                f"miss_range={miss_range[0]}:{miss_range[1]} "
                                f"actual_miss_mean={statistics.mean(case.target_misses):.3f} "
                                f"official_c8_li_us={native_us:.3f} "
                                f"c8_lidu_us={lidu_us:.3f} "
                                f"update_overhead_us={lidu_us - native_us:+.3f} "
                                f"warmup={args.warmup} iters={args.iters}",
                                flush=True,
                            )
    print("A5_FUSED_LI_MANAGE_C8_UT_OK", flush=True)


if __name__ == "__main__":
    main()
