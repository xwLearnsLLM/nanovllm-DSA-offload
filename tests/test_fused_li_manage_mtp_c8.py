#!/usr/bin/env python3
"""Behavior and latency test for one-kernel A5 C8 MTP LI management."""

from __future__ import annotations

import argparse
import random
import statistics
from dataclasses import dataclass

import torch
import nanovllm_dsa_a5

from _c8_lidu_case import (
    normalized_hadamard_128,
    official_c8_lightning_indexer,
    quantize_fp8,
)
from _lidu_utils import (
    MAX_CACHE_TOKENS,
    MAX_SOURCE_CAPACITY,
    TOPK,
    assert_pool_row,
    assert_request_pool_entries,
)
from _utils import csv_ints, require_a5


BLOCK_SIZE = 128
UNION_CAPACITY = 8192


@dataclass
class MtpC8Case:
    query: torch.Tensor
    key: torch.Tensor
    weights: torch.Tensor
    query_scale: torch.Tensor
    key_scale: torch.Tensor
    actual_q: torch.Tensor
    actual_q_cpu: list[int]
    req_entries: torch.Tensor
    req_entries_cpu: torch.Tensor
    initial_pool: torch.Tensor
    cache_tokens: torch.Tensor
    candidate_lens: torch.Tensor
    block_table: torch.Tensor
    native_topk: torch.Tensor
    query_counts: list[int]
    target_misses: list[int]
    source_capacity: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--batch-size", type=int, default=6)
    parser.add_argument("--heads", type=csv_ints, default=csv_ints("32,64"))
    parser.add_argument("--source-len", type=int, default=20096)
    parser.add_argument(
        "--queries-per-request",
        type=int,
        default=0,
        help="2/3/4, or 0 to cycle through MTP1/2/3",
    )
    parser.add_argument("--miss-min", type=int, default=0)
    parser.add_argument("--miss-max", type=int, default=300)
    parser.add_argument("--pool-extra", type=int, default=7)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--allow-non-a5", action="store_true")
    return parser.parse_args()


def check_args(args: argparse.Namespace) -> None:
    if args.batch_size <= 0:
        raise ValueError("batch-size must be positive")
    if any(heads not in (32, 64) for heads in args.heads):
        raise ValueError("C8 index heads must be 32 or 64")
    if args.source_len <= 0 or args.source_len % BLOCK_SIZE:
        raise ValueError("source-len must be a positive multiple of 128")
    if args.source_len > MAX_SOURCE_CAPACITY:
        raise ValueError("source-len exceeds the 18-bit token-ID capacity")
    if args.queries_per_request not in (0, 2, 3, 4):
        raise ValueError("queries-per-request must be 0, 2, 3, or 4")
    if not 0 <= args.miss_min <= args.miss_max <= UNION_CAPACITY:
        raise ValueError("require 0 <= miss-min <= miss-max <= 8192")
    if args.source_len < UNION_CAPACITY + args.miss_max:
        raise ValueError(
            "source-len must be >= 8192 + miss-max for controlled misses"
        )
    if args.pool_extra < 0:
        raise ValueError("pool-extra must be non-negative")
    if args.warmup < 0 or args.iters <= 0:
        raise ValueError("warmup must be non-negative and iters positive")


def query_ranges(actual_q: list[int]) -> list[tuple[int, int]]:
    result = []
    begin = 0
    for end in actual_q:
        result.append((begin, end))
        begin = end
    return result


def ordered_union(rows: torch.Tensor) -> list[int]:
    seen: set[int] = set()
    result: list[int] = []
    for token in rows.reshape(-1).tolist():
        token = int(token)
        if token >= 0 and token not in seen:
            seen.add(token)
            result.append(token)
    return result


def make_case(
    *,
    device: torch.device,
    batch: int,
    heads: int,
    source_len: int,
    query_counts: list[int],
    budgets: list[int],
    miss_range: tuple[int, int],
    pool_extra: int,
    seed: int,
    candidate_lens_cpu: list[int] | None = None,
) -> MtpC8Case:
    if len(query_counts) != batch or len(budgets) != batch:
        raise ValueError("query-count and budget lists must match batch")
    if candidate_lens_cpu is None:
        candidate_lens_cpu = [source_len] * batch
    if len(candidate_lens_cpu) != batch:
        raise ValueError("candidate-length list must match batch")
    if any(
        length < TOPK or length > source_len or length % BLOCK_SIZE
        for length in candidate_lens_cpu
    ):
        raise ValueError(
            "candidate lengths must be block-aligned values in "
            "[2048,source_len]"
        )
    if any(
        budget != 0
        and (
            budget < TOPK
            or budget > MAX_CACHE_TOKENS
            or budget % BLOCK_SIZE
        )
        for budget in budgets
    ):
        raise ValueError(
            "cache budgets must be 0 or block-aligned values in [2048,16256]"
        )
    torch.manual_seed(seed)
    torch.npu.manual_seed_all(seed)
    generator = torch.Generator().manual_seed(seed + 1)
    rng = random.Random(seed + 2)
    packed_queries = sum(query_counts)
    actual_q_cpu: list[int] = []
    total = 0
    for count in query_counts:
        if count not in (2, 3, 4):
            raise ValueError("every C8 MTP request must have 2, 3, or 4 queries")
        total += count
        actual_q_cpu.append(total)

    blocks = source_len // BLOCK_SIZE
    block_table_cpu = torch.stack(
        [
            torch.randperm(blocks, generator=generator).to(torch.int32)
            for _ in range(batch)
        ]
    )
    query_fp = torch.empty(
        (packed_queries, heads, 128), dtype=torch.bfloat16, device=device
    ).uniform_(-1, 1)
    key_fp = torch.empty(
        (blocks, BLOCK_SIZE, 1, 128),
        dtype=torch.bfloat16,
        device=device,
    ).uniform_(-1, 1)
    hadamard = normalized_hadamard_128(
        dtype=torch.bfloat16, device=device
    )
    query, query_scale = quantize_fp8(torch.matmul(query_fp, hadamard))
    key, key_scale = quantize_fp8(torch.matmul(key_fp, hadamard))
    weights = torch.empty(
        (packed_queries, heads), dtype=torch.bfloat16, device=device
    ).uniform_(0.01, 1.0).contiguous()
    actual_q = torch.tensor(
        actual_q_cpu, dtype=torch.int32, device=device
    )
    candidate_lens = torch.tensor(
        candidate_lens_cpu, dtype=torch.int32, device=device
    )
    block_table = block_table_cpu.to(device)
    native_topk = official_c8_lightning_indexer(
        query,
        key,
        weights,
        query_scale,
        key_scale,
        actual_q,
        candidate_lens,
        block_table,
    ).reshape(packed_queries, TOPK)
    torch.npu.synchronize()
    native_cpu = native_topk.cpu()

    pool_size = batch + pool_extra
    req_entries_cpu = torch.randperm(
        pool_size, generator=generator
    )[:batch].to(torch.int32)
    pool = torch.full(
        (pool_size, source_len), -1, dtype=torch.int32
    )
    target_misses: list[int] = []
    ranges = query_ranges(actual_q_cpu)
    for batch_row, ((begin, end), budget, candidate_len) in enumerate(
        zip(ranges, budgets, candidate_lens_cpu)
    ):
        pool_row = int(req_entries_cpu[batch_row])
        if budget == 0:
            target_misses.append(0)
            continue
        union = ordered_union(native_cpu[begin:end])
        if len(union) > budget:
            raise ValueError(
                f"request {batch_row}: union={len(union)} exceeds C={budget}"
            )
        if budget > candidate_len:
            raise ValueError(
                f"request {batch_row}: C={budget} exceeds "
                f"candidate_len={candidate_len}"
            )
        feasible_max = min(
            miss_range[1], len(union), candidate_len - budget
        )
        if miss_range[0] > feasible_max:
            raise ValueError(
                f"request {batch_row}: miss range {miss_range} is infeasible"
            )
        miss = rng.randint(miss_range[0], feasible_max)
        if batch_row == 0:
            miss = miss_range[0]
        elif batch_row == 1:
            miss = feasible_max
        target_misses.append(miss)

        union_tensor = torch.tensor(union, dtype=torch.int64)
        hit_count = len(union) - miss
        hit_ids = union_tensor[
            torch.randperm(len(union), generator=generator)[:hit_count]
        ]
        outside_mask = torch.ones(candidate_len, dtype=torch.bool)
        outside_mask[union_tensor] = False
        outside = torch.arange(candidate_len, dtype=torch.int64)[outside_mask]
        victim_count = budget - hit_count
        victim_ids = outside[
            torch.randperm(outside.numel(), generator=generator)[:victim_count]
        ]
        resident = torch.cat((hit_ids, victim_ids))
        slots = torch.randperm(
            budget, generator=generator, dtype=torch.int64
        ).to(torch.int32)
        pool[pool_row, resident] = slots

    return MtpC8Case(
        query=query,
        key=key,
        weights=weights,
        query_scale=query_scale,
        key_scale=key_scale,
        actual_q=actual_q,
        actual_q_cpu=actual_q_cpu,
        req_entries=req_entries_cpu.to(device),
        req_entries_cpu=req_entries_cpu,
        initial_pool=pool.to(device),
        cache_tokens=torch.tensor(
            budgets, dtype=torch.int32, device=device
        ),
        candidate_lens=candidate_lens,
        block_table=block_table,
        native_topk=native_topk,
        query_counts=query_counts,
        target_misses=target_misses,
        source_capacity=source_len,
    )


def launch(case: MtpC8Case, pool: torch.Tensor):
    return torch.ops.nanovllm_dsa.fused_li_manage_mtp_c8.default(
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
    case: MtpC8Case,
    pool: torch.Tensor,
    topk_slots: torch.Tensor,
    miss_sources: torch.Tensor,
    miss_slots: torch.Tensor,
    miss_counts: torch.Tensor,
):
    return torch.ops.nanovllm_dsa.fused_li_manage_mtp_c8_out.default(
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
        topk_slots,
        miss_sources,
        miss_slots,
        miss_counts,
    )


def benchmark(case: MtpC8Case, warmup: int, iters: int) -> float:
    batch = len(case.query_counts)
    packed_t = case.query.size(0)
    device = case.query.device
    pool = case.initial_pool.clone()
    topk_slots = torch.empty(
        (packed_t, 1, TOPK), dtype=torch.int32, device=device
    )
    miss_sources = torch.empty(
        (batch, UNION_CAPACITY), dtype=torch.int32, device=device
    )
    miss_slots = torch.empty_like(miss_sources)
    miss_counts = torch.empty((batch,), dtype=torch.int32, device=device)

    for _ in range(warmup):
        pool.copy_(case.initial_pool)
        launch_out(
            case, pool, topk_slots, miss_sources, miss_slots, miss_counts
        )
    torch.npu.synchronize()

    starts = [torch.npu.Event(enable_timing=True) for _ in range(iters)]
    ends = [torch.npu.Event(enable_timing=True) for _ in range(iters)]
    last_outputs = None
    for start, end in zip(starts, ends):
        # Restore the mutable request pool outside the timed interval so each
        # sample retains the requested miss distribution.
        pool.copy_(case.initial_pool)
        start.record()
        last_outputs = launch_out(
            case, pool, topk_slots, miss_sources, miss_slots, miss_counts
        )
        end.record()
    ends[-1].synchronize()
    if last_outputs is None:
        raise AssertionError("timed C8 MTP call returned no outputs")
    return statistics.mean(
        start.elapsed_time(end) for start, end in zip(starts, ends)
    ) * 1000.0


def validate_case(
    case: MtpC8Case,
    old_pool: torch.Tensor,
    new_pool: torch.Tensor,
    outputs: tuple[torch.Tensor, ...],
) -> None:
    topk_slots, miss_sources, miss_slots, miss_counts, alias = outputs
    if alias.data_ptr() != new_pool.data_ptr():
        raise AssertionError("C8 MTP LIDU cache output does not alias its input")
    expected_shapes = (
        (case.query.size(0), 1, TOPK),
        (len(case.query_counts), UNION_CAPACITY),
        (len(case.query_counts), UNION_CAPACITY),
        (len(case.query_counts),),
    )
    for tensor, shape in zip(outputs[:4], expected_shapes):
        if tensor.dtype != torch.int32 or tuple(tensor.shape) != shape:
            raise AssertionError(
                f"unexpected C8 MTP output {tensor.dtype}/{tuple(tensor.shape)}"
            )
    topk_slots_cpu = topk_slots.reshape(case.query.size(0), TOPK).cpu()
    miss_sources_cpu = miss_sources.cpu()
    miss_slots_cpu = miss_slots.cpu()
    miss_counts_cpu = miss_counts.cpu()
    native_cpu = case.native_topk.cpu().reshape(case.query.size(0), TOPK)
    old_cpu = old_pool.cpu()
    new_cpu = new_pool.cpu()
    ranges = query_ranges(case.actual_q_cpu)
    assert_request_pool_entries(
        case.req_entries_cpu, len(case.query_counts), new_cpu.size(0)
    )
    active_rows = set(case.req_entries_cpu.tolist())
    for pool_row in range(new_cpu.size(0)):
        if pool_row not in active_rows and not torch.equal(
            old_cpu[pool_row], new_cpu[pool_row]
        ):
            raise AssertionError(f"inactive request-pool row {pool_row} changed")

    for batch_row, (begin, end) in enumerate(ranges):
        pool_row = int(case.req_entries_cpu[batch_row])
        budget = int(case.cache_tokens[batch_row].cpu())
        candidate_len = int(case.candidate_lens[batch_row].cpu())
        assert_pool_row(old_cpu[pool_row], candidate_len, budget)
        assert_pool_row(new_cpu[pool_row], candidate_len, budget)
        count = int(miss_counts_cpu[batch_row])
        if budget == 0:
            if count != 0 or bool((topk_slots_cpu[begin:end] != -1).any()):
                raise AssertionError("C=0 MTP request is not a strict no-op")
            if not torch.equal(old_cpu[pool_row], new_cpu[pool_row]):
                raise AssertionError("C=0 MTP request changed cache state")
            continue

        for local_query, row in enumerate(range(begin, end)):
            selected = native_cpu[row].long()
            if (
                int(selected.min()) < 0
                or int(selected.max()) >= candidate_len
                or torch.unique(selected).numel() != TOPK
            ):
                raise AssertionError(
                    f"request {batch_row} query {local_query}: "
                    "top-K is outside the prefill-full-block source"
                )
            expected_slots = new_cpu[pool_row].gather(0, selected)
            if not torch.equal(topk_slots_cpu[row], expected_slots):
                raise AssertionError(
                    f"request {batch_row} query {local_query}: published slots mismatch"
                )
            if bool((expected_slots < 0).any()):
                raise AssertionError("updated cache does not contain a query top-K")
            if (
                bool((topk_slots_cpu[row] >= budget).any())
                or torch.unique(topk_slots_cpu[row]).numel() != TOPK
            ):
                raise AssertionError(
                    f"request {batch_row} query {local_query}: "
                    "published top-K slots are not unique values in [0,C)"
                )

        union = set(ordered_union(native_cpu[begin:end]))
        expected_misses = {
            token for token in union if int(old_cpu[pool_row, token]) < 0
        }
        if count != case.target_misses[batch_row] or count != len(expected_misses):
            raise AssertionError(
                f"request {batch_row}: miss_count={count}, "
                f"expected={case.target_misses[batch_row]}, "
                f"recomputed={len(expected_misses)}"
            )
        emitted_tokens = miss_sources_cpu[batch_row, :count].long()
        emitted_slots = miss_slots_cpu[batch_row, :count]
        if (
            torch.unique(emitted_tokens).numel() != count
            or set(emitted_tokens.tolist()) != expected_misses
        ):
            raise AssertionError("union miss output is not unique or complete")
        if (
            bool((emitted_slots < 0).any())
            or bool((emitted_slots >= budget).any())
            or torch.unique(emitted_slots).numel() != count
        ):
            raise AssertionError("union miss destination slots are invalid")

        old_valid = (old_cpu[pool_row] >= 0).nonzero().flatten()
        old_owner = torch.empty(budget, dtype=torch.long)
        old_owner[old_cpu[pool_row, old_valid].long()] = old_valid
        expected_new_row = old_cpu[pool_row].clone()
        for token, slot in zip(emitted_tokens.tolist(), emitted_slots.tolist()):
            if int(new_cpu[pool_row, token]) != slot:
                raise AssertionError("miss token was not assigned its output slot")
            victim = int(old_owner[slot])
            if victim in union or int(new_cpu[pool_row, victim]) != -1:
                raise AssertionError("MTP manager evicted a protected union token")
            expected_new_row[victim] = -1
            expected_new_row[token] = slot
        for token in union:
            old_slot = int(old_cpu[pool_row, token])
            if old_slot >= 0 and int(new_cpu[pool_row, token]) != old_slot:
                raise AssertionError("an existing union hit changed its slot")
        if not torch.equal(new_cpu[pool_row], expected_new_row):
            raise AssertionError(
                f"request {batch_row}: cache update changed entries other than "
                "the published miss/victim pairs"
            )


def check_case(case: MtpC8Case) -> None:
    pool = case.initial_pool.clone()
    old_pool = pool.clone()
    outputs = launch(case, pool)
    torch.npu.synchronize()
    validate_case(case, old_pool, pool, outputs)

    # Run every request alone from the same initial pool. This catches any
    # accidental dependence on neighboring batch rows or packed-query offsets,
    # while exercising the same public one-kernel LI + manager implementation.
    for batch_row, (begin, end) in enumerate(query_ranges(case.actual_q_cpu)):
        single_pool = case.initial_pool.clone()
        single_actual_q = torch.tensor(
            [end - begin], dtype=torch.int32, device=case.query.device
        )
        single = torch.ops.nanovllm_dsa.fused_li_manage_mtp_c8.default(
            case.query[begin:end].contiguous(),
            case.key,
            case.weights[begin:end].contiguous(),
            case.query_scale[begin:end].contiguous(),
            case.key_scale,
            single_actual_q,
            case.req_entries[batch_row : batch_row + 1],
            single_pool,
            case.cache_tokens[batch_row : batch_row + 1],
            case.candidate_lens[batch_row : batch_row + 1],
            case.block_table[batch_row : batch_row + 1].contiguous(),
        )
        torch.npu.synchronize()
        count = int(outputs[3].cpu()[batch_row])
        pool_row = int(case.req_entries_cpu[batch_row])
        if (
            not torch.equal(single[0].cpu(), outputs[0][begin:end].cpu())
            or int(single[3].cpu()[0]) != count
            or not torch.equal(
                single_pool.cpu()[pool_row], pool.cpu()[pool_row]
            )
        ):
            raise AssertionError(
                f"request {batch_row}: isolated update differs from batched update"
            )
        if not torch.equal(
            single[1][0, :count].cpu(),
            outputs[1][batch_row, :count].cpu(),
        ) or not torch.equal(
            single[2][0, :count].cpu(),
            outputs[2][batch_row, :count].cpu(),
        ):
            raise AssertionError(
                f"request {batch_row}: isolated miss prefix differs"
            )

    first_slots = outputs[0].cpu()
    stable_pool = pool.clone()
    repeated = launch(case, pool)
    torch.npu.synchronize()
    if bool((repeated[3].cpu() != 0).any()):
        raise AssertionError("repeated C8 MTP update did not become zero-miss")
    if not torch.equal(pool.cpu(), stable_pool.cpu()):
        raise AssertionError("repeated zero-miss update changed cache state")
    if not torch.equal(repeated[0].cpu(), first_slots):
        raise AssertionError("repeated update changed per-query slot mapping")

    out_pool = case.initial_pool.clone()
    out_topk = torch.full_like(outputs[0], -777777)
    out_sources = torch.full_like(outputs[1], -777777)
    out_slots = torch.full_like(outputs[2], -777777)
    out_counts = torch.full_like(outputs[3], -777777)
    out_outputs = launch_out(
        case,
        out_pool,
        out_topk,
        out_sources,
        out_slots,
        out_counts,
    )
    torch.npu.synchronize()
    expected_ptrs = (
        out_topk.data_ptr(),
        out_sources.data_ptr(),
        out_slots.data_ptr(),
        out_counts.data_ptr(),
        out_pool.data_ptr(),
    )
    if tuple(tensor.data_ptr() for tensor in out_outputs) != expected_ptrs:
        raise AssertionError("C8 MTP out path changed caller-owned addresses")
    if (
        not torch.equal(out_outputs[0].cpu(), outputs[0].cpu())
        or not torch.equal(out_outputs[3].cpu(), outputs[3].cpu())
        or not torch.equal(out_pool.cpu(), stable_pool.cpu())
    ):
        raise AssertionError("allocating and out C8 MTP paths disagree")
    for batch_row, count in enumerate(outputs[3].cpu().tolist()):
        if not torch.equal(
            out_outputs[1][batch_row, :count].cpu(),
            outputs[1][batch_row, :count].cpu(),
        ) or not torch.equal(
            out_outputs[2][batch_row, :count].cpu(),
            outputs[2][batch_row, :count].cpu(),
        ):
            raise AssertionError("out path changed a valid miss prefix")

    print(
        "A5_FUSED_LI_MANAGE_MTP_C8_CHECK "
        f"heads={case.query.size(1)} batch={len(case.query_counts)} "
        f"packed_t={case.query.size(0)} query_counts={case.query_counts} "
        f"source_capacity={case.source_capacity} "
        f"budgets={case.cache_tokens.cpu().tolist()} "
        f"misses={outputs[3].cpu().tolist()} "
        "official_c8_li_topk=1 source_range=1 union_dedup=1 "
        "unordered_unique_pool_entries=1 hit_slots_preserved=1 "
        "per_query_slots_unique=1 exact_cache_delta=1 "
        "single_request_update=1 isolated_request_match=1 repeat_zero_miss=1 "
        "one_device_kernel=1 out_alias=1 ok=1",
        flush=True,
    )


def check_heterogeneous_candidates_and_budgets(device: torch.device) -> None:
    case = make_case(
        device=device,
        batch=6,
        heads=32,
        source_len=20096,
        query_counts=[2, 3, 4, 2, 3, 4],
        budgets=[0, 8192, 12288, 6144, 12288, MAX_CACHE_TOKENS],
        candidate_lens_cpu=[4096, 12288, 20096, 8192, 16384, 19968],
        miss_range=(0, 300),
        pool_extra=7,
        seed=1707,
    )
    check_case(case)
    print(
        "A5_FUSED_LI_MANAGE_MTP_C8_HETEROGENEOUS_CHECK "
        f"candidates={case.candidate_lens.cpu().tolist()} "
        f"budgets={case.cache_tokens.cpu().tolist()} "
        "candidate_tail_unchanged=1 max_cache_budget=16256 ok=1",
        flush=True,
    )


def check_overlapping_union_zero_miss(device: torch.device) -> None:
    case = make_case(
        device=device,
        batch=1,
        heads=32,
        source_len=4096,
        query_counts=[4],
        budgets=[4096],
        miss_range=(0, 0),
        pool_extra=2,
        seed=2707,
    )
    union_size = len(ordered_union(case.native_topk.cpu()))
    if union_size >= 4 * TOPK:
        raise AssertionError("overlap case did not exercise union deduplication")
    check_case(case)
    print(
        "A5_FUSED_LI_MANAGE_MTP_C8_OVERLAP_ZERO_MISS_CHECK "
        f"queries=4 raw_topk_entries={4 * TOPK} union={union_size} "
        "first_update_zero_miss=1 union_dedup=1 ok=1",
        flush=True,
    )


def check_worst_union(device: torch.device) -> None:
    heads = 32
    source_len = UNION_CAPACITY * 2
    blocks = source_len // BLOCK_SIZE
    query_fp = torch.zeros(
        (4, heads, 128), dtype=torch.bfloat16, device=device
    )
    key_fp = torch.zeros(
        (blocks, BLOCK_SIZE, 1, 128),
        dtype=torch.bfloat16,
        device=device,
    )
    key_rows = key_fp.reshape(source_len, 1, 128)
    key_rows[:, 0, 4] = 0.25
    for query_row in range(4):
        query_fp[query_row, :, query_row] = 1
        begin = query_row * TOPK
        key_rows[begin : begin + TOPK, 0, query_row] = 1
    query, query_scale = quantize_fp8(query_fp)
    key, key_scale = quantize_fp8(key_fp)
    weights = torch.ones(
        (4, heads), dtype=torch.bfloat16, device=device
    )
    actual_q = torch.tensor([4], dtype=torch.int32, device=device)
    req = torch.tensor([1], dtype=torch.int32, device=device)
    pool_cpu = torch.full(
        (3, source_len), -1, dtype=torch.int32
    )
    pool_cpu[1, UNION_CAPACITY:] = torch.arange(
        UNION_CAPACITY, dtype=torch.int32
    )
    pool = pool_cpu.to(device)
    budget = torch.tensor(
        [UNION_CAPACITY], dtype=torch.int32, device=device
    )
    candidate = torch.tensor(
        [source_len], dtype=torch.int32, device=device
    )
    block_table = torch.arange(
        blocks, dtype=torch.int32, device=device
    ).reshape(1, blocks)
    native_topk = official_c8_lightning_indexer(
        query,
        key,
        weights,
        query_scale,
        key_scale,
        actual_q,
        candidate,
        block_table,
    ).reshape(4, TOPK)
    expected = torch.arange(UNION_CAPACITY, dtype=torch.int32).reshape(
        4, TOPK
    )
    if any(
        set(native_topk[row].cpu().tolist())
        != set(expected[row].tolist())
        for row in range(4)
    ):
        raise AssertionError("synthetic C8 MTP queries did not form an 8192 union")
    outputs = torch.ops.nanovllm_dsa.fused_li_manage_mtp_c8.default(
        query,
        key,
        weights,
        query_scale,
        key_scale,
        actual_q,
        req,
        pool,
        budget,
        candidate,
        block_table,
    )
    torch.npu.synchronize()
    if int(outputs[3].cpu()[0]) != UNION_CAPACITY:
        raise AssertionError("worst-case union did not report 8192 misses")
    sources = outputs[1].cpu()[0]
    slots = outputs[2].cpu()[0]
    if (
        torch.unique(sources).numel() != UNION_CAPACITY
        or set(sources.tolist()) != set(range(UNION_CAPACITY))
        or not torch.equal(
            torch.sort(slots).values,
            torch.arange(UNION_CAPACITY, dtype=torch.int32),
        )
    ):
        raise AssertionError("worst-case union miss buffers are incorrect")
    updated = pool.cpu()[1]
    if bool((updated[:UNION_CAPACITY] < 0).any()) or bool(
        (updated[UNION_CAPACITY:] >= 0).any()
    ):
        raise AssertionError("worst-case union cache update is incomplete")
    expected_slots = updated.gather(
        0, native_topk.cpu().reshape(-1).long()
    ).reshape(4, 1, TOPK)
    if not torch.equal(outputs[0].cpu(), expected_slots):
        raise AssertionError("worst-case per-query slots are incorrect")
    print(
        "A5_FUSED_LI_MANAGE_MTP_C8_WORST_UNION_CHECK "
        "queries=4 union=8192 misses=8192 ok=1",
        flush=True,
    )


def check_meta() -> None:
    query = torch.empty(
        (9, 32, 128), dtype=torch.float8_e4m3fn, device="meta"
    )
    key = torch.empty(
        (96, BLOCK_SIZE, 1, 128),
        dtype=torch.float8_e4m3fn,
        device="meta",
    )
    weights = torch.empty((9, 32), dtype=torch.bfloat16, device="meta")
    query_scale = torch.empty((9, 32), dtype=torch.float32, device="meta")
    key_scale = torch.empty(
        (96, BLOCK_SIZE, 1), dtype=torch.float32, device="meta"
    )
    ints = torch.empty((3,), dtype=torch.int32, device="meta")
    pool = torch.empty((7, 12288), dtype=torch.int32, device="meta")
    table = torch.empty((3, 96), dtype=torch.int32, device="meta")
    outputs = torch.ops.nanovllm_dsa.fused_li_manage_mtp_c8.default(
        query,
        key,
        weights,
        query_scale,
        key_scale,
        ints,
        ints,
        pool,
        ints,
        ints,
        table,
    )
    expected = [
        (9, 1, TOPK),
        (3, UNION_CAPACITY),
        (3, UNION_CAPACITY),
        (3,),
        (7, 12288),
    ]
    if [tuple(tensor.shape) for tensor in outputs] != expected:
        raise AssertionError("C8 MTP LIDU Meta returned wrong shapes")
    if outputs[4] is not pool and outputs[4]._cdata != pool._cdata:
        raise AssertionError("C8 MTP LIDU Meta cache output is not an alias")
    print("A5_FUSED_LI_MANAGE_MTP_C8_META_CHECK ok=1", flush=True)


def main() -> None:
    args = parse_args()
    check_args(args)
    device = torch.device(args.device)
    if device.type != "npu":
        raise ValueError("--device must select an NPU")
    torch.npu.set_device(device)
    torch.npu.config.allow_internal_format = False
    require_a5(device, args.allow_non_a5)
    check_meta()
    print(
        "A5_FUSED_LI_MANAGE_MTP_C8_CONFIG "
        f"opapi={nanovllm_dsa_a5.local_opapi_path()}",
        flush=True,
    )

    for heads in args.heads:
        if args.queries_per_request:
            query_counts = [args.queries_per_request] * args.batch_size
        else:
            query_counts = [2 + row % 3 for row in range(args.batch_size)]
        budgets = [query_count * TOPK for query_count in query_counts]
        if args.batch_size > 1:
            budgets[0] = 0
        case = make_case(
            device=device,
            batch=args.batch_size,
            heads=heads,
            source_len=args.source_len,
            query_counts=query_counts,
            budgets=budgets,
            miss_range=(args.miss_min, args.miss_max),
            pool_extra=args.pool_extra,
            seed=args.seed + heads,
        )
        check_case(case)
        avg_us = benchmark(case, args.warmup, args.iters)
        print(
            "A5_FUSED_LI_MANAGE_MTP_C8_RESULT "
            f"heads={heads} batch={args.batch_size} packed_t={case.query.size(0)} "
            f"source_len={args.source_len} "
            f"miss_mean={statistics.mean(case.target_misses):.3f} "
            f"avg_us={avg_us:.3f} warmup={args.warmup} iters={args.iters}",
            flush=True,
        )

    check_heterogeneous_candidates_and_budgets(device)
    check_overlapping_union_zero_miss(device)
    check_worst_union(device)
    print("A5_FUSED_LI_MANAGE_MTP_C8_UT_OK", flush=True)


if __name__ == "__main__":
    main()
