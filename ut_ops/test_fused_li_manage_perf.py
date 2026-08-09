"""Measure Fused LI Manage latency and its index-management increment.

The test times GLM's native LightningIndexer and the repository-local Fused LI Manage
kernel on identical inputs:

* LightningIndexer: score projection plus top-2048 selection.
* Fused LI Manage: the same selection plus hit/miss, eviction, request-pool mutation,
  destination-slot generation, and miss-count generation.

Their latency difference follows the same "index-management latency" proxy
used by the standalone ``ops_li_update`` benchmark.  SCATTER is deliberately
not called.
"""

from __future__ import annotations

import argparse
import gc
import statistics
from dataclasses import dataclass
from typing import Callable

import torch
import torch_npu  # type: ignore

import nanovllm.ops  # noqa: F401  # Load repository-local custom operators.
from ut_ops._op_utils import require_local_opapi


BLOCK_SIZE = 128
HEAD_DIM = 128
TOPK = 2048
MAX_EXACT_SOURCE_TOKENS = 1 << 18
MAX_CACHE_TOKENS = 16_382  # Packed slot value 0x3fff is reserved as invalid.


@dataclass(frozen=True)
class Result:
    baseline: str
    heads: int
    batch_size: int
    seq_len: int
    cache_tokens: int
    miss_min: int
    miss_max: int
    actual_miss_mean: float
    lightning_indexer_us: float
    fused_li_manage_us: float

    @property
    def management_us(self) -> float:
        return self.fused_li_manage_us - self.lightning_indexer_us


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark native LightningIndexer and Fused LI Manage without "
            "SCATTER."
        )
    )
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--heads", type=int, default=32, choices=(32, 64))
    parser.add_argument("--batch-sizes", default="24")
    parser.add_argument("--seq-lens", default="20992")
    parser.add_argument("--cache-tokens", type=int, default=6144)
    parser.add_argument(
        "--miss-ranges",
        default="0:300",
        help="Comma-separated inclusive ranges, for example 0:0,0:200,0:300.",
    )
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def parse_positive_csv(value: str, name: str) -> tuple[int, ...]:
    try:
        values = tuple(
            dict.fromkeys(int(item.strip()) for item in value.split(","))
        )
    except ValueError as exc:
        raise ValueError(f"{name} must contain comma-separated integers.") from exc
    if not values or any(item <= 0 for item in values):
        raise ValueError(f"{name} must contain positive integers.")
    return values


def parse_miss_ranges(value: str) -> tuple[tuple[int, int], ...]:
    ranges: list[tuple[int, int]] = []
    for item in value.split(","):
        fields = item.strip().split(":")
        if len(fields) != 2:
            raise ValueError(
                "--miss-ranges entries must use MIN:MAX, for example 0:300."
            )
        lower, upper = (int(field) for field in fields)
        if lower < 0 or lower > upper or upper > TOPK:
            raise ValueError(
                f"Invalid miss range {lower}:{upper}; require "
                "0 <= MIN <= MAX <= 2048."
            )
        ranges.append((lower, upper))
    if not ranges:
        raise ValueError("--miss-ranges must not be empty.")
    return tuple(dict.fromkeys(ranges))


def validate_case(
    batch_size: int,
    seq_len: int,
    cache_tokens: int,
    miss_min: int,
    miss_max: int,
) -> None:
    if batch_size <= 0:
        raise ValueError("batch size must be positive.")
    if (
        seq_len < TOPK
        or seq_len > MAX_EXACT_SOURCE_TOKENS
        or seq_len % BLOCK_SIZE
    ):
        raise ValueError(
            f"seq_len must be a multiple of {BLOCK_SIZE} in "
            f"[{TOPK}, {MAX_EXACT_SOURCE_TOKENS}]."
        )
    if (
        cache_tokens < TOPK
        or cache_tokens > MAX_CACHE_TOKENS
        or cache_tokens > seq_len
        or cache_tokens % BLOCK_SIZE
    ):
        raise ValueError(
            f"cache_tokens must be a multiple of {BLOCK_SIZE} in "
            f"[{TOPK}, min(seq_len, {MAX_CACHE_TOKENS})]."
        )
    feasible_misses = min(TOPK, seq_len - cache_tokens)
    if miss_max > feasible_misses:
        raise ValueError(
            f"miss_max={miss_max} exceeds the feasible maximum "
            f"{feasible_misses} for seq_len={seq_len}, "
            f"cache_tokens={cache_tokens}."
        )


def run_lightning_indexer(
    query: torch.Tensor,
    key: torch.Tensor,
    weights: torch.Tensor,
    query_lens: torch.Tensor,
    candidate_lens: torch.Tensor,
    block_table: torch.Tensor,
) -> tuple[torch.Tensor, str]:
    output = torch_npu.npu_lightning_indexer(
        query=query,
        key=key,
        weights=weights,
        actual_seq_lengths_query=query_lens,
        actual_seq_lengths_key=candidate_lens,
        block_table=block_table,
        layout_query="TND",
        layout_key="PA_BSND",
        sparse_count=TOPK,
        sparse_mode=3,
    )
    if isinstance(output, (tuple, list)):
        output = output[0]
    return output, "torch_npu_native"


def benchmark_npu_events(
    runner: Callable[[], object],
    *,
    warmup: int,
    iters: int,
    reset: Callable[[], None] | None = None,
) -> list[float]:
    for _ in range(warmup):
        if reset is not None:
            reset()
        runner()
    torch.npu.synchronize()

    times_ms: list[float] = []
    for _ in range(iters):
        if reset is not None:
            reset()
            # Keep request-state restoration outside the timed interval.
            torch.npu.synchronize()
        start = torch.npu.Event(enable_timing=True)
        end = torch.npu.Event(enable_timing=True)
        start.record()
        runner()
        end.record()
        end.synchronize()
        times_ms.append(float(start.elapsed_time(end)))
    return times_ms


def make_target_misses(
    batch_size: int,
    miss_min: int,
    miss_max: int,
    seed: int,
) -> torch.Tensor:
    if miss_min == miss_max:
        return torch.full((batch_size,), miss_min, dtype=torch.int32)
    generator = torch.Generator().manual_seed(seed)
    result = torch.randint(
        miss_min,
        miss_max + 1,
        (batch_size,),
        generator=generator,
        dtype=torch.int32,
    )
    return result


def build_initial_cache(
    *,
    batch_size: int,
    seq_len: int,
    cache_tokens: int,
    req_entries_cpu: torch.Tensor,
    target_misses: torch.Tensor,
    seed: int,
) -> torch.Tensor:
    pool_size = int(req_entries_cpu.max()) + 3
    cache_slots = torch.full(
        (pool_size, seq_len),
        -1,
        dtype=torch.int32,
    )
    topk = torch.arange(seq_len - TOPK, seq_len, dtype=torch.int64)
    generator = torch.Generator().manual_seed(seed)
    for row in range(batch_size):
        miss_count = int(target_misses[row])
        hit_count = TOPK - miss_count
        hits = topk[
            torch.randperm(TOPK, generator=generator)[:hit_count]
        ]
        other_count = cache_tokens - hits.numel()
        other_tokens = torch.randperm(
            seq_len - TOPK,
            generator=generator,
        )[:other_count].to(torch.int64)
        cached_tokens = torch.cat((hits, other_tokens))
        slots = torch.randperm(
            cache_tokens,
            generator=generator,
            dtype=torch.int32,
        )
        cache_slots[int(req_entries_cpu[row]), cached_tokens] = slots
    return cache_slots


def validate_outputs(
    *,
    seq_len: int,
    cache_tokens: int,
    req_entries_cpu: torch.Tensor,
    target_misses: torch.Tensor,
    initial_cache_slots: torch.Tensor,
    cache_slots: torch.Tensor,
    source_ids: torch.Tensor,
    destination_slots: torch.Tensor,
    miss_counts: torch.Tensor,
) -> None:
    counts_cpu = miss_counts.cpu()
    if not torch.equal(counts_cpu, target_misses):
        raise AssertionError(
            f"Fused LI Manage miss counts {counts_cpu.tolist()} do not match targets "
            f"{target_misses.tolist()}."
        )

    sources_cpu = source_ids.view(-1, TOPK).cpu().to(torch.int64)
    slots_cpu = destination_slots.view(-1, TOPK).cpu().to(torch.int64)
    initial_state_pool = initial_cache_slots
    state_pool = cache_slots.cpu()
    topk = torch.arange(seq_len - TOPK, seq_len, dtype=torch.int64)
    for row, miss_count_tensor in enumerate(target_misses):
        miss_count = int(miss_count_tensor)
        pool_row = int(req_entries_cpu[row])
        expected_misses = torch.sort(
            topk[initial_state_pool[pool_row, topk] < 0]
        ).values
        if expected_misses.numel() != miss_count:
            raise AssertionError(
                f"row={row} initial cache has {expected_misses.numel()} "
                f"misses, expected {miss_count}."
            )
        actual_misses = torch.sort(sources_cpu[row, :miss_count]).values
        if not torch.equal(actual_misses, expected_misses):
            raise AssertionError(f"row={row} active miss source set is incorrect.")
        if not torch.equal(torch.sort(sources_cpu[row]).values, topk):
            raise AssertionError(
                f"row={row} did not publish the full top-2048 source index set."
            )

        state = state_pool[pool_row]
        if bool((state[topk] < 0).any()):
            raise AssertionError(f"row={row} dropped a true top-2048 token.")
        valid_state = state[state >= 0].to(torch.int64)
        if (
            valid_state.numel() != cache_tokens
            or torch.unique(valid_state).numel() != cache_tokens
            or int(valid_state.min()) != 0
            or int(valid_state.max()) != cache_tokens - 1
        ):
            raise AssertionError(
                f"row={row} cache state is not a permutation of [0,C)."
            )

        full_slots = slots_cpu[row]
        if (
            bool((full_slots < 0).any())
            or bool((full_slots >= cache_tokens).any())
            or torch.unique(full_slots).numel() != TOPK
        ):
            raise AssertionError(
                f"row={row} full top-k destination slots are invalid."
            )


def run_case(
    *,
    device: torch.device,
    heads: int,
    batch_size: int,
    seq_len: int,
    cache_tokens_value: int,
    miss_min: int,
    miss_max: int,
    warmup: int,
    iters: int,
    seed: int,
) -> Result:
    validate_case(
        batch_size,
        seq_len,
        cache_tokens_value,
        miss_min,
        miss_max,
    )
    blocks_per_request = seq_len // BLOCK_SIZE
    total_blocks = batch_size * blocks_per_request

    query = torch.zeros(
        (batch_size, heads, HEAD_DIM),
        dtype=torch.bfloat16,
        device=device,
    )
    # Three base-64 digits encode every token ID below 2^18 exactly.
    query[:, 0, 0] = 1
    query[:, 0, 1] = 64
    query[:, 0, 2] = 4096
    weights = torch.zeros(
        (batch_size, heads),
        dtype=torch.bfloat16,
        device=device,
    )
    weights[:, 0] = 1

    key = torch.zeros(
        (total_blocks, BLOCK_SIZE, 1, HEAD_DIM),
        dtype=torch.bfloat16,
        device=device,
    )
    logical_ids = torch.arange(
        seq_len,
        dtype=torch.int32,
        device=device,
    ).view(1, blocks_per_request, BLOCK_SIZE)
    key_rows = key.view(
        batch_size,
        blocks_per_request,
        BLOCK_SIZE,
        1,
        HEAD_DIM,
    )
    key_rows[:, :, :, 0, 0] = (logical_ids % 64).to(torch.bfloat16)
    key_rows[:, :, :, 0, 1] = (
        (logical_ids // 64) % 64
    ).to(torch.bfloat16)
    key_rows[:, :, :, 0, 2] = (
        logical_ids // 4096
    ).to(torch.bfloat16)

    block_table = torch.arange(
        total_blocks,
        dtype=torch.int32,
        device=device,
    ).view(batch_size, blocks_per_request)
    query_lens = torch.arange(
        1,
        batch_size + 1,
        dtype=torch.int32,
        device=device,
    )
    candidate_lens = torch.full(
        (batch_size,),
        seq_len,
        dtype=torch.int32,
        device=device,
    )
    cache_tokens = torch.full(
        (batch_size,),
        cache_tokens_value,
        dtype=torch.int32,
        device=device,
    )

    pool_size = batch_size + 7
    pool_generator = torch.Generator().manual_seed(seed + 1)
    req_entries_cpu = torch.randperm(
        pool_size,
        generator=pool_generator,
    )[:batch_size].to(torch.int32)
    req_entries = req_entries_cpu.to(device)
    target_misses = make_target_misses(
        batch_size,
        miss_min,
        miss_max,
        seed + 2,
    )
    initial_cache_cpu = build_initial_cache(
        batch_size=batch_size,
        seq_len=seq_len,
        cache_tokens=cache_tokens_value,
        req_entries_cpu=req_entries_cpu,
        target_misses=target_misses,
        seed=seed + 3,
    )
    initial_cache = initial_cache_cpu.to(device)
    cache_slots = initial_cache.clone()

    source_ids = torch.full(
        (batch_size, 1, TOPK),
        -1,
        dtype=torch.int32,
        device=device,
    )
    destination_slots = torch.empty_like(source_ids)
    miss_counts = torch.empty(
        (batch_size,),
        dtype=torch.int32,
        device=device,
    )

    baseline = "torch_npu_native"

    def run_li() -> torch.Tensor:
        output, actual_baseline = run_lightning_indexer(
            query,
            key,
            weights,
            query_lens,
            candidate_lens,
            block_table,
        )
        if actual_baseline != baseline:
            raise AssertionError(
                f"LightningIndexer baseline changed from {baseline} to "
                f"{actual_baseline}."
            )
        return output

    def reset_fused_li_manage() -> None:
        cache_slots.copy_(initial_cache)

    def run_fused_li_manage() -> None:
        torch.ops.nanovllm_dsa.fused_li_manage.default(
            query,
            weights,
            key,
            block_table,
            candidate_lens,
            cache_tokens,
            req_entries,
            cache_slots,
            source_ids,
            destination_slots,
            miss_counts,
        )

    topk_indices = run_li()
    torch.npu.synchronize()
    expected_topk = torch.arange(
        seq_len - TOPK,
        seq_len,
        dtype=torch.int32,
    ).expand(batch_size, TOPK)
    actual_topk = torch.sort(
        topk_indices.view(batch_size, TOPK).cpu(),
        dim=1,
    ).values
    if not torch.equal(actual_topk, expected_topk):
        raise AssertionError(
            "LightningIndexer did not select the deterministic top-2048 set."
        )

    reset_fused_li_manage()
    run_fused_li_manage()
    torch.npu.synchronize()
    validate_outputs(
        seq_len=seq_len,
        cache_tokens=cache_tokens_value,
        req_entries_cpu=req_entries_cpu,
        target_misses=target_misses,
        initial_cache_slots=initial_cache_cpu,
        cache_slots=cache_slots,
        source_ids=source_ids,
        destination_slots=destination_slots,
        miss_counts=miss_counts,
    )

    lightning_times = benchmark_npu_events(
        run_li,
        warmup=warmup,
        iters=iters,
    )
    fused_li_manage_times = benchmark_npu_events(
        run_fused_li_manage,
        warmup=warmup,
        iters=iters,
        reset=reset_fused_li_manage,
    )
    lightning_us = statistics.mean(lightning_times) * 1000.0
    fused_li_manage_us = statistics.mean(fused_li_manage_times) * 1000.0
    result = Result(
        baseline=baseline,
        heads=heads,
        batch_size=batch_size,
        seq_len=seq_len,
        cache_tokens=cache_tokens_value,
        miss_min=miss_min,
        miss_max=miss_max,
        actual_miss_mean=float(target_misses.float().mean()),
        lightning_indexer_us=lightning_us,
        fused_li_manage_us=fused_li_manage_us,
    )
    print(
        "FUSED_LI_MANAGE_PERF_RESULT "
        f"baseline={baseline} heads={heads} batch={batch_size} "
        f"seq_len={seq_len} "
        f"cache_tokens={cache_tokens_value} "
        f"miss_range={miss_min}:{miss_max} "
        f"actual_miss_mean={result.actual_miss_mean:.2f} "
        f"lightning_indexer_us={lightning_us:.3f} "
        f"fused_li_manage_us={fused_li_manage_us:.3f} "
        f"index_management_us={result.management_us:+.3f} "
        f"timer=npu_event warmup={warmup} iters={iters}",
        flush=True,
    )
    return result


def print_table(results: list[Result]) -> None:
    print("FUSED_LI_MANAGE_PERF_TABLE")
    print(
        "| baseline | heads | bsz | seqlen | C | miss range | actual miss mean | "
        "LightningIndexer (us) | Fused LI Manage (us) | index management (us) |"
    )
    print(
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"
    )
    for result in results:
        print(
            f"| {result.baseline} | {result.heads} | {result.batch_size} | "
            f"{result.seq_len} | "
            f"{result.cache_tokens} | {result.miss_min}:{result.miss_max} | "
            f"{result.actual_miss_mean:.2f} | "
            f"{result.lightning_indexer_us:.3f} | {result.fused_li_manage_us:.3f} | "
            f"{result.management_us:+.3f} |"
        )


def main() -> None:
    args = parse_args()
    if args.warmup < 0 or args.iters <= 0:
        raise ValueError("--warmup must be >= 0 and --iters must be > 0.")
    batch_sizes = parse_positive_csv(args.batch_sizes, "--batch-sizes")
    seq_lens = parse_positive_csv(args.seq_lens, "--seq-lens")
    miss_ranges = parse_miss_ranges(args.miss_ranges)

    device = torch.device(args.device)
    if device.type != "npu":
        raise ValueError("--device must select one NPU, for example npu:0.")
    torch.npu.set_device(device)
    torch.npu.config.allow_internal_format = False
    opapi_path = require_local_opapi()
    print(
        f"FUSED_LI_MANAGE_PERF_OPAPI path={opapi_path} local=1",
        flush=True,
    )
    print(
        f"FUSED_LI_MANAGE_PERF_CONFIG device={device} heads={args.heads} "
        f"batch_sizes={list(batch_sizes)} seq_lens={list(seq_lens)} "
        f"cache_tokens={args.cache_tokens} miss_ranges={list(miss_ranges)} "
        f"seed={args.seed} warmup={args.warmup} iters={args.iters}",
        flush=True,
    )

    results: list[Result] = []
    case_id = 0
    for batch_size in batch_sizes:
        for seq_len in seq_lens:
            for miss_min, miss_max in miss_ranges:
                results.append(
                    run_case(
                        device=device,
                        heads=args.heads,
                        batch_size=batch_size,
                        seq_len=seq_len,
                        cache_tokens_value=args.cache_tokens,
                        miss_min=miss_min,
                        miss_max=miss_max,
                        warmup=args.warmup,
                        iters=args.iters,
                        seed=args.seed + case_id * 101,
                    )
                )
                case_id += 1
                gc.collect()
                torch.npu.empty_cache()
    print_table(results)
    print("FUSED_LI_MANAGE_PERF_UT_OK")


if __name__ == "__main__":
    main()
