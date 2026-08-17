#!/usr/bin/env python3
"""Benchmark staged C8 SFA against single QSFA for random MTP batches."""

from __future__ import annotations

import argparse
import math
import statistics
from collections.abc import Callable

import torch
import torch_npu  # type: ignore  # noqa: F401

import nanovllm_dsa_a5
from _c8_staged_attention_reference import full_attention_slots, make_case
from _utils import csv_ints, require_a5


BLOCK_SIZE = 128
TOPK = 2048
ATOL = 0.08
RTOL = 0.03


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="npu:0")
    parser.add_argument(
        "--batch-sizes",
        type=csv_ints,
        default=csv_ints("1,2,4,8,16,32,64"),
    )
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--query-dtype", choices=("bf16", "fp16"), default="bf16")
    parser.add_argument("--min-queries", type=int, default=0)
    parser.add_argument("--max-queries", type=int, default=4)
    parser.add_argument("--cache-tokens", type=int, default=6144)
    parser.add_argument("--tail-tokens", type=int, default=64)
    parser.add_argument("--miss-min", type=int, default=0)
    parser.add_argument("--miss-max", type=int, default=300)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument(
        "--execution-mode",
        choices=("eager", "npugraph_ex"),
        default="npugraph_ex",
        help=(
            "run each fixed-shape attention path eagerly or through an "
            "npugraph_ex capture/replay graph"
        ),
    )
    parser.add_argument("--allow-non-a5", action="store_true")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if any(batch_size < 1 or batch_size > 64 for batch_size in args.batch_sizes):
        raise ValueError("--batch-sizes values must satisfy 1 <= BS <= 64")
    if not 1 <= args.heads <= 64:
        raise ValueError("--heads must satisfy 1 <= heads <= 64")
    if not 0 <= args.min_queries <= args.max_queries <= 4:
        raise ValueError("query-count range must be within [0,4]")
    if args.max_queries == 0:
        raise ValueError("at least one request must be able to produce a query")
    if args.cache_tokens != 0 and (
        args.cache_tokens < TOPK or args.cache_tokens % BLOCK_SIZE
    ):
        raise ValueError("--cache-tokens must be 0 or block-aligned >= 2048")
    if args.tail_tokens < args.max_queries - 1:
        raise ValueError("--tail-tokens must cover the maximum MTP causal tail")
    if not 0 <= args.miss_min <= args.miss_max <= TOPK:
        raise ValueError("miss-count range must be within [0,2048]")
    if args.warmup < 0 or args.iters <= 0:
        raise ValueError("--warmup must be non-negative and --iters positive")


def random_counts(
    batch_size: int,
    minimum: int,
    maximum: int,
    seed: int,
) -> tuple[int, ...]:
    generator = torch.Generator().manual_seed(seed)
    counts = torch.randint(
        minimum,
        maximum + 1,
        (batch_size,),
        generator=generator,
    ).tolist()
    if sum(counts) == 0:
        counts[seed % batch_size] = max(1, minimum)
    return tuple(int(count) for count in counts)


def random_misses(
    query_counts: tuple[int, ...],
    cache_tokens: int,
    minimum: int,
    maximum: int,
    seed: int,
) -> tuple[int, ...]:
    packed_queries = sum(query_counts)
    if cache_tokens == 0:
        return (0,) * packed_queries
    generator = torch.Generator().manual_seed(seed)
    return tuple(
        int(value)
        for value in torch.randint(
            minimum,
            maximum + 1,
            (packed_queries,),
            generator=generator,
        ).tolist()
    )


def event_benchmark(
    launch: Callable[
        [], torch.Tensor | tuple[torch.Tensor, ...] | None
    ],
    warmup: int,
    iters: int,
) -> float:
    for _ in range(warmup):
        launch()
    torch.npu.synchronize()
    starts = [torch.npu.Event(enable_timing=True) for _ in range(iters)]
    ends = [torch.npu.Event(enable_timing=True) for _ in range(iters)]
    for start, end in zip(starts, ends):
        start.record()
        launch()
        end.record()
    ends[-1].synchronize()
    return statistics.mean(
        start.elapsed_time(end) for start, end in zip(starts, ends)
    ) * 1000.0


def stage1_step(
    query: torch.Tensor,
    packed_kv: torch.Tensor,
    actual_q: torch.Tensor,
    resident_lengths: torch.Tensor,
    cache_tokens: torch.Tensor,
    block_table: torch.Tensor,
    topk_slots: torch.Tensor,
    miss_counts: torch.Tensor,
    partial: torch.Tensor,
    maximum: torch.Tensor,
    denominator: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    torch.ops.nanovllm_dsa.sparse_tail_attention_c8_mtp_stage1.default(
        query,
        packed_kv,
        actual_q,
        resident_lengths,
        cache_tokens,
        block_table,
        topk_slots,
        miss_counts,
        scale,
        partial,
        maximum,
        denominator,
    )
    return partial


def stage2_step(
    query: torch.Tensor,
    packed_kv: torch.Tensor,
    actual_q: torch.Tensor,
    resident_lengths: torch.Tensor,
    block_table: torch.Tensor,
    topk_slots: torch.Tensor,
    miss_counts: torch.Tensor,
    partial: torch.Tensor,
    maximum: torch.Tensor,
    denominator: torch.Tensor,
    output: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    torch.ops.nanovllm_dsa.sparse_tail_attention_c8_mtp_stage2.default(
        query,
        packed_kv,
        actual_q,
        resident_lengths,
        block_table,
        topk_slots,
        miss_counts,
        scale,
        partial,
        maximum,
        denominator,
        output,
    )
    return output


def staged_step(
    query: torch.Tensor,
    packed_kv: torch.Tensor,
    actual_q: torch.Tensor,
    resident_lengths: torch.Tensor,
    cache_tokens: torch.Tensor,
    block_table: torch.Tensor,
    topk_slots: torch.Tensor,
    miss_counts: torch.Tensor,
    partial: torch.Tensor,
    maximum: torch.Tensor,
    denominator: torch.Tensor,
    output: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    stage1_step(
        query,
        packed_kv,
        actual_q,
        resident_lengths,
        cache_tokens,
        block_table,
        topk_slots,
        miss_counts,
        partial,
        maximum,
        denominator,
        scale,
    )
    return stage2_step(
        query,
        packed_kv,
        actual_q,
        resident_lengths,
        block_table,
        topk_slots,
        miss_counts,
        partial,
        maximum,
        denominator,
        output,
        scale,
    )


def qsfa_step(
    query: torch.Tensor,
    packed_kv: torch.Tensor,
    full_slots: torch.Tensor,
    block_table: torch.Tensor,
    actual_q: torch.Tensor,
    resident_lengths: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    return nanovllm_dsa_a5.sparse_tail_attention_c8(
        query,
        packed_kv,
        full_slots,
        block_table,
        actual_q,
        resident_lengths,
        scale,
    )


def execution_runners(
    execution_mode: str,
) -> tuple[
    Callable[..., torch.Tensor],
    Callable[..., torch.Tensor],
    Callable[..., torch.Tensor],
    Callable[..., torch.Tensor],
]:
    if execution_mode == "eager":
        return staged_step, qsfa_step, stage1_step, stage2_step
    graph_options = {
        # Benchmark inputs and caller-owned outputs keep stable addresses for
        # the lifetime of each fixed-shape graph. Avoid graph-side clones so
        # the measurement contains one replay of exactly two staged kernels
        # versus one replay of the baseline QSFA kernel.
        "clone_input": False,
        "clone_output": False,
    }
    return (
        torch.compile(
            staged_step,
            backend="npugraph_ex",
            fullgraph=True,
            dynamic=False,
            options=graph_options,
        ),
        torch.compile(
            qsfa_step,
            backend="npugraph_ex",
            fullgraph=True,
            dynamic=False,
            options=graph_options,
        ),
        torch.compile(
            stage1_step,
            backend="npugraph_ex",
            fullgraph=True,
            dynamic=False,
            options=graph_options,
        ),
        torch.compile(
            stage2_step,
            backend="npugraph_ex",
            fullgraph=True,
            dynamic=False,
            options=graph_options,
        ),
    )


def benchmark_batch(
    args: argparse.Namespace,
    device: torch.device,
    batch_size: int,
    seed: int,
) -> dict[str, float | int | str | tuple[int, ...]]:
    query_counts = random_counts(
        batch_size,
        args.min_queries,
        args.max_queries,
        seed,
    )
    miss_counts = random_misses(
        query_counts,
        args.cache_tokens,
        args.miss_min,
        args.miss_max,
        seed + 1,
    )
    query_dtype = torch.bfloat16 if args.query_dtype == "bf16" else torch.float16
    case = make_case(
        device=device,
        query_counts=query_counts,
        heads=args.heads,
        cache_tokens=(args.cache_tokens,) * batch_size,
        final_tail_tokens=(args.tail_tokens,) * batch_size,
        miss_counts=miss_counts,
        query_dtype=query_dtype,
        seed=seed + 2,
    )
    actual_q = case.actual_q.to(torch.int32)
    full_slots = full_attention_slots(case).to(device)
    partial = torch.empty(
        (*case.query.shape[:-1], 512),
        dtype=torch.float32,
        device=device,
    )
    maximum = torch.empty(
        (1, case.query.shape[0], args.heads),
        dtype=torch.float32,
        device=device,
    )
    denominator = torch.empty_like(maximum)
    staged_output = torch.empty(
        (*case.query.shape[:-1], 512),
        dtype=case.query.dtype,
        device=device,
    )

    staged_runner, qsfa_runner, stage1_runner, stage2_runner = (
        execution_runners(args.execution_mode)
    )

    def launch_staged() -> torch.Tensor:
        return staged_runner(
            case.query,
            case.packed,
            actual_q,
            case.resident_lengths,
            case.cache_tokens,
            case.block_table,
            case.topk_slots,
            case.miss_counts,
            partial,
            maximum,
            denominator,
            staged_output,
            case.scale,
        )

    def launch_qsfa() -> torch.Tensor:
        return qsfa_runner(
            case.query,
            case.packed,
            full_slots,
            case.block_table,
            actual_q,
            case.resident_lengths,
            case.scale,
        )

    def launch_stage1() -> torch.Tensor:
        return stage1_runner(
            case.query,
            case.packed,
            actual_q,
            case.resident_lengths,
            case.cache_tokens,
            case.block_table,
            case.topk_slots,
            case.miss_counts,
            partial,
            maximum,
            denominator,
            case.scale,
        )

    def launch_stage2() -> torch.Tensor:
        return stage2_runner(
            case.query,
            case.packed,
            actual_q,
            case.resident_lengths,
            case.block_table,
            case.topk_slots,
            case.miss_counts,
            partial,
            maximum,
            denominator,
            staged_output,
            case.scale,
        )

    staged = launch_staged()
    qsfa = launch_qsfa()
    torch.npu.synchronize()
    if staged.data_ptr() != staged_output.data_ptr():
        raise AssertionError("Stage2 did not return caller-owned output")
    staged_cpu = staged.cpu().float()
    qsfa_cpu = qsfa.cpu().float()
    if not torch.isfinite(staged_cpu).all() or not torch.isfinite(qsfa_cpu).all():
        raise AssertionError("performance case produced NaN or Inf")
    torch.testing.assert_close(staged_cpu, qsfa_cpu, atol=ATOL, rtol=RTOL)
    absolute = (staged_cpu - qsfa_cpu).abs()
    tolerance = ATOL + RTOL * qsfa_cpu.abs()

    stage1_us = event_benchmark(launch_stage1, args.warmup, args.iters)
    stage2_us = event_benchmark(launch_stage2, args.warmup, args.iters)
    staged_us = event_benchmark(launch_staged, args.warmup, args.iters)
    qsfa_us = event_benchmark(launch_qsfa, args.warmup, args.iters)
    component_sum_us = stage1_us + stage2_us
    speedup = qsfa_us / staged_us
    change_pct = (qsfa_us - staged_us) / qsfa_us * 100.0
    histogram = tuple(query_counts.count(count) for count in range(5))
    return {
        "execution_mode": args.execution_mode,
        "batch_size": batch_size,
        "packed_queries": sum(query_counts),
        "zero_query_requests": histogram[0],
        "query_histogram": histogram,
        "staged_us": staged_us,
        "stage1_us": stage1_us,
        "stage2_us": stage2_us,
        "component_sum_us": component_sum_us,
        "component_gap_us": staged_us - component_sum_us,
        "qsfa_us": qsfa_us,
        "speedup": speedup,
        "change_pct": change_pct,
        "max_abs": float(absolute.max()),
        "max_tolerance_ratio": float((absolute / tolerance).max()),
    }


def main() -> None:
    args = parse_args()
    validate_args(args)
    device = torch.device(args.device)
    if device.type != "npu":
        raise ValueError("--device must select an NPU")
    torch.npu.set_device(device)
    torch.npu.config.allow_internal_format = False
    device_name = require_a5(device, args.allow_non_a5)
    print(
        "A5_C8_STAGED_PERF_CONFIG "
        f"device={device} device_name={device_name!r} "
        f"batch_sizes={args.batch_sizes} heads={args.heads} "
        f"query_dtype={args.query_dtype} "
        f"queries={args.min_queries}:{args.max_queries} "
        f"cache_tokens={args.cache_tokens} tail_tokens={args.tail_tokens} "
        f"misses={args.miss_min}:{args.miss_max} seed={args.seed} "
        f"warmup={args.warmup} iters={args.iters} "
        f"execution_mode={args.execution_mode}",
        flush=True,
    )

    results = []
    for index, batch_size in enumerate(args.batch_sizes):
        result = benchmark_batch(
            args,
            device,
            batch_size,
            args.seed + index * 1009,
        )
        results.append(result)
        print(
            "A5_C8_STAGED_PERF_RESULT "
            f"execution_mode={result['execution_mode']} "
            f"bs={result['batch_size']} T={result['packed_queries']} "
            f"zero_query_requests={result['zero_query_requests']} "
            f"query_histogram_0_to_4={result['query_histogram']} "
            f"staged_us={result['staged_us']:.3f} "
            f"stage1_us={result['stage1_us']:.3f} "
            f"stage2_us={result['stage2_us']:.3f} "
            f"component_sum_us={result['component_sum_us']:.3f} "
            f"component_gap_us={result['component_gap_us']:+.3f} "
            f"qsfa_us={result['qsfa_us']:.3f} "
            f"speedup={result['speedup']:.6f} "
            f"staged_change_pct={result['change_pct']:+.3f} "
            f"staged_faster={int(result['speedup'] > 1.0)} "
            f"max_abs={result['max_abs']:.9f} "
            f"max_tolerance_ratio={result['max_tolerance_ratio']:.9f}",
            flush=True,
        )

    staged_total = sum(float(result["staged_us"]) for result in results)
    qsfa_total = sum(float(result["qsfa_us"]) for result in results)
    speedups = [float(result["speedup"]) for result in results]
    aggregate_speedup = qsfa_total / staged_total
    aggregate_change = (qsfa_total - staged_total) / qsfa_total * 100.0
    geometric_speedup = math.exp(
        statistics.mean(math.log(speedup) for speedup in speedups)
    )
    print(
        "A5_C8_STAGED_PERF_SUMMARY "
        f"execution_mode={args.execution_mode} cases={len(results)} "
        f"aggregate_speedup={aggregate_speedup:.6f} "
        f"aggregate_staged_change_pct={aggregate_change:+.3f} "
        f"geomean_speedup={geometric_speedup:.6f} "
        f"min_speedup={min(speedups):.6f} "
        f"max_speedup={max(speedups):.6f} "
        f"faster_cases={sum(speedup > 1.0 for speedup in speedups)} "
        f"slower_cases={sum(speedup < 1.0 for speedup in speedups)}",
        flush=True,
    )


if __name__ == "__main__":
    main()
