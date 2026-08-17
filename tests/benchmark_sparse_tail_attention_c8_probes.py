#!/usr/bin/env python3
"""Isolate C8 SFA P/M/L writeback and TND slot-resolution overhead."""

from __future__ import annotations

import argparse
import statistics
from collections.abc import Callable

import torch
import torch_npu  # type: ignore  # noqa: F401

import nanovllm_dsa_a5
from _c8_staged_attention_reference import full_attention_slots, make_case
from _utils import require_a5


CASES = ("q4_miss0", "q4_miss_random", "varq_miss_random", "half01")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--query-dtype", choices=("bf16", "fp16"), default="bf16")
    parser.add_argument("--cache-tokens", type=int, default=6144)
    parser.add_argument("--tail-tokens", type=int, default=128)
    parser.add_argument("--miss-max", type=int, default=300)
    parser.add_argument("--cases", nargs="+", choices=CASES, default=list(CASES))
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--allow-non-a5", action="store_true")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if not 1 <= args.batch_size <= 64:
        raise ValueError("--batch-size must be within [1,64]")
    if not 1 <= args.heads <= 64:
        raise ValueError("--heads must be within [1,64]")
    if args.cache_tokens < 2048 or args.cache_tokens % 128:
        raise ValueError("--cache-tokens must be block-aligned and >= 2048")
    if args.tail_tokens < 3:
        raise ValueError("--tail-tokens must cover four-query MTP causality")
    if not 0 <= args.miss_max <= 2048:
        raise ValueError("--miss-max must be within [0,2048]")
    if args.warmup < 0 or args.iters <= 0 or args.repeats <= 0:
        raise ValueError("warmup/iters/repeats are invalid")


def case_metadata(
    name: str,
    batch_size: int,
    miss_max: int,
    seed: int,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    generator = torch.Generator().manual_seed(seed)
    if name.startswith("q4"):
        query_counts = (4,) * batch_size
    elif name == "varq_miss_random":
        counts = torch.randint(0, 5, (batch_size,), generator=generator).tolist()
        if sum(counts) == 0:
            counts[0] = 1
        query_counts = tuple(int(value) for value in counts)
    else:
        query_counts = tuple(index & 1 for index in range(batch_size))
        if sum(query_counts) == 0:
            query_counts = (1,)

    packed_queries = sum(query_counts)
    if name == "q4_miss0":
        miss_counts = (0,) * packed_queries
    else:
        misses = torch.randint(
            0, miss_max + 1, (packed_queries,), generator=generator
        ).tolist()
        miss_counts = tuple(int(value) for value in misses)
    return query_counts, miss_counts


def pml_probe_step(
    query: torch.Tensor,
    packed_kv: torch.Tensor,
    slots: torch.Tensor,
    block_table: torch.Tensor,
    actual_q: torch.Tensor,
    resident: torch.Tensor,
    miss_counts: torch.Tensor,
    cache_tokens: torch.Tensor,
    scale: float,
    enabled: bool,
    attention_out: torch.Tensor,
    partial: torch.Tensor,
    maximum: torch.Tensor,
    denominator: torch.Tensor,
) -> torch.Tensor:
    torch.ops.nanovllm_dsa._sparse_tail_attention_c8_pml_probe_out.default(
        query,
        packed_kv,
        slots,
        block_table,
        actual_q,
        resident,
        miss_counts,
        cache_tokens,
        scale,
        enabled,
        attention_out,
        partial,
        maximum,
        denominator,
    )
    return attention_out


def tnd_probe_step(
    query: torch.Tensor,
    packed_kv: torch.Tensor,
    slots: torch.Tensor,
    block_table: torch.Tensor,
    actual_q: torch.Tensor,
    resident: torch.Tensor,
    miss_counts: torch.Tensor,
    cache_tokens: torch.Tensor,
    scale: float,
    enabled: bool,
    attention_out: torch.Tensor,
) -> torch.Tensor:
    torch.ops.nanovllm_dsa._sparse_tail_attention_c8_tnd_probe_out.default(
        query,
        packed_kv,
        slots,
        block_table,
        actual_q,
        resident,
        miss_counts,
        cache_tokens,
        scale,
        enabled,
        attention_out,
    )
    return attention_out


def stage1_step(
    query: torch.Tensor,
    packed_kv: torch.Tensor,
    actual_q: torch.Tensor,
    resident: torch.Tensor,
    cache_tokens: torch.Tensor,
    block_table: torch.Tensor,
    topk_slots: torch.Tensor,
    miss_counts: torch.Tensor,
    partial: torch.Tensor,
    maximum: torch.Tensor,
    denominator: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    torch.ops.nanovllm_dsa.sparse_tail_attention_c8_stage1_out.default(
        query,
        packed_kv,
        actual_q,
        resident,
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


def qsfa_step(
    query: torch.Tensor,
    packed_kv: torch.Tensor,
    slots: torch.Tensor,
    block_table: torch.Tensor,
    actual_q: torch.Tensor,
    resident: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    return nanovllm_dsa_a5.sparse_tail_attention_c8(
        query, packed_kv, slots, block_table, actual_q, resident, scale
    )


def graph(fn: Callable[..., torch.Tensor]) -> Callable[..., torch.Tensor]:
    return torch.compile(
        fn,
        backend="npugraph_ex",
        fullgraph=True,
        dynamic=False,
        options={"clone_input": False, "clone_output": False},
    )


def event_mean_us(
    launch: Callable[[], torch.Tensor], warmup: int, iters: int
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


def repeated_median_us(
    launch: Callable[[], torch.Tensor], args: argparse.Namespace
) -> tuple[float, float]:
    samples = [
        event_mean_us(launch, args.warmup, args.iters)
        for _ in range(args.repeats)
    ]
    median = statistics.median(samples)
    spread = max(samples) - min(samples)
    return median, spread


def benchmark_case(
    args: argparse.Namespace,
    device: torch.device,
    name: str,
    seed: int,
) -> dict[str, float | int | str]:
    query_counts, miss_counts = case_metadata(
        name, args.batch_size, args.miss_max, seed
    )
    query_dtype = torch.bfloat16 if args.query_dtype == "bf16" else torch.float16
    case = make_case(
        device=device,
        query_counts=query_counts,
        heads=args.heads,
        cache_tokens=(args.cache_tokens,) * len(query_counts),
        final_tail_tokens=(args.tail_tokens,) * len(query_counts),
        miss_counts=miss_counts,
        query_dtype=query_dtype,
        seed=seed + 1,
    )
    actual_q = case.actual_q.to(torch.int32)
    full_slots = full_attention_slots(case).to(device)
    partial = torch.empty(
        (*case.query.shape[:-1], 512), dtype=torch.float32, device=device
    )
    maximum = torch.empty(
        (1, case.query.size(0), args.heads), dtype=torch.float32, device=device
    )
    denominator = torch.empty_like(maximum)
    attention = torch.empty(
        (*case.query.shape[:-1], 512), dtype=case.query.dtype, device=device
    )

    pml_control = graph(pml_probe_step)
    pml_enabled = graph(pml_probe_step)
    tnd_control = graph(tnd_probe_step)
    tnd_enabled = graph(tnd_probe_step)
    stage1 = graph(stage1_step)
    qsfa = graph(qsfa_step)

    common = (
        case.query,
        case.packed,
        case.block_table,
        actual_q,
        case.resident_lengths,
        case.miss_counts,
        case.cache_tokens,
        case.scale,
    )

    def launch_pml(enabled: bool) -> torch.Tensor:
        return (pml_enabled if enabled else pml_control)(
            common[0], common[1], full_slots, *common[2:], enabled,
            attention, partial, maximum, denominator
        )

    def launch_tnd(enabled: bool) -> torch.Tensor:
        slots = case.topk_slots if enabled else full_slots
        return (tnd_enabled if enabled else tnd_control)(
            common[0], common[1], slots, *common[2:], enabled, attention
        )

    def launch_stage1() -> torch.Tensor:
        return stage1(
            case.query, case.packed, actual_q, case.resident_lengths,
            case.cache_tokens, case.block_table, case.topk_slots,
            case.miss_counts, partial, maximum, denominator, case.scale
        )

    def launch_qsfa() -> torch.Tensor:
        return qsfa(
            case.query, case.packed, full_slots, case.block_table, actual_q,
            case.resident_lengths, case.scale
        )

    launches = {
        "native_qsfa": launch_qsfa,
        "pml_control": lambda: launch_pml(False),
        "pml_enabled": lambda: launch_pml(True),
        "tnd_control": lambda: launch_tnd(False),
        "tnd_enabled": lambda: launch_tnd(True),
        "stage1": launch_stage1,
    }
    timings: dict[str, float] = {}
    for label, launch in launches.items():
        median, spread = repeated_median_us(launch, args)
        timings[f"{label}_us"] = median
        timings[f"{label}_spread_us"] = spread

    timings["pml_delta_us"] = (
        timings["pml_enabled_us"] - timings["pml_control_us"]
    )
    timings["tnd_delta_us"] = (
        timings["tnd_enabled_us"] - timings["tnd_control_us"]
    )
    timings["pml_delta_pct"] = (
        timings["pml_delta_us"] / timings["pml_control_us"] * 100.0
    )
    timings["tnd_delta_pct"] = (
        timings["tnd_delta_us"] / timings["tnd_control_us"] * 100.0
    )
    timings["pml_control_vs_native_us"] = (
        timings["pml_control_us"] - timings["native_qsfa_us"]
    )
    timings["tnd_control_vs_native_us"] = (
        timings["tnd_control_us"] - timings["native_qsfa_us"]
    )
    return {
        "case": name,
        "batch_size": len(query_counts),
        "packed_queries": sum(query_counts),
        "zero_query_requests": query_counts.count(0),
        **timings,
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
        f"C8_PROBE_CONFIG device={device_name!r} bs={args.batch_size} "
        f"heads={args.heads} dtype={args.query_dtype} repeats={args.repeats}"
    )
    for index, name in enumerate(args.cases):
        result = benchmark_case(args, device, name, args.seed + index * 17)
        print(
            "C8_PROBE "
            f"case={result['case']} T={result['packed_queries']} "
            f"zero_requests={result['zero_query_requests']} "
            f"native_qsfa_us={result['native_qsfa_us']:.3f} "
            f"pml_control_us={result['pml_control_us']:.3f} "
            f"pml_control_vs_native_us={result['pml_control_vs_native_us']:.3f} "
            f"pml_enabled_us={result['pml_enabled_us']:.3f} "
            f"pml_delta_us={result['pml_delta_us']:.3f} "
            f"pml_delta_pct={result['pml_delta_pct']:.2f} "
            f"pml_spread_us={result['pml_enabled_spread_us']:.3f} "
            f"tnd_control_us={result['tnd_control_us']:.3f} "
            f"tnd_control_vs_native_us={result['tnd_control_vs_native_us']:.3f} "
            f"tnd_enabled_us={result['tnd_enabled_us']:.3f} "
            f"tnd_delta_us={result['tnd_delta_us']:.3f} "
            f"tnd_delta_pct={result['tnd_delta_pct']:.2f} "
            f"tnd_spread_us={result['tnd_enabled_spread_us']:.3f} "
            f"stage1_us={result['stage1_us']:.3f}"
        )


if __name__ == "__main__":
    main()
