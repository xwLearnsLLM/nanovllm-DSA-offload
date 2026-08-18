#!/usr/bin/env python3
"""Repeated correctness stress test for no-MTP staged C8 SFA.

This test intentionally keeps one set of caller-owned P/M/L/output buffers,
uses heterogeneous per-row miss counts, and validates every launch.  It is
separate from the boundary and benchmark tests so intermittent pipeline bugs
cannot be hidden by checking only the launch before a timing loop.
"""

from __future__ import annotations

import argparse

import torch
import torch_npu  # type: ignore  # noqa: F401

import nanovllm_dsa_a5
from _utils import csv_ints, require_a5
from test_sparse_tail_attention_c8 import make_inputs


NOPE_DIM = 512
TOPK = 2048
DEFAULT_ATOL = 0.08
DEFAULT_RTOL = 0.03


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--batch-size", type=int, default=5)
    parser.add_argument("--heads", type=int, default=32)
    parser.add_argument("--query-dtypes", default="bf16,fp16")
    parser.add_argument("--seeds", type=csv_ints, default=csv_ints("31,33"))
    parser.add_argument("--cache-tokens", type=int, default=6144)
    parser.add_argument("--tail-tokens", type=int, default=128)
    parser.add_argument("--max-tail-tokens", type=int, default=512)
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--modes", default="eager,graph")
    parser.add_argument("--phases", default="reuse,nan")
    parser.add_argument("--atol", type=float, default=DEFAULT_ATOL)
    parser.add_argument("--rtol", type=float, default=DEFAULT_RTOL)
    parser.add_argument("--absolute-output-limit", type=float, default=5.0)
    parser.add_argument("--allow-non-a5", action="store_true")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if not 1 <= args.heads <= 64:
        raise ValueError("--heads must be in [1,64]")
    if args.cache_tokens < TOPK or args.cache_tokens % 128:
        raise ValueError("--cache-tokens must be block-aligned and >=2048")
    if not 0 <= args.tail_tokens <= args.max_tail_tokens:
        raise ValueError("tail tokens must be in [0,max-tail-tokens]")
    if args.repeats <= 0:
        raise ValueError("--repeats must be positive")
    if args.atol < 0 or args.rtol < 0:
        raise ValueError("allclose tolerances must be non-negative")
    if args.absolute_output_limit <= 0:
        raise ValueError("--absolute-output-limit must be positive")


def csv_strings(value: str) -> list[str]:
    result = [item.strip() for item in value.split(",") if item.strip()]
    if not result:
        raise ValueError("expected a non-empty comma-separated list")
    return result


def make_case(args: argparse.Namespace, dtype: str, seed: int) -> dict[str, object]:
    case_args = argparse.Namespace(
        device=args.device,
        batch_size=args.batch_size,
        heads=args.heads,
        cache_tokens=args.cache_tokens,
        tail_tokens=args.tail_tokens,
        max_tail_tokens=args.max_tail_tokens,
        seed=seed,
    )
    inputs = make_inputs(case_args)
    if dtype == "fp16":
        inputs["query"] = inputs["query"].to(torch.float16)
    elif dtype != "bf16":
        raise ValueError(f"unsupported query dtype: {dtype!r}")
    return inputs


def miss_patterns(batch: int, device: torch.device) -> list[torch.Tensor]:
    # Boundary-heavy and online-like heterogeneous rows.  Repeat/truncate the
    # templates for non-default batch sizes while retaining per-row variation.
    templates = (
        (0, 1, 127, 128, 129),
        (2048, 2047, 1025, 257, 1),
        (13, 64, 191, 300, 511),
        (300, 0, 128, 2048, 37),
    )
    result = []
    for template in templates:
        values = [template[index % len(template)] for index in range(batch)]
        result.append(torch.tensor(values, dtype=torch.int32, device=device))
    return result


def staged_buffers(inputs: dict[str, object]) -> tuple[torch.Tensor, ...]:
    query = inputs["query"]
    assert isinstance(query, torch.Tensor)
    p = torch.empty((*query.shape[:-1], NOPE_DIM), dtype=torch.float32,
                    device=query.device)
    m = torch.empty((1, query.size(0), query.size(1)), dtype=torch.float32,
                    device=query.device)
    l = torch.empty_like(m)
    out = torch.empty((*query.shape[:-1], NOPE_DIM), dtype=query.dtype,
                      device=query.device)
    return p, m, l, out


def eager_runner(inputs: dict[str, object], buffers: tuple[torch.Tensor, ...]):
    p, m, l, out = buffers

    def run(misses: torch.Tensor) -> torch.Tensor:
        nanovllm_dsa_a5.sparse_tail_attention_c8_stage1(
            inputs["query"], inputs["packed"], inputs["slots"],
            inputs["block_table"], inputs["actual_q"],
            inputs["resident_lengths"], misses, inputs["scale"], p, m, l,
        )
        nanovllm_dsa_a5.sparse_tail_attention_c8_stage2(
            inputs["query"], inputs["packed"], inputs["slots"],
            inputs["block_table"], inputs["actual_q"],
            inputs["resident_lengths"], misses, inputs["scale"],
            p, m, l, out,
        )
        return out

    return run


def graph_runner(inputs: dict[str, object], buffers: tuple[torch.Tensor, ...],
                 example_misses: torch.Tensor):
    p, m, l, out = buffers

    def step(query, packed, slots, table, actual_q, resident, misses,
             partial, maximum, denominator, output):
        torch.ops.nanovllm_dsa.sparse_tail_attention_c8_stage1.default(
            query, packed, slots, table, actual_q, resident, misses,
            float(inputs["scale"]), partial, maximum, denominator,
        )
        torch.ops.nanovllm_dsa.sparse_tail_attention_c8_stage2.default(
            query, packed, slots, table, actual_q, resident, misses,
            float(inputs["scale"]), partial, maximum, denominator, output,
        )
        return output

    compiled = torch.compile(
        step,
        backend="npugraph_ex",
        fullgraph=True,
        dynamic=False,
        options={"clone_input": False, "clone_output": False},
    )
    fixed_args = (
        inputs["query"], inputs["packed"], inputs["slots"],
        inputs["block_table"], inputs["actual_q"],
        inputs["resident_lengths"], p, m, l, out,
    )

    def run(misses: torch.Tensor) -> torch.Tensor:
        return compiled(
            fixed_args[0], fixed_args[1], fixed_args[2], fixed_args[3],
            fixed_args[4], fixed_args[5], misses, fixed_args[6],
            fixed_args[7], fixed_args[8], fixed_args[9],
        )

    # Materialize and warm the graph before recording stress outputs.
    run(example_misses)
    run(example_misses)
    torch.npu.synchronize()
    return run


def native_output(inputs: dict[str, object]) -> torch.Tensor:
    output = nanovllm_dsa_a5.sparse_tail_attention_c8(
        inputs["query"], inputs["packed"], inputs["slots"],
        inputs["block_table"], inputs["actual_q"],
        inputs["resident_lengths"], inputs["scale"],
    )
    torch.npu.synchronize()
    return output.cpu().float()


def check_snapshots(
    snapshots: torch.Tensor,
    golden: torch.Tensor,
    pattern_ids: list[int],
    patterns: list[torch.Tensor],
    label: str,
    atol: float,
    rtol: float,
    absolute_output_limit: float,
) -> float:
    actual = snapshots.cpu().float()
    expected = golden.unsqueeze(0).expand_as(actual)
    finite = torch.isfinite(actual)
    absolute = (actual - expected).abs()
    tolerance = atol + rtol * expected.abs()
    mismatch = (~finite) | absolute.gt(tolerance)
    oversized = actual.abs().gt(absolute_output_limit)

    if bool(mismatch.any()) or bool(oversized.any()):
        combined = mismatch | oversized
        first = torch.nonzero(combined, as_tuple=False)[0]
        iteration, row, head, dim = (int(value) for value in first)
        worst_flat = int(torch.nan_to_num(absolute, nan=float("inf")).argmax())
        worst = []
        remaining = worst_flat
        for size in reversed(absolute.shape):
            worst.append(remaining % size)
            remaining //= size
        worst = list(reversed(worst))
        bad_by_head = combined[iteration].sum(dim=-1)
        bad_heads = [
            (int(index), int(bad_by_head[row, index]))
            for index in torch.nonzero(
                bad_by_head[row], as_tuple=False).flatten()
        ]
        pattern = patterns[pattern_ids[iteration]].cpu().tolist()
        raise AssertionError(
            f"{label} failed at iteration={iteration} pattern={pattern}: "
            f"first=(row={row},head={head},dim={dim}) "
            f"actual={float(actual[iteration,row,head,dim])} "
            f"expected={float(expected[iteration,row,head,dim])} "
            f"abs={float(absolute[iteration,row,head,dim])} "
            f"tol={float(tolerance[iteration,row,head,dim])}; "
            f"worst_index={tuple(worst)} "
            f"worst_abs={float(absolute[tuple(worst)])}; "
            f"bad_heads_in_first_row={bad_heads}"
        )
    return float(absolute.max())


def run_stress(
    args: argparse.Namespace,
    inputs: dict[str, object],
    dtype: str,
    seed: int,
    mode: str,
    phase: str,
) -> None:
    query = inputs["query"]
    assert isinstance(query, torch.Tensor)
    patterns = miss_patterns(args.batch_size, query.device)
    buffers = staged_buffers(inputs)
    runner = eager_runner(inputs, buffers)
    if mode == "graph":
        runner = graph_runner(inputs, buffers, patterns[0])
    elif mode != "eager":
        raise ValueError(f"unsupported execution mode: {mode!r}")

    golden = native_output(inputs)
    snapshots = torch.empty(
        (args.repeats, *query.shape[:-1], NOPE_DIM),
        dtype=query.dtype,
        device=query.device,
    )
    pattern_ids = []
    output = buffers[-1]
    for iteration in range(args.repeats):
        pattern_id = iteration % len(patterns)
        pattern_ids.append(pattern_id)
        if phase == "nan":
            output.fill_(float("nan"))
        elif phase != "reuse":
            raise ValueError(f"unsupported stress phase: {phase!r}")
        result = runner(patterns[pattern_id])
        snapshots[iteration].copy_(result)

    torch.npu.synchronize()
    max_abs = check_snapshots(
        snapshots,
        golden,
        pattern_ids,
        patterns,
        f"dtype={dtype} seed={seed} mode={mode} phase={phase}",
        args.atol,
        args.rtol,
        args.absolute_output_limit,
    )
    print(
        "A5_C8_STAGED_NOMTP_STRESS_OK "
        f"dtype={dtype} seed={seed} mode={mode} phase={phase} "
        f"batch={args.batch_size} heads={args.heads} "
        f"repeats={args.repeats} max_abs={max_abs:.9f}",
        flush=True,
    )


def main() -> None:
    args = parse_args()
    validate_args(args)
    device = torch.device(args.device)
    require_a5(device, allow_non_a5=args.allow_non_a5)
    dtypes = csv_strings(args.query_dtypes)
    modes = csv_strings(args.modes)
    phases = csv_strings(args.phases)
    for dtype in dtypes:
        for seed in args.seeds:
            inputs = make_case(args, dtype, seed)
            for mode in modes:
                for phase in phases:
                    run_stress(args, inputs, dtype, seed, mode, phase)


if __name__ == "__main__":
    main()
