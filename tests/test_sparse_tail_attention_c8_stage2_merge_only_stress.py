#!/usr/bin/env python3
"""Stress only the previous-state merge path of no-MTP C8 Stage2.

Stage1 first produces a fixed non-empty P/M/L state using heterogeneous miss
counts.  Stage2 is then called repeatedly with zero misses, so its current
state is exactly {P=0, M=-inf, L=0}; every output must equal Stage1 P/L.  Each
launch is snapshotted before the next launch to catch intermittent failures.
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
    parser.add_argument(
        "--stage1-misses",
        type=csv_ints,
        default=csv_ints("0,37,128,257,300"),
        help="Per-row misses used only to construct the fixed Stage1 state",
    )
    parser.add_argument("--repeats", type=int, default=200)
    parser.add_argument("--modes", default="eager,graph")
    parser.add_argument("--phases", default="reuse,nan")
    parser.add_argument("--atol", type=float, default=0.08)
    parser.add_argument("--rtol", type=float, default=0.03)
    parser.add_argument("--absolute-output-limit", type=float, default=5.0)
    parser.add_argument("--allow-non-a5", action="store_true")
    return parser.parse_args()


def csv_strings(value: str) -> list[str]:
    result = [item.strip() for item in value.split(",") if item.strip()]
    if not result:
        raise ValueError("expected a non-empty comma-separated list")
    return result


def validate_args(args: argparse.Namespace) -> None:
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if not 1 <= args.heads <= 64:
        raise ValueError("--heads must be in [1,64]")
    if args.cache_tokens < TOPK or args.cache_tokens % 128:
        raise ValueError("--cache-tokens must be block-aligned and >=2048")
    if not 0 <= args.tail_tokens <= args.max_tail_tokens:
        raise ValueError("tail tokens must be in [0,max-tail-tokens]")
    if len(args.stage1_misses) != args.batch_size:
        raise ValueError("--stage1-misses must contain exactly batch-size values")
    if any(not 0 <= value <= TOPK for value in args.stage1_misses):
        raise ValueError("Stage1 miss counts must be in [0,2048]")
    if args.repeats <= 0:
        raise ValueError("--repeats must be positive")
    if args.atol < 0 or args.rtol < 0:
        raise ValueError("allclose tolerances must be non-negative")
    if args.absolute_output_limit <= 0:
        raise ValueError("--absolute-output-limit must be positive")


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


def allocate_state(inputs: dict[str, object]) -> tuple[torch.Tensor, ...]:
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


def produce_stage1_state(
    inputs: dict[str, object],
    misses: torch.Tensor,
    state: tuple[torch.Tensor, ...],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    p, m, l, _ = state
    nanovllm_dsa_a5.sparse_tail_attention_c8_stage1(
        inputs["query"], inputs["packed"], inputs["slots"],
        inputs["block_table"], inputs["actual_q"],
        inputs["resident_lengths"], misses, inputs["scale"], p, m, l,
    )
    torch.npu.synchronize()
    p_cpu = p.cpu().float()
    m_cpu = m.cpu().float()
    l_cpu = l.cpu().float()
    if not (torch.isfinite(p_cpu).all() and torch.isfinite(m_cpu).all()
            and torch.isfinite(l_cpu).all()):
        raise AssertionError("Stage1 produced non-finite P/M/L")
    if bool(l_cpu.lt(0).any()):
        raise AssertionError("Stage1 produced a negative softmax sum")
    return p_cpu, m_cpu, l_cpu


def expected_from_state(
    p: torch.Tensor,
    l: torch.Tensor,
    output_dtype: torch.dtype,
) -> torch.Tensor:
    denominator = l.squeeze(0).unsqueeze(-1)
    expected = torch.where(
        denominator > 0,
        p / denominator,
        torch.zeros_like(p),
    )
    return expected.to(output_dtype).float()


def eager_stage2_runner(
    inputs: dict[str, object],
    zero_misses: torch.Tensor,
    state: tuple[torch.Tensor, ...],
):
    p, m, l, out = state

    def run() -> torch.Tensor:
        nanovllm_dsa_a5.sparse_tail_attention_c8_stage2(
            inputs["query"], inputs["packed"], inputs["slots"],
            inputs["block_table"], inputs["actual_q"],
            inputs["resident_lengths"], zero_misses, inputs["scale"],
            p, m, l, out,
        )
        return out

    return run


def graph_stage2_runner(
    inputs: dict[str, object],
    zero_misses: torch.Tensor,
    state: tuple[torch.Tensor, ...],
):
    p, m, l, out = state

    def step(query, packed, slots, table, actual_q, resident, misses,
             previous_p, previous_m, previous_l, output):
        torch.ops.nanovllm_dsa.sparse_tail_attention_c8_stage2.default(
            query, packed, slots, table, actual_q, resident, misses,
            float(inputs["scale"]), previous_p, previous_m, previous_l,
            output,
        )
        return output

    compiled = torch.compile(
        step,
        backend="npugraph_ex",
        fullgraph=True,
        dynamic=False,
        options={"clone_input": False, "clone_output": False},
    )
    args = (
        inputs["query"], inputs["packed"], inputs["slots"],
        inputs["block_table"], inputs["actual_q"],
        inputs["resident_lengths"], zero_misses, p, m, l, out,
    )

    def run() -> torch.Tensor:
        return compiled(*args)

    run()
    run()
    torch.npu.synchronize()
    return run


def assert_state_unchanged(
    state: tuple[torch.Tensor, ...],
    original: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
) -> None:
    for name, device_value, expected in zip(("P", "M", "L"), state[:3], original):
        actual = device_value.cpu().float()
        if not torch.equal(actual, expected):
            difference = (actual - expected).abs()
            raise AssertionError(
                f"Stage2 modified previous {name}: "
                f"max_abs={float(difference.max())}"
            )


def check_snapshots(
    snapshots: torch.Tensor,
    expected: torch.Tensor,
    label: str,
    atol: float,
    rtol: float,
    absolute_output_limit: float,
) -> float:
    actual = snapshots.cpu().float()
    golden = expected.unsqueeze(0).expand_as(actual)
    finite = torch.isfinite(actual)
    absolute = (actual - golden).abs()
    tolerance = atol + rtol * golden.abs()
    bad = (~finite) | absolute.gt(tolerance)
    bad |= actual.abs().gt(absolute_output_limit)
    if bool(bad.any()):
        first = torch.nonzero(bad, as_tuple=False)[0]
        iteration, row, head, dim = (int(value) for value in first)
        bad_by_head = bad[iteration, row].sum(dim=-1)
        bad_heads = [
            (int(index), int(bad_by_head[index]))
            for index in torch.nonzero(bad_by_head, as_tuple=False).flatten()
        ]
        raise AssertionError(
            f"{label} mismatch: iteration={iteration} row={row} "
            f"head={head} dim={dim} "
            f"actual={float(actual[iteration,row,head,dim])} "
            f"expected={float(golden[iteration,row,head,dim])} "
            f"abs={float(absolute[iteration,row,head,dim])} "
            f"tol={float(tolerance[iteration,row,head,dim])} "
            f"bad_heads={bad_heads}"
        )
    return float(absolute.max())


def run_case(
    args: argparse.Namespace,
    inputs: dict[str, object],
    dtype: str,
    seed: int,
    mode: str,
    phase: str,
) -> None:
    query = inputs["query"]
    assert isinstance(query, torch.Tensor)
    state = allocate_state(inputs)
    stage1_misses = torch.tensor(
        args.stage1_misses, dtype=torch.int32, device=query.device)
    zero_misses = torch.zeros_like(stage1_misses)
    original_state = produce_stage1_state(inputs, stage1_misses, state)
    expected = expected_from_state(original_state[0], original_state[2], query.dtype)

    runner = eager_stage2_runner(inputs, zero_misses, state)
    if mode == "graph":
        runner = graph_stage2_runner(inputs, zero_misses, state)
    elif mode != "eager":
        raise ValueError(f"unsupported execution mode: {mode!r}")

    snapshots = torch.empty(
        (args.repeats, *query.shape[:-1], NOPE_DIM),
        dtype=query.dtype,
        device=query.device,
    )
    output = state[-1]
    for iteration in range(args.repeats):
        if phase == "nan":
            output.fill_(float("nan"))
        elif phase != "reuse":
            raise ValueError(f"unsupported phase: {phase!r}")
        snapshots[iteration].copy_(runner())

    torch.npu.synchronize()
    assert_state_unchanged(state, original_state)
    label = f"dtype={dtype} seed={seed} mode={mode} phase={phase}"
    max_abs = check_snapshots(
        snapshots,
        expected,
        label,
        args.atol,
        args.rtol,
        args.absolute_output_limit,
    )
    print(
        "A5_C8_STAGE2_MERGE_ONLY_STRESS_OK "
        f"{label} batch={args.batch_size} heads={args.heads} "
        f"repeats={args.repeats} max_abs={max_abs:.9f}",
        flush=True,
    )


def main() -> None:
    args = parse_args()
    validate_args(args)
    require_a5(torch.device(args.device), allow_non_a5=args.allow_non_a5)
    for dtype in csv_strings(args.query_dtypes):
        for seed in args.seeds:
            inputs = make_case(args, dtype, seed)
            for mode in csv_strings(args.modes):
                for phase in csv_strings(args.phases):
                    run_case(args, inputs, dtype, seed, mode, phase)


if __name__ == "__main__":
    main()
