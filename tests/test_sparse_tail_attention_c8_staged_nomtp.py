#!/usr/bin/env python3
"""Correctness, boundary, graph, and timing checks for no-MTP staged C8 SFA."""

from __future__ import annotations

import argparse
import statistics

import torch
import torch_npu  # type: ignore  # noqa: F401

import nanovllm_dsa_a5
from _utils import require_a5
from test_sparse_tail_attention_c8 import make_inputs


ATOL = 0.08
RTOL = 0.03
NOPE_DIM = 512


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--heads", default="1,8,64")
    parser.add_argument("--query-dtypes", default="bf16,fp16")
    parser.add_argument("--batch-sizes", default="1,4,16,64")
    parser.add_argument("--cache-tokens", type=int, default=6144)
    parser.add_argument("--tail-tokens", type=int, default=128)
    parser.add_argument("--max-tail-tokens", type=int, default=512)
    parser.add_argument("--miss-counts", default="0,1,127,128,2047,2048")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--seed", type=int, default=31)
    parser.add_argument("--graph", action="store_true")
    parser.add_argument("--allow-non-a5", action="store_true")
    return parser.parse_args()


def csv_ints(value: str) -> list[int]:
    return [int(item) for item in value.split(",") if item]


def staged_buffers(inputs: dict[str, object]) -> tuple[torch.Tensor, ...]:
    query = inputs["query"]
    assert isinstance(query, torch.Tensor)
    partial = torch.empty((*query.shape[:-1], NOPE_DIM), dtype=torch.float32,
                          device=query.device)
    maximum = torch.empty((1, query.size(0), query.size(1)), dtype=torch.float32,
                          device=query.device)
    denominator = torch.empty_like(maximum)
    output = torch.empty((*query.shape[:-1], NOPE_DIM), dtype=query.dtype,
                         device=query.device)
    return partial, maximum, denominator, output


def eager_staged(
    inputs: dict[str, object], miss_counts: torch.Tensor,
    buffers: tuple[torch.Tensor, ...],
) -> torch.Tensor:
    partial, maximum, denominator, output = buffers
    nanovllm_dsa_a5.sparse_tail_attention_c8_stage1(
        inputs["query"], inputs["packed"], inputs["slots"],
        inputs["block_table"], inputs["actual_q"],
        inputs["resident_lengths"], miss_counts, inputs["scale"],
        partial, maximum, denominator,
    )
    nanovllm_dsa_a5.sparse_tail_attention_c8_stage2(
        inputs["query"], inputs["packed"], inputs["slots"],
        inputs["block_table"], inputs["actual_q"],
        inputs["resident_lengths"], miss_counts, inputs["scale"],
        partial, maximum, denominator, output,
    )
    return output


def native(inputs: dict[str, object]) -> torch.Tensor:
    return nanovllm_dsa_a5.sparse_tail_attention_c8(
        inputs["query"], inputs["packed"], inputs["slots"],
        inputs["block_table"], inputs["actual_q"],
        inputs["resident_lengths"], inputs["scale"],
    )


def event_us(fn, warmup: int, iters: int) -> float:
    for _ in range(warmup):
        fn()
    torch.npu.synchronize()
    starts = [torch.npu.Event(enable_timing=True) for _ in range(iters)]
    ends = [torch.npu.Event(enable_timing=True) for _ in range(iters)]
    for start, end in zip(starts, ends):
        start.record()
        fn()
        end.record()
    ends[-1].synchronize()
    return statistics.mean(s.elapsed_time(e) for s, e in zip(starts, ends)) * 1000


def graph_runner(inputs: dict[str, object], miss_counts: torch.Tensor,
                 buffers: tuple[torch.Tensor, ...]):
    partial, maximum, denominator, output = buffers

    def step(query, packed, slots, table, actual_q, resident, misses,
             p, m, l, out):
        torch.ops.nanovllm_dsa.sparse_tail_attention_c8_stage1.default(
            query, packed, slots, table, actual_q, resident, misses,
            float(inputs["scale"]), p, m, l)
        torch.ops.nanovllm_dsa.sparse_tail_attention_c8_stage2.default(
            query, packed, slots, table, actual_q, resident, misses,
            float(inputs["scale"]), p, m, l, out)
        return out

    compiled = torch.compile(
        step, backend="npugraph_ex", fullgraph=True, dynamic=False,
        options={"clone_input": False, "clone_output": False},
    )
    args = (inputs["query"], inputs["packed"], inputs["slots"],
            inputs["block_table"], inputs["actual_q"],
            inputs["resident_lengths"], miss_counts,
            partial, maximum, denominator, output)
    compiled(*args)
    compiled(*args)
    torch.npu.synchronize()
    return lambda: compiled(*args)


def native_graph_runner(inputs: dict[str, object]):
    def step(query, packed, slots, table, actual_q, resident):
        return torch.ops.nanovllm_dsa.sparse_tail_attention_c8.default(
            query, packed, slots, table, actual_q, resident,
            float(inputs["scale"]))

    compiled = torch.compile(
        step, backend="npugraph_ex", fullgraph=True, dynamic=False,
        options={"clone_input": False, "clone_output": False},
    )
    args = (inputs["query"], inputs["packed"], inputs["slots"],
            inputs["block_table"], inputs["actual_q"],
            inputs["resident_lengths"])
    compiled(*args)
    compiled(*args)
    torch.npu.synchronize()
    return lambda: compiled(*args)


def make_case(args: argparse.Namespace, batch: int, heads: int,
              query_dtype: str, cache: int, tail: int,
              seed: int) -> dict[str, object]:
    case = argparse.Namespace(**vars(args))
    case.batch_size = batch
    case.heads = heads
    case.cache_tokens = cache
    case.tail_tokens = tail
    case.seed = seed
    inputs = make_inputs(case)
    if query_dtype == "fp16":
        inputs["query"] = inputs["query"].to(torch.float16)
    elif query_dtype != "bf16":
        raise ValueError(f"unsupported query dtype: {query_dtype}")
    return inputs


def run_case(args: argparse.Namespace, batch: int, heads: int,
             query_dtype: str, cache: int, tail: int,
             miss: int, seed: int) -> None:
    inputs = make_case(args, batch, heads, query_dtype, cache, tail, seed)
    valid_sparse = 0 if cache == 0 else 2048
    miss = min(miss, valid_sparse)
    misses = torch.full((batch,), miss, dtype=torch.int32,
                        device=inputs["query"].device)
    buffers = staged_buffers(inputs)
    staged = eager_staged(inputs, misses, buffers)
    expected = native(inputs)
    torch.npu.synchronize()
    staged_cpu = staged.cpu().float()
    expected_cpu = expected.cpu().float()
    if not torch.isfinite(staged_cpu).all():
        raise AssertionError("staged no-MTP C8 SFA produced NaN/Inf")
    torch.testing.assert_close(staged_cpu, expected_cpu, atol=ATOL, rtol=RTOL)
    absolute = (staged_cpu - expected_cpu).abs()

    staged_fn = lambda: eager_staged(inputs, misses, buffers)
    native_fn = lambda: native(inputs)
    if args.graph:
        staged_fn = graph_runner(inputs, misses, buffers)
        native_fn = native_graph_runner(inputs)
    staged_us = event_us(staged_fn, args.warmup, args.iters)
    native_us = event_us(native_fn, args.warmup, args.iters)
    print(
        "A5_C8_STAGED_NOMTP_RESULT "
        f"batch={batch} heads={heads} dtype={query_dtype} "
        f"cache={cache} tail={tail} "
        f"miss={miss} mode={'npugraph_ex' if args.graph else 'eager'} "
        f"staged_us={staged_us:.3f} native_us={native_us:.3f} "
        f"ratio={staged_us / native_us:.4f} "
        f"delta_us={staged_us - native_us:.3f} "
        f"max_abs={float(absolute.max()):.9f} ok=1",
        flush=True,
    )


def main() -> None:
    args = parse_args()
    require_a5(args.device, allow_non_a5=args.allow_non_a5)
    heads_list = csv_ints(args.heads)
    query_dtypes = [item for item in args.query_dtypes.split(",") if item]
    batches = csv_ints(args.batch_sizes)
    misses = csv_ints(args.miss_counts)
    seed = args.seed
    # Main split boundaries: empty miss, tile edges, and all-miss.
    for heads in heads_list:
        for query_dtype in query_dtypes:
            for batch in batches:
                for miss in misses:
                    run_case(
                        args, batch, heads, query_dtype,
                        args.cache_tokens, args.tail_tokens, miss, seed)
                    seed += 1
    # Dense C=0 has no miss region; Stage1 must equal native by itself.
    for tail in (1, 127, 128, args.tail_tokens):
        run_case(args, batches[0], heads_list[-1], query_dtypes[0],
                 0, tail, 0, seed)
        seed += 1
    # Hit/tail boundary variants, including no tail and exact block tails.
    for tail in (0, 1, 127, 128):
        run_case(args, batches[0], heads_list[-1], query_dtypes[-1],
                 args.cache_tokens, tail, min(128, args.cache_tokens), seed)
        seed += 1


if __name__ == "__main__":
    main()
