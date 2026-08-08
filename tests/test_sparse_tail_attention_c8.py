#!/usr/bin/env python3
"""Semantic/performance test for native A5 packed-C8 topK+tail QSFA."""

from __future__ import annotations

import argparse
import statistics

import torch

import nanovllm_dsa_a5
import torch_npu  # type: ignore  # noqa: E402,F401

from _utils import csv_ints, require_a5


BLOCK_SIZE = 128
NOPE_DIM = 512
ROPE_DIM = 64
QUERY_DIM = NOPE_DIM + ROPE_DIM
TILE_SIZE = 128
SCALE_COUNT = NOPE_DIM // TILE_SIZE
PACKED_DIM = 656
TOPK = 2048


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--batch-sizes", type=csv_ints, default=csv_ints("24"))
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--cache-tokens", type=csv_ints, default=csv_ints("6144"))
    parser.add_argument("--tail-tokens", type=csv_ints, default=csv_ints("64"))
    parser.add_argument("--max-tail-tokens", type=int, default=512)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--allow-non-a5", action="store_true")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if any(batch <= 0 for batch in args.batch_sizes):
        raise ValueError("batch sizes must be positive")
    if not 1 <= args.heads <= 64:
        raise ValueError("this project intentionally supports Q_HEAD <= 64")
    if any(
        cache_tokens != 0
        and (cache_tokens < TOPK or cache_tokens % BLOCK_SIZE)
        for cache_tokens in args.cache_tokens
    ):
        raise ValueError("cache tokens must be 0 or block-aligned >= 2048")
    if args.max_tail_tokens < 0 or any(
        not 0 <= tail_tokens <= args.max_tail_tokens
        for tail_tokens in args.tail_tokens
    ):
        raise ValueError("tail tokens must be in [0,max_tail_tokens]")
    if any(cache_tokens == 0 and tail_tokens == 0
           for cache_tokens in args.cache_tokens
           for tail_tokens in args.tail_tokens):
        raise ValueError("dense C=0 test requires at least one resident token")
    if args.warmup < 0 or args.iters < 0:
        raise ValueError("warmup and iters must be non-negative")


def case_args(
    args: argparse.Namespace,
    batch_size: int,
    cache_tokens: int,
    tail_tokens: int,
    max_tail_tokens: int,
    seed: int,
) -> argparse.Namespace:
    values = vars(args).copy()
    values.update(
        batch_size=batch_size,
        cache_tokens=cache_tokens,
        tail_tokens=tail_tokens,
        max_tail_tokens=max_tail_tokens,
        seed=seed,
    )
    return argparse.Namespace(**values)


def pack_cache(
    nope: torch.Tensor,
    rope: torch.Tensor,
    scales: torch.Tensor,
) -> torch.Tensor:
    packed_bytes = torch.cat(
        (
            nope.contiguous().view(torch.uint8),
            rope.contiguous().view(torch.uint8),
            scales.contiguous().view(torch.uint8),
        ),
        dim=-1,
    )
    if packed_bytes.shape[-1] != PACKED_DIM:
        raise AssertionError(f"packed row has {packed_bytes.shape[-1]} bytes")
    return packed_bytes.view(torch.float8_e4m3fn)


def make_inputs(args: argparse.Namespace) -> dict[str, object]:
    torch.manual_seed(args.seed)
    generator = torch.Generator().manual_seed(args.seed)
    device = torch.device(args.device)
    resident_len = args.cache_tokens + args.tail_tokens
    blocks_per_row = (resident_len + BLOCK_SIZE - 1) // BLOCK_SIZE
    physical_blocks = args.batch_size * blocks_per_row
    block_table_cpu = torch.randperm(
        physical_blocks, generator=generator, dtype=torch.int64
    ).reshape(args.batch_size, blocks_per_row).to(torch.int32)

    nope_cpu = torch.randint(
        -3,
        4,
        (physical_blocks, BLOCK_SIZE, 1, NOPE_DIM),
        generator=generator,
        dtype=torch.int16,
    ).float().to(torch.float8_e4m3fn)
    rope_cpu = torch.empty(
        (physical_blocks, BLOCK_SIZE, 1, ROPE_DIM), dtype=torch.float32
    ).uniform_(-0.5, 0.5, generator=generator).to(torch.bfloat16)
    scales_cpu = torch.empty(
        (physical_blocks, BLOCK_SIZE, 1, SCALE_COUNT), dtype=torch.float32
    ).uniform_(0.02, 0.08, generator=generator)
    packed_cpu = pack_cache(nope_cpu, rope_cpu, scales_cpu)

    q_nope_cpu = torch.empty(
        (args.batch_size, args.heads, NOPE_DIM), dtype=torch.float32
    ).uniform_(-0.5, 0.5, generator=generator).to(torch.bfloat16)
    q_rope_cpu = torch.empty(
        (args.batch_size, args.heads, ROPE_DIM), dtype=torch.float32
    ).uniform_(-0.5, 0.5, generator=generator).to(torch.bfloat16)
    query_cpu = torch.cat((q_nope_cpu, q_rope_cpu), dim=-1).contiguous()

    slots_cpu = torch.full(
        (args.batch_size, 1, TOPK + args.max_tail_tokens),
        -1,
        dtype=torch.int32,
    )
    for row in range(args.batch_size):
        if args.cache_tokens == 0:
            slots_cpu[row, 0, : args.tail_tokens] = torch.arange(
                args.tail_tokens, dtype=torch.int32
            )
        else:
            slots_cpu[row, 0, :TOPK] = torch.randperm(
                args.cache_tokens, generator=generator
            )[:TOPK].to(torch.int32)
            slots_cpu[row, 0, TOPK : TOPK + args.tail_tokens] = torch.arange(
                args.cache_tokens, resident_len, dtype=torch.int32
            )

    return {
        "device": device,
        "query_cpu": query_cpu,
        "nope_cpu": nope_cpu,
        "rope_cpu": rope_cpu,
        "scales_cpu": scales_cpu,
        "packed": packed_cpu.to(device),
        "query": query_cpu.to(device),
        "slots_cpu": slots_cpu,
        "slots": slots_cpu.to(device),
        "block_table_cpu": block_table_cpu,
        "block_table": block_table_cpu.to(device),
        "actual_q": torch.arange(
            1, args.batch_size + 1, dtype=torch.int32, device=device
        ),
        "resident_lengths": torch.full(
            (args.batch_size,), resident_len, dtype=torch.int32, device=device
        ),
        "scale": QUERY_DIM**-0.5,
    }


def cpu_reference(
    inputs: dict[str, object], args: argparse.Namespace
) -> torch.Tensor:
    output = torch.empty(
        (args.batch_size, args.heads, NOPE_DIM), dtype=torch.float32
    )
    scales = inputs["scales_cpu"].repeat_interleave(TILE_SIZE, dim=-1)
    nope = inputs["nope_cpu"].float() * scales
    value = nope.to(torch.bfloat16).float()
    key = torch.cat(
        (nope.to(torch.bfloat16), inputs["rope_cpu"]), dim=-1
    ).float()
    query = inputs["query_cpu"].float()
    block_table = inputs["block_table_cpu"].to(torch.int64)
    valid_count = (
        args.tail_tokens
        if args.cache_tokens == 0
        else TOPK + args.tail_tokens
    )
    for row in range(args.batch_size):
        indices = inputs["slots_cpu"][row, 0, :valid_count].to(torch.int64)
        physical = block_table[row, indices // BLOCK_SIZE]
        offsets = indices % BLOCK_SIZE
        selected_key = key[physical, offsets, 0]
        selected_value = value[physical, offsets, 0]
        scores = query[row] @ selected_key.T * inputs["scale"]
        probabilities = torch.softmax(scores, dim=-1)
        output[row] = (
            probabilities.to(torch.bfloat16).float() @ selected_value
        )
    return output


def launch(inputs: dict[str, object]) -> torch.Tensor:
    return nanovllm_dsa_a5.sparse_tail_attention_c8(
        inputs["query"],
        inputs["packed"],
        inputs["slots"],
        inputs["block_table"],
        inputs["actual_q"],
        inputs["resident_lengths"],
        inputs["scale"],
    )


def check(inputs: dict[str, object], args: argparse.Namespace) -> None:
    expected = cpu_reference(inputs, args)
    actual = launch(inputs)
    torch.npu.synchronize()
    actual_cpu = actual.cpu().float()
    if not bool(torch.isfinite(actual_cpu).all()):
        raise AssertionError("C8 QSFA produced NaN or Inf")
    torch.testing.assert_close(actual_cpu, expected, atol=0.08, rtol=0.03)
    max_abs = float((actual_cpu - expected).abs().max())
    attended = (
        args.tail_tokens
        if args.cache_tokens == 0
        else TOPK + args.tail_tokens
    )
    print(
        "A5_SPARSE_TAIL_ATTENTION_C8_CHECK "
        f"batch={args.batch_size} heads={args.heads} "
        f"cache_tokens={args.cache_tokens} tail_tokens={args.tail_tokens} "
        f"attended_tokens={attended} max_abs={max_abs:.9f} "
        "finite=1 ok=1",
        flush=True,
    )


def benchmark(inputs: dict[str, object], args: argparse.Namespace) -> None:
    for _ in range(args.warmup):
        launch(inputs)
    torch.npu.synchronize()
    starts = [torch.npu.Event(enable_timing=True) for _ in range(args.iters)]
    ends = [torch.npu.Event(enable_timing=True) for _ in range(args.iters)]
    retained = []
    for start, end in zip(starts, ends):
        start.record()
        retained.append(launch(inputs))
        end.record()
    ends[-1].synchronize()
    avg_us = statistics.mean(
        start.elapsed_time(end) for start, end in zip(starts, ends)
    ) * 1000
    print(
        "A5_SPARSE_TAIL_ATTENTION_C8_RESULT "
        f"batch={args.batch_size} heads={args.heads} "
        f"cache_tokens={args.cache_tokens} tail_tokens={args.tail_tokens} "
        f"avg_us={avg_us:.3f} warmup={args.warmup} iters={args.iters}",
        flush=True,
    )


def check_meta(heads: int, max_tail_tokens: int) -> None:
    query = torch.empty((3, heads, QUERY_DIM), dtype=torch.bfloat16, device="meta")
    packed = torch.empty(
        (96, BLOCK_SIZE, 1, PACKED_DIM),
        dtype=torch.float8_e4m3fn,
        device="meta",
    )
    slots = torch.empty(
        (3, 1, TOPK + max_tail_tokens), dtype=torch.int32, device="meta"
    )
    table = torch.empty((3, 96), dtype=torch.int32, device="meta")
    lengths = torch.empty((3,), dtype=torch.int32, device="meta")
    output = nanovllm_dsa_a5.sparse_tail_attention_c8(
        query, packed, slots, table, lengths, lengths, 1.0
    )
    if tuple(output.shape) != (3, heads, NOPE_DIM) or output.dtype != query.dtype:
        raise AssertionError("C8 SFA Meta implementation returned wrong shape/dtype")
    print(
        f"A5_SPARSE_TAIL_ATTENTION_C8_META_CHECK heads={heads} ok=1",
        flush=True,
    )


def main() -> None:
    args = parse_args()
    validate_args(args)
    device = torch.device(args.device)
    if device.type != "npu":
        raise ValueError("--device must select an NPU")
    torch.npu.set_device(device)
    torch.npu.config.allow_internal_format = False
    device_name = require_a5(device, args.allow_non_a5)
    check_meta(args.heads, args.max_tail_tokens)
    print(
        "A5_SPARSE_TAIL_ATTENTION_C8_CONFIG "
        f"device={device} device_name={device_name!r} heads={args.heads} "
        f"batch_sizes={args.batch_sizes} cache_tokens={args.cache_tokens} "
        f"tail_tokens={args.tail_tokens} max_tail_tokens={args.max_tail_tokens}",
        flush=True,
    )

    # C8-specific representatives cover the packed dense row, sparse-only,
    # the common cache budget and the largest production budget/tail pair.
    mandatory = (
        (1, 0, 2048, 2048),
        (1, 2048, 0, 512),
        (1, 6144, 64, 512),
        (1, 12288, 257, 512),
    )
    for index, (batch, cache_tokens, tail_tokens, max_tail) in enumerate(mandatory):
        current = case_args(
            args,
            batch,
            cache_tokens,
            tail_tokens,
            max_tail,
            args.seed + 10 + index,
        )
        check(make_inputs(current), current)

    case_index = 0
    for batch in args.batch_sizes:
        for cache_tokens in args.cache_tokens:
            for tail_tokens in args.tail_tokens:
                current = case_args(
                    args,
                    batch,
                    cache_tokens,
                    tail_tokens,
                    args.max_tail_tokens,
                    args.seed + 1000 + case_index,
                )
                inputs = make_inputs(current)
                check(inputs, current)
                if args.iters > 0:
                    benchmark(inputs, current)
                case_index += 1
    print("A5_SPARSE_TAIL_ATTENTION_C8_UT_OK", flush=True)


if __name__ == "__main__":
    main()
