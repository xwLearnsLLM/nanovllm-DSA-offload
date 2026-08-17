#!/usr/bin/env python3
"""Semantic/performance test for local A5 packed-C8 MTP topK+tail attention.

Each request packs 1..4 verification queries (root + drafts). Every global
query row r = prefix_b + i attends its OWN top-2048 HBM slots plus the causal
dense tail C, C+1, ..., C+i, encoded as [topk || tail || -1 pad] in one
sparse_and_tail_slots row.
"""

from __future__ import annotations

import argparse
import statistics
from pathlib import Path

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
MAX_QUERIES_PER_REQUEST = 4


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--batch-sizes", type=csv_ints, default=csv_ints("24"))
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--query-counts", type=csv_ints, default=csv_ints("4"))
    parser.add_argument("--cache-tokens", type=csv_ints, default=csv_ints("6144"))
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
        not 1 <= query_count <= MAX_QUERIES_PER_REQUEST
        for query_count in args.query_counts
    ):
        raise ValueError("query counts must be in [1,4]")
    if any(
        cache_tokens != 0
        and (cache_tokens < TOPK or cache_tokens % BLOCK_SIZE)
        for cache_tokens in args.cache_tokens
    ):
        raise ValueError("cache tokens must be 0 or block-aligned >= 2048")
    if args.warmup < 0 or args.iters <= 0:
        raise ValueError("warmup must be non-negative and iters positive")


def case_args(
    args: argparse.Namespace,
    batch_size: int,
    query_counts: tuple[int, ...],
    cache_tokens: int,
    seed: int,
) -> argparse.Namespace:
    values = vars(args).copy()
    values.update(
        batch_size=batch_size,
        query_counts=query_counts,
        cache_tokens=cache_tokens,
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
    query_counts = tuple(args.query_counts)
    batch = args.batch_size
    total_rows = sum(query_counts)
    max_count = max(query_counts)
    resident_lens = tuple(args.cache_tokens + count for count in query_counts)
    max_resident = max(resident_lens)
    blocks_per_row = (max_resident + BLOCK_SIZE - 1) // BLOCK_SIZE
    physical_blocks = batch * blocks_per_row
    block_table_cpu = torch.randperm(
        physical_blocks, generator=generator, dtype=torch.int64
    ).reshape(batch, blocks_per_row).to(torch.int32)

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
        (total_rows, args.heads, NOPE_DIM), dtype=torch.float32
    ).uniform_(-0.5, 0.5, generator=generator).to(torch.bfloat16)
    q_rope_cpu = torch.empty(
        (total_rows, args.heads, ROPE_DIM), dtype=torch.float32
    ).uniform_(-0.5, 0.5, generator=generator).to(torch.bfloat16)
    query_cpu = torch.cat((q_nope_cpu, q_rope_cpu), dim=-1).contiguous()

    slots_cpu = torch.full(
        (total_rows, 1, TOPK + max_count), -1, dtype=torch.int32
    )
    for request in range(batch):
        cache_tokens = args.cache_tokens
        for path in range(query_counts[request]):
            row = sum(query_counts[:request]) + path
            if cache_tokens == 0:
                slots_cpu[row, 0, : path + 1] = torch.arange(
                    path + 1, dtype=torch.int32
                )
            else:
                # Per-path independent top-2048 selection.
                slots_cpu[row, 0, :TOPK] = torch.randperm(
                    cache_tokens, generator=generator
                )[:TOPK].to(torch.int32)
                slots_cpu[row, 0, TOPK : TOPK + path + 1] = torch.arange(
                    cache_tokens, cache_tokens + path + 1, dtype=torch.int32
                )

    return {
        "device": device,
        "query_counts": query_counts,
        "total_rows": total_rows,
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
        "actual_q": torch.tensor(
            [sum(query_counts[: b + 1]) for b in range(batch)],
            dtype=torch.int32,
            device=device,
        ),
        "resident_lengths": torch.tensor(
            resident_lens, dtype=torch.int32, device=device
        ),
        "scale": QUERY_DIM**-0.5,
    }


def cpu_reference(
    inputs: dict[str, object], args: argparse.Namespace
) -> torch.Tensor:
    query_counts: tuple[int, ...] = inputs["query_counts"]
    total_rows: int = inputs["total_rows"]
    output = torch.empty(
        (total_rows, args.heads, NOPE_DIM), dtype=torch.float32
    )
    scales = inputs["scales_cpu"].repeat_interleave(TILE_SIZE, dim=-1)
    nope = inputs["nope_cpu"].float() * scales
    value = nope.to(torch.bfloat16).float()
    key = torch.cat(
        (nope.to(torch.bfloat16), inputs["rope_cpu"]), dim=-1
    ).float()
    query = inputs["query_cpu"].float()
    block_table = inputs["block_table_cpu"].to(torch.int64)
    for row in range(total_rows):
        request = next(
            b
            for b in range(len(query_counts))
            if row < sum(query_counts[: b + 1])
        )
        path = row - sum(query_counts[:request])
        cache_tokens = args.cache_tokens
        # Row (request, path): own top-2048 plus causal tail C..C+path.
        valid_count = (
            path + 1 if cache_tokens == 0 else TOPK + path + 1
        )
        indices = inputs["slots_cpu"][row, 0, :valid_count].to(torch.int64)
        physical = block_table[request, indices // BLOCK_SIZE]
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
    return nanovllm_dsa_a5.sparse_tail_attention_mtp_c8(
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
        raise AssertionError("C8 MTP QSFA produced NaN or Inf")
    torch.testing.assert_close(actual_cpu, expected, atol=0.08, rtol=0.03)
    max_abs = float((actual_cpu - expected).abs().max())
    query_counts: tuple[int, ...] = inputs["query_counts"]
    print(
        "A5_SPARSE_TAIL_ATTENTION_MTP_C8_CHECK "
        f"batch={args.batch_size} query_counts={query_counts} "
        f"heads={args.heads} cache_tokens={args.cache_tokens} "
        f"max_abs={max_abs:.9f} finite=1 ok=1",
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
        "A5_SPARSE_TAIL_ATTENTION_MTP_C8_RESULT "
        f"batch={args.batch_size} query_counts={tuple(args.query_counts)} "
        f"heads={args.heads} cache_tokens={args.cache_tokens} "
        f"avg_us={avg_us:.3f} warmup={args.warmup} iters={args.iters}",
        flush=True,
    )


def check_meta(heads: int) -> None:
    query = torch.empty((5, heads, QUERY_DIM), dtype=torch.bfloat16, device="meta")
    packed = torch.empty(
        (96, BLOCK_SIZE, 1, PACKED_DIM),
        dtype=torch.float8_e4m3fn,
        device="meta",
    )
    slots = torch.empty(
        (5, 1, TOPK + MAX_QUERIES_PER_REQUEST),
        dtype=torch.int32,
        device="meta",
    )
    table = torch.empty((2, 96), dtype=torch.int32, device="meta")
    actual_q = torch.tensor([3, 5], dtype=torch.int32, device="meta")
    lengths = torch.tensor([2051, 2051], dtype=torch.int32, device="meta")
    output = nanovllm_dsa_a5.sparse_tail_attention_mtp_c8(
        query, packed, slots, table, actual_q, lengths, 1.0
    )
    if tuple(output.shape) != (5, heads, NOPE_DIM) or output.dtype != query.dtype:
        raise AssertionError("C8 MTP SFA Meta implementation returned wrong shape/dtype")
    print(
        f"A5_SPARSE_TAIL_ATTENTION_MTP_C8_META_CHECK heads={heads} ok=1",
        flush=True,
    )


def check_local_kernel_registration() -> None:
    vendor = Path(nanovllm_dsa_a5.local_opapi_path()).resolve().parents[2]
    metadata = tuple(
        vendor.glob(
            "op_impl/ai_core/tbe/kernel/config/**/binary_info_config.json"
        )
    )
    if not metadata or not any(
        "A5SparseTailAttentionMtpC8" in path.read_text(
            encoding="utf-8", errors="ignore"
        )
        for path in metadata
    ):
        raise AssertionError(
            "local A5SparseTailAttentionMtpC8 kernel metadata is missing; "
            "run bash build_c8.sh"
        )
    if not torch._C._dispatch_has_kernel_for_dispatch_key(
        "nanovllm_dsa::sparse_tail_attention_mtp_c8", "PrivateUse1"
    ):
        raise AssertionError(
            "sparse_tail_attention_mtp_c8 has no C++ PrivateUse1 kernel"
        )
    print(
        "A5_SPARSE_TAIL_ATTENTION_MTP_C8_LOCAL_KERNEL_CHECK "
        "cann_op=A5SparseTailAttentionMtpC8 privateuse1=1 python_custom_op=0 "
        "ok=1",
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
    check_meta(args.heads)
    check_local_kernel_registration()
    print(
        "A5_SPARSE_TAIL_ATTENTION_MTP_C8_CONFIG "
        f"device={device} device_name={device_name!r} heads={args.heads} "
        f"batch_sizes={args.batch_sizes} query_counts={args.query_counts} "
        f"cache_tokens={args.cache_tokens}",
        flush=True,
    )

    # Gate coverage: dense C=0 with the full 4-path causal tail, sparse-only
    # minimum cache, the common cache budget with a middle path count, the
    # production budget, and a mixed per-request count batch (TND rows with
    # non-uniform diffs, exercising the s1>1 branch with different counts).
    mandatory = (
        (1, (4,), 0),
        (1, (2,), 2048),
        (1, (3,), 6144),
        (1, (4,), 12288),
        (2, (2, 3), 4096),
        (3, (1, 4, 2), 6144),
    )
    for index, (batch, query_counts, cache_tokens) in enumerate(mandatory):
        current = case_args(
            args, batch, query_counts, cache_tokens, args.seed + 10 + index
        )
        check(make_inputs(current), current)

    case_index = 0
    for batch in args.batch_sizes:
        for query_count in args.query_counts:
            for cache_tokens in args.cache_tokens:
                current = case_args(
                    args,
                    batch,
                    (query_count,) * batch,
                    cache_tokens,
                    args.seed + 1000 + case_index,
                )
                inputs = make_inputs(current)
                check(inputs, current)
                benchmark(inputs, current)
                case_index += 1
    print("A5_SPARSE_TAIL_ATTENTION_MTP_C8_UT_OK", flush=True)


if __name__ == "__main__":
    main()
