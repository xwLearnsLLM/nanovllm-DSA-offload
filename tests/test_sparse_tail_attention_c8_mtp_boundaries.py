#!/usr/bin/env python3
"""Compare staged and single C8 SFA across MTP boundary conditions."""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import torch
import torch_npu  # type: ignore  # noqa: F401

import nanovllm_dsa_a5
from _c8_staged_attention_reference import (
    full_attention_slots,
    make_case,
)
from _utils import require_a5


ATOL = 0.08
RTOL = 0.03


@dataclass(frozen=True)
class BoundaryCase:
    name: str
    query_counts: tuple[int, ...]
    cache_tokens: tuple[int, ...]
    final_tail_tokens: tuple[int, ...]
    miss_counts: tuple[int, ...]


FIXED_CASES = (
    BoundaryCase(
        name="mixed_mtp_2_3_4",
        query_counts=(2, 3, 4),
        cache_tokens=(0, 2048, 6144),
        final_tail_tokens=(257, 2, 3),
        miss_counts=(0, 0, 0, 1, 2048, 2047, 37, 2048, 0),
    ),
    BoundaryCase(
        name="tail_block_minus_one",
        query_counts=(2,),
        cache_tokens=(2048,),
        final_tail_tokens=(127,),
        miss_counts=(1, 2048),
    ),
    BoundaryCase(
        name="tail_block_exact",
        query_counts=(3,),
        cache_tokens=(6144,),
        final_tail_tokens=(128,),
        miss_counts=(2048, 0, 2047),
    ),
    BoundaryCase(
        name="tail_block_plus_one",
        query_counts=(4,),
        cache_tokens=(2048,),
        final_tail_tokens=(129,),
        miss_counts=(0, 1, 37, 2048),
    ),
    BoundaryCase(
        name="all_miss_minimum_mtp_tail",
        query_counts=(4,),
        cache_tokens=(2048,),
        final_tail_tokens=(3,),
        miss_counts=(2048, 2048, 2048, 2048),
    ),
    BoundaryCase(
        name="all_hit_minimum_mtp_tail",
        query_counts=(4,),
        cache_tokens=(6144,),
        final_tail_tokens=(3,),
        miss_counts=(0, 0, 0, 0),
    ),
)


QUERY_DTYPES = {
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
}
LENGTH_DTYPES = {
    "int32": torch.int32,
}
MISS_BOUNDARIES = (0, 1, 37, 2047, 2048)


def parse_csv(raw: str, choices: dict[str, object], name: str) -> list[str]:
    values = [value.strip() for value in raw.split(",") if value.strip()]
    if not values or any(value not in choices for value in values):
        allowed = ",".join(choices)
        raise ValueError(f"{name} must be a comma list drawn from {allowed}")
    return values


def parse_heads(raw: str) -> list[int]:
    heads = [int(value) for value in raw.split(",") if value.strip()]
    if not heads or any(head < 1 or head > 64 for head in heads):
        raise ValueError("--heads values must satisfy 1 <= heads <= 64")
    return heads


def parse_batch_sizes(raw: str) -> list[int]:
    batch_sizes = [int(value) for value in raw.split(",") if value.strip()]
    if not batch_sizes or any(batch_size < 1 for batch_size in batch_sizes):
        raise ValueError("--batch-sizes values must be positive")
    return batch_sizes


def make_variable_batch_case(batch_size: int) -> BoundaryCase:
    query_counts = tuple(index % 5 for index in range(batch_size))
    # A batch cannot contain only empty requests because native attention
    # requires T > 0. Keep BS=1 useful while BS>=2 still exercises Q_b=0.
    if sum(query_counts) == 0:
        query_counts = (4,)

    cache_pattern = (0, 2048, 6144)
    sparse_tail_pattern = (3, 127, 128, 129)
    cache_tokens = tuple(
        cache_pattern[index % len(cache_pattern)]
        for index in range(batch_size)
    )
    final_tail_tokens = tuple(
        257
        if cache_tokens[index] == 0
        else max(
            query_counts[index] - 1,
            sparse_tail_pattern[index % len(sparse_tail_pattern)],
        )
        for index in range(batch_size)
    )
    miss_counts: list[int] = []
    for request, query_count in enumerate(query_counts):
        if cache_tokens[request] == 0:
            miss_counts.extend([0] * query_count)
            continue
        miss_counts.extend(
            MISS_BOUNDARIES[(request + query) % len(MISS_BOUNDARIES)]
            for query in range(query_count)
        )
    return BoundaryCase(
        name=f"variable_batch_{batch_size}",
        query_counts=query_counts,
        cache_tokens=cache_tokens,
        final_tail_tokens=final_tail_tokens,
        miss_counts=tuple(miss_counts),
    )


def comparison_metrics(
    actual: torch.Tensor,
    expected: torch.Tensor,
) -> tuple[float, float, float]:
    absolute = (actual.float() - expected.float()).abs()
    tolerance = ATOL + RTOL * expected.float().abs()
    return (
        float(absolute.max()),
        float((absolute / tolerance).max()),
        float(absolute.mean()),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--heads", default="8,64")
    parser.add_argument("--batch-sizes", default="1,2,4,8")
    parser.add_argument("--query-dtypes", default="bf16")
    parser.add_argument("--length-dtypes", default="int32")
    parser.add_argument("--seed", type=int, default=31)
    parser.add_argument("--allow-non-a5", action="store_true")
    return parser.parse_args()


def run_case(
    *,
    spec: BoundaryCase,
    device: torch.device,
    heads: int,
    query_dtype_name: str,
    length_dtype_name: str,
    seed: int,
) -> tuple[float, float, float]:
    case = make_case(
        device=device,
        query_counts=spec.query_counts,
        heads=heads,
        cache_tokens=spec.cache_tokens,
        final_tail_tokens=spec.final_tail_tokens,
        miss_counts=spec.miss_counts,
        query_dtype=QUERY_DTYPES[query_dtype_name],
        seed=seed,
    )
    actual_q = case.actual_q.to(LENGTH_DTYPES[length_dtype_name])
    partial = torch.empty(
        (*case.query.shape[:-1], 512),
        dtype=torch.float32,
        device=device,
    )
    maximum = torch.empty(
        (1, case.query.shape[0], heads),
        dtype=torch.float32,
        device=device,
    )
    denominator = torch.empty_like(maximum)
    staged = torch.empty(
        (*case.query.shape[:-1], 512),
        dtype=case.query.dtype,
        device=device,
    )

    nanovllm_dsa_a5.sparse_tail_attention_c8_stage1_out(
        case.query,
        case.packed,
        actual_q,
        case.resident_lengths,
        case.cache_tokens,
        case.block_table,
        case.topk_slots,
        case.miss_counts,
        case.scale,
        partial,
        maximum,
        denominator,
    )
    returned = nanovllm_dsa_a5.sparse_tail_attention_c8_stage2_out(
        case.query,
        case.packed,
        actual_q,
        case.resident_lengths,
        case.block_table,
        case.topk_slots,
        case.miss_counts,
        case.scale,
        partial,
        maximum,
        denominator,
        staged,
    )
    if returned.data_ptr() != staged.data_ptr():
        raise AssertionError("Stage2 did not return caller-owned output")

    single = nanovllm_dsa_a5.sparse_tail_attention_c8(
        case.query,
        case.packed,
        full_attention_slots(case).to(device),
        case.block_table,
        actual_q,
        case.resident_lengths,
        case.scale,
    )
    torch.npu.synchronize()
    staged_cpu = staged.cpu().float()
    single_cpu = single.cpu().float()
    if not torch.isfinite(staged_cpu).all():
        raise AssertionError("staged C8 SFA produced NaN or Inf")
    if not torch.isfinite(single_cpu).all():
        raise AssertionError("single C8 SFA produced NaN or Inf")
    try:
        torch.testing.assert_close(
            staged_cpu,
            single_cpu,
            atol=ATOL,
            rtol=RTOL,
        )
    except AssertionError as error:
        context = (
            f"case={spec.name} heads={heads} query_dtype={query_dtype_name} "
            f"length_dtype={length_dtype_name}"
        )
        raise AssertionError(context) from error
    return comparison_metrics(staged_cpu, single_cpu)


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    if device.type != "npu":
        raise ValueError("--device must select an NPU")
    torch.npu.set_device(device)
    torch.npu.config.allow_internal_format = False
    require_a5(device, args.allow_non_a5)

    heads_values = parse_heads(args.heads)
    batch_sizes = parse_batch_sizes(args.batch_sizes)
    query_dtype_names = parse_csv(
        args.query_dtypes,
        QUERY_DTYPES,
        "--query-dtypes",
    )
    length_dtype_names = parse_csv(
        args.length_dtypes,
        LENGTH_DTYPES,
        "--length-dtypes",
    )
    runs = 0
    worst_abs = 0.0
    worst_tolerance_ratio = 0.0
    worst_mean = 0.0
    cases = (*FIXED_CASES, *(make_variable_batch_case(bs) for bs in batch_sizes))
    for case_index, spec in enumerate(cases):
        for heads in heads_values:
            for query_dtype_name in query_dtype_names:
                for length_dtype_name in length_dtype_names:
                    metrics = run_case(
                        spec=spec,
                        device=device,
                        heads=heads,
                        query_dtype_name=query_dtype_name,
                        length_dtype_name=length_dtype_name,
                        seed=args.seed + case_index,
                    )
                    max_abs, max_tolerance_ratio, mean_abs = metrics
                    worst_abs = max(worst_abs, max_abs)
                    worst_tolerance_ratio = max(
                        worst_tolerance_ratio,
                        max_tolerance_ratio,
                    )
                    worst_mean = max(worst_mean, mean_abs)
                    runs += 1
                    print(
                        "A5_C8_STAGED_MTP_CASE_OK "
                        f"case={spec.name} heads={heads} "
                        f"query_dtype={query_dtype_name} "
                        f"length_dtype={length_dtype_name} "
                        f"max_abs={max_abs:.9f} "
                        f"max_tolerance_ratio={max_tolerance_ratio:.9f} "
                        f"mean_abs={mean_abs:.9f}",
                        flush=True,
                    )
    print(
        "A5_C8_STAGED_MTP_BOUNDARIES_OK "
        f"runs={runs} worst_abs={worst_abs:.9f} "
        f"worst_tolerance_ratio={worst_tolerance_ratio:.9f} "
        f"worst_mean={worst_mean:.9f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
