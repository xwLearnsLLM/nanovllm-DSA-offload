#!/usr/bin/env python3
"""End-to-end correctness test for C8 SFA Stage1/Stage2 merging."""

from __future__ import annotations

import argparse

import torch
import torch_npu  # type: ignore  # noqa: F401

import nanovllm_dsa_a5
from _c8_staged_attention_reference import (
    cpu_state,
    error_metrics,
    full_attention_slots,
    make_case,
)
from _utils import require_a5


ATOL = 0.08
RTOL = 0.03


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--allow-non-a5", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    if device.type != "npu":
        raise ValueError("--device must select an NPU")
    torch.npu.set_device(device)
    torch.npu.config.allow_internal_format = False
    require_a5(device, args.allow_non_a5)
    case = make_case(device=device, heads=args.heads, seed=args.seed)
    partial = torch.empty(
        (*case.query.shape[:-1], 512),
        dtype=torch.float32,
        device=device,
    )
    maximum = torch.empty(
        (1, case.query.shape[0], case.query.shape[1]),
        dtype=torch.float32,
        device=device,
    )
    denominator = torch.empty_like(maximum)
    output = torch.empty(
        (*case.query.shape[:-1], 512),
        dtype=case.query.dtype,
        device=device,
    )
    nanovllm_dsa_a5.sparse_tail_attention_c8_mtp_stage1(
        case.query,
        case.packed,
        case.actual_q,
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
    output_ptr = output.data_ptr()
    returned = nanovllm_dsa_a5.sparse_tail_attention_c8_mtp_stage2(
        case.query,
        case.packed,
        case.actual_q,
        case.resident_lengths,
        case.block_table,
        case.topk_slots,
        case.miss_counts,
        case.scale,
        partial,
        maximum,
        denominator,
        output,
    )
    if returned.data_ptr() != output_ptr:
        raise AssertionError("Stage2 did not return caller-owned attention_out")

    full_slots = full_attention_slots(case).to(device)
    baseline = nanovllm_dsa_a5.sparse_tail_attention_c8(
        case.query,
        case.packed,
        full_slots,
        case.block_table,
        case.actual_q,
        case.resident_lengths,
        case.scale,
    )
    torch.npu.synchronize()
    actual = output.cpu().float()
    baseline_cpu = baseline.cpu().float()
    expected = cpu_state(case, "full")[3]
    if not torch.isfinite(actual).all():
        raise AssertionError("Stage2 produced NaN or Inf")
    torch.testing.assert_close(actual, expected, atol=ATOL, rtol=RTOL)
    torch.testing.assert_close(actual, baseline_cpu, atol=ATOL, rtol=RTOL)

    # C is request metadata, not a compile-time constant. Exercise dense C=0
    # and sparse C=6144 in one batch, including per-request TND row mapping.
    variable_c_case = make_case(
        device=device,
        query_counts=(1, 1),
        heads=args.heads,
        cache_tokens=(0, 6144),
        final_tail_tokens=(257, 65),
        miss_counts=(0, 37),
        seed=args.seed + 10,
    )
    variable_p = torch.empty(
        (2, args.heads, 512), dtype=torch.float32, device=device
    )
    variable_m = torch.empty(
        (1, 2, args.heads), dtype=torch.float32, device=device
    )
    variable_l = torch.empty_like(variable_m)
    variable_out = torch.empty(
        (2, args.heads, 512),
        dtype=variable_c_case.query.dtype,
        device=device,
    )
    nanovllm_dsa_a5.sparse_tail_attention_c8_mtp_stage1(
        variable_c_case.query,
        variable_c_case.packed,
        variable_c_case.actual_q,
        variable_c_case.resident_lengths,
        variable_c_case.cache_tokens,
        variable_c_case.block_table,
        variable_c_case.topk_slots,
        variable_c_case.miss_counts,
        variable_c_case.scale,
        variable_p,
        variable_m,
        variable_l,
    )
    nanovllm_dsa_a5.sparse_tail_attention_c8_mtp_stage2(
        variable_c_case.query,
        variable_c_case.packed,
        variable_c_case.actual_q,
        variable_c_case.resident_lengths,
        variable_c_case.block_table,
        variable_c_case.topk_slots,
        variable_c_case.miss_counts,
        variable_c_case.scale,
        variable_p,
        variable_m,
        variable_l,
        variable_out,
    )
    variable_slots = full_attention_slots(variable_c_case).to(device)
    variable_baseline = nanovllm_dsa_a5.sparse_tail_attention_c8(
        variable_c_case.query,
        variable_c_case.packed,
        variable_slots,
        variable_c_case.block_table,
        variable_c_case.actual_q,
        variable_c_case.resident_lengths,
        variable_c_case.scale,
    )
    torch.npu.synchronize()
    variable_actual = variable_out.cpu().float()
    variable_expected = cpu_state(variable_c_case, "full")[3]
    torch.testing.assert_close(
        variable_actual, variable_expected, atol=ATOL, rtol=RTOL
    )
    torch.testing.assert_close(
        variable_actual,
        variable_baseline.cpu().float(),
        atol=ATOL,
        rtol=RTOL,
    )

    replay = output.clone()
    nanovllm_dsa_a5.sparse_tail_attention_c8_mtp_stage2(
        case.query,
        case.packed,
        case.actual_q,
        case.resident_lengths,
        case.block_table,
        case.topk_slots,
        case.miss_counts,
        case.scale,
        partial,
        maximum,
        denominator,
        replay,
    )
    torch.npu.synchronize()
    torch.testing.assert_close(
        replay.cpu(), output.cpu(), atol=0.001, rtol=0.001
    )

    poison_case = make_case(
        device=device,
        query_counts=(1,),
        heads=args.heads,
        cache_tokens=(2048,),
        final_tail_tokens=(5,),
        miss_counts=(37,),
        seed=args.seed + 1,
    )
    poison_p = torch.empty(
        (1, args.heads, 512), dtype=torch.float32, device=device
    )
    poison_m = torch.empty(
        (1, 1, args.heads), dtype=torch.float32, device=device
    )
    poison_l = torch.empty_like(poison_m)
    nanovllm_dsa_a5.sparse_tail_attention_c8_mtp_stage1(
        poison_case.query,
        poison_case.packed,
        poison_case.actual_q,
        poison_case.resident_lengths,
        poison_case.cache_tokens,
        poison_case.block_table,
        poison_case.topk_slots,
        poison_case.miss_counts,
        poison_case.scale,
        poison_p,
        poison_m,
        poison_l,
    )
    poisoned_cpu = poison_case.packed_cpu.clone()
    hit_slots = poison_case.topk_slots_cpu[0, 0, 37:].to(torch.int64)
    tail_slots = torch.arange(2048, 2053, dtype=torch.int64)
    ignored_slots = torch.cat((hit_slots, tail_slots))
    physical = poison_case.block_table_cpu[
        0, ignored_slots // 128
    ].to(torch.int64)
    offsets = ignored_slots % 128
    poisoned_cpu[physical, offsets, 0] = 0
    poison_original = torch.empty(
        (1, args.heads, 512),
        dtype=poison_case.query.dtype,
        device=device,
    )
    poison_replay = torch.empty_like(poison_original)
    nanovllm_dsa_a5.sparse_tail_attention_c8_mtp_stage2(
        poison_case.query,
        poison_case.packed,
        poison_case.actual_q,
        poison_case.resident_lengths,
        poison_case.block_table,
        poison_case.topk_slots,
        poison_case.miss_counts,
        poison_case.scale,
        poison_p,
        poison_m,
        poison_l,
        poison_original,
    )
    nanovllm_dsa_a5.sparse_tail_attention_c8_mtp_stage2(
        poison_case.query,
        poisoned_cpu.to(device),
        poison_case.actual_q,
        poison_case.resident_lengths,
        poison_case.block_table,
        poison_case.topk_slots,
        poison_case.miss_counts,
        poison_case.scale,
        poison_p,
        poison_m,
        poison_l,
        poison_replay,
    )
    torch.npu.synchronize()
    torch.testing.assert_close(
        poison_replay.cpu(), poison_original.cpu(), atol=0.001, rtol=0.001
    )
    max_abs, max_rel, mean_abs = error_metrics(actual, expected)
    print(
        "A5_SPARSE_TAIL_ATTENTION_C8_STAGE2_OK "
        f"max_abs={max_abs:.9f} max_rel={max_rel:.9f} "
        f"mean_abs={mean_abs:.9f} finite_rate=1.0",
        flush=True,
    )


if __name__ == "__main__":
    main()
