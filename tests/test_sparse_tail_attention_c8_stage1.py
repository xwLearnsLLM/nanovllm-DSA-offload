#!/usr/bin/env python3
"""Correctness test for caller-owned C8 SFA Stage1 P/M/L."""

from __future__ import annotations

import argparse

import torch
import torch_npu  # type: ignore  # noqa: F401

import nanovllm_dsa_a5
from _c8_staged_attention_reference import cpu_state, error_metrics, make_case
from _utils import require_a5


ATOL = 0.08
RTOL = 0.03


def check_meta(heads: int) -> None:
    query = torch.empty(
        (3, heads, 576), dtype=torch.bfloat16, device="meta"
    )
    packed = torch.empty(
        (17, 128, 1, 656), dtype=torch.float8_e4m3fn, device="meta"
    )
    actual_q = torch.empty((1,), dtype=torch.int32, device="meta")
    resident = torch.empty((1,), dtype=torch.int32, device="meta")
    cache_tokens = torch.empty((1,), dtype=torch.int32, device="meta")
    table = torch.empty((1, 17), dtype=torch.int32, device="meta")
    slots = torch.empty((3, 1, 2048), dtype=torch.int32, device="meta")
    misses = torch.empty((3,), dtype=torch.int32, device="meta")
    partial = torch.empty((3, heads, 512), device="meta")
    maximum = torch.empty((1, 3, heads), device="meta")
    denominator = torch.empty_like(maximum)
    outputs = nanovllm_dsa_a5.sparse_tail_attention_c8_stage1_out(
        query,
        packed,
        actual_q,
        resident,
        cache_tokens,
        table,
        slots,
        misses,
        1.0,
        partial,
        maximum,
        denominator,
    )
    expected_outputs = (partial, maximum, denominator)
    if any(
        actual is not expected and actual._cdata != expected._cdata
        for actual, expected in zip(outputs, expected_outputs)
    ):
        raise AssertionError("Stage1 Meta outputs must alias caller buffers")
    attention = torch.empty(
        (3, heads, 512), dtype=torch.bfloat16, device="meta"
    )
    returned = nanovllm_dsa_a5.sparse_tail_attention_c8_stage2_out(
        query,
        packed,
        actual_q,
        resident,
        table,
        slots,
        misses,
        1.0,
        partial,
        maximum,
        denominator,
        attention,
    )
    if returned is not attention and returned._cdata != attention._cdata:
        raise AssertionError("Stage2 Meta output must alias caller buffer")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=17)
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
    check_meta(args.heads)
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
    pointers = (partial.data_ptr(), maximum.data_ptr(), denominator.data_ptr())
    outputs = nanovllm_dsa_a5.sparse_tail_attention_c8_stage1_out(
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
    torch.npu.synchronize()
    if tuple(tensor.data_ptr() for tensor in outputs) != pointers:
        raise AssertionError("Stage1 did not return its caller-owned buffers")
    expected_p, expected_m, expected_l, expected_o = cpu_state(case, "stage1")
    actual_p = partial.cpu()
    actual_m = maximum.cpu()
    actual_l = denominator.cpu()
    nonempty = torch.isfinite(expected_m)
    empty = ~nonempty
    if not torch.equal(actual_p[empty.squeeze(0)], torch.zeros_like(actual_p[empty.squeeze(0)])):
        raise AssertionError("empty Stage1 P must be zero")
    if not torch.equal(actual_l[empty], torch.zeros_like(actual_l[empty])):
        raise AssertionError("empty Stage1 L must be zero")
    if not torch.isneginf(actual_m[empty]).all():
        raise AssertionError("empty Stage1 M must be -inf")
    torch.testing.assert_close(
        actual_m[nonempty], expected_m[nonempty], atol=ATOL, rtol=RTOL
    )
    torch.testing.assert_close(
        actual_l[nonempty], expected_l[nonempty], atol=ATOL, rtol=RTOL
    )
    torch.testing.assert_close(actual_p, expected_p, atol=ATOL, rtol=RTOL)
    normalized = torch.where(
        actual_l.squeeze(0).unsqueeze(-1) > 0,
        actual_p / actual_l.squeeze(0).unsqueeze(-1),
        torch.zeros_like(actual_p),
    )
    torch.testing.assert_close(normalized, expected_o, atol=ATOL, rtol=RTOL)
    max_abs, max_rel, mean_abs = error_metrics(normalized, expected_o)

    empty_case = make_case(
        device=device,
        query_counts=(1,),
        heads=args.heads,
        cache_tokens=(2048,),
        final_tail_tokens=(0,),
        miss_counts=(2048,),
        seed=args.seed + 1,
    )
    empty_p = torch.empty(
        (1, args.heads, 512), dtype=torch.float32, device=device
    )
    empty_m = torch.empty(
        (1, 1, args.heads), dtype=torch.float32, device=device
    )
    empty_l = torch.empty_like(empty_m)
    nanovllm_dsa_a5.sparse_tail_attention_c8_stage1_out(
        empty_case.query,
        empty_case.packed,
        empty_case.actual_q,
        empty_case.resident_lengths,
        empty_case.cache_tokens,
        empty_case.block_table,
        empty_case.topk_slots,
        empty_case.miss_counts,
        empty_case.scale,
        empty_p,
        empty_m,
        empty_l,
    )
    torch.npu.synchronize()
    if not torch.equal(empty_p.cpu(), torch.zeros_like(empty_p.cpu())):
        raise AssertionError("all-miss/no-tail Stage1 P must be zero")
    if not torch.equal(empty_l.cpu(), torch.zeros_like(empty_l.cpu())):
        raise AssertionError("all-miss/no-tail Stage1 L must be zero")
    if not torch.isneginf(empty_m.cpu()).all():
        raise AssertionError("all-miss/no-tail Stage1 M must be -inf")

    poison_case = make_case(
        device=device,
        query_counts=(1,),
        heads=args.heads,
        cache_tokens=(2048,),
        final_tail_tokens=(5,),
        miss_counts=(37,),
        seed=args.seed + 2,
    )
    poison_p = torch.empty(
        (1, args.heads, 512), dtype=torch.float32, device=device
    )
    poison_m = torch.empty(
        (1, 1, args.heads), dtype=torch.float32, device=device
    )
    poison_l = torch.empty_like(poison_m)
    nanovllm_dsa_a5.sparse_tail_attention_c8_stage1_out(
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
    miss_slots = poison_case.topk_slots_cpu[0, 0, :37].to(torch.int64)
    physical = poison_case.block_table_cpu[0, miss_slots // 128].to(torch.int64)
    offsets = miss_slots % 128
    poisoned_cpu[physical, offsets, 0] = 0
    poison_p_replay = torch.empty_like(poison_p)
    poison_m_replay = torch.empty_like(poison_m)
    poison_l_replay = torch.empty_like(poison_l)
    nanovllm_dsa_a5.sparse_tail_attention_c8_stage1_out(
        poison_case.query,
        poisoned_cpu.to(device),
        poison_case.actual_q,
        poison_case.resident_lengths,
        poison_case.cache_tokens,
        poison_case.block_table,
        poison_case.topk_slots,
        poison_case.miss_counts,
        poison_case.scale,
        poison_p_replay,
        poison_m_replay,
        poison_l_replay,
    )
    torch.npu.synchronize()
    torch.testing.assert_close(
        poison_p_replay.cpu(), poison_p.cpu(), atol=0.001, rtol=0.001
    )
    torch.testing.assert_close(
        poison_m_replay.cpu(), poison_m.cpu(), atol=0.001, rtol=0.001
    )
    torch.testing.assert_close(
        poison_l_replay.cpu(), poison_l.cpu(), atol=0.001, rtol=0.001
    )
    print(
        "A5_SPARSE_TAIL_ATTENTION_C8_STAGE1_OK "
        f"max_abs={max_abs:.9f} max_rel={max_rel:.9f} "
        f"mean_abs={mean_abs:.9f} finite_rate=1.0",
        flush=True,
    )


if __name__ == "__main__":
    main()
