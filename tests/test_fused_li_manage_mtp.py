#!/usr/bin/env python3
"""Correctness test for the BF16/FP16 fused MTP LI manager."""

import argparse
import random

import torch
import torch_npu  # type: ignore  # noqa: F401

import nanovllm_dsa_a5

from _utils import require_a5


TOPK = 2048
CACHE_SIZE = 8192
CACHE_ROW_CAPACITY = 262144
MAX_MTP_QUERIES = 4


def parse_args():
    parser = argparse.ArgumentParser(
        description="Validate request-level MTP TopK union and cache update on A5."
    )
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--bs", type=int, default=24)
    parser.add_argument("--min-seqlen", type=int, default=32768)
    parser.add_argument("--max-seqlen", type=int, default=65536)
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument("--q-heads", type=int, choices=(32, 64), default=64)
    parser.add_argument(
        "--queries-per-request",
        type=int,
        choices=range(0, MAX_MTP_QUERIES + 1),
        default=0,
        help="0 cycles through 1,2,3,4; otherwise fixes Q for every request",
    )
    parser.add_argument("--dtype", choices=("bf16", "fp16"), default="bf16")
    parser.add_argument("--min-miss-count", type=int, default=0)
    parser.add_argument("--max-miss-count", type=int, default=300)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--allow-non-a5", action="store_true")
    return parser.parse_args()


def validate_args(args):
    if args.bs <= 0:
        raise ValueError("bs must be positive")
    if not (CACHE_SIZE + args.max_miss_count <= args.min_seqlen <= args.max_seqlen):
        raise ValueError(
            "require 8192 + max_miss_count <= min_seqlen <= max_seqlen"
        )
    if args.max_seqlen > CACHE_ROW_CAPACITY:
        raise ValueError("max-seqlen must be <= 262144")
    if args.block_size <= 0 or args.block_size % 16 != 0:
        raise ValueError("block-size must be a positive multiple of 16")
    if not (0 <= args.min_miss_count <= args.max_miss_count <= CACHE_SIZE):
        raise ValueError("require 0 <= min_miss_count <= max_miss_count <= 8192")


def find_native_op():
    op = getattr(torch_npu, "npu_lightning_indexer", None)
    if op is not None:
        return op
    namespace = getattr(torch.ops, "_C_ascend", None)
    op = getattr(namespace, "npu_lightning_indexer", None) if namespace else None
    if op is None:
        raise RuntimeError("the installed vLLM/torch_npu LightningIndexer is unavailable")
    return op


def extract_indices(output, packed_tokens):
    if isinstance(output, torch.Tensor):
        candidates = (output,)
    elif isinstance(output, (tuple, list)):
        candidates = tuple(output)
    else:
        candidates = ()
    for tensor in candidates:
        if (
            isinstance(tensor, torch.Tensor)
            and tensor.dtype == torch.int32
            and tensor.numel() == packed_tokens * TOPK
        ):
            return tensor.reshape(packed_tokens, TOPK)
    raise RuntimeError("native LightningIndexer returned no [T, 1, 2048] INT32 output")


def make_inputs(args, device):
    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)
    torch.npu.manual_seed_all(args.seed)
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16

    query_counts = (
        [(row % MAX_MTP_QUERIES) + 1 for row in range(args.bs)]
        if args.queries_per_request == 0
        else [args.queries_per_request] * args.bs
    )
    final_seqlens = [
        rng.randint(args.min_seqlen, args.max_seqlen) for _ in range(args.bs)
    ]
    cumulative_q = []
    packed_tokens = 0
    for count in query_counts:
        packed_tokens += count
        cumulative_q.append(packed_tokens)

    blocks_per_row = (max(final_seqlens) + args.block_size - 1) // args.block_size
    block_table = torch.arange(
        args.bs * blocks_per_row, dtype=torch.int32, device=device
    ).reshape(args.bs, blocks_per_row)
    key = torch.empty(
        (args.bs * blocks_per_row, args.block_size, 1, 128),
        dtype=dtype,
        device=device,
    ).uniform_(-1, 1)
    query = torch.empty(
        (packed_tokens, args.q_heads, 128), dtype=dtype, device=device
    ).uniform_(-1, 1)
    weights = torch.empty(
        (packed_tokens, args.q_heads), dtype=dtype, device=device
    ).uniform_(-1, 1)
    actual_q = torch.tensor(cumulative_q, dtype=torch.int32, device=device)
    actual_k = torch.tensor(final_seqlens, dtype=torch.int32, device=device)
    return {
        "query": query,
        "key": key,
        "weights": weights,
        "actual_q": actual_q,
        "actual_k": actual_k,
        "block_table": block_table,
        "query_counts": query_counts,
        "cumulative_q": cumulative_q,
        "final_seqlens": final_seqlens,
        "packed_tokens": packed_tokens,
    }


def run_reference(case):
    output = find_native_op()(
        query=case["query"],
        key=case["key"],
        weights=case["weights"],
        actual_seq_lengths_query=case["actual_q"],
        actual_seq_lengths_key=case["actual_k"],
        block_table=case["block_table"],
        layout_query="TND",
        layout_key="PA_BSND",
        sparse_count=TOPK,
        sparse_mode=3,
        pre_tokens=(1 << 63) - 1,
        next_tokens=(1 << 63) - 1,
        return_value=False,
    )
    return extract_indices(output, case["packed_tokens"])


def request_rows(case, batch_idx):
    end = case["cumulative_q"][batch_idx]
    begin = 0 if batch_idx == 0 else case["cumulative_q"][batch_idx - 1]
    return begin, end


def ordered_union(rows):
    seen = set()
    result = []
    for token in rows.reshape(-1).tolist():
        if token >= 0 and token not in seen:
            seen.add(token)
            result.append(token)
    return result


def make_cache(args, case, reference_cpu):
    rng = random.Random(args.seed + 1)
    generator = torch.Generator().manual_seed(args.seed + 2)
    cache = torch.full(
        (args.bs, CACHE_ROW_CAPACITY), -1, dtype=torch.int32
    )
    target_misses = []
    reference_unions = []

    for batch_idx in range(args.bs):
        begin, end = request_rows(case, batch_idx)
        union = ordered_union(reference_cpu[begin:end])
        reference_unions.append(set(union))
        final_len = case["final_seqlens"][batch_idx]
        feasible_max = min(
            args.max_miss_count, len(union), final_len - CACHE_SIZE
        )
        if args.min_miss_count > feasible_max:
            raise RuntimeError(
                f"batch {batch_idx}: requested miss range is infeasible; "
                f"union_size={len(union)} final_seqlen={final_len}"
            )
        miss_count = rng.randint(args.min_miss_count, feasible_max)
        target_misses.append(miss_count)

        union_tensor = torch.tensor(union, dtype=torch.int64)
        hit_count = len(union) - miss_count
        hit_order = torch.randperm(len(union), generator=generator)[:hit_count]
        hit_tokens = union_tensor[hit_order]

        outside_mask = torch.ones(final_len, dtype=torch.bool)
        outside_mask[union_tensor] = False
        outside = torch.arange(final_len, dtype=torch.int64)[outside_mask]
        other_count = CACHE_SIZE - hit_count
        other_order = torch.randperm(outside.numel(), generator=generator)[:other_count]
        cached_tokens = torch.cat((hit_tokens, outside[other_order]))
        if cached_tokens.numel() != CACHE_SIZE:
            raise RuntimeError("cache construction did not produce exactly 8192 tokens")
        shuffled_slots = torch.randperm(
            CACHE_SIZE, generator=generator, dtype=torch.int32
        )
        cache[batch_idx, cached_tokens] = shuffled_slots

    return cache.to(case["query"].device), target_misses, reference_unions


def run_mtp(case, cache):
    return torch.ops.nanovllm_dsa.fused_li_manage_mtp.default(
        case["query"],
        case["key"],
        case["weights"],
        cache,
        case["actual_q"],
        case["actual_k"],
        case["block_table"],
    )


def check_meta():
    query = torch.empty((7, 32, 128), dtype=torch.bfloat16, device="meta")
    key = torch.empty((96, 128, 1, 128), dtype=torch.bfloat16, device="meta")
    weights = torch.empty((7, 32), dtype=torch.bfloat16, device="meta")
    cache = torch.empty((3, CACHE_ROW_CAPACITY), dtype=torch.int32, device="meta")
    actual_q = torch.empty((3,), dtype=torch.int32, device="meta")
    actual_k = torch.empty((3,), dtype=torch.int32, device="meta")
    table = torch.empty((3, 96), dtype=torch.int32, device="meta")
    outputs = torch.ops.nanovllm_dsa.fused_li_manage_mtp.default(
        query, key, weights, cache, actual_q, actual_k, table
    )
    expected = [
        (7, 1, TOPK),
        (7, 1, TOPK),
        (3, CACHE_SIZE),
        (3, CACHE_SIZE),
        (3,),
    ]
    if [tuple(tensor.shape) for tensor in outputs] != expected:
        raise AssertionError("fused_li_manage_mtp Meta returned wrong shapes")
    print("A5_FUSED_LI_MANAGE_MTP_META_CHECK ok=1", flush=True)


def check_cache_row(cache_row, final_len, batch_idx):
    valid_tokens = (cache_row >= 0).nonzero(as_tuple=False).flatten()
    if valid_tokens.numel() != CACHE_SIZE:
        raise AssertionError(
            f"batch {batch_idx}: valid cached token count={valid_tokens.numel()}, "
            f"expected {CACHE_SIZE}"
        )
    if bool((valid_tokens >= final_len).any()):
        raise AssertionError(f"batch {batch_idx}: cached token exceeds final seqlen")
    slots = cache_row[valid_tokens]
    expected = torch.arange(CACHE_SIZE, dtype=torch.int32)
    if not torch.equal(torch.sort(slots).values, expected):
        raise AssertionError(
            f"batch {batch_idx}: slots are not the unique range [0, 8192)"
        )


def validate(case, reference, old_cache, new_cache, output, targets, reference_unions):
    if not isinstance(output, (tuple, list)) or len(output) != 5:
        raise AssertionError("MTP operator must return five tensors")
    topk_index, topk_slots, miss_index, miss_slots, miss_count = output
    expected_shapes = (
        (case["packed_tokens"], 1, TOPK),
        (case["packed_tokens"], 1, TOPK),
        (len(case["query_counts"]), CACHE_SIZE),
        (len(case["query_counts"]), CACHE_SIZE),
        (len(case["query_counts"]),),
    )
    for name, tensor, shape in zip(
        ("topk_index", "topk_slots", "miss_index", "miss_slots", "miss_count"),
        output,
        expected_shapes,
    ):
        if tensor.dtype != torch.int32 or tuple(tensor.shape) != shape:
            raise AssertionError(
                f"{name}: dtype/shape={tensor.dtype}/{tuple(tensor.shape)}, "
                f"expected int32/{shape}"
            )

    topk_index = topk_index.reshape(case["packed_tokens"], TOPK).cpu()
    topk_slots = topk_slots.reshape(case["packed_tokens"], TOPK).cpu()
    miss_index = miss_index.cpu()
    miss_slots = miss_slots.cpu()
    miss_count = miss_count.cpu()
    reference = reference.cpu()

    for row in range(case["packed_tokens"]):
        if not torch.equal(
            torch.sort(topk_index[row]).values,
            torch.sort(reference[row]).values,
        ):
            raise AssertionError(f"query row {row}: TopK multiset differs from native LI")

    for batch_idx, query_count in enumerate(case["query_counts"]):
        begin, end = request_rows(case, batch_idx)
        final_len = case["final_seqlens"][batch_idx]
        for local_q, row in enumerate(range(begin, end)):
            visible_len = final_len - query_count + local_q + 1
            indices = topk_index[row].to(torch.long)
            if int(indices.min()) < 0 or int(indices.max()) >= visible_len:
                raise AssertionError(
                    f"batch {batch_idx} query {local_q}: index outside [0, {visible_len})"
                )
            expected_slots = new_cache[batch_idx].gather(0, indices)
            if not torch.equal(expected_slots, topk_slots[row]):
                raise AssertionError(
                    f"batch {batch_idx} query {local_q}: topk_slots != updated cache"
                )

        output_union = set(ordered_union(topk_index[begin:end]))
        if output_union != reference_unions[batch_idx]:
            raise AssertionError(f"batch {batch_idx}: request-level TopK union differs")
        expected_misses = {
            token
            for token in reference_unions[batch_idx]
            if int(old_cache[batch_idx, token]) < 0
        }
        count = int(miss_count[batch_idx])
        if count != targets[batch_idx] or count != len(expected_misses):
            raise AssertionError(
                f"batch {batch_idx}: miss_count={count}, target={targets[batch_idx]}, "
                f"old-cache union misses={len(expected_misses)}"
            )
        emitted_tokens = miss_index[batch_idx, :count].to(torch.long)
        emitted_slots = miss_slots[batch_idx, :count]
        if emitted_tokens.unique().numel() != count:
            raise AssertionError(f"batch {batch_idx}: miss_index contains duplicates")
        if set(emitted_tokens.tolist()) != expected_misses:
            raise AssertionError(f"batch {batch_idx}: miss_index set is incorrect")
        if bool((emitted_slots < 0).any()) or bool((emitted_slots >= CACHE_SIZE).any()):
            raise AssertionError(f"batch {batch_idx}: miss_slots is outside [0, 8192)")
        if emitted_slots.unique().numel() != count:
            raise AssertionError(f"batch {batch_idx}: miss_slots contains duplicates")

        old_valid_tokens = (old_cache[batch_idx] >= 0).nonzero(as_tuple=False).flatten()
        old_slot_owner = torch.empty(CACHE_SIZE, dtype=torch.long)
        old_slot_owner[old_cache[batch_idx, old_valid_tokens].to(torch.long)] = old_valid_tokens
        for token, slot in zip(emitted_tokens.tolist(), emitted_slots.tolist()):
            if int(new_cache[batch_idx, token]) != slot:
                raise AssertionError(f"batch {batch_idx}: miss token was not assigned its output slot")
            victim = int(old_slot_owner[slot])
            if victim in reference_unions[batch_idx]:
                raise AssertionError(f"batch {batch_idx}: evicted token belongs to the union")
            if int(new_cache[batch_idx, victim]) != -1:
                raise AssertionError(f"batch {batch_idx}: victim token was not cleared")

        for token in reference_unions[batch_idx]:
            old_slot = int(old_cache[batch_idx, token])
            if old_slot >= 0 and int(new_cache[batch_idx, token]) != old_slot:
                raise AssertionError(f"batch {batch_idx}: an old union hit changed slot")
        check_cache_row(new_cache[batch_idx], final_len, batch_idx)


def main():
    args = parse_args()
    validate_args(args)
    check_meta()
    device = torch.device(args.device)
    if device.type != "npu":
        raise ValueError("--device must select an NPU")
    torch.npu.set_device(device)
    torch.npu.config.allow_internal_format = False
    require_a5(device, args.allow_non_a5)
    case = make_inputs(args, device)
    reference = run_reference(case)
    torch.npu.synchronize()
    reference_cpu = reference.cpu()
    cache, targets, reference_unions = make_cache(args, case, reference_cpu)
    old_cache = cache.cpu()
    output = run_mtp(case, cache)
    torch.npu.synchronize()
    new_cache = cache.cpu()
    validate(
        case, reference_cpu, old_cache, new_cache, output, targets, reference_unions
    )
    print(
        f"case bs={args.bs} packed_t={case['packed_tokens']} "
        f"q_heads={args.q_heads} query_counts={case['query_counts']}"
    )
    print(f"actual_seq_lengths_query={case['cumulative_q']}")
    print(f"actual_seq_lengths_key={case['final_seqlens']}")
    print(f"target_union_miss_count={targets}")
    print("per_query_native_topk_multiset_check=passed")
    print("mtp_causal_key_prefix_check=passed")
    print("request_union_dedup_check=passed")
    print("union_miss_index_and_slot_check=passed")
    print("single_cache_update_check=passed")
    print("A5_FUSED_LI_MANAGE_MTP_UT_OK")


if __name__ == "__main__":
    main()
