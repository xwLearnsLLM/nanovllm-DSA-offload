"""Correctness and latency test for the Fused LI Manage operator.

The native ``torch_npu.npu_lightning_indexer`` is timed on the same inputs.
The LI-Manage-minus-LightningIndexer delta is the index-management latency proxy.
"""

from __future__ import annotations

import argparse
import gc
import statistics
from collections.abc import Callable

import torch
import torch_npu  # type: ignore

import nanovllm.ops  # noqa: F401  # Load repository-local custom operators.
from ut_ops._op_utils import require_local_opapi


BLOCK_SIZE = 128
HEAD_DIM = 128
TOPK = 2048
EXACT_PAYLOAD_MAX_SOURCE_TOKENS = 1 << 18
MAX_ACTUAL_SEQ_LEN = (1 << 21) - 1
MAX_CAPACITY = 1 << 21
MAX_CACHE_TOKENS = (1 << 14) - 1


def parse_csv(value: str, name: str) -> tuple[int, ...]:
    try:
        values = tuple(dict.fromkeys(int(item.strip()) for item in value.split(",")))
    except ValueError as exc:
        raise ValueError(f"{name} must contain comma-separated integers.") from exc
    if not values:
        raise ValueError(f"{name} must not be empty.")
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate Fused LI Manage across the 18-bit and 21-bit boundaries."
    )
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--heads", default="32,64")
    parser.add_argument("--seq-lens", default="262272")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--cache-tokens", type=int, default=6144)
    parser.add_argument("--miss-count", type=int, default=300)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--iters", type=int, default=3)
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def validate_args(
    heads: tuple[int, ...],
    seq_lens: tuple[int, ...],
    batch_size: int,
    cache_tokens: int,
    miss_count: int,
    warmup: int,
    iters: int,
) -> None:
    if any(head not in (32, 64) for head in heads):
        raise ValueError("--heads only supports 32 and 64.")
    if batch_size <= 0:
        raise ValueError("--batch-size must be positive.")
    if any(length < TOPK or length > MAX_ACTUAL_SEQ_LEN for length in seq_lens):
        raise ValueError(f"--seq-lens must be in [{TOPK}, {MAX_ACTUAL_SEQ_LEN}].")
    if cache_tokens < TOPK or cache_tokens % BLOCK_SIZE:
        raise ValueError("--cache-tokens must be a multiple of 128 and >= 2048.")
    if cache_tokens > MAX_CACHE_TOKENS:
        raise ValueError(f"--cache-tokens must not exceed {MAX_CACHE_TOKENS}.")
    if any(cache_tokens > length for length in seq_lens):
        raise ValueError("--cache-tokens must not exceed any sequence length.")
    if miss_count < 0 or miss_count > TOPK:
        raise ValueError("--miss-count must be in [0, 2048].")
    if any(cache_tokens - (TOPK - miss_count) > length - TOPK for length in seq_lens):
        raise ValueError("The requested cache/miss configuration is infeasible.")
    if warmup < 0 or iters < 0:
        raise ValueError("--warmup and --iters must be non-negative.")


def validate_score_tag_codec() -> None:
    indices = torch.tensor(
        [
            0,
            EXACT_PAYLOAD_MAX_SOURCE_TOKENS - 1,
            EXACT_PAYLOAD_MAX_SOURCE_TOKENS,
            EXACT_PAYLOAD_MAX_SOURCE_TOKENS + 1,
            (2 << 18) - 1,
            2 << 18,
            (7 << 18) + 12345,
            MAX_ACTUAL_SEQ_LEN - 1,
        ],
        dtype=torch.int64,
    )
    low18 = indices & ((1 << 18) - 1)
    high3 = indices >> 18
    tags = high3
    decoded = low18 | (tags << 18)
    if not torch.equal(decoded, indices):
        raise AssertionError("The score-low3 index codec is not reversible.")

    # Eviction candidates carry slot14 + index_low18 in their payload. Their
    # negated score key must retain index_high3 so the packed fast path can
    # reconstruct the full 21-bit source index without reading cache_slots.
    slots = torch.tensor([0, 1, 6143, 12287, 16382, 7, 99, 2047], dtype=torch.int64)
    payloads = (slots << 18) | low18
    score_values = torch.tensor(
        [0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0], dtype=torch.float32
    )
    score_bits = score_values.view(torch.int32).to(torch.int64) & 0xFFFFFFFF
    tagged_bits = (score_bits & ~0x7) | tags
    tagged_scores = (tagged_bits.to(torch.int32)).view(torch.float32)
    evict_key_bits = (-tagged_scores).view(torch.int32).to(torch.int64) & 0xFFFFFFFF
    decoded_indices = (payloads & ((1 << 18) - 1)) | ((evict_key_bits & 0x7) << 18)
    decoded_slots = payloads >> 18
    if not torch.equal(decoded_indices, indices) or not torch.equal(decoded_slots, slots):
        raise AssertionError("The packed eviction-candidate codec is not reversible.")
    print(
        "FUSED_LI_MANAGE_SCORE_TAG_CODEC_CHECK "
        "index_low_bits=18 index_high_bits=3 score_tag_bits=3 "
        "encoding=index_high3_direct evict_payload=slot14_indexlow18 ok=1",
        flush=True,
    )


def build_case(
    *,
    device: torch.device,
    heads: int,
    batch_size: int,
    seq_len: int,
    cache_tokens_value: int,
    miss_count: int,
    seed: int,
) -> dict[str, torch.Tensor | int]:
    blocks_per_request = (seq_len + BLOCK_SIZE - 1) // BLOCK_SIZE
    capacity = blocks_per_request * BLOCK_SIZE
    if capacity > MAX_CAPACITY:
        raise AssertionError("Aligned source capacity exceeds 2^21.")
    total_blocks = batch_size * blocks_per_request

    query = torch.zeros(
        (batch_size, heads, HEAD_DIM), dtype=torch.bfloat16, device=device
    )
    # Four base-64 digits encode every index in [0, 2^21) exactly.
    query[:, 0, 0] = 1
    query[:, 0, 1] = 64
    query[:, 0, 2] = 4096
    query[:, 0, 3] = 262144
    weights = torch.zeros(
        (batch_size, heads), dtype=torch.bfloat16, device=device
    )
    weights[:, 0] = 1

    key = torch.zeros(
        (total_blocks, BLOCK_SIZE, 1, HEAD_DIM),
        dtype=torch.bfloat16,
        device=device,
    )
    logical_ids = torch.arange(capacity, dtype=torch.int32, device=device).view(
        1, blocks_per_request, BLOCK_SIZE
    )
    key_rows = key.view(
        batch_size, blocks_per_request, BLOCK_SIZE, 1, HEAD_DIM
    )
    key_rows[:, :, :, 0, 0] = (logical_ids % 64).to(torch.bfloat16)
    key_rows[:, :, :, 0, 1] = ((logical_ids // 64) % 64).to(torch.bfloat16)
    key_rows[:, :, :, 0, 2] = ((logical_ids // 4096) % 64).to(torch.bfloat16)
    key_rows[:, :, :, 0, 3] = (logical_ids // 262144).to(torch.bfloat16)

    block_table = torch.arange(
        total_blocks, dtype=torch.int32, device=device
    ).view(batch_size, blocks_per_request)
    candidate_lens = torch.full(
        (batch_size,), seq_len, dtype=torch.int32, device=device
    )
    query_lens = torch.arange(
        1, batch_size + 1, dtype=torch.int32, device=device
    )
    cache_tokens = torch.full(
        (batch_size,), cache_tokens_value, dtype=torch.int32, device=device
    )

    pool_size = batch_size + 1
    req_entries_cpu = torch.arange(
        batch_size, 0, -1, dtype=torch.int32
    )
    req_entries = req_entries_cpu.to(device)
    initial_cache_cpu = torch.full(
        (pool_size, capacity), -1, dtype=torch.int32
    )
    topk = torch.arange(seq_len - TOPK, seq_len, dtype=torch.int64)
    generator = torch.Generator().manual_seed(seed)
    hit_tokens = topk[miss_count:]
    other_count = cache_tokens_value - hit_tokens.numel()
    lower_count = seq_len - TOPK
    other_tokens = (
        torch.div(
            torch.arange(other_count, dtype=torch.int64) * lower_count,
            other_count,
            rounding_mode="floor",
        )
        if other_count
        else torch.empty(0, dtype=torch.int64)
    )
    cached_tokens = torch.cat((hit_tokens, other_tokens))
    for row in range(batch_size):
        slots = torch.randperm(
            cache_tokens_value, generator=generator, dtype=torch.int32
        )
        initial_cache_cpu[int(req_entries_cpu[row]), cached_tokens] = slots

    initial_cache = initial_cache_cpu.to(device)
    cache_slots = initial_cache.clone()
    source_ids = torch.full(
        (batch_size, 1, TOPK), -1, dtype=torch.int32, device=device
    )
    destination_slots = torch.full_like(source_ids, -1)
    miss_counts = torch.full(
        (batch_size,), -1, dtype=torch.int32, device=device
    )
    return {
        "capacity": capacity,
        "query": query,
        "key": key,
        "weights": weights,
        "req_entries": req_entries,
        "req_entries_cpu": req_entries_cpu,
        "initial_cache": initial_cache,
        "initial_cache_cpu": initial_cache_cpu,
        "cache_slots": cache_slots,
        "cache_tokens": cache_tokens,
        "query_lens": query_lens,
        "candidate_lens": candidate_lens,
        "block_table": block_table,
        "source_ids": source_ids,
        "destination_slots": destination_slots,
        "miss_counts": miss_counts,
    }


def call_fused_li_manage(case: dict[str, torch.Tensor | int]) -> None:
    torch.ops.nanovllm_dsa.fused_li_manage.default(
        case["query"],
        case["weights"],
        case["key"],
        case["block_table"],
        case["candidate_lens"],
        case["cache_tokens"],
        case["req_entries"],
        case["cache_slots"],
        case["source_ids"],
        case["destination_slots"],
        case["miss_counts"],
    )


def call_lightning_indexer(
    case: dict[str, torch.Tensor | int],
) -> torch.Tensor:
    output = torch_npu.npu_lightning_indexer(
        query=case["query"],
        key=case["key"],
        weights=case["weights"],
        actual_seq_lengths_query=case["query_lens"],
        actual_seq_lengths_key=case["candidate_lens"],
        block_table=case["block_table"],
        layout_query="TND",
        layout_key="PA_BSND",
        sparse_count=TOPK,
        sparse_mode=3,
    )
    if isinstance(output, (tuple, list)):
        output = output[0]
    return output


def validate_lightning_indexer(
    output: torch.Tensor, *, batch_size: int, seq_len: int
) -> None:
    expected = torch.arange(
        seq_len - TOPK, seq_len, dtype=torch.int32
    ).expand(batch_size, TOPK)
    actual = torch.sort(output.view(batch_size, TOPK).cpu(), dim=1).values
    if not torch.equal(actual, expected):
        raise AssertionError(
            "torch_npu.npu_lightning_indexer did not select the expected "
            "top-2048 set."
        )


def validate_manage_matches_lightning(
    lightning_output: torch.Tensor, case: dict[str, torch.Tensor | int]
) -> int:
    batch_size = int(case["candidate_lens"].numel())
    lightning_sorted = torch.sort(
        lightning_output.view(batch_size, TOPK).cpu(), dim=1
    ).values
    manage_sorted = torch.sort(
        case["source_ids"].view(batch_size, TOPK).cpu(), dim=1
    ).values
    set_diff = int((lightning_sorted != manage_sorted).sum().item())
    if set_diff:
        raise AssertionError(
            f"Fused LI Manage differs from LightningIndexer at {set_diff} sorted positions."
        )
    return set_diff


def validate_outputs(
    case: dict[str, torch.Tensor | int],
    *,
    seq_len: int,
    cache_tokens_value: int,
    expected_miss_count: int,
    reference_topk: torch.Tensor,
    old_cache_pool: torch.Tensor,
) -> torch.Tensor:
    batch_size = int(case["candidate_lens"].numel())
    counts = case["miss_counts"].cpu()
    expected_counts = torch.full(
        (batch_size,), expected_miss_count, dtype=torch.int32
    )
    if not torch.equal(counts, expected_counts):
        raise AssertionError(
            f"miss_counts={counts.tolist()}, expected={expected_counts.tolist()}."
        )

    sources = case["source_ids"].view(batch_size, TOPK).cpu().to(torch.int64)
    slots = case["destination_slots"].view(batch_size, TOPK).cpu().to(torch.int64)
    state_pool = case["cache_slots"].cpu()
    old_pool = old_cache_pool.cpu()
    req_entries_cpu = case["req_entries_cpu"].to(torch.int64)
    reference = reference_topk.view(batch_size, TOPK).cpu().to(torch.int64)

    if case["req_entries"].dtype != torch.int32:
        raise AssertionError("req_pool_entries must be int32.")
    if not torch.equal(case["req_entries"].cpu().to(torch.int64), req_entries_cpu):
        raise AssertionError("Device and host req_pool_entries mappings differ.")
    if torch.unique(req_entries_cpu).numel() != batch_size:
        raise AssertionError("req_pool_entries must be unique within the active batch.")
    if bool((req_entries_cpu < 0).any()) or bool(
        (req_entries_cpu >= state_pool.shape[0]).any()
    ):
        raise AssertionError("req_pool_entries contains an out-of-range pool row.")

    mapped_rows = set(int(value) for value in req_entries_cpu.tolist())
    for pool_row in range(state_pool.shape[0]):
        if pool_row not in mapped_rows and not torch.equal(
            state_pool[pool_row], old_pool[pool_row]
        ):
            raise AssertionError(f"Unmapped request-pool row {pool_row} was modified.")

    for row in range(batch_size):
        source_row = sources[row]
        slot_row = slots[row]
        if bool((source_row < 0).any()) or bool((source_row >= seq_len).any()):
            raise AssertionError(f"row={row} topk_index contains an out-of-range token.")
        if torch.unique(source_row).numel() != TOPK:
            raise AssertionError(f"row={row} topk_index contains duplicates.")
        if not torch.equal(
            torch.sort(source_row).values,
            torch.sort(reference[row]).values,
        ):
            raise AssertionError(
                f"row={row} topk_index differs from LightningIndexer reference."
            )
        if bool((slot_row < 0).any()) or bool(
            (slot_row >= cache_tokens_value).any()
        ):
            raise AssertionError(f"row={row} contains an invalid destination slot.")
        if torch.unique(slot_row).numel() != TOPK:
            raise AssertionError(f"row={row} destination slots are not unique.")

        pool_row = int(req_entries_cpu[row])
        old_state = old_pool[pool_row]
        new_state = state_pool[pool_row]
        old_topk_slots = old_state[source_row].to(torch.int64)
        actual_miss_mask = old_topk_slots == -1
        actual_miss_count = int(actual_miss_mask.sum())
        if actual_miss_count != expected_miss_count:
            raise AssertionError(
                f"row={row} actual old-cache miss count is {actual_miss_count}, "
                f"expected {expected_miss_count}."
            )
        if expected_miss_count and not bool(
            actual_miss_mask[:expected_miss_count].all()
        ):
            raise AssertionError(f"row={row} miss prefix contains an old-cache hit.")
        if bool(actual_miss_mask[expected_miss_count:].any()):
            raise AssertionError(f"row={row} hit suffix contains an old-cache miss.")
        if not torch.equal(
            slot_row[expected_miss_count:],
            old_topk_slots[expected_miss_count:],
        ):
            raise AssertionError(
                f"row={row} hit suffix did not preserve its old cache slots."
            )
        if not torch.equal(new_state[source_row].to(torch.int64), slot_row):
            raise AssertionError(
                f"row={row} new_cache_slots[topk_index] does not equal topk_slots itemwise."
            )

        old_valid_slots = old_state[old_state >= 0].to(torch.int64)
        if (
            old_valid_slots.numel() != cache_tokens_value
            or torch.unique(old_valid_slots).numel() != cache_tokens_value
            or int(old_valid_slots.min()) != 0
            or int(old_valid_slots.max()) != cache_tokens_value - 1
        ):
            raise AssertionError(
                f"row={row} old cache cardinality/permutation is invalid."
            )
        valid_slots = new_state[new_state >= 0].to(torch.int64)
        if (
            valid_slots.numel() != cache_tokens_value
            or torch.unique(valid_slots).numel() != cache_tokens_value
            or int(valid_slots.min()) != 0
            or int(valid_slots.max()) != cache_tokens_value - 1
        ):
            raise AssertionError(f"row={row} cache cardinality/permutation is invalid.")
    return state_pool


def benchmark(
    runner: Callable[[], object],
    *,
    warmup: int,
    iters: int,
    reset: Callable[[], None] | None = None,
) -> float | None:
    if iters == 0:
        return None
    for _ in range(warmup):
        if reset is not None:
            reset()
        runner()
    torch.npu.synchronize()

    times_ms: list[float] = []
    for _ in range(iters):
        if reset is not None:
            reset()
            # Request-state restoration is outside the timed interval.
            torch.npu.synchronize()
        start = torch.npu.Event(enable_timing=True)
        end = torch.npu.Event(enable_timing=True)
        start.record()
        runner()
        end.record()
        end.synchronize()
        times_ms.append(float(start.elapsed_time(end)))
    return statistics.mean(times_ms) * 1000.0


def run_case(
    *,
    device: torch.device,
    heads: int,
    batch_size: int,
    seq_len: int,
    cache_tokens: int,
    miss_count: int,
    warmup: int,
    iters: int,
    seed: int,
) -> None:
    case = build_case(
        device=device,
        heads=heads,
        batch_size=batch_size,
        seq_len=seq_len,
        cache_tokens_value=cache_tokens,
        miss_count=miss_count,
        seed=seed,
    )
    capacity = int(case["capacity"])
    key_mib = case["key"].numel() * case["key"].element_size() / (1 << 20)
    print(
        "FUSED_LI_MANAGE_CASE "
        f"heads={heads} batch={batch_size} seq_len={seq_len} capacity={capacity} "
        f"cache_tokens={cache_tokens} miss_count={miss_count} key_mib={key_mib:.1f}",
        flush=True,
    )

    lightning_output = call_lightning_indexer(case)
    torch.npu.synchronize()
    validate_lightning_indexer(
        lightning_output, batch_size=batch_size, seq_len=seq_len
    )
    reference_topk = lightning_output.view(batch_size, TOPK).cpu()

    def reset_state() -> None:
        case["cache_slots"].copy_(case["initial_cache"])

    reset_state()
    call_fused_li_manage(case)
    torch.npu.synchronize()
    manage_state = validate_outputs(
        case,
        seq_len=seq_len,
        cache_tokens_value=cache_tokens,
        expected_miss_count=miss_count,
        reference_topk=reference_topk,
        old_cache_pool=case["initial_cache_cpu"],
    )
    topk_set_diff = validate_manage_matches_lightning(lightning_output, case)
    encoding = (
        "exact_packed"
        if seq_len <= EXACT_PAYLOAD_MAX_SOURCE_TOKENS
        else "score_low3"
    )
    print(
        "FUSED_LI_MANAGE_TOPK_COMPARE "
        f"heads={heads} batch={batch_size} seq_len={seq_len} encoding={encoding} "
        f"sorted_position_diff={topk_set_diff} ok=1",
        flush=True,
    )
    crossed_18bit = (
        int(case["source_ids"].max()) >= EXACT_PAYLOAD_MAX_SOURCE_TOKENS
    )
    if seq_len > EXACT_PAYLOAD_MAX_SOURCE_TOKENS and not crossed_18bit:
        raise AssertionError("The boundary case did not cross the old 18-bit limit.")

    # A second update without restoring state must see the same top-k entirely cached.
    call_fused_li_manage(case)
    torch.npu.synchronize()
    validate_outputs(
        case,
        seq_len=seq_len,
        cache_tokens_value=cache_tokens,
        expected_miss_count=0,
        reference_topk=reference_topk,
        old_cache_pool=manage_state,
    )
    print(
        "FUSED_LI_MANAGE_SEMANTICS_CHECK "
        f"heads={heads} batch={batch_size} seq_len={seq_len} "
        "reference_topk=1 actual_miss_recomputed=1 miss_prefix=1 "
        "hit_suffix=1 hit_slot_preserved=1 itemwise_slot_map=1 "
        "cache_permutation=1 unmapped_pool_unchanged=1 repeated_update=1 ok=1",
        flush=True,
    )

    lightning_us = benchmark(
        lambda: call_lightning_indexer(case),
        warmup=warmup,
        iters=iters,
    )
    fused_li_manage_us = benchmark(
        lambda: call_fused_li_manage(case),
        warmup=warmup,
        iters=iters,
        reset=reset_state,
    )
    if lightning_us is not None and fused_li_manage_us is not None:
        print(
            "FUSED_LI_MANAGE_RESULT "
            f"heads={heads} batch={batch_size} seq_len={seq_len} "
            f"cache_tokens={cache_tokens} miss_count={miss_count} "
            f"lightning_indexer_us={lightning_us:.3f} "
            f"fused_li_manage_us={fused_li_manage_us:.3f} "
            f"index_management_us={fused_li_manage_us - lightning_us:+.3f} "
            f"timer=npu_event warmup={warmup} iters={iters}",
            flush=True,
        )
    print(
        f"FUSED_LI_MANAGE_CHECK heads={heads} seq_len={seq_len} "
        f"crossed_18bit={int(crossed_18bit)} repeated_update=1 ok=1",
        flush=True,
    )


def main() -> None:
    args = parse_args()
    heads = parse_csv(args.heads, "--heads")
    seq_lens = parse_csv(args.seq_lens, "--seq-lens")
    validate_args(
        heads,
        seq_lens,
        args.batch_size,
        args.cache_tokens,
        args.miss_count,
        args.warmup,
        args.iters,
    )
    device = torch.device(args.device)
    if device.type != "npu":
        raise ValueError("--device must select one NPU, for example npu:0.")
    torch.npu.set_device(device)
    torch.npu.config.allow_internal_format = False
    opapi_path = require_local_opapi()
    if not callable(getattr(torch_npu, "npu_lightning_indexer", None)):
        raise RuntimeError(
            "This torch_npu build does not expose npu_lightning_indexer."
        )
    print(
        "FUSED_LI_MANAGE_CONFIG "
        f"device={device} heads={list(heads)} seq_lens={list(seq_lens)} "
        f"max_actual_seq_len={MAX_ACTUAL_SEQ_LEN} "
        f"payload=exact_packed_le_{EXACT_PAYLOAD_MAX_SOURCE_TOKENS}_else_score_low3 "
        "score_tag=index_high3_direct "
        "baseline=torch_npu.npu_lightning_indexer "
        f"opapi={opapi_path}",
        flush=True,
    )
    validate_score_tag_codec()

    case_id = 0
    for seq_len in seq_lens:
        for head_count in heads:
            run_case(
                device=device,
                heads=head_count,
                batch_size=args.batch_size,
                seq_len=seq_len,
                cache_tokens=args.cache_tokens,
                miss_count=args.miss_count,
                warmup=args.warmup,
                iters=args.iters,
                seed=args.seed + case_id * 101,
            )
            case_id += 1
            gc.collect()
            torch.npu.empty_cache()
    print("FUSED_LI_MANAGE_UT_OK", flush=True)


if __name__ == "__main__":
    main()
