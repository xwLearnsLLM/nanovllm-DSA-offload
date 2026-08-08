"""Shared index-management construction and assertions for BF16/C8 LIDU."""

from __future__ import annotations

import argparse

import torch


TOPK = 2048
MAX_SOURCE_CAPACITY = 1 << 18
MAX_CACHE_TOKENS = 16256


def miss_ranges(value: str) -> list[tuple[int, int]]:
    result: list[tuple[int, int]] = []
    for item in value.split(","):
        lo_text, separator, hi_text = item.strip().partition(":")
        if not separator:
            raise argparse.ArgumentTypeError("miss ranges use MIN:MAX")
        lo, hi = int(lo_text), int(hi_text)
        if not 0 <= lo <= hi <= TOPK:
            raise argparse.ArgumentTypeError(
                f"invalid miss range {item!r}; require 0 <= MIN <= MAX <= {TOPK}"
            )
        result.append((lo, hi))
    if not result:
        raise argparse.ArgumentTypeError("expected at least one miss range")
    return result


def feasible_miss(candidate_len: int, cache_tokens: int, miss: int) -> bool:
    if cache_tokens == 0:
        return miss == 0
    hits = TOPK - miss
    victims = cache_tokens - hits
    return candidate_len >= cache_tokens and 0 <= victims <= candidate_len - TOPK


def build_pool(
    native_topk: torch.Tensor,
    source_capacity: int,
    candidate_lens: list[int],
    budgets: list[int],
    target_misses: list[int],
    req_entries_cpu: torch.Tensor,
    pool_size: int,
    seed: int,
) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    pool = torch.full((pool_size, source_capacity), -1, dtype=torch.int32)
    topk_cpu = native_topk.cpu().to(torch.int64)
    for batch_row, (candidate_len, budget, miss) in enumerate(
        zip(candidate_lens, budgets, target_misses)
    ):
        pool_row = int(req_entries_cpu[batch_row])
        if budget == 0:
            continue
        if not feasible_miss(candidate_len, budget, miss):
            raise ValueError(
                f"cannot construct C={budget}, miss={miss}, "
                f"candidate_len={candidate_len}"
            )
        selected = topk_cpu[batch_row]
        hit_count = TOPK - miss
        hit_ids = selected[
            torch.randperm(TOPK, generator=generator)[:hit_count]
        ]
        outside_mask = torch.ones(candidate_len, dtype=torch.bool)
        outside_mask[selected] = False
        outside = torch.arange(candidate_len, dtype=torch.int64)[outside_mask]
        victim_count = budget - hit_count
        victim_ids = outside[
            torch.randperm(outside.numel(), generator=generator)[:victim_count]
        ]
        token_ids = torch.cat((hit_ids, victim_ids))
        slots = torch.randperm(
            budget, generator=generator, dtype=torch.int64
        ).to(torch.int32)
        pool[pool_row, token_ids] = slots
    return pool


def assert_pool_row(row: torch.Tensor, candidate_len: int, budget: int) -> None:
    valid_tokens = (row[:candidate_len] >= 0).nonzero().flatten()
    if bool((row[candidate_len:] >= 0).any()):
        raise AssertionError("cache row contains a token outside candidate_len")
    if budget == 0:
        if valid_tokens.numel() != 0:
            raise AssertionError("C=0 request mutated its cache row")
        return
    if valid_tokens.numel() != budget:
        raise AssertionError(
            f"cache cardinality={valid_tokens.numel()}, expected {budget}"
        )
    slots = row[valid_tokens]
    expected = torch.arange(budget, dtype=torch.int32)
    if not torch.equal(torch.sort(slots).values, expected):
        raise AssertionError("cache slots are not the exact unique range [0,C)")


def assert_request_pool_entries(
    entries: torch.Tensor, batch: int, pool_size: int
) -> None:
    if entries.dtype != torch.int32 or entries.shape != (batch,):
        raise AssertionError("req_pool_entries must be int32[B]")
    if int(entries.min()) < 0 or int(entries.max()) >= pool_size:
        raise AssertionError("req_pool_entries contains an invalid request-pool row")
    if torch.unique(entries).numel() != batch:
        raise AssertionError("active req_pool_entries are not unique")


def assert_update_row(
    *,
    label: str,
    sources: torch.Tensor,
    slots: torch.Tensor,
    reference: torch.Tensor,
    old_row: torch.Tensor,
    new_row: torch.Tensor,
    candidate_len: int,
    budget: int,
    expected_miss: int,
    actual_miss: int,
) -> None:
    if budget == 0:
        if (
            actual_miss != 0
            or bool((sources != -1).any())
            or bool((slots != -1).any())
            or not torch.equal(old_row, new_row)
        ):
            raise AssertionError(f"{label}: C=0 row is not a strict no-op")
        return

    if sources.numel() != TOPK or slots.numel() != TOPK:
        raise AssertionError(f"{label}: expected exactly {TOPK} source IDs and slots")
    if int(sources.min()) < 0 or int(sources.max()) >= candidate_len:
        raise AssertionError(f"{label}: topk_index is outside [0,{candidate_len})")
    if torch.unique(sources).numel() != TOPK:
        raise AssertionError(f"{label}: topk_index is not unique")
    if (
        int(reference.min()) < 0
        or int(reference.max()) >= candidate_len
        or torch.unique(reference).numel() != TOPK
    ):
        raise AssertionError(f"{label}: LightningIndexer reference is invalid")
    if not torch.equal(torch.sort(sources).values, torch.sort(reference).values):
        raise AssertionError(f"{label}: top-2048 set differs from LightningIndexer")

    if not 0 <= actual_miss <= TOPK:
        raise AssertionError(
            f"{label}: miss_count={actual_miss} is outside [0,{TOPK}]"
        )
    if actual_miss > candidate_len - budget:
        raise AssertionError(
            f"{label}: miss_count={actual_miss} exceeds candidate_len-C"
        )
    old_selected_slots = old_row.gather(0, sources.long())
    new_selected_slots = new_row.gather(0, sources.long())
    recomputed_miss = int((old_selected_slots < 0).sum())
    if actual_miss != expected_miss or actual_miss != recomputed_miss:
        raise AssertionError(
            f"{label}: miss_count={actual_miss}, expected={expected_miss}, "
            f"recomputed={recomputed_miss}"
        )
    if actual_miss and bool((old_selected_slots[:actual_miss] >= 0).any()):
        raise AssertionError(f"{label}: miss prefix contains a cache hit")
    if actual_miss < TOPK:
        old_hit_slots = old_selected_slots[actual_miss:]
        if bool((old_hit_slots < 0).any()):
            raise AssertionError(f"{label}: hit suffix contains a cache miss")
        if not torch.equal(slots[actual_miss:], old_hit_slots):
            raise AssertionError(f"{label}: an old hit changed its HBM slot")

    if int(slots.min()) < 0 or int(slots.max()) >= budget:
        raise AssertionError(f"{label}: topk_slots is outside [0,{budget})")
    if torch.unique(slots).numel() != TOPK:
        raise AssertionError(f"{label}: topk_slots is not unique")
    if not torch.equal(slots, new_selected_slots):
        raise AssertionError(f"{label}: published slots differ from updated cache state")


def assert_18bit_boundary_selected(case: object, label: str) -> None:
    source_capacity = getattr(case, "source_capacity")
    query = getattr(case, "query")
    native_topk = getattr(case, "native_topk")
    if source_capacity != MAX_SOURCE_CAPACITY:
        raise AssertionError(f"{label}: boundary case has the wrong source capacity")
    selected = native_topk.cpu().reshape(query.size(0), TOPK)
    if int(selected.max()) < (1 << 17):
        raise AssertionError(f"{label}: top-k did not exercise token-index bit 17")
    print(
        f"{label} source_capacity={source_capacity} "
        f"selected_max={int(selected.max())} high_index_bit17=1 ok=1",
        flush=True,
    )
