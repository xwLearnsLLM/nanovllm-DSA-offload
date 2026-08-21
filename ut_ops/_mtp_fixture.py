"""Deterministic MTP3 metadata fixtures for COPYSFA-MTP tests."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import torch
import torch_npu  # type: ignore


QUERY_COUNT = 4
BLOCK_SIZE = 128
TOPK = 2048
UNION_CAPACITY = QUERY_COUNT * TOPK
MAX_SOURCE_CAPACITY = 1 << 18
KPE_DIM = 64
CKV_DIM = 512


@dataclass
class MtpCase:
    name: str
    device: torch.device
    batch_size: int
    source_capacity: int
    cache_tokens: torch.Tensor
    cache_tokens_cpu: torch.Tensor
    req_pool_entries_cpu: torch.Tensor
    initial_cache_cpu: torch.Tensor
    topk_cpu: list[torch.Tensor]
    union_cpu: list[torch.Tensor]


def random_block_table(
    batch_size: int,
    blocks_per_request: int,
    generator: torch.Generator,
) -> tuple[torch.Tensor, int]:
    total_blocks = batch_size * blocks_per_request
    table = torch.randperm(total_blocks, generator=generator).to(torch.int32)
    return table.view(batch_size, blocks_per_request).contiguous(), total_blocks


def ordered_union(rows: list[torch.Tensor]) -> torch.Tensor:
    ordered: list[int] = []
    seen: set[int] = set()
    for row in rows:
        for value in row.tolist():
            token = int(value)
            if token not in seen:
                seen.add(token)
                ordered.append(token)
    return torch.tensor(ordered, dtype=torch.int64)


def _make_request_topk(
    source_capacity: int,
    *,
    profile: str,
    generator: torch.Generator,
    unique_miss_count: int | None = None,
    miss_overlap_rate: float | None = None,
    hit_overlap_rate: float | None = None,
) -> tuple[list[torch.Tensor], torch.Tensor | None]:
    permutation = torch.randperm(source_capacity, generator=generator).to(torch.int64)
    if profile == "broad":
        # Four disjoint rows retain the full 8192-copy stress coverage.
        rows = [
            row.contiguous()
            for row in permutation[:UNION_CAPACITY].view(QUERY_COUNT, TOPK)
        ]
        return rows, None
    if profile != "miss_overlap":
        raise ValueError(f"unknown TopK profile: {profile}")

    if unique_miss_count is None or miss_overlap_rate is None:
        raise ValueError("miss-overlap profile requires miss count and overlap rate")
    if not 0 <= unique_miss_count <= UNION_CAPACITY:
        raise ValueError("unique miss count must be in [0,8192]")
    if not 0.0 <= miss_overlap_rate <= 1.0:
        raise ValueError("miss overlap rate must be in [0,1]")
    if hit_overlap_rate is None:
        hit_overlap_rate = 0.0
    if not 0.0 <= hit_overlap_rate <= 1.0:
        raise ValueError("hit overlap rate must be in [0,1]")

    # Every unique miss has one unavoidable query occurrence and at most three
    # duplicates.  Spread the duplicates as evenly as possible across unique
    # miss tokens, then balance each membership degree over the four queries.
    extra_occurrences = round(3 * unique_miss_count * miss_overlap_rate)
    total_occurrences = unique_miss_count + extra_occurrences
    if total_occurrences > UNION_CAPACITY:
        raise ValueError(
            "requested miss workload exceeds four TopK2048 rows: "
            f"unique={unique_miss_count}, overlap_rate={miss_overlap_rate:.6f}, "
            f"query_occurrences={total_occurrences}"
        )
    if unique_miss_count:
        base_extra, higher_degree_count = divmod(
            extra_occurrences, unique_miss_count
        )
    else:
        base_extra = higher_degree_count = 0
    base_degree = 1 + base_extra
    degree_counts = {degree: 0 for degree in range(1, QUERY_COUNT + 1)}
    if unique_miss_count:
        degree_counts[base_degree] = unique_miss_count - higher_degree_count
        if higher_degree_count:
            degree_counts[base_degree + 1] = higher_degree_count

    rows_parts: list[list[torch.Tensor]] = [[] for _ in range(QUERY_COUNT)]
    query_loads = [0] * QUERY_COUNT
    cursor = 0
    preferred_misses = permutation[:unique_miss_count].contiguous()
    for degree, count in degree_counts.items():
        memberships = list(combinations(range(QUERY_COUNT), degree))
        for local_idx in range(count):
            token = permutation[cursor : cursor + 1]
            cursor += 1
            offset = local_idx % len(memberships)
            ordered_memberships = memberships[offset:] + memberships[:offset]

            def load_score(members: tuple[int, ...]) -> tuple[int, int, int]:
                next_loads = [
                    load + int(query_idx in members)
                    for query_idx, load in enumerate(query_loads)
                ]
                return (
                    max(next_loads),
                    max(next_loads) - min(next_loads),
                    sum(query_loads[query_idx] for query_idx in members),
                )

            members = min(ordered_memberships, key=load_score)
            for query_idx in members:
                rows_parts[query_idx].append(token)
                query_loads[query_idx] += 1

    # Share a configurable prefix of the remaining hit positions across all
    # four queries. The rest stays query-local, preserving the former fixture
    # when hit_overlap_rate is zero.
    hit_counts = [
        TOPK - sum(int(part.numel()) for part in parts)
        for parts in rows_parts
    ]
    shared_hit_count = round(min(hit_counts) * hit_overlap_rate)
    shared_hits = permutation[cursor : cursor + shared_hit_count]
    cursor += shared_hit_count
    for query_idx in range(QUERY_COUNT):
        needed = hit_counts[query_idx]
        if needed < 0:
            raise ValueError(
                f"query={query_idx} receives {-needed + TOPK} misses, "
                f"exceeding TopK={TOPK}"
            )
        rows_parts[query_idx].append(shared_hits)
        unique_hit_count = needed - shared_hit_count
        rows_parts[query_idx].append(
            permutation[cursor : cursor + unique_hit_count]
        )
        cursor += unique_hit_count
    if cursor > source_capacity:
        raise AssertionError("miss-overlap fixture exceeds source capacity")

    rows: list[torch.Tensor] = []
    for query_idx in range(QUERY_COUNT):
        row = torch.cat(rows_parts[query_idx])
        if row.numel() != TOPK:
            raise AssertionError("miss-overlap fixture TopK row size changed")
        row = row[torch.randperm(TOPK, generator=generator)]
        rows.append(row.contiguous())
    return rows, preferred_misses


def make_cache_state(
    *,
    topk_rows: list[torch.Tensor],
    source_capacity: int,
    cache_tokens: tuple[int, ...],
    req_pool_entries: torch.Tensor,
    miss_fractions: tuple[float, ...],
    generator: torch.Generator,
    exact_miss_counts: tuple[int, ...] | None = None,
    preferred_misses: list[torch.Tensor] | None = None,
) -> torch.Tensor:
    batch_size = len(cache_tokens)
    if len(miss_fractions) != batch_size:
        raise ValueError("cache token and miss-fraction counts differ")
    if exact_miss_counts is not None and len(exact_miss_counts) != batch_size:
        raise ValueError("exact miss counts must match the batch size")
    if preferred_misses is not None and len(preferred_misses) != batch_size:
        raise ValueError("preferred misses must match the batch size")

    state = torch.full(
        (batch_size + 3, source_capacity), -777, dtype=torch.int32
    )
    all_tokens = torch.arange(source_capacity, dtype=torch.int64)
    for request, (budget, miss_fraction) in enumerate(
        zip(cache_tokens, miss_fractions)
    ):
        pool_row = int(req_pool_entries[request])
        state[pool_row].fill_(-1)
        rows = topk_rows[request * QUERY_COUNT : (request + 1) * QUERY_COUNT]
        union = ordered_union(rows)
        if union.numel() > budget:
            raise ValueError(
                f"request={request}: TopK union={union.numel()} exceeds C={budget}"
            )
        if budget == source_capacity:
            requested = 0 if exact_miss_counts is None else exact_miss_counts[request]
            preferred = (
                torch.empty(0, dtype=torch.int64)
                if preferred_misses is None
                else preferred_misses[request]
            )
            if requested or preferred.numel():
                raise ValueError("fully cached rows require miss_count=0")
            cached = all_tokens
        else:
            if exact_miss_counts is None:
                miss_count = min(
                    int(round(float(union.numel()) * miss_fraction)),
                    int(union.numel()),
                )
                misses = union[:miss_count]
            else:
                miss_count = int(exact_miss_counts[request])
                if not 0 <= miss_count <= int(union.numel()):
                    raise ValueError(
                        f"request={request}: invalid exact miss_count={miss_count}"
                    )
                if preferred_misses is not None:
                    misses = preferred_misses[request]
                    if misses.numel() != miss_count:
                        raise ValueError(
                            f"request={request}: preferred miss count="
                            f"{misses.numel()}, expected={miss_count}"
                        )
                    if (
                        torch.unique(misses).numel() != miss_count
                        or not bool(torch.isin(misses, union).all())
                    ):
                        raise ValueError("preferred misses must be unique TopK tokens")
                else:
                    occurrences = torch.bincount(
                        torch.cat(rows), minlength=source_capacity
                    )
                    ordered_candidates = torch.cat(
                        (
                            union[occurrences[union] == 2],
                            union[occurrences[union] > 2],
                            union[occurrences[union] == 1],
                        )
                    )
                    misses = ordered_candidates[:miss_count]
            miss_mask = torch.zeros(source_capacity, dtype=torch.bool)
            miss_mask[misses] = True
            hits = union[~miss_mask[union]]
            union_mask = torch.zeros(source_capacity, dtype=torch.bool)
            union_mask[union] = True
            fillers = all_tokens[~union_mask]
            needed = budget - int(hits.numel())
            if needed < 0 or fillers.numel() < needed:
                raise AssertionError("cannot construct the requested cache state")
            cached = torch.cat((hits, fillers[:needed]))

        if cached.numel() != budget or torch.unique(cached).numel() != budget:
            raise AssertionError("initial cache must contain exactly C unique tokens")
        state[pool_row, cached] = torch.randperm(
            budget, generator=generator
        ).to(torch.int32)
    return state.contiguous()


def make_case(
    *,
    name: str,
    device: torch.device,
    batch_size: int,
    source_capacity: int,
    cache_tokens: int,
    miss_fractions: tuple[float, ...],
    seed: int,
    topk_profile: str,
    exact_miss_count: int | None = None,
    miss_overlap_rate: float | None = None,
    hit_overlap_rate: float | None = None,
) -> MtpCase:
    if source_capacity < UNION_CAPACITY or source_capacity % BLOCK_SIZE:
        raise ValueError("source capacity must be block aligned and >=8192")
    if not UNION_CAPACITY <= cache_tokens <= source_capacity:
        raise ValueError("cache tokens must be in [8192, source capacity]")
    if len(miss_fractions) != batch_size:
        raise ValueError("miss fractions must match batch size")
    if miss_overlap_rate is not None and topk_profile != "miss_overlap":
        raise ValueError("miss overlap construction requires topk_profile=miss_overlap")
    if hit_overlap_rate is not None and topk_profile != "miss_overlap":
        raise ValueError("hit overlap construction requires topk_profile=miss_overlap")

    generator = torch.Generator().manual_seed(seed)
    req_pool_entries = torch.randperm(
        batch_size + 3, generator=generator
    )[:batch_size].to(torch.int32)
    topk_rows: list[torch.Tensor] = []
    unions: list[torch.Tensor] = []
    preferred_misses: list[torch.Tensor] | None = (
        [] if miss_overlap_rate is not None else None
    )
    for _ in range(batch_size):
        rows, request_preferred_misses = _make_request_topk(
            source_capacity,
            profile=topk_profile,
            generator=generator,
            unique_miss_count=exact_miss_count,
            miss_overlap_rate=miss_overlap_rate,
            hit_overlap_rate=hit_overlap_rate,
        )
        topk_rows.extend(rows)
        unions.append(ordered_union(rows))
        if preferred_misses is not None:
            if request_preferred_misses is None:
                raise AssertionError("miss-overlap fixture omitted preferred misses")
            preferred_misses.append(request_preferred_misses)
    budgets = (cache_tokens,) * batch_size
    exact_counts = (
        None if exact_miss_count is None else (exact_miss_count,) * batch_size
    )
    initial_cache = make_cache_state(
        topk_rows=topk_rows,
        source_capacity=source_capacity,
        cache_tokens=budgets,
        req_pool_entries=req_pool_entries,
        miss_fractions=miss_fractions,
        generator=generator,
        exact_miss_counts=exact_counts,
        preferred_misses=preferred_misses,
    )
    cache_tokens_cpu = torch.full(
        (batch_size,), cache_tokens, dtype=torch.int32
    )
    return MtpCase(
        name=name,
        device=device,
        batch_size=batch_size,
        source_capacity=source_capacity,
        cache_tokens=cache_tokens_cpu.to(device),
        cache_tokens_cpu=cache_tokens_cpu,
        req_pool_entries_cpu=req_pool_entries,
        initial_cache_cpu=initial_cache,
        topk_cpu=topk_rows,
        union_cpu=unions,
    )


def materialize_metadata(
    case: MtpCase,
    *,
    cache_state_cpu: torch.Tensor | None = None,
) -> tuple[torch.Tensor, ...]:
    """Create caller-owned COPYSFA-MTP metadata from the synthetic fixture."""

    before_state = (
        case.initial_cache_cpu if cache_state_cpu is None else cache_state_cpu
    )
    topk_dst_slots = torch.full(
        (case.batch_size * QUERY_COUNT, 1, TOPK), -313, dtype=torch.int32
    )
    topk_src_ids = torch.full_like(topk_dst_slots, -313)
    miss_src_ids = torch.full(
        (case.batch_size, UNION_CAPACITY), -313, dtype=torch.int32
    )
    miss_dst_slots = torch.full_like(miss_src_ids, -313)
    miss_counts = torch.zeros(case.batch_size, dtype=torch.int32)
    topk_miss_counts = torch.zeros(
        case.batch_size * QUERY_COUNT, dtype=torch.int32
    )

    for request in range(case.batch_size):
        pool_row = int(case.req_pool_entries_cpu[request])
        budget = int(case.cache_tokens_cpu[request])
        before = before_state[pool_row, : case.source_capacity].to(torch.int64)
        rows = case.topk_cpu[
            request * QUERY_COUNT : (request + 1) * QUERY_COUNT
        ]
        union = case.union_cpu[request]
        misses = union[before[union] < 0]
        count = int(misses.numel())

        union_mask = torch.zeros(case.source_capacity, dtype=torch.bool)
        union_mask[union] = True
        cached_tokens = torch.nonzero(before >= 0).flatten()
        victims = cached_tokens[~union_mask[cached_tokens]][:count]
        if victims.numel() != count:
            raise AssertionError(
                f"request={request}: not enough non-TopK cache victims"
            )
        destinations = before[victims]
        after = before.clone()
        after[victims] = -1
        after[misses] = destinations

        for query_idx, row in enumerate(rows):
            output_row = request * QUERY_COUNT + query_idx
            miss_mask = before[row] < 0
            # Match fused_li_manage_mtp's source-aware contract: misses are
            # the contiguous prefix and hits follow it in the same TopK row.
            order = torch.cat(
                (torch.nonzero(miss_mask).flatten(),
                 torch.nonzero(~miss_mask).flatten())
            )
            ordered_row = row[order]
            slots = after[ordered_row]
            if bool((slots < 0).any()) or torch.unique(slots).numel() != TOPK:
                raise AssertionError("materialized TopK slots are invalid")
            topk_dst_slots[output_row, 0] = slots.to(torch.int32)
            topk_src_ids[output_row, 0] = torch.where(
                before[ordered_row] < 0,
                ordered_row,
                torch.full_like(ordered_row, -1),
            ).to(torch.int32)
            topk_miss_counts[output_row] = int(miss_mask.sum())
        miss_src_ids[request, :count] = misses.to(torch.int32)
        miss_dst_slots[request, :count] = destinations.to(torch.int32)
        miss_counts[request] = count

    return tuple(
        tensor.to(case.device)
        for tensor in (
            topk_dst_slots,
            topk_src_ids,
            miss_src_ids,
            miss_dst_slots,
            miss_counts,
            topk_miss_counts,
        )
    )


def metadata_stats(
    case: MtpCase, metadata: tuple[torch.Tensor, ...]
) -> tuple[float, float]:
    total_query_miss_occurrences = int((metadata[1] >= 0).sum().cpu())
    total_unique_misses = int(metadata[4].sum().cpu())
    query_miss_occurrences_mean = (
        total_query_miss_occurrences / case.batch_size
    )
    overlap_rate = (
        0.0
        if total_unique_misses == 0
        else (total_query_miss_occurrences - total_unique_misses)
        / (3 * total_unique_misses)
    )
    return query_miss_occurrences_mean, overlap_rate


def swapped_from_cpu(cpu: torch.Tensor, device: torch.device) -> torch.Tensor:
    tensor = torch_npu.empty_with_swapped_memory(
        cpu.shape,
        dtype=cpu.dtype,
        device=device,
    )
    tensor.fill_(0)
    tensor.add_(cpu.to(device))
    return tensor


def apply_scatter_reference(
    expected_kpe: torch.Tensor,
    expected_ckv: torch.Tensor,
    dram_kpe: torch.Tensor,
    dram_ckv: torch.Tensor,
    hbm_block_table: torch.Tensor,
    dram_block_table: torch.Tensor,
    source_ids: torch.Tensor,
    destination_slots: torch.Tensor,
    copy_counts: torch.Tensor,
) -> None:
    for request, count_value in enumerate(copy_counts.tolist()):
        count = int(count_value)
        if count == 0:
            continue
        sources = source_ids[request, :count].to(torch.int64)
        destinations = destination_slots[request, :count].to(torch.int64)
        src_blocks = dram_block_table[
            request, sources // BLOCK_SIZE
        ].to(torch.int64)
        src_offsets = sources % BLOCK_SIZE
        dst_blocks = hbm_block_table[
            request, destinations // BLOCK_SIZE
        ].to(torch.int64)
        dst_offsets = destinations % BLOCK_SIZE
        expected_kpe[dst_blocks, dst_offsets] = dram_kpe[src_blocks, src_offsets]
        expected_ckv[dst_blocks, dst_offsets] = dram_ckv[src_blocks, src_offsets]
