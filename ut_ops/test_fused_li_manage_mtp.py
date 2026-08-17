"""Semantic, graph, and latency checks for the bundled GLM MTP3 LIM op."""

from __future__ import annotations

import argparse
import statistics
from collections.abc import Callable
from dataclasses import dataclass
from itertools import accumulate

import torch
import torch_npu  # type: ignore

import nanovllm.ops  # noqa: F401  # Load repository-local custom operators.
from ut_ops._op_utils import require_local_opapi


QUERY_COUNT = 4
HEADS = 32
HEAD_DIM = 128
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
    dtype: torch.dtype
    batch_size: int
    source_capacity: int
    query: torch.Tensor
    key: torch.Tensor
    weights: torch.Tensor
    query_scale: torch.Tensor
    key_scale: torch.Tensor
    actual_query_lens: torch.Tensor
    actual_key_lens: torch.Tensor
    offload_key_lens: torch.Tensor
    req_valid: torch.Tensor
    cache_state: torch.Tensor
    req_pool_entries: torch.Tensor
    cache_tokens: torch.Tensor
    candidate_lens: torch.Tensor
    block_table: torch.Tensor
    req_pool_entries_cpu: torch.Tensor
    cache_tokens_cpu: torch.Tensor
    candidate_lens_cpu: torch.Tensor
    block_table_cpu: torch.Tensor
    initial_cache_cpu: torch.Tensor
    topk_cpu: list[torch.Tensor]
    union_cpu: list[torch.Tensor]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate NanovllmFusedLiManageMtp on one NPU."
    )
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument(
        "--q-heads",
        type=int,
        choices=(32, 64),
        default=32,
        help="Number of query heads per packed MTP query.",
    )
    parser.add_argument("--source-len", type=int, default=20992)
    parser.add_argument("--cache-tokens", type=int, default=8192)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--graph-replays", type=int, default=3)
    parser.add_argument(
        "--perf-query-miss-count",
        type=int,
        default=200,
        help="Exact old-cache misses in each of the four TopK2048 rows.",
    )
    parser.add_argument(
        "--perf-query-noise",
        type=float,
        default=0.25,
        help=(
            "Noise applied to four correlated queries. The resulting TopK union "
            "should be about 3000-4000 tokens per request."
        ),
    )
    parser.add_argument(
        "--skip-performance",
        action="store_true",
        help="Run semantic and graph checks without the B=24 benchmark.",
    )
    return parser.parse_args()


def _validate_cli(args: argparse.Namespace) -> None:
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive.")
    if (
        args.source_len < UNION_CAPACITY
        or args.source_len > MAX_SOURCE_CAPACITY
        or args.source_len % BLOCK_SIZE
    ):
        raise ValueError(
            "--source-len must be block aligned and in [8192, 2^18]."
        )
    if not UNION_CAPACITY <= args.cache_tokens <= args.source_len:
        raise ValueError("--cache-tokens must be in [8192, source-len].")
    if args.warmup < 0 or args.iters <= 0 or args.graph_replays < 0:
        raise ValueError(
            "--warmup/--graph-replays must be >=0 and --iters must be >0."
        )
    if not 0 <= args.perf_query_miss_count <= TOPK:
        raise ValueError("--perf-query-miss-count must be in [0,2048].")
    if args.perf_query_noise <= 0:
        raise ValueError("--perf-query-noise must be positive.")
    if (
        not args.skip_performance
        and args.cache_tokens == args.source_len
        and args.perf_query_miss_count != 0
    ):
        raise ValueError(
            "--perf-query-miss-count must be 0 when the full source is cached."
        )


def _random_block_table(
    batch_size: int,
    blocks_per_request: int,
    generator: torch.Generator,
) -> tuple[torch.Tensor, int]:
    total_blocks = batch_size * blocks_per_request
    table = torch.randperm(total_blocks, generator=generator).to(torch.int32)
    return table.view(batch_size, blocks_per_request).contiguous(), total_blocks


def _ordered_union(rows: list[torch.Tensor]) -> torch.Tensor:
    ordered: list[int] = []
    seen: set[int] = set()
    for row in rows:
        for value in row.tolist():
            token = int(value)
            if token not in seen:
                seen.add(token)
                ordered.append(token)
    return torch.tensor(ordered, dtype=torch.int64)


def _call_native_li(
    query: torch.Tensor,
    key: torch.Tensor,
    weights: torch.Tensor,
    block_table: torch.Tensor,
    candidate_lens: torch.Tensor,
    query_ends: torch.Tensor,
) -> torch.Tensor:
    result = torch_npu.npu_lightning_indexer(
        query=query,
        key=key,
        weights=weights,
        actual_seq_lengths_query=query_ends,
        actual_seq_lengths_key=candidate_lens,
        block_table=block_table,
        layout_query="TND",
        layout_key="PA_BSND",
        sparse_count=TOPK,
        sparse_mode=3,
    )
    topk = result[0] if isinstance(result, (tuple, list)) else result
    if not isinstance(topk, torch.Tensor):
        raise TypeError("native LightningIndexer did not return a Tensor")
    return topk


def _native_topk(
    query: torch.Tensor,
    key: torch.Tensor,
    weights: torch.Tensor,
    block_table: torch.Tensor,
    candidate_lens: torch.Tensor,
    cache_tokens_cpu: torch.Tensor,
) -> list[torch.Tensor]:
    """Use native LightningIndexer as the four independent-query golden."""

    batch_size = int(candidate_lens.numel())
    active_query_rows: list[int] = []
    active_request_rows: list[int] = []
    candidate_cpu = candidate_lens.cpu()
    for request in range(batch_size):
        if int(cache_tokens_cpu[request]) == 0:
            continue
        if int(candidate_cpu[request]) < TOPK:
            raise AssertionError("active MTP LIM rows must have >=2048 candidates")
        for query_idx in range(QUERY_COUNT):
            active_query_rows.append(request * QUERY_COUNT + query_idx)
            active_request_rows.append(request)

    result_rows = [torch.empty(0, dtype=torch.int64) for _ in range(batch_size * QUERY_COUNT)]
    if not active_query_rows:
        return result_rows

    query_index = torch.tensor(active_query_rows, dtype=torch.int64, device=query.device)
    request_index = torch.tensor(active_request_rows, dtype=torch.int64, device=query.device)
    active_query = query.index_select(0, query_index).contiguous()
    active_weights = weights.index_select(0, query_index).contiguous()
    active_table = block_table.index_select(0, request_index).contiguous()
    active_lens = candidate_lens.index_select(0, request_index).contiguous()
    query_ends = torch.arange(
        1,
        len(active_query_rows) + 1,
        dtype=torch.int32,
        device=query.device,
    )
    topk = _call_native_li(
        active_query,
        key,
        active_weights,
        active_table,
        active_lens,
        query_ends,
    )
    expected_shape = (len(active_query_rows), 1, TOPK)
    if tuple(topk.shape) != expected_shape:
        raise AssertionError(
            f"native LightningIndexer shape={tuple(topk.shape)}, "
            f"expected={expected_shape}"
        )
    topk_cpu = topk.reshape(len(active_query_rows), TOPK).cpu().to(torch.int64)
    for local_row, query_row in enumerate(active_query_rows):
        result_rows[query_row] = topk_cpu[local_row].contiguous()
    return result_rows


def _make_cache_state(
    *,
    topk_rows: list[torch.Tensor],
    candidate_lens: tuple[int, ...],
    cache_tokens: tuple[int, ...],
    req_pool_entries: torch.Tensor,
    source_capacity: int,
    miss_fractions: tuple[float, ...],
    generator: torch.Generator,
    pool_size: int,
    exact_miss_counts: tuple[int, ...] | None = None,
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    if exact_miss_counts is not None and len(exact_miss_counts) != len(
        candidate_lens
    ):
        raise ValueError("exact miss counts must match the batch size")
    # Non-active rows use a distinct sentinel so accidental writes are visible.
    state = torch.full((pool_size, source_capacity), -777, dtype=torch.int32)
    unions: list[torch.Tensor] = []
    for request, (candidate_len, budget, miss_fraction) in enumerate(
        zip(candidate_lens, cache_tokens, miss_fractions)
    ):
        pool_row = int(req_pool_entries[request])
        state[pool_row].fill_(-65536)
        if budget == 0:
            if exact_miss_counts is not None and exact_miss_counts[request] != 0:
                raise ValueError("C=0 rows require exact miss_count=0")
            unions.append(torch.empty(0, dtype=torch.int64))
            continue

        request_topk = topk_rows[
            request * QUERY_COUNT : (request + 1) * QUERY_COUNT
        ]
        union = _ordered_union(request_topk)
        unions.append(union)
        if union.numel() > budget:
            raise AssertionError(
                f"request={request}: union={union.numel()} exceeds C={budget}"
            )

        if budget == candidate_len:
            if exact_miss_counts is not None and exact_miss_counts[request] != 0:
                raise ValueError("fully cached rows require exact miss_count=0")
            cached = torch.arange(candidate_len, dtype=torch.int64)
        else:
            if exact_miss_counts is None:
                miss_count = min(
                    int(round(float(union.numel()) * miss_fraction)),
                    int(union.numel()),
                )
                hits = union[miss_count:]
            else:
                miss_count = int(exact_miss_counts[request])
                if miss_count < 0 or miss_count > int(union.numel()):
                    raise ValueError(
                        f"request={request}: exact miss_count={miss_count} "
                        f"must be in [0,{union.numel()}]"
                    )
                # Prefer source tokens shared by multiple query rows. This
                # models the GLM MTP3 workload: four TopK sets overlap, and
                # about 300 unique union tokens need one physical copy each.
                occurrences = torch.bincount(
                    torch.cat(request_topk), minlength=candidate_len
                )
                repeated = union[occurrences[union] > 1]
                single = union[occurrences[union] == 1]
                misses = torch.cat((repeated, single))[:miss_count]
                miss_mask = torch.zeros(candidate_len, dtype=torch.bool)
                miss_mask[misses] = True
                hits = union[~miss_mask[union]]
            union_mask = torch.zeros(candidate_len, dtype=torch.bool)
            union_mask[union] = True
            fillers = torch.arange(candidate_len, dtype=torch.int64)[~union_mask]
            needed = budget - int(hits.numel())
            if needed < 0 or fillers.numel() < needed:
                raise AssertionError(
                    f"cannot construct request={request} C={budget} state"
                )
            cached = torch.cat((hits, fillers[:needed]))

        if cached.numel() != budget or torch.unique(cached).numel() != budget:
            raise AssertionError("initial cache tokens must be unique and exactly C")
        slot_permutation = torch.randperm(budget, generator=generator).to(torch.int32)
        state[pool_row, cached] = slot_permutation
    return state.contiguous(), unions


def _make_balanced_mtp_cache_state(
    *,
    topk_rows: list[torch.Tensor],
    candidate_lens: tuple[int, ...],
    cache_tokens: tuple[int, ...],
    req_pool_entries: torch.Tensor,
    source_capacity: int,
    per_query_miss_count: int,
    generator: torch.Generator,
    pool_size: int,
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    """Build the MTP3 performance state without redefining miss as union miss.

    Up to half of each query's misses are shared by all four queries. The rest
    use exact pair-membership buckets on opposite query-pair edges. For the
    target value 200 this gives exactly 200 misses in every TopK row and
    300-400 unique union misses per request.
    """

    state = torch.full((pool_size, source_capacity), -777, dtype=torch.int32)
    unions: list[torch.Tensor] = []
    for request, (candidate_len, budget) in enumerate(
        zip(candidate_lens, cache_tokens)
    ):
        pool_row = int(req_pool_entries[request])
        state[pool_row].fill_(-65536)
        rows = topk_rows[request * QUERY_COUNT : (request + 1) * QUERY_COUNT]
        union = _ordered_union(rows)
        unions.append(union)
        if budget == 0:
            if per_query_miss_count:
                raise ValueError("C=0 rows cannot have performance misses")
            continue
        if union.numel() > budget:
            raise AssertionError(
                f"request={request}: TopK union={union.numel()} exceeds C={budget}"
            )
        if budget == candidate_len and per_query_miss_count:
            raise ValueError("fully cached rows cannot have performance misses")

        membership = torch.zeros(candidate_len, dtype=torch.uint8)
        for query_idx, row in enumerate(rows):
            membership[row] |= 1 << query_idx
        available_by_mask = {
            mask: torch.nonzero(membership == mask).flatten().to(torch.int64)
            for mask in range(1, 1 << QUERY_COUNT)
        }
        selected_parts: list[torch.Tensor] = []
        common = available_by_mask[0b1111]
        common_count = min(per_query_miss_count // 2, int(common.numel()))
        selected_parts.append(common[:common_count])
        pair_degree = per_query_miss_count - common_count
        opposite_pair_masks = (
            (0b0011, 0b1100),
            (0b0101, 0b1010),
            (0b1001, 0b0110),
        )
        capacities = [
            min(
                int(available_by_mask[left].numel()),
                int(available_by_mask[right].numel()),
            )
            for left, right in opposite_pair_masks
        ]
        if sum(capacities) < pair_degree:
            raise AssertionError(
                f"request={request}: pair-overlap capacity={capacities} cannot "
                f"supply degree={pair_degree}; adjust --perf-query-noise"
            )
        pair_counts = [min(pair_degree // 3, capacity) for capacity in capacities]
        remaining = pair_degree - sum(pair_counts)
        while remaining:
            progressed = False
            for idx, capacity in enumerate(capacities):
                if pair_counts[idx] < capacity:
                    pair_counts[idx] += 1
                    remaining -= 1
                    progressed = True
                    if remaining == 0:
                        break
            if not progressed:
                raise AssertionError("failed to distribute pair-overlap misses")
        wanted_by_mask: dict[int, int] = {}
        for (left, right), count in zip(opposite_pair_masks, pair_counts):
            wanted_by_mask[left] = count
            wanted_by_mask[right] = count
        for mask, wanted in wanted_by_mask.items():
            available = available_by_mask[mask]
            selected_parts.append(available[:wanted])
        selected_misses = torch.cat(selected_parts)
        if torch.unique(selected_misses).numel() != selected_misses.numel():
            raise AssertionError("balanced MTP miss construction produced duplicates")

        per_query_counts = [
            int(torch.isin(selected_misses, row).sum()) for row in rows
        ]
        if per_query_counts != [per_query_miss_count] * QUERY_COUNT:
            raise AssertionError(
                f"request={request}: constructed per-query misses="
                f"{per_query_counts}, expected={per_query_miss_count}"
            )

        missing_mask = torch.zeros(candidate_len, dtype=torch.bool)
        missing_mask[selected_misses] = True
        hits = union[~missing_mask[union]]
        union_mask = torch.zeros(candidate_len, dtype=torch.bool)
        union_mask[union] = True
        fillers = torch.arange(candidate_len, dtype=torch.int64)[~union_mask]
        needed = budget - int(hits.numel())
        if needed < 0 or fillers.numel() < needed:
            raise AssertionError(
                f"request={request}: cannot build C={budget} cache with "
                f"{selected_misses.numel()} selected misses"
            )
        cached = torch.cat((hits, fillers[:needed]))
        if cached.numel() != budget or torch.unique(cached).numel() != budget:
            raise AssertionError("balanced initial cache must contain exactly C tokens")
        state[pool_row, cached] = torch.randperm(
            budget, generator=generator
        ).to(torch.int32)

    return state.contiguous(), unions


def make_case(
    *,
    name: str,
    device: torch.device,
    dtype: torch.dtype,
    candidate_lens: tuple[int, ...],
    cache_tokens: tuple[int, ...],
    miss_fractions: tuple[float, ...],
    seed: int,
    source_capacity: int | None = None,
    exact_miss_counts: tuple[int, ...] | None = None,
    correlated_query_noise: float | None = None,
    balanced_query_miss_count: int | None = None,
) -> MtpCase:
    batch_size = len(candidate_lens)
    if not (
        len(cache_tokens) == batch_size == len(miss_fractions)
        and batch_size > 0
    ):
        raise ValueError("candidate/cache/miss tuples must have equal nonzero length")
    capacity = source_capacity or max(candidate_lens)
    if capacity % BLOCK_SIZE or capacity > MAX_SOURCE_CAPACITY:
        raise ValueError("source capacity must be block aligned and <=2^18")
    if any(length <= 0 or length > capacity or length % BLOCK_SIZE for length in candidate_lens):
        raise ValueError("candidate lengths must be positive, aligned, and <= capacity")
    for length, budget in zip(candidate_lens, cache_tokens):
        if budget == 0:
            continue
        if budget < min(length, UNION_CAPACITY) or budget > length:
            raise ValueError(
                f"active row requires min(candidate,8192)<=C<=candidate, got {length=}, {budget=}"
            )

    generator = torch.Generator().manual_seed(seed)
    block_table_cpu, physical_blocks = _random_block_table(
        batch_size, capacity // BLOCK_SIZE, generator
    )
    req_entries_cpu = torch.randperm(batch_size + 3, generator=generator)[
        :batch_size
    ].to(torch.int32)
    candidate_cpu = torch.tensor(candidate_lens, dtype=torch.int32)
    cache_tokens_cpu = torch.tensor(cache_tokens, dtype=torch.int32)

    if correlated_query_noise is None:
        query_cpu = torch.randn(
            batch_size * QUERY_COUNT,
            HEADS,
            HEAD_DIM,
            generator=generator,
            dtype=torch.float32,
        )
        # Positive weights match GLM indexer usage and avoid unstable cancellation.
        weights_cpu = torch.rand(
            batch_size * QUERY_COUNT,
            HEADS,
            generator=generator,
            dtype=torch.float32,
        )
    else:
        if correlated_query_noise <= 0:
            raise ValueError("correlated query noise must be positive")
        base_query = torch.randn(
            batch_size, 1, HEADS, HEAD_DIM, generator=generator
        )
        query_noise = torch.randn(
            batch_size, QUERY_COUNT, HEADS, HEAD_DIM, generator=generator
        )
        query_cpu = (
            base_query + correlated_query_noise * query_noise
        ).reshape(batch_size * QUERY_COUNT, HEADS, HEAD_DIM)
        # One request's four nearby decode positions use the same positive
        # head weighting in this focused index-management benchmark.
        base_weights = torch.rand(
            batch_size, 1, HEADS, generator=generator, dtype=torch.float32
        )
        weights_cpu = base_weights.expand(-1, QUERY_COUNT, -1).reshape(
            batch_size * QUERY_COUNT, HEADS
        ).contiguous()
    query_cpu = query_cpu.to(dtype)
    weights_cpu = weights_cpu.to(dtype)
    torch.manual_seed(seed + 991)
    key = torch.randn(
        physical_blocks,
        BLOCK_SIZE,
        1,
        HEAD_DIM,
        dtype=dtype,
        device=device,
    )
    query = query_cpu.to(device)
    weights = weights_cpu.to(device)
    block_table = block_table_cpu.to(device)
    candidate = candidate_cpu.to(device)
    topk_rows = _native_topk(
        query,
        key,
        weights,
        block_table,
        candidate,
        cache_tokens_cpu,
    )
    if balanced_query_miss_count is None:
        cache_cpu, union_rows = _make_cache_state(
            topk_rows=topk_rows,
            candidate_lens=candidate_lens,
            cache_tokens=cache_tokens,
            req_pool_entries=req_entries_cpu,
            source_capacity=capacity,
            miss_fractions=miss_fractions,
            generator=generator,
            pool_size=batch_size + 3,
            exact_miss_counts=exact_miss_counts,
        )
    else:
        if exact_miss_counts is not None:
            raise ValueError(
                "balanced per-query misses and exact union misses are exclusive"
            )
        cache_cpu, union_rows = _make_balanced_mtp_cache_state(
            topk_rows=topk_rows,
            candidate_lens=candidate_lens,
            cache_tokens=cache_tokens,
            req_pool_entries=req_entries_cpu,
            source_capacity=capacity,
            per_query_miss_count=balanced_query_miss_count,
            generator=generator,
            pool_size=batch_size + 3,
        )
    return MtpCase(
        name=name,
        device=device,
        dtype=dtype,
        batch_size=batch_size,
        source_capacity=capacity,
        query=query,
        key=key,
        weights=weights,
        query_scale=torch.empty(
            query.shape[:2], dtype=torch.float32, device=device
        ),
        key_scale=torch.empty(
            (physical_blocks, BLOCK_SIZE, 1), dtype=torch.float32, device=device
        ),
        actual_query_lens=torch.arange(
            QUERY_COUNT, batch_size * QUERY_COUNT + 1, QUERY_COUNT,
            dtype=torch.int32, device=device,
        ),
        actual_key_lens=candidate,
        offload_key_lens=candidate,
        req_valid=(cache_tokens_cpu > 0).to(torch.int32).to(device),
        cache_state=torch.full(
            (batch_size + 3,), -1, dtype=torch.int32, device=device
        ),
        req_pool_entries=req_entries_cpu.to(device),
        cache_tokens=cache_tokens_cpu.to(device),
        candidate_lens=candidate,
        block_table=block_table,
        req_pool_entries_cpu=req_entries_cpu,
        cache_tokens_cpu=cache_tokens_cpu,
        candidate_lens_cpu=candidate_cpu,
        block_table_cpu=block_table_cpu,
        initial_cache_cpu=cache_cpu,
        topk_cpu=topk_rows,
        union_cpu=union_rows,
    )


def call_mtp(case: MtpCase, cache_slots: torch.Tensor):
    outputs = make_outputs(case)
    call_mtp_with_buffers(case, cache_slots, *outputs)
    return outputs


def call_mtp_with_buffers(
    case: MtpCase,
    cache_slots: torch.Tensor,
    topk_slots: torch.Tensor,
    topk_source_ids: torch.Tensor,
    miss_source_ids: torch.Tensor,
    miss_destination_slots: torch.Tensor,
    miss_counts: torch.Tensor,
):
    torch.ops.nanovllm_dsa.fused_li_manage_mtp.default(
        case.weights,
        case.query_scale,
        case.query,
        case.key_scale,
        case.key,
        case.block_table,
        case.actual_query_lens,
        case.actual_key_lens,
        case.offload_key_lens,
        case.req_valid,
        case.req_pool_entries,
        case.cache_state,
        cache_slots,
        topk_source_ids,
        topk_slots,
        miss_source_ids,
        miss_destination_slots,
        miss_counts,
    )
    return (
        topk_slots,
        topk_source_ids,
        miss_source_ids,
        miss_destination_slots,
        miss_counts,
    )


def make_outputs(case: MtpCase) -> tuple[torch.Tensor, ...]:
    topk_slots = torch.full(
        (case.query.size(0), 1, TOPK),
        -313,
        dtype=torch.int32,
        device=case.device,
    )
    topk_source_ids = torch.full_like(topk_slots, -313)
    miss_sources = torch.full(
        (case.batch_size, UNION_CAPACITY),
        -313,
        dtype=torch.int32,
        device=case.device,
    )
    miss_destinations = torch.full_like(miss_sources, -313)
    miss_counts = torch.full(
        (case.batch_size,), -313, dtype=torch.int32, device=case.device
    )
    return (
        topk_slots,
        topk_source_ids,
        miss_sources,
        miss_destinations,
        miss_counts,
    )


def _pack_query_counts(case: MtpCase, counts: tuple[int, ...]) -> list[int]:
    """Select the first 1-4 query rows per request and update packed metadata."""

    if len(counts) != case.batch_size or any(count < 1 or count > 4 for count in counts):
        raise ValueError("query counts must contain one value in [1,4] per request")
    rows = [
        request * QUERY_COUNT + query_idx
        for request, count in enumerate(counts)
        for query_idx in range(count)
    ]
    row_index = torch.tensor(rows, dtype=torch.int64, device=case.device)
    case.query = case.query.index_select(0, row_index).contiguous()
    case.weights = case.weights.index_select(0, row_index).contiguous()
    case.query_scale = torch.empty(
        case.query.shape[:2], dtype=torch.float32, device=case.device
    )
    case.actual_query_lens = torch.tensor(
        list(accumulate(counts)),
        dtype=torch.int32,
        device=case.device,
    )
    return rows


def _native_packed_topk(case: MtpCase, lengths: torch.Tensor) -> torch.Tensor:
    """Native LI golden with each packed query evaluated independently."""

    ends = case.actual_query_lens.cpu().tolist()
    starts = [0, *ends[:-1]]
    request_rows = [
        request
        for request, (start, end) in enumerate(zip(starts, ends))
        for _ in range(end - start)
    ]
    request_index = torch.tensor(
        request_rows, dtype=torch.int64, device=case.device
    )
    return _call_native_li(
        case.query,
        case.key,
        case.weights,
        case.block_table.index_select(0, request_index).contiguous(),
        lengths.index_select(0, request_index).contiguous(),
        torch.arange(
            1, case.query.size(0) + 1,
            dtype=torch.int32, device=case.device,
        ),
    ).reshape(case.query.size(0), TOPK).cpu().to(torch.int64)


def run_meta_check() -> None:
    batch_size = 2
    token_rows = 3  # mixed MTP0 (one query) + MTP1 (two queries)
    source_capacity = 8192
    meta = torch.device("meta")
    query = torch.empty(
        token_rows, HEADS, HEAD_DIM, device=meta, dtype=torch.bfloat16
    )
    key = torch.empty(
        batch_size * source_capacity // BLOCK_SIZE,
        BLOCK_SIZE,
        1,
        HEAD_DIM,
        device=meta,
        dtype=torch.bfloat16,
    )
    weights = torch.empty(
        token_rows, HEADS, device=meta, dtype=torch.bfloat16
    )
    query_scale = torch.empty(
        token_rows, HEADS, device=meta, dtype=torch.float32
    )
    key_scale = torch.empty(
        batch_size * source_capacity // BLOCK_SIZE, BLOCK_SIZE, 1,
        device=meta, dtype=torch.float32,
    )
    req_entries = torch.empty(batch_size, device=meta, dtype=torch.int32)
    cache_slots = torch.empty(
        batch_size + 1, source_capacity, device=meta, dtype=torch.int32
    )
    cache_tokens = torch.empty(batch_size, device=meta, dtype=torch.int32)
    candidate_lens = torch.empty(batch_size, device=meta, dtype=torch.int32)
    actual_query_lens = torch.tensor([1, 3], device=meta, dtype=torch.int32)
    req_valid = torch.ones(batch_size, device=meta, dtype=torch.int32)
    cache_state = torch.full(
        (batch_size + 1,), -1, device=meta, dtype=torch.int32
    )
    block_table = torch.empty(
        batch_size,
        source_capacity // BLOCK_SIZE,
        device=meta,
        dtype=torch.int32,
    )
    expected_shapes = (
        (token_rows, 1, TOPK),
        (token_rows, 1, TOPK),
        (batch_size, UNION_CAPACITY),
        (batch_size, UNION_CAPACITY),
        (batch_size,),
    )
    buffers = (
        torch.empty(expected_shapes[0], device=meta, dtype=torch.int32),
        torch.empty(expected_shapes[1], device=meta, dtype=torch.int32),
        torch.empty(expected_shapes[2], device=meta, dtype=torch.int32),
        torch.empty(expected_shapes[3], device=meta, dtype=torch.int32),
        torch.empty(expected_shapes[4], device=meta, dtype=torch.int32),
    )
    result = torch.ops.nanovllm_dsa.fused_li_manage_mtp.default(
        weights,
        query_scale,
        query,
        key_scale,
        key,
        block_table,
        actual_query_lens,
        candidate_lens,
        candidate_lens,
        req_valid,
        req_entries,
        cache_state,
        cache_slots,
        buffers[1],
        buffers[0],
        buffers[2],
        buffers[3],
        buffers[4],
    )
    if result is not None:
        raise AssertionError("Fused LIM Manage MTP must return None")
    if tuple(tuple(output.shape) for output in buffers) != expected_shapes:
        raise AssertionError("Fused LIM Manage MTP Meta buffers changed shape")
    if any(output.dtype != torch.int32 for output in buffers):
        raise AssertionError("Fused LIM Manage MTP Meta buffers changed dtype")
    print(
        "FUSED_LI_MANAGE_MTP_META_CHECK caller_owned_buffers=1 return_none=1 ok=1",
        flush=True,
    )


def validate_result(
    case: MtpCase,
    before_cpu: torch.Tensor,
    cache_slots: torch.Tensor,
    outputs: tuple[torch.Tensor, ...],
    *,
    label: str,
) -> list[int]:
    (
        topk_slots,
        topk_source_ids,
        miss_sources,
        miss_destinations,
        miss_counts,
    ) = outputs
    after_cpu = cache_slots.cpu()
    topk_slots_cpu = topk_slots.reshape(-1, TOPK).cpu().to(torch.int64)
    topk_source_ids_cpu = (
        topk_source_ids.reshape(-1, TOPK).cpu().to(torch.int64)
    )
    sources_cpu = miss_sources.cpu().to(torch.int64)
    destinations_cpu = miss_destinations.cpu().to(torch.int64)
    counts_cpu = miss_counts.cpu().to(torch.int64)
    active_pool_rows = set(int(value) for value in case.req_pool_entries_cpu.tolist())

    for pool_row in range(before_cpu.shape[0]):
        if pool_row not in active_pool_rows and not torch.equal(
            before_cpu[pool_row], after_cpu[pool_row]
        ):
            raise AssertionError(f"{label}: unused pool row {pool_row} changed")

    expected_counts: list[int] = []
    for request in range(case.batch_size):
        pool_row = int(case.req_pool_entries_cpu[request])
        budget = int(case.cache_tokens_cpu[request])
        candidate_len = int(case.candidate_lens_cpu[request])
        before = before_cpu[pool_row]
        after = after_cpu[pool_row]
        if budget == 0:
            expected_counts.append(0)
            if int(counts_cpu[request]) != 0:
                raise AssertionError(f"{label}: C=0 row has nonzero miss count")
            if not torch.equal(before, after):
                raise AssertionError(f"{label}: C=0 pool row changed")
            continue

        valid_slots = after[:candidate_len]
        valid_slots = valid_slots[valid_slots >= 0].to(torch.int64)
        if (
            valid_slots.numel() != budget
            or torch.unique(valid_slots).numel() != budget
            or int(valid_slots.min()) != 0
            or int(valid_slots.max()) != budget - 1
        ):
            raise AssertionError(
                f"{label}: request={request} state is not a permutation of [0,C)"
            )

        # LightningIndexer guarantees the top-k set, but its merge order is
        # not a public contract (ties may be ordered differently by the fused
        # MTP kernel).  Reconstruct token IDs from the returned logical slots,
        # compare each query as a set against the native golden, and retain the
        # fused operator's deterministic order for the ordered-union checks.
        cached_tokens = torch.nonzero(after[:candidate_len] >= 0).flatten()
        slot_to_token = torch.full((budget,), -1, dtype=torch.int64)
        slot_to_token[after[cached_tokens].to(torch.int64)] = cached_tokens
        actual_topk_rows: list[torch.Tensor] = []
        for query_idx in range(QUERY_COUNT):
            query_row = request * QUERY_COUNT + query_idx
            actual_slots = topk_slots_cpu[query_row]
            if (
                bool((actual_slots < 0).any())
                or bool((actual_slots >= budget).any())
                or torch.unique(actual_slots).numel() != TOPK
            ):
                negative = int((actual_slots < 0).sum())
                out_of_range = int((actual_slots >= budget).sum())
                unique = int(torch.unique(actual_slots).numel())
                raise AssertionError(
                    f"{label}: request={request} query={query_idx} slots invalid "
                    f"(negative={negative}, out_of_range={out_of_range}, "
                    f"unique={unique}/{TOPK}, min={int(actual_slots.min())}, "
                    f"max={int(actual_slots.max())})"
                )
            actual_tokens = slot_to_token[actual_slots]
            expected_sources = torch.where(
                before[actual_tokens] < 0,
                actual_tokens,
                torch.full_like(actual_tokens, -1),
            )
            if not torch.equal(
                topk_source_ids_cpu[query_row], expected_sources
            ):
                raise AssertionError(
                    f"{label}: request={request} query={query_idx} "
                    "aligned source IDs differ"
                )
            golden_tokens = case.topk_cpu[query_row]
            if not torch.equal(
                torch.sort(actual_tokens).values,
                torch.sort(golden_tokens).values,
            ):
                missing = torch.isin(golden_tokens, actual_tokens, invert=True)
                extra = torch.isin(actual_tokens, golden_tokens, invert=True)
                raise AssertionError(
                    f"{label}: request={request} query={query_idx} topk token "
                    f"set differs (missing={int(missing.sum())}, "
                    f"extra={int(extra.sum())})"
                )
            actual_topk_rows.append(actual_tokens)

        union = _ordered_union(actual_topk_rows)
        golden_union = case.union_cpu[request]
        if not torch.equal(
            torch.sort(union).values,
            torch.sort(golden_union).values,
        ):
            raise AssertionError(
                f"{label}: request={request} topk union set differs"
            )
        if bool((after[union] < 0).any()):
            raise AssertionError(f"{label}: request={request} union is not cached")

        expected_misses = union[before[union] < 0]
        expected_count = int(expected_misses.numel())
        expected_counts.append(expected_count)
        actual_count = int(counts_cpu[request])
        if actual_count != expected_count:
            raise AssertionError(
                f"{label}: request={request} miss_count={actual_count}, "
                f"expected={expected_count}"
            )
        if not torch.equal(
            sources_cpu[request, :actual_count], expected_misses
        ):
            raise AssertionError(
                f"{label}: request={request} ordered union misses differ"
            )
        active_destinations = destinations_cpu[request, :actual_count]
        if actual_count and (
            bool((active_destinations < 0).any())
            or bool((active_destinations >= budget).any())
            or torch.unique(active_destinations).numel() != actual_count
        ):
            raise AssertionError(
                f"{label}: request={request} miss destination slots invalid"
            )

        old_hits = union[before[union] >= 0]
        if old_hits.numel() and not torch.equal(after[old_hits], before[old_hits]):
            raise AssertionError(
                f"{label}: request={request} changed an existing hit slot"
            )
        if actual_count and not torch.equal(
            after[expected_misses].to(torch.int64), active_destinations
        ):
            raise AssertionError(
                f"{label}: request={request} miss-to-slot mapping is wrong"
            )

        old_tokens = torch.nonzero(before[:candidate_len] >= 0).flatten()
        evicted = old_tokens[after[old_tokens] < 0]
        if evicted.numel():
            union_mask = torch.zeros(candidate_len, dtype=torch.bool)
            union_mask[union] = True
            if bool(union_mask[evicted].any()):
                raise AssertionError(
                    f"{label}: request={request} evicted a union token"
                )

    return expected_counts


def _compare_valid_outputs(
    case: MtpCase,
    left: tuple[torch.Tensor, ...],
    right: tuple[torch.Tensor, ...],
    *,
    label: str,
) -> None:
    left_topk = left[0].reshape(case.batch_size, QUERY_COUNT, TOPK).cpu()
    right_topk = right[0].reshape(case.batch_size, QUERY_COUNT, TOPK).cpu()
    left_topk_sources = left[1].reshape(
        case.batch_size, QUERY_COUNT, TOPK
    ).cpu()
    right_topk_sources = right[1].reshape(
        case.batch_size, QUERY_COUNT, TOPK
    ).cpu()
    for request in range(case.batch_size):
        if int(case.cache_tokens_cpu[request]) == 0:
            continue
        if not torch.equal(left_topk[request], right_topk[request]):
            raise AssertionError(f"{label}: request={request} topk_slots differ")
        if not torch.equal(
            left_topk_sources[request], right_topk_sources[request]
        ):
            raise AssertionError(
                f"{label}: request={request} topk_source_ids differ"
            )
    left_counts = left[4].cpu()
    right_counts = right[4].cpu()
    if not torch.equal(left_counts, right_counts):
        raise AssertionError(f"{label}: miss_counts differ")
    for request, count_value in enumerate(left_counts.tolist()):
        count = int(count_value)
        if not torch.equal(
            left[2][request, :count].cpu(), right[2][request, :count].cpu()
        ):
            raise AssertionError(f"{label}: request={request} miss IDs differ")
        if not torch.equal(
            left[3][request, :count].cpu(), right[3][request, :count].cpu()
        ):
            raise AssertionError(f"{label}: request={request} miss slots differ")


def run_semantic_case(case: MtpCase) -> None:
    fresh_cache = case.initial_cache_cpu.to(case.device)
    before = case.initial_cache_cpu.clone()
    fresh_outputs = call_mtp(case, fresh_cache)
    torch.npu.synchronize()
    counts = validate_result(
        case, before, fresh_cache, fresh_outputs, label=f"{case.name}/fresh"
    )

    persistent_cache = case.initial_cache_cpu.to(case.device)
    persistent_buffers = make_outputs(case)
    persistent_outputs = call_mtp_with_buffers(
        case, persistent_cache, *persistent_buffers
    )
    torch.npu.synchronize()
    validate_result(
        case,
        before,
        persistent_cache,
        persistent_outputs,
        label=f"{case.name}/persistent",
    )
    for returned, supplied in zip(persistent_outputs, persistent_buffers):
        if returned.data_ptr() != supplied.data_ptr():
            raise AssertionError(
                f"{case.name}: caller-owned output buffer was replaced"
            )
    _compare_valid_outputs(
        case, fresh_outputs, persistent_outputs, label=case.name
    )
    if not torch.equal(fresh_cache.cpu(), persistent_cache.cpu()):
        raise AssertionError(f"{case.name}: cache states differ")

    repeat_before = fresh_cache.cpu()
    repeat_outputs = call_mtp(case, fresh_cache)
    torch.npu.synchronize()
    repeat_counts = validate_result(
        case,
        repeat_before,
        fresh_cache,
        repeat_outputs,
        label=f"{case.name}/repeat",
    )
    if any(repeat_counts):
        raise AssertionError(f"{case.name}: identical repeat must be zero miss")
    print(
        "FUSED_LI_MANAGE_MTP_SEMANTIC_CHECK "
        f"case={case.name} dtype={case.dtype} batch={case.batch_size} "
        f"candidate_lens={case.candidate_lens_cpu.tolist()} "
        f"cache_tokens={case.cache_tokens_cpu.tolist()} "
        f"miss_counts={counts} shuffled_pool_entries=1 random_block_table=1 ok=1",
        flush=True,
    )


def run_new_state_cases(device: torch.device, seed: int) -> None:
    case = make_case(
        name="new_state_protocol", device=device, dtype=torch.bfloat16,
        candidate_lens=(4096,), cache_tokens=(4096,), miss_fractions=(0.0,),
        seed=seed,
    )
    pool_row = int(case.req_pool_entries_cpu[0])

    # Empty hot buffer: negative values use 1-based slot encoding.
    empty = torch.full_like(case.initial_cache_cpu, -777)
    empty[pool_row].fill_(-65536)
    empty[pool_row, :4096] = -torch.arange(1, 4097, dtype=torch.int32)
    case.cache_state.fill_(-1)
    case.cache_state[pool_row] = 0
    cache = empty.to(device)
    outputs = call_mtp(case, cache)
    torch.npu.synchronize()
    union = case.union_cpu[0]
    after = cache.cpu()[pool_row]
    if bool((after[union] < 0).any()):
        raise AssertionError("free-slot path did not cache the complete TopK union")
    count = int(outputs[4][0].cpu())
    if count != int(union.numel()):
        raise AssertionError("free-slot path miss union count differs")
    if not torch.equal(
        torch.sort(outputs[2][0, :count].cpu().to(torch.int64)).values,
        torch.sort(union).values,
    ):
        raise AssertionError("free-slot path source union differs")
    state = int(case.cache_state[pool_row].cpu())
    if state < 0 or not (-4096 <= int(after[state]) <= -1):
        raise AssertionError("partially free cache_state does not point to a free binding")

    # -3 bypasses cache management and writes raw 0-based token IDs to both
    # TopK outputs.
    case.cache_state[pool_row] = -3
    plain_before = cache.clone()
    plain = call_mtp(case, cache)
    torch.npu.synchronize()
    if not torch.equal(cache, plain_before) or int(plain[4][0].cpu()) != 0:
        raise AssertionError("plain LI path modified cache state or reported misses")
    for query_idx in range(QUERY_COUNT):
        src = plain[1][query_idx].reshape(-1).cpu().to(torch.int64)
        dst = plain[0][query_idx].reshape(-1).cpu().to(torch.int64)
        if not torch.equal(src, dst) or not torch.equal(
            torch.sort(src).values, torch.sort(case.topk_cpu[query_idx]).values
        ):
            raise AssertionError("plain LI outputs differ from native TopK")
    print("FUSED_LI_MANAGE_MTP_NEW_STATE_CHECK free_slots=1 plain_li=1 ok=1", flush=True)


def run_variable_query_case(device: torch.device, seed: int) -> None:
    counts = (1, 2, 3, 4)
    lengths = (4096, 8192, 12288, 16384)
    case = make_case(
        name="variable_mtp0123", device=device, dtype=torch.bfloat16,
        candidate_lens=lengths, cache_tokens=lengths,
        miss_fractions=(0.0,) * len(counts), seed=seed,
    )
    _pack_query_counts(case, counts)
    golden = _native_packed_topk(case, case.offload_key_lens)
    cache = case.initial_cache_cpu.to(device)
    outputs = call_mtp(case, cache)
    torch.npu.synchronize()
    actual_slots = outputs[0].reshape(-1, TOPK).cpu().to(torch.int64)
    actual_sources = outputs[1].reshape(-1, TOPK).cpu().to(torch.int64)
    ends = case.actual_query_lens.cpu().tolist()
    starts = [0, *ends[:-1]]
    after = cache.cpu()
    for request, (start, end) in enumerate(zip(starts, ends)):
        pool_row = int(case.req_pool_entries_cpu[request])
        length = lengths[request]
        slot_to_token = torch.empty(length, dtype=torch.int64)
        tokens = torch.arange(length, dtype=torch.int64)
        slot_to_token[after[pool_row, :length].to(torch.int64)] = tokens
        for row in range(start, end):
            actual_tokens = slot_to_token[actual_slots[row]]
            if not torch.equal(
                torch.sort(actual_tokens).values,
                torch.sort(golden[row]).values,
            ):
                raise AssertionError(
                    f"variable MTP request={request} row={row} TopK differs"
                )
            if not bool((actual_sources[row] == -1).all()):
                raise AssertionError("fully cached variable MTP row reported misses")
    if not bool((outputs[4].cpu() == 0).all()):
        raise AssertionError("fully cached variable MTP requests reported misses")
    print(
        "FUSED_LI_MANAGE_MTP_VARIABLE_QUERY_CHECK counts=[1,2,3,4] ok=1",
        flush=True,
    )


def run_offload_tail_and_plain_case(device: torch.device, seed: int) -> None:
    case = make_case(
        name="offload_tail_plain", device=device, dtype=torch.bfloat16,
        candidate_lens=(4096,), cache_tokens=(4096,), miss_fractions=(0.0,),
        source_capacity=8192, seed=seed,
    )
    case.actual_key_lens = torch.tensor([8192], dtype=torch.int32, device=device)
    case.offload_key_lens = torch.tensor([4096], dtype=torch.int32, device=device)
    offload_golden = _native_packed_topk(case, case.offload_key_lens)
    full_golden = _native_packed_topk(case, case.actual_key_lens)
    if all(
        torch.equal(torch.sort(left).values, torch.sort(right).values)
        for left, right in zip(offload_golden, full_golden)
    ):
        raise AssertionError("tail boundary workload did not distinguish full/offload LI")

    pool_row = int(case.req_pool_entries_cpu[0])
    cache = case.initial_cache_cpu.to(device)
    managed = call_mtp(case, cache)
    torch.npu.synchronize()
    managed_slots = managed[0].reshape(-1, TOPK).cpu().to(torch.int64)
    after = cache.cpu()[pool_row]
    slot_to_token = torch.empty(4096, dtype=torch.int64)
    slot_to_token[after[:4096].to(torch.int64)] = torch.arange(4096)
    for row in range(QUERY_COUNT):
        tokens = slot_to_token[managed_slots[row]]
        if not torch.equal(
            torch.sort(tokens).values, torch.sort(offload_golden[row]).values
        ):
            raise AssertionError("managed LI included dense tail tokens")

    case.cache_state[pool_row] = -3
    before_plain = cache.clone()
    plain = call_mtp(case, cache)
    torch.npu.synchronize()
    if not torch.equal(cache, before_plain) or int(plain[4][0].cpu()) != 0:
        raise AssertionError("plain LI changed cache or miss count")
    for row in range(QUERY_COUNT):
        src = plain[1][row].reshape(-1).cpu().to(torch.int64)
        dst = plain[0][row].reshape(-1).cpu().to(torch.int64)
        if not torch.equal(src, dst) or not torch.equal(
            torch.sort(src).values, torch.sort(full_golden[row]).values
        ):
            raise AssertionError("plain LI did not use full actual key range")
    print(
        "FUSED_LI_MANAGE_MTP_SOURCE_RANGE_CHECK offload_prefix=4096 "
        "dense_tail=4096 plain_full=8192 ok=1",
        flush=True,
    )


def run_free_scan_transition_case(device: torch.device, seed: int) -> None:
    case = make_case(
        name="free_scan_transition", device=device, dtype=torch.bfloat16,
        candidate_lens=(4096,), cache_tokens=(4096,), miss_fractions=(0.0,),
        seed=seed,
    )
    _pack_query_counts(case, (1,))
    golden = _native_packed_topk(case, case.offload_key_lens)[0]
    pool_row = int(case.req_pool_entries_cpu[0])
    initial = torch.full_like(case.initial_cache_cpu, -777)
    initial[pool_row].fill_(-65536)
    initial[pool_row, :TOPK] = -torch.arange(1, TOPK + 1, dtype=torch.int32)
    direct = golden[golden < TOPK]
    scanned = golden[golden >= TOPK]
    if direct.numel() == 0 or scanned.numel() == 0:
        raise AssertionError("free-slot workload did not cover both allocation paths")
    case.cache_state.fill_(-1)
    case.cache_state[pool_row] = 0
    cache = initial.to(device)
    outputs = call_mtp(case, cache)
    torch.npu.synchronize()
    after = cache.cpu()[pool_row]
    if int(outputs[4][0].cpu()) != TOPK or bool((after[golden] < 0).any()):
        raise AssertionError("free-slot scan failed to cache the complete TopK")
    slots = after[golden].to(torch.int64)
    if torch.unique(slots).numel() != TOPK or int(slots.min()) != 0 or int(slots.max()) != TOPK - 1:
        raise AssertionError("free-slot scan did not produce a slot permutation")
    if int(case.cache_state[pool_row].cpu()) != -1:
        raise AssertionError("consuming the final free slot did not set cache_state=-1")
    print(
        "FUSED_LI_MANAGE_MTP_FREE_SCAN_CHECK direct_binding=1 "
        "unbound_scan=1 transition_full=1 ok=1",
        flush=True,
    )


def run_skip_boundary_case(device: torch.device, seed: int) -> None:
    case = make_case(
        name="skip_boundaries", device=device, dtype=torch.bfloat16,
        candidate_lens=(4096, 4096, 4096, 4096),
        cache_tokens=(4096, 4096, 4096, 4096),
        miss_fractions=(0.0, 0.0, 0.0, 0.0), seed=seed,
    )
    case.req_valid[0] = 0
    case.cache_state[int(case.req_pool_entries_cpu[1])] = -2
    case.offload_key_lens[2] = 4095  # not block aligned
    case.actual_key_lens[3] = 2048
    case.offload_key_lens[3] = 4096  # offload > actual
    before = case.initial_cache_cpu.to(device)
    cache = before.clone()
    outputs = call_mtp(case, cache)
    torch.npu.synchronize()
    if not torch.equal(cache, before):
        raise AssertionError("skipped/invalid requests changed cache slots")
    if not bool((outputs[4].cpu() == 0).all()):
        raise AssertionError("skipped/invalid requests must write zero miss counts")
    # TopK rows of skipped requests are intentionally unspecified and must not
    # be consumed by the caller. Only cache immutability and miss_counts=0 are
    # part of the skip contract.
    print(
        "FUSED_LI_MANAGE_MTP_SKIP_CHECK req_valid=0 state_minus2=1 "
        "unaligned_offload=1 offload_gt_actual=1 ok=1",
        flush=True,
    )


def run_graph_case(device: torch.device, seed: int, replays: int) -> None:
    if replays <= 0:
        return
    batch_size = 6
    case = make_case(
        name="graph",
        device=device,
        dtype=torch.bfloat16,
        candidate_lens=(20992,) * batch_size,
        cache_tokens=(8192,) * batch_size,
        miss_fractions=(0.75,) * batch_size,
        source_capacity=32768,
        seed=seed + 3000,
    )
    graph_cache = case.initial_cache_cpu.to(device)
    eager_cache = case.initial_cache_cpu.to(device)
    graph_buffers = make_outputs(case)

    # Warm up the caller-owned-buffer contract on disposable state.
    warm_cache = case.initial_cache_cpu.to(device)
    call_mtp_with_buffers(case, warm_cache, *make_outputs(case))
    torch.npu.synchronize()

    graph = torch.npu.NPUGraph()
    pool = torch.npu.graph_pool_handle()
    with torch.npu.graph(graph, pool=pool):
        graph_outputs = call_mtp_with_buffers(
            case, graph_cache, *graph_buffers
        )
    torch.npu.synchronize()
    for returned, supplied in zip(graph_outputs, graph_buffers):
        if returned.data_ptr() != supplied.data_ptr():
            raise AssertionError("graph capture lost caller-owned LIM buffers")

    # Capture may execute the op. Start replay and eager reference from exactly
    # the same state, then let both states evolve across all replays.
    graph_cache.copy_(case.initial_cache_cpu.to(device))
    eager_cache.copy_(case.initial_cache_cpu.to(device))
    torch.npu.synchronize()
    generator = torch.Generator().manual_seed(seed + 3017)
    max_blocks = case.source_capacity // BLOCK_SIZE
    base_table = case.block_table_cpu.clone()
    for replay in range(replays):
        query_cpu = torch.randn(
            case.query.shape, generator=generator, dtype=torch.float32
        ).to(case.dtype)
        weights_cpu = torch.rand(
            case.weights.shape, generator=generator, dtype=torch.float32
        ).to(case.dtype)
        request_order = torch.roll(
            torch.arange(batch_size, dtype=torch.int64), shifts=replay + 1
        )
        pool_entries_cpu = torch.roll(
            case.req_pool_entries_cpu, shifts=replay + 1
        )
        candidate_len = min(20992 + replay * 1024, case.source_capacity)
        candidate_len -= candidate_len % BLOCK_SIZE
        candidate_cpu = torch.full(
            (batch_size,), candidate_len, dtype=torch.int32
        )
        table_cpu = base_table.index_select(0, request_order).contiguous()
        if table_cpu.shape != (batch_size, max_blocks):
            raise AssertionError("graph block-table refresh changed shape")

        case.query.copy_(query_cpu.to(device))
        case.weights.copy_(weights_cpu.to(device))
        case.req_pool_entries.copy_(pool_entries_cpu.to(device))
        case.candidate_lens.copy_(candidate_cpu.to(device))
        case.actual_key_lens.copy_(candidate_cpu.to(device))
        case.offload_key_lens.copy_(candidate_cpu.to(device))
        case.block_table.copy_(table_cpu.to(device))
        torch.npu.current_stream().synchronize()
        graph.replay()
        torch.npu.synchronize()

        eager_outputs = call_mtp(case, eager_cache)
        torch.npu.synchronize()
        _compare_valid_outputs(
            case, graph_outputs, eager_outputs, label=f"graph/replay={replay}"
        )
        if not torch.equal(graph_cache.cpu(), eager_cache.cpu()):
            raise AssertionError(f"graph replay={replay} cache state differs")

    print(
        "FUSED_LI_MANAGE_MTP_GRAPH_CHECK "
        f"batch={batch_size} replays={replays} dynamic_query=1 "
        "dynamic_weights=1 dynamic_pool_entries=1 dynamic_lengths=1 "
        "dynamic_block_table=1 evolving_state=1 ok=1",
        flush=True,
    )
    del case, graph_cache, eager_cache, warm_cache
    torch.npu.empty_cache()


def _event_us(
    runner: Callable[[], object],
    *,
    warmup: int,
    iters: int,
    reset: Callable[[], None] | None = None,
) -> float:
    """Measure only NPU work; mutable request-state reset is not timed."""

    for _ in range(warmup):
        if reset is not None:
            reset()
        runner()
    torch.npu.synchronize()

    samples_ms: list[float] = []
    for _ in range(iters):
        if reset is not None:
            reset()
            torch.npu.synchronize()
        start = torch.npu.Event(enable_timing=True)
        end = torch.npu.Event(enable_timing=True)
        start.record()
        runner()
        end.record()
        end.synchronize()
        samples_ms.append(float(start.elapsed_time(end)))
    return statistics.mean(samples_ms) * 1000.0


def _assert_topk_sets(
    actual: torch.Tensor,
    expected_rows: list[torch.Tensor],
    *,
    label: str,
) -> None:
    actual_cpu = actual.reshape(-1, TOPK).cpu().to(torch.int64)
    expected = torch.stack(expected_rows)
    if not torch.equal(
        torch.sort(actual_cpu, dim=1).values,
        torch.sort(expected, dim=1).values,
    ):
        raise AssertionError(f"{label}: TopK sets differ from native golden")


def _query_miss_counts(case: MtpCase) -> tuple[list[int], list[int]]:
    per_query_totals = [0] * QUERY_COUNT
    union_counts: list[int] = []
    for request in range(case.batch_size):
        pool_row = int(case.req_pool_entries_cpu[request])
        before = case.initial_cache_cpu[pool_row]
        rows = case.topk_cpu[
            request * QUERY_COUNT : (request + 1) * QUERY_COUNT
        ]
        for query_idx, row in enumerate(rows):
            per_query_totals[query_idx] += int((before[row] < 0).sum())
        union = case.union_cpu[request]
        union_counts.append(int((before[union] < 0).sum()))
    return per_query_totals, union_counts


def run_performance_case(
    device: torch.device,
    *,
    batch_size: int,
    source_len: int,
    cache_tokens: int,
    seed: int,
    warmup: int,
    iters: int,
    perf_query_miss_count: int,
    perf_query_noise: float,
) -> None:
    case = make_case(
        name="performance",
        device=device,
        dtype=torch.bfloat16,
        candidate_lens=(source_len,) * batch_size,
        cache_tokens=(cache_tokens,) * batch_size,
        miss_fractions=(0.0,) * batch_size,
        seed=seed + 6000,
        correlated_query_noise=perf_query_noise,
        balanced_query_miss_count=perf_query_miss_count,
    )

    union_sizes = [int(row.numel()) for row in case.union_cpu]
    union_mean = statistics.mean(union_sizes)
    union_min = min(union_sizes)
    union_max = max(union_sizes)
    union_target_ok = 3000.0 <= union_mean <= 4000.0
    print(
        "FUSED_LI_MANAGE_MTP_WORKLOAD_CHECK "
        f"batch={batch_size} candidate_len={source_len} "
        f"query_noise={perf_query_noise:.4f} "
        f"topk_union_min={union_min} topk_union_mean={union_mean:.2f} "
        f"topk_union_max={union_max} target_range=[3000,4000] "
        f"target_ok={int(union_target_ok)}",
        flush=True,
    )
    if not union_target_ok:
        raise AssertionError(
            "MTP3 TopK union is outside the intended 3000-4000 range; "
            "adjust --perf-query-noise and rerun."
        )

    per_query_totals, expected_union_counts = _query_miss_counts(case)
    expected_query_total = batch_size * perf_query_miss_count
    minimum_unique_per_request = (
        2 * perf_query_miss_count - perf_query_miss_count // 2
    )
    maximum_unique_per_request = 2 * perf_query_miss_count
    if per_query_totals != [expected_query_total] * QUERY_COUNT:
        raise AssertionError(
            f"performance workload per-query misses={per_query_totals}, "
            f"expected={expected_query_total} for every query"
        )
    if any(
        count < minimum_unique_per_request
        or count > maximum_unique_per_request
        for count in expected_union_counts
    ):
        raise AssertionError(
            f"performance workload union misses={expected_union_counts}, "
            f"expected range=[{minimum_unique_per_request},"
            f"{maximum_unique_per_request}] per request"
        )
    unique_union_mean = statistics.mean(expected_union_counts)

    correctness_cache = case.initial_cache_cpu.to(device)
    correctness_outputs = call_mtp(case, correctness_cache)
    torch.npu.synchronize()
    correctness_counts = validate_result(
        case,
        case.initial_cache_cpu,
        correctness_cache,
        correctness_outputs,
        label="performance/correctness",
    )
    if correctness_counts != expected_union_counts:
        raise AssertionError(
            f"MTP LIM miss_counts={correctness_counts}, "
            f"expected={expected_union_counts}"
        )
    print(
        "FUSED_LI_MANAGE_MTP_TARGET_BATCH_CHECK "
        f"batch={batch_size} candidate_len={source_len} "
        f"cache_tokens={cache_tokens} "
        f"per_query_misses={perf_query_miss_count} "
        f"unique_union_misses_min={min(expected_union_counts)} "
        f"unique_union_misses_mean={unique_union_mean:.2f} "
        f"unique_union_misses_max={max(expected_union_counts)} "
        f"per_query_miss_totals={per_query_totals} "
        f"total_union_misses={sum(correctness_counts)} "
        "one_request_per_owner=1 ok=1",
        flush=True,
    )

    query_view = case.query.view(batch_size, QUERY_COUNT, HEADS, HEAD_DIM)
    weights_view = case.weights.view(batch_size, QUERY_COUNT, HEADS)
    single_query = query_view[:, 0].contiguous()
    single_weights = weights_view[:, 0].contiguous()
    single_query_ends = torch.arange(
        1, batch_size + 1, dtype=torch.int32, device=device
    )
    mtp_query_ends = torch.arange(
        QUERY_COUNT,
        batch_size * QUERY_COUNT + 1,
        QUERY_COUNT,
        dtype=torch.int32,
        device=device,
    )

    native_single = _call_native_li(
        single_query,
        case.key,
        single_weights,
        case.block_table,
        case.candidate_lens,
        single_query_ends,
    )
    native_mtp = _call_native_li(
        case.query,
        case.key,
        case.weights,
        case.block_table,
        case.candidate_lens,
        mtp_query_ends,
    )
    torch.npu.synchronize()
    _assert_topk_sets(
        native_single,
        [case.topk_cpu[request * QUERY_COUNT] for request in range(batch_size)],
        label="official_li_single",
    )
    expected_native_mtp_shape = (batch_size * QUERY_COUNT, 1, TOPK)
    if tuple(native_mtp.shape) != expected_native_mtp_shape:
        raise AssertionError(
            f"official LI MTP3 shape={tuple(native_mtp.shape)}, "
            f"expected={expected_native_mtp_shape}"
        )
    # This call is the score/topk timing baseline, not the semantic golden.
    # With sparse_mode=3, a qlen=4 TND sequence uses right-down causal
    # boundaries, whereas LIM-MTP intentionally searches the same immutable
    # prefill source for all four rows. The independent qlen=1 native golden
    # used to build `case.topk_cpu` validates LIM-MTP semantics above.
    print(
        "FUSED_LI_MANAGE_MTP_OFFICIAL_BASELINE_CHECK "
        "layout=TND query_len=4 sparse_mode=3 "
        "role=score_topk_timing semantic_golden=independent_qlen1 ok=1",
        flush=True,
    )

    mtp_initial = case.initial_cache_cpu.to(device)
    mtp_cache = torch.empty_like(mtp_initial)
    mtp_buffers = make_outputs(case)

    def native_single_step() -> object:
        return _call_native_li(
            single_query,
            case.key,
            single_weights,
            case.block_table,
            case.candidate_lens,
            single_query_ends,
        )

    def native_mtp_step() -> object:
        return _call_native_li(
            case.query,
            case.key,
            case.weights,
            case.block_table,
            case.candidate_lens,
            mtp_query_ends,
        )

    def fused_mtp_step() -> object:
        return call_mtp_with_buffers(case, mtp_cache, *mtp_buffers)

    official_li_single_us = _event_us(
        native_single_step, warmup=warmup, iters=iters
    )
    official_li_mtp3_us = _event_us(
        native_mtp_step, warmup=warmup, iters=iters
    )
    fused_lim_mtp3_us = _event_us(
        fused_mtp_step,
        warmup=warmup,
        iters=iters,
        reset=lambda: mtp_cache.copy_(mtp_initial),
    )
    management_mtp3_us = fused_lim_mtp3_us - official_li_mtp3_us
    print(
        "FUSED_LI_MANAGE_MTP_MANAGEMENT_RESULT "
        f"batch={batch_size} candidate_len={source_len} "
        f"cache_tokens={cache_tokens} "
        f"per_query_misses={perf_query_miss_count} "
        f"unique_union_misses_mean={unique_union_mean:.2f} "
        f"topk_union_mean={union_mean:.2f} "
        f"official_li_single_us={official_li_single_us:.3f} "
        f"official_li_mtp3_us={official_li_mtp3_us:.3f} "
        "official_li_mtp3_layout=TND_qlen4_sparse3 "
        f"fused_lim_mtp3_us={fused_lim_mtp3_us:.3f} "
        f"index_management_mtp3_us={management_mtp3_us:+.3f} "
        "single_lim_baseline=external "
        f"timer=npu_event performance_assert=0 warmup={warmup} iters={iters}",
        flush=True,
    )
    del case
    torch.npu.empty_cache()


def _swapped_from_cpu(cpu: torch.Tensor, device: torch.device) -> torch.Tensor:
    tensor = torch_npu.empty_with_swapped_memory(
        cpu.shape,
        dtype=cpu.dtype,
        device=device,
    )
    tensor.fill_(0)
    tensor.add_(cpu.to(device))
    return tensor


def _apply_scatter_reference(
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
        expected_kpe[dst_blocks, dst_offsets] = dram_kpe[
            src_blocks, src_offsets
        ]
        expected_ckv[dst_blocks, dst_offsets] = dram_ckv[
            src_blocks, src_offsets
        ]


def main() -> None:
    global HEADS
    args = parse_args()
    HEADS = args.q_heads
    _validate_cli(args)
    device = torch.device(args.device)
    if device.type != "npu":
        raise ValueError("--device must select one NPU, for example npu:0.")
    torch.npu.set_device(device)
    torch.npu.config.allow_internal_format = False
    opapi_path = require_local_opapi()
    print(
        f"FUSED_LI_MANAGE_MTP_OPAPI path={opapi_path} local=1",
        flush=True,
    )
    print(
        "FUSED_LI_MANAGE_MTP_CONFIG "
        f"device={device} query_len={QUERY_COUNT} heads={HEADS} "
        f"topk={TOPK} union_capacity={UNION_CAPACITY} seed={args.seed}",
        flush=True,
    )
    run_meta_check()

    mixed = make_case(
        name="mixed_b6_bf16",
        device=device,
        dtype=torch.bfloat16,
        candidate_lens=(1024, 4096, 8192, 20992, 32768, 65536),
        cache_tokens=(0, 4096, 8192, 8192, 12288, 12288),
        miss_fractions=(0.0, 0.0, 0.0, 0.02, 0.5, 1.0),
        seed=args.seed,
    )
    run_semantic_case(mixed)
    del mixed
    torch.npu.empty_cache()

    fp16 = make_case(
        name="fp16_b1_full_source",
        device=device,
        dtype=torch.float16,
        candidate_lens=(8192,),
        cache_tokens=(8192,),
        miss_fractions=(0.0,),
        seed=args.seed + 1000,
    )
    run_semantic_case(fp16)
    del fp16
    torch.npu.empty_cache()

    minimum = make_case(
        name="minimum_topk_source_bf16",
        device=device,
        dtype=torch.bfloat16,
        candidate_lens=(TOPK,),
        cache_tokens=(TOPK,),
        miss_fractions=(0.0,),
        seed=args.seed + 1500,
    )
    run_semantic_case(minimum)
    del minimum
    torch.npu.empty_cache()

    run_new_state_cases(device, args.seed + 2000)
    run_variable_query_case(device, args.seed + 2100)
    run_offload_tail_and_plain_case(device, args.seed + 2200)
    run_free_scan_transition_case(device, args.seed + 2300)
    run_skip_boundary_case(device, args.seed + 2400)

    run_graph_case(device, args.seed, args.graph_replays)
    if not args.skip_performance:
        run_performance_case(
            device,
            batch_size=args.batch_size,
            source_len=args.source_len,
            cache_tokens=args.cache_tokens,
            seed=args.seed,
            warmup=args.warmup,
            iters=args.iters,
            perf_query_miss_count=args.perf_query_miss_count,
            perf_query_noise=args.perf_query_noise,
        )
    print("FUSED_LI_MANAGE_MTP_UT_OK", flush=True)


if __name__ == "__main__":
    main()
