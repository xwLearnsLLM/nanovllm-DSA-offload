"""Semantic, graph, and latency checks for the bundled GLM MTP3 LIDU op."""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from time import perf_counter

import torch
import torch_npu  # type: ignore

import nanovllm.ops  # noqa: F401  # Load repository-local custom operators.


QUERY_COUNT = 4
HEADS = 32
HEAD_DIM = 128
BLOCK_SIZE = 128
TOPK = 2048
UNION_CAPACITY = QUERY_COUNT * TOPK
MAX_SOURCE_CAPACITY = 1 << 18


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
        description="Validate NanovllmLiduDecodeUpdateMtp on one NPU."
    )
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--source-len", type=int, default=20992)
    parser.add_argument("--cache-tokens", type=int, default=8192)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--graph-replays", type=int, default=3)
    parser.add_argument("--min-speedup", type=float, default=1.0)
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
    if args.min_speedup <= 0:
        raise ValueError("--min-speedup must be positive.")


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
            raise AssertionError("active MTP-LIDU rows must have >=2048 candidates")
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
    result = torch_npu.npu_lightning_indexer(
        query=active_query,
        key=key,
        weights=active_weights,
        actual_seq_lengths_query=query_ends,
        actual_seq_lengths_key=active_lens,
        block_table=active_table,
        layout_query="TND",
        layout_key="PA_BSND",
        sparse_count=TOPK,
        sparse_mode=3,
    )
    topk = result[0] if isinstance(result, (tuple, list)) else result
    if not isinstance(topk, torch.Tensor):
        raise TypeError("native LightningIndexer did not return a Tensor")
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
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    # Non-active rows use a distinct sentinel so accidental writes are visible.
    state = torch.full((pool_size, source_capacity), -777, dtype=torch.int32)
    unions: list[torch.Tensor] = []
    for request, (candidate_len, budget, miss_fraction) in enumerate(
        zip(candidate_lens, cache_tokens, miss_fractions)
    ):
        pool_row = int(req_pool_entries[request])
        state[pool_row].fill_(-1)
        if budget == 0:
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
            cached = torch.arange(candidate_len, dtype=torch.int64)
        else:
            miss_count = min(
                int(round(float(union.numel()) * miss_fraction)),
                int(union.numel()),
            )
            hits = union[miss_count:]
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

    query_cpu = torch.randn(
        batch_size * QUERY_COUNT,
        HEADS,
        HEAD_DIM,
        generator=generator,
        dtype=torch.float32,
    ).to(dtype)
    # Positive weights match GLM indexer usage and avoid unstable cancellation.
    weights_cpu = torch.rand(
        batch_size * QUERY_COUNT,
        HEADS,
        generator=generator,
        dtype=torch.float32,
    ).to(dtype)
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
    cache_cpu, union_rows = _make_cache_state(
        topk_rows=topk_rows,
        candidate_lens=candidate_lens,
        cache_tokens=cache_tokens,
        req_pool_entries=req_entries_cpu,
        source_capacity=capacity,
        miss_fractions=miss_fractions,
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
    return torch.ops.nanovllm_dsa.lidu_decode_update_mtp.default(
        case.query,
        case.key,
        case.weights,
        case.req_pool_entries,
        cache_slots,
        case.cache_tokens,
        case.candidate_lens,
        case.block_table,
    )


def call_mtp_out(
    case: MtpCase,
    cache_slots: torch.Tensor,
    topk_slots: torch.Tensor,
    miss_source_ids: torch.Tensor,
    miss_destination_slots: torch.Tensor,
    miss_counts: torch.Tensor,
):
    return torch.ops.nanovllm_dsa.lidu_decode_update_mtp_out.default(
        case.query,
        case.key,
        case.weights,
        case.req_pool_entries,
        cache_slots,
        case.cache_tokens,
        case.candidate_lens,
        case.block_table,
        topk_slots,
        miss_source_ids,
        miss_destination_slots,
        miss_counts,
    )


def make_outputs(case: MtpCase) -> tuple[torch.Tensor, ...]:
    topk_slots = torch.full(
        (case.batch_size * QUERY_COUNT, 1, TOPK),
        -313,
        dtype=torch.int32,
        device=case.device,
    )
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
    return topk_slots, miss_sources, miss_destinations, miss_counts


def run_meta_check() -> None:
    batch_size = 2
    source_capacity = 8192
    meta = torch.device("meta")
    query = torch.empty(
        batch_size * QUERY_COUNT, HEADS, HEAD_DIM, device=meta, dtype=torch.bfloat16
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
        batch_size * QUERY_COUNT, HEADS, device=meta, dtype=torch.bfloat16
    )
    req_entries = torch.empty(batch_size, device=meta, dtype=torch.int32)
    cache_slots = torch.empty(
        batch_size + 1, source_capacity, device=meta, dtype=torch.int32
    )
    cache_tokens = torch.empty(batch_size, device=meta, dtype=torch.int32)
    candidate_lens = torch.empty(batch_size, device=meta, dtype=torch.int32)
    block_table = torch.empty(
        batch_size,
        source_capacity // BLOCK_SIZE,
        device=meta,
        dtype=torch.int32,
    )
    outputs = torch.ops.nanovllm_dsa.lidu_decode_update_mtp.default(
        query,
        key,
        weights,
        req_entries,
        cache_slots,
        cache_tokens,
        candidate_lens,
        block_table,
    )
    expected_shapes = (
        (batch_size * QUERY_COUNT, 1, TOPK),
        (batch_size, UNION_CAPACITY),
        (batch_size, UNION_CAPACITY),
        (batch_size,),
        (batch_size + 1, source_capacity),
    )
    if tuple(tuple(output.shape) for output in outputs) != expected_shapes:
        raise AssertionError("MTP-LIDU Meta output shapes are incorrect")
    if any(output.dtype != torch.int32 for output in outputs):
        raise AssertionError("MTP-LIDU Meta output dtypes are not int32")
    if not torch._C._is_alias_of(outputs[-1], cache_slots):
        raise AssertionError("MTP-LIDU Meta cache output lost its alias")
    out_buffers = (
        torch.empty(expected_shapes[0], device=meta, dtype=torch.int32),
        torch.empty(expected_shapes[1], device=meta, dtype=torch.int32),
        torch.empty(expected_shapes[2], device=meta, dtype=torch.int32),
        torch.empty(expected_shapes[3], device=meta, dtype=torch.int32),
    )
    out_outputs = torch.ops.nanovllm_dsa.lidu_decode_update_mtp_out.default(
        query,
        key,
        weights,
        req_entries,
        cache_slots,
        cache_tokens,
        candidate_lens,
        block_table,
        *out_buffers,
    )
    if tuple(tuple(output.shape) for output in out_outputs) != expected_shapes:
        raise AssertionError("MTP-LIDU _out Meta output shapes are incorrect")
    if any(output.dtype != torch.int32 for output in out_outputs):
        raise AssertionError("MTP-LIDU _out Meta output dtypes are not int32")
    for returned, supplied in zip(out_outputs[:4], out_buffers):
        if not torch._C._is_alias_of(returned, supplied):
            raise AssertionError("MTP-LIDU _out Meta output lost its alias")
    if not torch._C._is_alias_of(out_outputs[-1], cache_slots):
        raise AssertionError("MTP-LIDU _out Meta cache output lost its alias")
    print(
        "MTP_LIDU_META_CHECK alloc=1 out=1 mutable_cache_alias=1 ok=1",
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
    topk_slots, miss_sources, miss_destinations, miss_counts, cache_alias = outputs
    if cache_alias.data_ptr() != cache_slots.data_ptr():
        raise AssertionError(f"{label}: cache output does not alias mutable pool")
    after_cpu = cache_slots.cpu()
    topk_slots_cpu = topk_slots.reshape(-1, TOPK).cpu().to(torch.int64)
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

        union = case.union_cpu[request]
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
        if bool((after[union] < 0).any()):
            raise AssertionError(f"{label}: request={request} union is not cached")

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

        for query_idx in range(QUERY_COUNT):
            query_row = request * QUERY_COUNT + query_idx
            expected_slots = after[case.topk_cpu[query_row]].to(torch.int64)
            if not torch.equal(topk_slots_cpu[query_row], expected_slots):
                raise AssertionError(
                    f"{label}: request={request} query={query_idx} topk_slots differ"
                )
            if torch.unique(topk_slots_cpu[query_row]).numel() != TOPK:
                raise AssertionError(
                    f"{label}: request={request} query={query_idx} slots repeat"
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
    for request in range(case.batch_size):
        if int(case.cache_tokens_cpu[request]) == 0:
            continue
        if not torch.equal(left_topk[request], right_topk[request]):
            raise AssertionError(f"{label}: request={request} topk_slots differ")
    left_counts = left[3].cpu()
    right_counts = right[3].cpu()
    if not torch.equal(left_counts, right_counts):
        raise AssertionError(f"{label}: miss_counts differ")
    for request, count_value in enumerate(left_counts.tolist()):
        count = int(count_value)
        if not torch.equal(
            left[1][request, :count].cpu(), right[1][request, :count].cpu()
        ):
            raise AssertionError(f"{label}: request={request} miss IDs differ")
        if not torch.equal(
            left[2][request, :count].cpu(), right[2][request, :count].cpu()
        ):
            raise AssertionError(f"{label}: request={request} miss slots differ")


def run_semantic_case(case: MtpCase) -> None:
    alloc_cache = case.initial_cache_cpu.to(case.device)
    before = case.initial_cache_cpu.clone()
    alloc_outputs = call_mtp(case, alloc_cache)
    torch.npu.synchronize()
    counts = validate_result(
        case, before, alloc_cache, alloc_outputs, label=f"{case.name}/alloc"
    )

    out_cache = case.initial_cache_cpu.to(case.device)
    out_buffers = make_outputs(case)
    out_outputs = call_mtp_out(case, out_cache, *out_buffers)
    torch.npu.synchronize()
    validate_result(
        case, before, out_cache, out_outputs, label=f"{case.name}/out"
    )
    for returned, supplied in zip(out_outputs[:4], out_buffers):
        if returned.data_ptr() != supplied.data_ptr():
            raise AssertionError(f"{case.name}: _out did not preserve output buffer")
    _compare_valid_outputs(case, alloc_outputs, out_outputs, label=case.name)
    if not torch.equal(alloc_cache.cpu(), out_cache.cpu()):
        raise AssertionError(f"{case.name}: alloc and _out cache states differ")

    repeat_before = alloc_cache.cpu()
    repeat_outputs = call_mtp(case, alloc_cache)
    torch.npu.synchronize()
    repeat_counts = validate_result(
        case,
        repeat_before,
        alloc_cache,
        repeat_outputs,
        label=f"{case.name}/repeat",
    )
    if any(repeat_counts):
        raise AssertionError(f"{case.name}: identical repeat must be zero miss")
    print(
        "MTP_LIDU_SEMANTIC_CHECK "
        f"case={case.name} dtype={case.dtype} batch={case.batch_size} "
        f"candidate_lens={case.candidate_lens_cpu.tolist()} "
        f"cache_tokens={case.cache_tokens_cpu.tolist()} "
        f"miss_counts={counts} shuffled_pool_entries=1 random_block_table=1 ok=1",
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

    # Warm up the out contract on disposable state before capture.
    warm_cache = case.initial_cache_cpu.to(device)
    call_mtp_out(case, warm_cache, *make_outputs(case))
    torch.npu.synchronize()

    graph = torch.npu.NPUGraph()
    pool = torch.npu.graph_pool_handle()
    with torch.npu.graph(graph, pool=pool):
        graph_outputs = call_mtp_out(case, graph_cache, *graph_buffers)
    torch.npu.synchronize()
    for returned, supplied in zip(graph_outputs[:4], graph_buffers):
        if returned.data_ptr() != supplied.data_ptr():
            raise AssertionError("graph capture lost caller-owned MTP-LIDU buffers")
    if graph_outputs[4].data_ptr() != graph_cache.data_ptr():
        raise AssertionError("graph capture lost mutable cache alias")

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
        "MTP_LIDU_GRAPH_CHECK "
        f"batch={batch_size} replays={replays} dynamic_query=1 "
        "dynamic_weights=1 dynamic_pool_entries=1 dynamic_lengths=1 "
        "dynamic_block_table=1 evolving_state=1 ok=1",
        flush=True,
    )
    del case, graph_cache, eager_cache, warm_cache
    torch.npu.empty_cache()


def _single_query_out(
    case: MtpCase,
    query: torch.Tensor,
    weights: torch.Tensor,
    cache_slots: torch.Tensor,
    source_ids: torch.Tensor,
    destination_slots: torch.Tensor,
    miss_counts: torch.Tensor,
):
    return torch.ops.nanovllm_dsa.lidu_decode_update_out.default(
        query,
        case.key,
        weights,
        case.req_pool_entries,
        cache_slots,
        case.cache_tokens,
        case.candidate_lens,
        case.block_table,
        source_ids,
        destination_slots,
        miss_counts,
    )


def _wall_ms(fn, warmup: int, iters: int) -> float:
    for _ in range(warmup):
        fn()
    torch.npu.synchronize()
    start = perf_counter()
    for _ in range(iters):
        fn()
    torch.npu.synchronize()
    return (perf_counter() - start) * 1000.0 / iters


def _fresh_state_ms(
    case: MtpCase,
    initial_cpu: torch.Tensor,
    warmup: int,
    iters: int,
) -> tuple[float, int]:
    cache = initial_cpu.to(case.device)
    initial = initial_cpu.to(case.device)
    buffers = make_outputs(case)
    elapsed = 0.0
    last_misses = 0
    for iteration in range(warmup + iters):
        cache.copy_(initial)
        torch.npu.synchronize()
        start = perf_counter()
        outputs = call_mtp_out(case, cache, *buffers)
        torch.npu.synchronize()
        if iteration >= warmup:
            elapsed += (perf_counter() - start) * 1000.0
        last_misses = int(outputs[3].sum().cpu())
    return elapsed / iters, last_misses


def run_performance_case(
    device: torch.device,
    *,
    batch_size: int,
    source_len: int,
    cache_tokens: int,
    seed: int,
    warmup: int,
    iters: int,
    min_speedup: float,
) -> None:
    case = make_case(
        name="performance",
        device=device,
        dtype=torch.bfloat16,
        candidate_lens=(source_len,) * batch_size,
        cache_tokens=(cache_tokens,) * batch_size,
        miss_fractions=(0.5,) * batch_size,
        seed=seed + 6000,
    )
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
    print(
        "MTP_LIDU_TARGET_BATCH_CHECK "
        f"batch={batch_size} candidate_len={source_len} "
        f"cache_tokens={cache_tokens} total_union_misses={sum(correctness_counts)} "
        "one_request_per_owner=1 ok=1",
        flush=True,
    )
    fused_cache = case.initial_cache_cpu.to(device)
    serial_cache = case.initial_cache_cpu.to(device)
    fused_buffers = make_outputs(case)
    serial_buffers = tuple(
        (
            torch.empty(
                (batch_size, 1, TOPK), dtype=torch.int32, device=device
            ),
            torch.empty(
                (batch_size, 1, TOPK), dtype=torch.int32, device=device
            ),
            torch.empty((batch_size,), dtype=torch.int32, device=device),
        )
        for _ in range(QUERY_COUNT)
    )
    query_rows = tuple(
        case.query.view(case.batch_size, QUERY_COUNT, HEADS, HEAD_DIM)[
            :, query_idx
        ].contiguous()
        for query_idx in range(QUERY_COUNT)
    )
    weight_rows = tuple(
        case.weights.view(case.batch_size, QUERY_COUNT, HEADS)[
            :, query_idx
        ].contiguous()
        for query_idx in range(QUERY_COUNT)
    )

    def fused_step():
        return call_mtp_out(case, fused_cache, *fused_buffers)

    def serial_step():
        result = None
        for query_idx in range(QUERY_COUNT):
            result = _single_query_out(
                case,
                query_rows[query_idx],
                weight_rows[query_idx],
                serial_cache,
                *serial_buffers[query_idx],
            )
        return result

    # Report union copy reduction from an identical typical-miss state.
    union_cache = case.initial_cache_cpu.to(device)
    serial_count_cache = case.initial_cache_cpu.to(device)
    union_result = call_mtp_out(case, union_cache, *make_outputs(case))
    per_query_misses: list[int] = []
    for query_idx in range(QUERY_COUNT):
        result = _single_query_out(
            case,
            query_rows[query_idx],
            weight_rows[query_idx],
            serial_count_cache,
            *serial_buffers[query_idx],
        )
        per_query_misses.append(int(result[2].sum().cpu()))
    torch.npu.synchronize()
    union_misses = int(union_result[3].sum().cpu())

    fused_ms = _wall_ms(fused_step, warmup, iters)
    serial_ms = _wall_ms(serial_step, warmup, iters)
    speedup = serial_ms / fused_ms
    if speedup < min_speedup:
        raise AssertionError(
            f"MTP-LIDU speedup={speedup:.4f} is below {min_speedup:.4f}"
        )

    level_results: list[tuple[str, int, float]] = []
    for label, fraction in (("zero", 0.0), ("typical", 0.5), ("high", 1.0)):
        level_cpu, _ = _make_cache_state(
            topk_rows=case.topk_cpu,
            candidate_lens=(source_len,) * batch_size,
            cache_tokens=(cache_tokens,) * batch_size,
            req_pool_entries=case.req_pool_entries_cpu,
            source_capacity=case.source_capacity,
            miss_fractions=(fraction,) * batch_size,
            generator=torch.Generator().manual_seed(seed + 7000),
            pool_size=batch_size + 3,
        )
        avg_ms, misses = _fresh_state_ms(
            case, level_cpu, min(warmup, 3), min(iters, 10)
        )
        level_results.append((label, misses, avg_ms))
        print(
            "MTP_LIDU_MISS_LEVEL_RESULT "
            f"level={label} batch={batch_size} candidate_len={source_len} "
            f"cache_tokens={cache_tokens} total_union_misses={misses} "
            f"avg_ms={avg_ms:.6f}",
            flush=True,
        )

    print(
        "MTP_LIDU_UNION_COMPARE "
        f"per_query_misses={per_query_misses} "
        f"sum_per_query_misses={sum(per_query_misses)} "
        f"union_misses={union_misses}",
        flush=True,
    )
    print(
        "MTP_LIDU_PERF_RESULT "
        f"batch={batch_size} candidate_len={source_len} "
        f"cache_tokens={cache_tokens} fused_ms={fused_ms:.6f} "
        f"four_serial_lidu_ms={serial_ms:.6f} speedup={speedup:.4f} "
        f"min_speedup={min_speedup:.4f} warmup={warmup} iters={iters}",
        flush=True,
    )
    del case
    torch.npu.empty_cache()


def main() -> None:
    args = parse_args()
    _validate_cli(args)
    device = torch.device(args.device)
    if device.type != "npu":
        raise ValueError("--device must select one NPU, for example npu:0.")
    torch.npu.set_device(device)
    torch.npu.config.allow_internal_format = False
    opapi_path = os.environ.get("NANOVLLM_CUST_OPAPI_LIB", "")
    if not opapi_path or not os.path.isfile(opapi_path):
        raise RuntimeError(
            "Repository-local libcust_opapi.so was not selected; rebuild "
            "with `bash scripts/build_nanovllm_ops.sh`."
        )
    print(f"MTP_LIDU_OPAPI path={opapi_path} local=1", flush=True)
    print(
        "MTP_LIDU_CONFIG "
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
            min_speedup=args.min_speedup,
        )
    print("MTP_LIDU_UT_OK", flush=True)


if __name__ == "__main__":
    main()
