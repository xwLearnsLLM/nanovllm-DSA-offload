"""Single-NPU semantic and latency test for Fused LI Manage + SCATTER."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from time import perf_counter

import torch
import torch_npu  # type: ignore

import nanovllm.ops  # noqa: F401  # Load repository-local custom operators.
from ut_ops._op_utils import require_local_opapi


BLOCK_SIZE = 128
INDEX_DIM = 128
TOPK = 2048
KPE_DIM = 64
CKV_DIM = 512
CACHE_TOKEN_BUDGETS = (3072, 6144, 8192, 12288)
CACHE_TIERS = (0, TOPK, *CACHE_TOKEN_BUDGETS)
CANDIDATE_LENS = (1024, 4096, 8192, 16384, 32768, 65536)


@dataclass
class Case:
    device: torch.device
    heads: int
    req_entries_cpu: torch.Tensor
    cache_tokens_cpu: torch.Tensor
    candidate_lens_cpu: torch.Tensor
    index_block_table_cpu: torch.Tensor
    dram_block_table_cpu: torch.Tensor
    hbm_block_table_cpu: torch.Tensor
    query: torch.Tensor
    weights: torch.Tensor
    index_cache: torch.Tensor
    cache_slots: torch.Tensor
    cache_tokens: torch.Tensor
    candidate_lens: torch.Tensor
    req_entries: torch.Tensor
    index_block_table: torch.Tensor
    dram_block_table: torch.Tensor
    hbm_block_table: torch.Tensor
    dram_kpe: torch.Tensor
    dram_ckv: torch.Tensor
    hbm_kpe: torch.Tensor
    hbm_ckv: torch.Tensor
    init_source_ids: torch.Tensor
    init_destination_slots: torch.Tensor
    init_counts: torch.Tensor
    topk_src_ids: torch.Tensor
    topk_dst_slots: torch.Tensor
    miss_counts: torch.Tensor
    expected_initial_sources: list[torch.Tensor]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate bundled LightningIndexerDecodeUpdate and KvcacheScatterCopy."
    )
    parser.add_argument("--device", default="npu:0")
    parser.add_argument(
        "--heads",
        default="32,64",
        help="Comma-separated index head counts; supported values are 32 and 64.",
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument(
        "--graph-replays",
        type=int,
        default=3,
        help="Replay a persistent-output LI-Manage-to-SCATTER graph this many times.",
    )
    return parser.parse_args()


def _parse_heads(value: str) -> tuple[int, ...]:
    heads = tuple(dict.fromkeys(int(item.strip()) for item in value.split(",")))
    if not heads or any(head not in (32, 64) for head in heads):
        raise ValueError("--heads only accepts a non-empty subset of 32,64.")
    return heads


def _swapped_from_cpu(cpu: torch.Tensor, device: torch.device) -> torch.Tensor:
    tensor = torch_npu.empty_with_swapped_memory(
        cpu.shape,
        dtype=cpu.dtype,
        device=device,
    )
    tensor.fill_(0)
    tensor.add_(cpu.to(device))
    return tensor


def _random_tables(
    row_block_counts: tuple[int, ...],
    *,
    generator: torch.Generator,
) -> tuple[torch.Tensor, int]:
    max_columns = max(row_block_counts)
    total_blocks = sum(row_block_counts)
    permutation = torch.randperm(total_blocks, generator=generator).to(torch.int32)
    table = torch.zeros(len(row_block_counts), max_columns, dtype=torch.int32)
    cursor = 0
    for row, count in enumerate(row_block_counts):
        table[row, :count] = permutation[cursor : cursor + count]
        cursor += count
    return table.contiguous(), total_blocks


def _logical_token_rows(
    physical_cache: torch.Tensor,
    block_table: torch.Tensor,
    row: int,
    token_ids: torch.Tensor,
) -> torch.Tensor:
    token_ids = token_ids.to(device=physical_cache.device, dtype=torch.int64)
    blocks = block_table[row].to(
        device=physical_cache.device, dtype=torch.int64
    )[token_ids // BLOCK_SIZE]
    offsets = token_ids % BLOCK_SIZE
    return physical_cache[blocks, offsets]


def make_case(heads: int, device: torch.device, seed: int) -> Case:
    generator = torch.Generator().manual_seed(seed + heads)
    batch = len(CACHE_TIERS)
    candidate_blocks = tuple(length // BLOCK_SIZE for length in CANDIDATE_LENS)
    hbm_blocks = tuple(max(1, count // BLOCK_SIZE) for count in CACHE_TIERS)
    index_table_cpu, index_block_count = _random_tables(
        candidate_blocks, generator=generator
    )
    dram_table_cpu, dram_block_count = _random_tables(
        candidate_blocks, generator=generator
    )
    hbm_table_cpu, hbm_block_count = _random_tables(
        hbm_blocks, generator=generator
    )

    index_cpu = torch.zeros(
        index_block_count,
        BLOCK_SIZE,
        1,
        INDEX_DIM,
        dtype=torch.bfloat16,
    )
    for row, candidate_len in enumerate(CANDIDATE_LENS):
        logical_ids = torch.arange(candidate_len, dtype=torch.int64)
        physical_blocks = index_table_cpu[row].to(torch.int64)[
            logical_ids // BLOCK_SIZE
        ]
        offsets = logical_ids % BLOCK_SIZE
        # q=[256, 1, 0...] makes the exact score equal to logical token id.
        index_cpu[physical_blocks, offsets, 0, 0] = (
            logical_ids // 256
        ).to(torch.bfloat16)
        index_cpu[physical_blocks, offsets, 0, 1] = (
            logical_ids % 256
        ).to(torch.bfloat16)

    dram_kpe_cpu = torch.randn(
        dram_block_count,
        BLOCK_SIZE,
        KPE_DIM,
        generator=generator,
        dtype=torch.float32,
    ).to(torch.bfloat16)
    dram_ckv_cpu = torch.randn(
        dram_block_count,
        BLOCK_SIZE,
        CKV_DIM,
        generator=generator,
        dtype=torch.float32,
    ).to(torch.bfloat16)

    query = torch.zeros(batch, heads, INDEX_DIM, dtype=torch.bfloat16)
    query[:, 0, 0] = 256
    query[:, 0, 1] = 1
    weights = torch.zeros(batch, heads, dtype=torch.bfloat16)
    weights[:, 0] = 1

    req_entries_cpu = torch.tensor([4, 1, 7, 0, 6, 2], dtype=torch.int32)
    pool_size = int(req_entries_cpu.max()) + 2
    state_width = max(CANDIDATE_LENS)
    cache_slots_cpu = torch.full(
        (pool_size, state_width), -1, dtype=torch.int32
    )
    max_c = max(CACHE_TIERS)
    init_source_ids_cpu = torch.full(
        (batch, max_c), -1, dtype=torch.int32
    )
    init_destination_cpu = torch.full_like(init_source_ids_cpu, -1)
    expected_initial_sources: list[torch.Tensor] = []
    hit_count = TOPK // 2
    for row, (cache_tokens, candidate_len) in enumerate(
        zip(CACHE_TIERS, CANDIDATE_LENS)
    ):
        if cache_tokens == 0:
            expected_initial_sources.append(torch.empty(0, dtype=torch.int64))
            continue
        high_hits = torch.arange(
            candidate_len - hit_count,
            candidate_len,
            dtype=torch.int64,
        )
        low = torch.arange(cache_tokens - hit_count, dtype=torch.int64)
        sources = torch.cat((high_hits, low))
        sources = sources[torch.randperm(cache_tokens, generator=generator)]
        destinations = torch.randperm(cache_tokens, generator=generator).to(
            torch.int64
        )
        pool_row = int(req_entries_cpu[row])
        cache_slots_cpu[pool_row, sources] = destinations.to(torch.int32)
        init_source_ids_cpu[row, :cache_tokens] = sources.to(torch.int32)
        init_destination_cpu[row, :cache_tokens] = destinations.to(torch.int32)
        expected_initial_sources.append(sources)

    return Case(
        device=device,
        heads=heads,
        req_entries_cpu=req_entries_cpu,
        cache_tokens_cpu=torch.tensor(CACHE_TIERS, dtype=torch.int32),
        candidate_lens_cpu=torch.tensor(CANDIDATE_LENS, dtype=torch.int32),
        index_block_table_cpu=index_table_cpu,
        dram_block_table_cpu=dram_table_cpu,
        hbm_block_table_cpu=hbm_table_cpu,
        query=query.to(device).contiguous(),
        weights=weights.to(device).contiguous(),
        index_cache=index_cpu.to(device).contiguous(),
        cache_slots=cache_slots_cpu.to(device).contiguous(),
        cache_tokens=torch.tensor(CACHE_TIERS, dtype=torch.int32, device=device),
        candidate_lens=torch.tensor(CANDIDATE_LENS, dtype=torch.int32, device=device),
        req_entries=req_entries_cpu.to(device),
        index_block_table=index_table_cpu.to(device),
        dram_block_table=dram_table_cpu.to(device),
        hbm_block_table=hbm_table_cpu.to(device),
        dram_kpe=_swapped_from_cpu(dram_kpe_cpu, device),
        dram_ckv=_swapped_from_cpu(dram_ckv_cpu, device),
        hbm_kpe=torch.zeros(
            hbm_block_count,
            BLOCK_SIZE,
            KPE_DIM,
            dtype=torch.bfloat16,
            device=device,
        ),
        hbm_ckv=torch.zeros(
            hbm_block_count,
            BLOCK_SIZE,
            CKV_DIM,
            dtype=torch.bfloat16,
            device=device,
        ),
        init_source_ids=init_source_ids_cpu.to(device),
        init_destination_slots=init_destination_cpu.to(device),
        init_counts=torch.tensor(CACHE_TIERS, dtype=torch.int32, device=device),
        topk_src_ids=torch.empty(
            (batch, 1, TOPK), dtype=torch.int32, device=device
        ),
        topk_dst_slots=torch.empty(
            (batch, 1, TOPK), dtype=torch.int32, device=device
        ),
        miss_counts=torch.empty((batch,), dtype=torch.int32, device=device),
        expected_initial_sources=expected_initial_sources,
    )


def call_scatter(
    case: Case,
    source_ids: torch.Tensor,
    destination_slots: torch.Tensor,
    counts: torch.Tensor,
) -> None:
    torch.ops.nanovllm_dsa.scatter_copy.default(
        source_ids,
        destination_slots,
        counts,
        case.hbm_block_table,
        case.dram_block_table,
        case.hbm_kpe,
        case.hbm_ckv,
        case.dram_kpe,
        case.dram_ckv,
    )


def call_fused_li_manage(case: Case):
    torch.ops.nanovllm_dsa.fused_li_manage.default(
        case.query,
        case.weights,
        case.index_cache,
        case.index_block_table,
        case.candidate_lens,
        case.cache_tokens,
        case.req_entries,
        case.cache_slots,
        case.topk_src_ids,
        case.topk_dst_slots,
        case.miss_counts,
    )
    return case.topk_src_ids, case.topk_dst_slots, case.miss_counts


def call_fused_li_manage_with_buffers(
    case: Case,
    *,
    query: torch.Tensor,
    weights: torch.Tensor,
    req_entries: torch.Tensor,
    cache_tokens: torch.Tensor,
    candidate_lens: torch.Tensor,
    index_block_table: torch.Tensor,
    source_ids: torch.Tensor,
    destination_slots: torch.Tensor,
    miss_counts: torch.Tensor,
):
    torch.ops.nanovllm_dsa.fused_li_manage.default(
        query,
        weights,
        case.index_cache,
        index_block_table,
        candidate_lens,
        cache_tokens,
        req_entries,
        case.cache_slots,
        source_ids,
        destination_slots,
        miss_counts,
    )
    return source_ids, destination_slots, miss_counts


def validate_cache_data(case: Case, row: int, sources: torch.Tensor) -> None:
    if sources.numel() == 0:
        return
    pool_row = int(case.req_entries_cpu[row])
    slots = case.cache_slots[pool_row].detach().cpu()[sources].to(torch.int64)
    if bool((slots < 0).any()):
        raise AssertionError(f"row={row} has uncached expected source tokens")
    actual_kpe = _logical_token_rows(
        case.hbm_kpe, case.hbm_block_table, row, slots
    ).cpu()
    actual_ckv = _logical_token_rows(
        case.hbm_ckv, case.hbm_block_table, row, slots
    ).cpu()
    expected_kpe = _logical_token_rows(
        case.dram_kpe, case.dram_block_table, row, sources
    ).cpu()
    expected_ckv = _logical_token_rows(
        case.dram_ckv, case.dram_block_table, row, sources
    ).cpu()
    if not torch.equal(actual_kpe, expected_kpe):
        raise AssertionError(f"row={row} SCATTER KPE data mismatch")
    if not torch.equal(actual_ckv, expected_ckv):
        raise AssertionError(f"row={row} SCATTER CKV data mismatch")


def validate_state(case: Case, expected_topk: list[torch.Tensor]) -> None:
    slots_pool = case.cache_slots.detach().cpu()
    used_rows = {int(value) for value in case.req_entries_cpu.tolist()}
    for pool_row in range(slots_pool.shape[0]):
        if pool_row not in used_rows and bool((slots_pool[pool_row] != -1).any()):
            raise AssertionError(f"unused request-pool row {pool_row} was modified")
    for row, cache_tokens in enumerate(CACHE_TIERS):
        pool_row = int(case.req_entries_cpu[row])
        state = slots_pool[pool_row, : CANDIDATE_LENS[row]]
        valid_slots = state[state >= 0].to(torch.int64)
        if valid_slots.numel() != cache_tokens:
            raise AssertionError(
                f"row={row} expected {cache_tokens} cached tokens, got {valid_slots.numel()}"
            )
        if cache_tokens and (
            torch.unique(valid_slots).numel() != cache_tokens
            or int(valid_slots.min()) != 0
            or int(valid_slots.max()) != cache_tokens - 1
        ):
            raise AssertionError(f"row={row} destination slots are not a permutation of [0,C)")
        top = expected_topk[row]
        if top.numel() and bool((state[top] < 0).any()):
            raise AssertionError(f"row={row} does not contain every true top-2048 token")


def validate_full_topk_slots(
    case: Case,
    destination_slots: torch.Tensor,
    expected_topk: list[torch.Tensor],
) -> None:
    """The full Fused LI Manage slot row is the sparse Attention index input."""

    state_pool = case.cache_slots.detach().cpu()
    actual_rows = destination_slots.detach().cpu().view(len(CACHE_TIERS), TOPK)
    for row, (cache_tokens, topk_sources) in enumerate(
        zip(CACHE_TIERS, expected_topk)
    ):
        if cache_tokens == 0:
            continue
        pool_row = int(case.req_entries_cpu[row])
        expected_slots = state_pool[pool_row, topk_sources].to(torch.int64)
        actual_slots = actual_rows[row].to(torch.int64)
        if bool((actual_slots < 0).any()):
            raise AssertionError(f"row={row} full top-k slots contain invalid values")
        if torch.unique(actual_slots).numel() != TOPK:
            raise AssertionError(f"row={row} full top-k slots are not unique")
        if not torch.equal(
            torch.sort(actual_slots).values,
            torch.sort(expected_slots).values,
        ):
            raise AssertionError(f"row={row} full top-k slot set is incorrect")


def validate_full_topk_indices(
    source_ids: torch.Tensor,
    expected_topk: list[torch.Tensor],
) -> None:
    """Every active request publishes all 2048 selected token indices."""

    actual_rows = source_ids.detach().cpu().view(len(CACHE_TIERS), TOPK)
    for row, (cache_tokens, expected_sources) in enumerate(
        zip(CACHE_TIERS, expected_topk)
    ):
        if cache_tokens == 0:
            continue
        actual_sources = actual_rows[row].to(torch.int64)
        if torch.unique(actual_sources).numel() != TOPK:
            raise AssertionError(f"row={row} full top-k source indices are not unique")
        if not torch.equal(
            torch.sort(actual_sources).values,
            torch.sort(expected_sources).values,
        ):
            raise AssertionError(f"row={row} full top-k source index set is incorrect")


def run_case(case: Case, warmup: int, iters: int) -> None:
    # Initialization path: one SCATTER call at the configured maximum C.
    call_scatter(
        case,
        case.init_source_ids,
        case.init_destination_slots,
        case.init_counts,
    )
    torch.npu.synchronize()
    for row, sources in enumerate(case.expected_initial_sources):
        validate_cache_data(case, row, sources)
    print(
        f"FUSED_LI_MANAGE_SCATTER_INIT_CHECK heads={case.heads} "
        f"max_c={max(CACHE_TIERS)} ok=1"
    )

    source_ids, destination_slots, miss_counts = call_fused_li_manage(case)
    call_scatter(
        case,
        source_ids.view(len(CACHE_TIERS), TOPK),
        destination_slots.view(len(CACHE_TIERS), TOPK),
        miss_counts,
    )
    torch.npu.synchronize()

    expected_misses = torch.tensor(
        [
            0
            if cache_tokens == 0
            else min(TOPK // 2, candidate_len - cache_tokens)
            for cache_tokens, candidate_len in zip(
                CACHE_TIERS,
                CANDIDATE_LENS,
            )
        ],
        dtype=torch.int32,
    )
    actual_misses = miss_counts.detach().cpu()
    if not torch.equal(actual_misses, expected_misses):
        raise AssertionError(
            f"unexpected first-update miss counts: {actual_misses.tolist()}"
        )
    expected_high = [
        torch.empty(0, dtype=torch.int64)
        if cache_tokens == 0
        else torch.arange(candidate_len - TOPK, candidate_len, dtype=torch.int64)
        for cache_tokens, candidate_len in zip(CACHE_TIERS, CANDIDATE_LENS)
    ]
    validate_state(case, expected_high)
    validate_full_topk_indices(source_ids, expected_high)
    validate_full_topk_slots(case, destination_slots, expected_high)
    for row, sources in enumerate(expected_high):
        validate_cache_data(case, row, sources)
    print(
        f"FUSED_LI_MANAGE_SCATTER_UPDATE_CHECK heads={case.heads} "
        f"miss_counts={actual_misses.tolist()} ok=1"
    )
    print(f"FUSED_LI_MANAGE_FULL_TOPK_INDEX_SLOTS_CHECK heads={case.heads} ok=1")

    # Identical query must be a zero-miss repeat and leave every pool row valid.
    repeat_source, repeat_slots, repeat_counts = call_fused_li_manage(case)
    call_scatter(
        case,
        repeat_source.view(len(CACHE_TIERS), TOPK),
        repeat_slots.view(len(CACHE_TIERS), TOPK),
        repeat_counts,
    )
    torch.npu.synchronize()
    repeat_counts_cpu = repeat_counts.cpu()
    if bool((repeat_counts_cpu != 0).any()):
        raise AssertionError(
            f"repeat update must have zero misses, got {repeat_counts_cpu.tolist()}"
        )
    validate_state(case, expected_high)
    validate_full_topk_indices(repeat_source, expected_high)
    validate_full_topk_slots(case, repeat_slots, expected_high)
    print(f"FUSED_LI_MANAGE_SCATTER_ZERO_MISS_CHECK heads={case.heads} ok=1")

    for _ in range(warmup):
        source, slots, counts = call_fused_li_manage(case)
        call_scatter(
            case,
            source.view(len(CACHE_TIERS), TOPK),
            slots.view(len(CACHE_TIERS), TOPK),
            counts,
        )
    torch.npu.synchronize()
    start = perf_counter()
    for _ in range(iters):
        source, slots, counts = call_fused_li_manage(case)
        call_scatter(
            case,
            source.view(len(CACHE_TIERS), TOPK),
            slots.view(len(CACHE_TIERS), TOPK),
            counts,
        )
    torch.npu.synchronize()
    avg_ms = (perf_counter() - start) * 1000.0 / iters
    print(
        f"FUSED_LI_MANAGE_SCATTER_RESULT heads={case.heads} batch={len(CACHE_TIERS)} "
        f"candidate_max={max(CANDIDATE_LENS)} avg_chain_ms={avg_ms:.6f} "
        f"warmup={warmup} iters={iters}"
    )


def run_local_bs24_case(
    heads: int,
    device: torch.device,
    seed: int,
) -> None:
    """Cover the one-request-per-core schedule used by the target batch 24."""

    batch = 24
    candidate_len = 19_968
    cache_budget = int(CACHE_TOKEN_BUDGETS[1])
    miss_target = 300
    if not TOPK <= cache_budget <= candidate_len:
        raise ValueError(
            "The 16K-32K Fused LI Manage budget is invalid for the bs24 local-schedule "
            f"test: C={cache_budget}, source={candidate_len}."
        )

    generator = torch.Generator().manual_seed(seed + heads + 2400)
    candidate_blocks = candidate_len // BLOCK_SIZE
    block_table_cpu = torch.arange(
        candidate_blocks, dtype=torch.int32
    ).repeat(batch, 1)
    index_cpu = torch.zeros(
        candidate_blocks,
        BLOCK_SIZE,
        1,
        INDEX_DIM,
        dtype=torch.bfloat16,
    )
    logical_ids = torch.arange(candidate_len, dtype=torch.int64)
    index_cpu[:, :, 0, 0] = (
        logical_ids.reshape(candidate_blocks, BLOCK_SIZE) // 256
    ).to(torch.bfloat16)
    index_cpu[:, :, 0, 1] = (
        logical_ids.reshape(candidate_blocks, BLOCK_SIZE) % 256
    ).to(torch.bfloat16)

    query = torch.zeros(
        batch, heads, INDEX_DIM, dtype=torch.bfloat16, device=device
    )
    query[:, 0, 0] = 256
    query[:, 0, 1] = 1
    weights = torch.zeros(
        batch, heads, dtype=torch.bfloat16, device=device
    )
    weights[:, 0] = 1

    req_entries_cpu = torch.randperm(
        batch, generator=generator
    ).to(torch.int32)
    cache_slots_cpu = torch.full(
        (batch, candidate_len), -1, dtype=torch.int32
    )
    cached_top_count = TOPK - miss_target
    cached_sources = torch.cat(
        (
            torch.arange(
                candidate_len - cached_top_count,
                candidate_len,
                dtype=torch.int64,
            ),
            torch.arange(
                cache_budget - cached_top_count,
                dtype=torch.int64,
            ),
        )
    )
    for row in range(batch):
        destinations = torch.randperm(
            cache_budget, generator=generator
        ).to(torch.int32)
        pool_row = int(req_entries_cpu[row])
        cache_slots_cpu[pool_row, cached_sources] = destinations

    cache_slots = cache_slots_cpu.to(device)
    cache_tokens = torch.full(
        (batch,), cache_budget, dtype=torch.int32, device=device
    )
    candidate_lens = torch.full(
        (batch,), candidate_len, dtype=torch.int32, device=device
    )
    req_entries = req_entries_cpu.to(device)
    block_table = block_table_cpu.to(device)
    index_cache = index_cpu.to(device)

    source_ids = torch.empty(
        (batch, 1, TOPK), dtype=torch.int32, device=device
    )
    destination_slots = torch.empty_like(source_ids)
    miss_counts = torch.empty((batch,), dtype=torch.int32, device=device)
    torch.ops.nanovllm_dsa.fused_li_manage.default(
        query,
        weights,
        index_cache,
        block_table,
        candidate_lens,
        cache_tokens,
        req_entries,
        cache_slots,
        source_ids,
        destination_slots,
        miss_counts,
    )
    torch.npu.synchronize()
    counts_cpu = miss_counts.cpu()
    expected_counts = torch.full_like(counts_cpu, miss_target)
    if not torch.equal(counts_cpu, expected_counts):
        raise AssertionError(
            "bs24 local schedule returned unexpected miss counts: "
            f"{counts_cpu.tolist()}"
        )

    expected_topk = torch.arange(
        candidate_len - TOPK, candidate_len, dtype=torch.int64
    )
    expected_misses = expected_topk[:miss_target]
    state_pool = cache_slots.cpu()
    sources_cpu = source_ids.view(batch, TOPK).cpu().to(torch.int64)
    slots_cpu = destination_slots.view(batch, TOPK).cpu().to(torch.int64)
    for row in range(batch):
        pool_row = int(req_entries_cpu[row])
        state = state_pool[pool_row]
        valid_slots = state[state >= 0].to(torch.int64)
        if (
            valid_slots.numel() != cache_budget
            or torch.unique(valid_slots).numel() != cache_budget
            or int(valid_slots.min()) != 0
            or int(valid_slots.max()) != cache_budget - 1
        ):
            raise AssertionError(
                f"bs24 row={row} cache state is not a permutation of [0,C)."
            )
        if bool((state[expected_topk] < 0).any()):
            raise AssertionError(
                f"bs24 row={row} dropped a true top-2048 token."
            )
        if not torch.equal(
            torch.sort(sources_cpu[row]).values,
            expected_topk,
        ):
            raise AssertionError(
                f"bs24 row={row} full top-k source index set is incorrect."
            )
        active_sources = torch.sort(
            sources_cpu[row, :miss_target]
        ).values
        if not torch.equal(active_sources, expected_misses):
            raise AssertionError(
                f"bs24 row={row} active miss source set is incorrect."
            )
        active_slots = slots_cpu[row, :miss_target]
        if (
            bool((active_slots < 0).any())
            or bool((active_slots >= cache_budget).any())
            or torch.unique(active_slots).numel() != miss_target
        ):
            raise AssertionError(
                f"bs24 row={row} active destination exceeds [0,C)."
            )
        expected_slots = state[expected_topk].to(torch.int64)
        if not torch.equal(
            torch.sort(slots_cpu[row]).values,
            torch.sort(expected_slots).values,
        ):
            raise AssertionError(
                f"bs24 row={row} full top-k slot set is incorrect."
            )

    torch.ops.nanovllm_dsa.fused_li_manage.default(
        query,
        weights,
        index_cache,
        block_table,
        candidate_lens,
        cache_tokens,
        req_entries,
        cache_slots,
        source_ids,
        destination_slots,
        miss_counts,
    )
    repeat_sources = source_ids
    repeat_slots = destination_slots
    repeat_counts = miss_counts
    torch.npu.synchronize()
    if bool((repeat_counts.cpu() != 0).any()):
        raise AssertionError("bs24 identical-query repeat must have zero misses.")
    repeat_slots_cpu = repeat_slots.view(batch, TOPK).cpu().to(torch.int64)
    repeat_sources_cpu = repeat_sources.view(batch, TOPK).cpu().to(torch.int64)
    repeat_state_pool = cache_slots.cpu()
    for row in range(batch):
        if not torch.equal(
            torch.sort(repeat_sources_cpu[row]).values,
            expected_topk,
        ):
            raise AssertionError(
                f"bs24 row={row} repeat top-k source index set is incorrect."
            )
        pool_row = int(req_entries_cpu[row])
        expected_slots = repeat_state_pool[pool_row, expected_topk].to(
            torch.int64
        )
        if not torch.equal(
            torch.sort(repeat_slots_cpu[row]).values,
            torch.sort(expected_slots).values,
        ):
            raise AssertionError(
                f"bs24 row={row} repeat top-k slot set is incorrect."
            )
    print(
        f"FUSED_LI_MANAGE_LOCAL_BS24_CHECK heads={heads} batch={batch} "
        f"candidate_len={candidate_len} cache_tokens={cache_budget} "
        f"misses_per_row={miss_target} shuffled_pool_entries=1 ok=1"
    )


def run_graph_case(case: Case, replays: int) -> None:
    """Capture with zero misses, then replay with changing nonzero misses."""

    if replays <= 0:
        return
    row = 2  # The configurable GLM 8.2K inference tier.
    query = case.query[row : row + 1].clone().contiguous()
    weights = case.weights[row : row + 1].contiguous()
    req_entries = case.req_entries[row : row + 1].contiguous()
    cache_tokens = case.cache_tokens[row : row + 1].contiguous()
    candidate_lens = case.candidate_lens[row : row + 1].contiguous()
    index_table = case.index_block_table[row : row + 1].contiguous()
    hbm_table = case.hbm_block_table[row : row + 1].contiguous()
    dram_table = case.dram_block_table[row : row + 1].contiguous()
    source_ids = torch.zeros(
        (1, 1, TOPK), dtype=torch.int32, device=case.device
    )
    destination_slots = torch.zeros_like(source_ids)
    miss_counts = torch.zeros((1,), dtype=torch.int32, device=case.device)

    def chain():
        outputs = call_fused_li_manage_with_buffers(
            case,
            query=query,
            weights=weights,
            req_entries=req_entries,
            cache_tokens=cache_tokens,
            candidate_lens=candidate_lens,
            index_block_table=index_table,
            source_ids=source_ids,
            destination_slots=destination_slots,
            miss_counts=miss_counts,
        )
        torch.ops.nanovllm_dsa.scatter_copy.default(
            outputs[0].view(1, TOPK),
            outputs[1].view(1, TOPK),
            outputs[2],
            hbm_table,
            dram_table,
            case.hbm_kpe,
            case.hbm_ckv,
            case.dram_kpe,
            case.dram_ckv,
        )
        return outputs

    # The preceding semantic test leaves this row at its high-token top-2048,
    # so capture follows the same zero-miss path as startup pre-capture.
    chain()
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    pool = torch.npu.graph_pool_handle()
    with torch.npu.graph(graph, pool=pool):
        graph_outputs = chain()
    torch.npu.synchronize()
    if any(
        output.data_ptr() != expected.data_ptr()
        for output, expected in zip(
            graph_outputs, (source_ids, destination_slots, miss_counts)
        )
    ):
        raise AssertionError("graph did not reuse caller-owned output buffers")
    if int(graph_outputs[2].cpu()[0]) != 0:
        raise AssertionError("graph capture precondition must be zero miss")

    candidate_len = CANDIDATE_LENS[row]
    expected_sets = (
        torch.arange(0, TOPK, dtype=torch.int64),
        torch.arange(candidate_len - TOPK, candidate_len, dtype=torch.int64),
    )
    for replay in range(replays):
        select_low = replay % 2 == 0
        query.zero_()
        query[:, 0, 0] = -256 if select_low else 256
        query[:, 0, 1] = -1 if select_low else 1
        torch.npu.current_stream().synchronize()
        graph.replay()
        torch.npu.synchronize()

        count = int(graph_outputs[2].cpu()[0])
        if count < 0 or count > TOPK:
            raise AssertionError(f"graph replay={replay} invalid miss_count={count}")
        active_sources = graph_outputs[0].view(-1)[:count].cpu()
        active_slots = graph_outputs[1].view(-1)[:count].cpu()
        if bool((active_sources < 0).any()) or bool(
            (active_sources >= candidate_len).any()
        ):
            raise AssertionError(
                f"graph replay={replay} source token exceeds candidate range"
            )
        cache_budget = CACHE_TIERS[row]
        if bool((active_slots < 0).any()) or bool(
            (active_slots >= cache_budget).any()
        ):
            raise AssertionError(
                f"graph replay={replay} destination slot exceeds cache budget"
            )

        expected = expected_sets[0 if select_low else 1]
        full_sources = graph_outputs[0].view(-1).cpu().to(torch.int64)
        if not torch.equal(
            torch.sort(full_sources).values,
            expected,
        ):
            raise AssertionError(
                f"graph replay={replay} did not publish the full top-2048 "
                "source index set"
            )
        pool_row = int(case.req_entries_cpu[row])
        state = case.cache_slots[pool_row].cpu()
        if bool((state[expected] < 0).any()):
            raise AssertionError(
                f"graph replay={replay} dropped a true top-2048 token"
            )
        full_slots = graph_outputs[1].view(-1).cpu().to(torch.int64)
        expected_slots = state[expected].to(torch.int64)
        if not torch.equal(
            torch.sort(full_slots).values,
            torch.sort(expected_slots).values,
        ):
            raise AssertionError(
                f"graph replay={replay} did not publish the full top-2048 "
                "HBM slot set"
            )
        validate_cache_data(case, row, expected)
    print(
        f"FUSED_LI_MANAGE_SCATTER_GRAPH_CHECK heads={case.heads} replays={replays} "
        "capture_zero_miss=1 replay_nonzero_miss=1 ok=1"
    )


def main() -> None:
    args = parse_args()
    if args.warmup < 0 or args.iters <= 0 or args.graph_replays < 0:
        raise ValueError(
            "--warmup and --graph-replays must be >=0; --iters must be >0."
        )
    device = torch.device(args.device)
    if device.type != "npu":
        raise ValueError("--device must select one NPU, for example npu:0.")
    heads = _parse_heads(args.heads)
    torch.npu.set_device(device)
    torch.npu.config.allow_internal_format = False
    opapi_path = require_local_opapi()
    print(f"FUSED_LI_MANAGE_SCATTER_OPAPI path={opapi_path} local=1")
    print(
        f"FUSED_LI_MANAGE_SCATTER_CONFIG device={device} heads={list(heads)} "
        f"cache_tiers={list(CACHE_TIERS)} candidate_lens={list(CANDIDATE_LENS)} "
        f"seed={args.seed}"
    )
    for head_count in heads:
        case = make_case(head_count, device, args.seed)
        run_case(case, args.warmup, args.iters)
        run_graph_case(case, args.graph_replays)
        del case
        torch.npu.empty_cache()
        run_local_bs24_case(head_count, device, args.seed)
        torch.npu.empty_cache()
    print("FUSED_LI_MANAGE_SCATTER_UT_OK")


if __name__ == "__main__":
    main()
