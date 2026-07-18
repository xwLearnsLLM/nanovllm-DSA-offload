"""Single-NPU semantic and latency test for repository-bundled LIDU+SCATTER."""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from time import perf_counter

import torch
import torch_npu  # type: ignore

import nanovllm.ops as ascend_ops


BLOCK_SIZE = 128
INDEX_DIM = 128
TOPK = 2048
KPE_DIM = 64
CKV_DIM = 512
CACHE_TIERS = (0, 2048, 3072, 5120, 8192, 12288)
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
        "--skip-gs-compare",
        action="store_true",
        help="Skip the same-input LightningIndexer+GatherSelection timing.",
    )
    return parser.parse_args()


def _parse_heads(value: str) -> tuple[int, ...]:
    heads = tuple(dict.fromkeys(int(item.strip()) for item in value.split(",")))
    if not heads or any(item not in (32, 64) for item in heads):
        raise ValueError("--heads only accepts 32 and/or 64.")
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
        expected_initial_sources=expected_initial_sources,
    )


def call_scatter(
    case: Case,
    source_ids: torch.Tensor,
    destination_slots: torch.Tensor,
    counts: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    return torch.ops.nanovllm_dsa.scatter_copy.default(
        case.hbm_kpe,
        case.hbm_ckv,
        case.dram_kpe,
        case.dram_ckv,
        case.hbm_block_table,
        case.dram_block_table,
        source_ids,
        destination_slots,
        counts,
    )


def call_lidu(case: Case):
    return torch.ops.nanovllm_dsa.lidu_decode_update.default(
        case.query,
        case.index_cache,
        case.weights,
        case.req_entries,
        case.cache_slots,
        case.cache_tokens,
        case.candidate_lens,
        case.index_block_table,
    )


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


def run_case(case: Case, warmup: int, iters: int) -> None:
    # Initialization path: one SCATTER call with capacities up to C=12288.
    call_scatter(
        case,
        case.init_source_ids,
        case.init_destination_slots,
        case.init_counts,
    )
    torch.npu.synchronize()
    for row, sources in enumerate(case.expected_initial_sources):
        validate_cache_data(case, row, sources)
    print(f"LIDU_SCATTER_INIT_CHECK heads={case.heads} max_c=12288 ok=1")

    source_ids, destination_slots, miss_counts, cache_alias = call_lidu(case)
    if cache_alias.data_ptr() != case.cache_slots.data_ptr():
        raise AssertionError("LIDU cache_slots output does not alias its mutable input")
    call_scatter(
        case,
        source_ids.view(len(CACHE_TIERS), TOPK),
        destination_slots.view(len(CACHE_TIERS), TOPK),
        miss_counts,
    )
    torch.npu.synchronize()

    expected_misses = torch.tensor([0, 1024, 1024, 1024, 1024, 1024])
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
    for row, sources in enumerate(expected_high):
        validate_cache_data(case, row, sources)
    print(
        f"LIDU_SCATTER_UPDATE_CHECK heads={case.heads} "
        f"miss_counts={actual_misses.tolist()} ok=1"
    )

    # Identical query must be a zero-miss repeat and leave every pool row valid.
    repeat_source, repeat_slots, repeat_counts, _ = call_lidu(case)
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
    print(f"LIDU_SCATTER_ZERO_MISS_CHECK heads={case.heads} ok=1")

    for _ in range(warmup):
        source, slots, counts, _ = call_lidu(case)
        call_scatter(
            case,
            source.view(len(CACHE_TIERS), TOPK),
            slots.view(len(CACHE_TIERS), TOPK),
            counts,
        )
    torch.npu.synchronize()
    start = perf_counter()
    for _ in range(iters):
        source, slots, counts, _ = call_lidu(case)
        call_scatter(
            case,
            source.view(len(CACHE_TIERS), TOPK),
            slots.view(len(CACHE_TIERS), TOPK),
            counts,
        )
    torch.npu.synchronize()
    avg_ms = (perf_counter() - start) * 1000.0 / iters
    print(
        f"LIDU_SCATTER_RESULT heads={case.heads} batch={len(CACHE_TIERS)} "
        f"candidate_max={max(CANDIDATE_LENS)} avg_chain_ms={avg_ms:.6f} "
        f"warmup={warmup} iters={iters}"
    )


def compare_with_gs(case: Case, warmup: int, iters: int) -> None:
    """Time both stable chains on the same five offloaded request rows."""

    row_slice = slice(1, None)
    batch = len(CACHE_TIERS) - 1
    query = case.query[row_slice].contiguous()
    weights = case.weights[row_slice].contiguous()
    req_entries = case.req_entries[row_slice].contiguous()
    candidate_lens = case.candidate_lens[row_slice].contiguous()
    index_tables = case.index_block_table[row_slice].contiguous()
    hbm_tables = case.hbm_block_table[row_slice].contiguous()
    dram_tables = case.dram_block_table[row_slice].contiguous()
    cache_tokens = case.cache_tokens[row_slice].contiguous()
    gather_lens = candidate_lens + 1
    query_lens = torch.arange(
        1, batch + 1, dtype=torch.int32, device=case.device
    )

    selection_blocks_per_row = TOPK // BLOCK_SIZE
    selection_kpe = torch.zeros(
        batch * selection_blocks_per_row,
        BLOCK_SIZE,
        KPE_DIM,
        dtype=torch.bfloat16,
        device=case.device,
    )
    selection_ckv = torch.zeros(
        batch * selection_blocks_per_row,
        BLOCK_SIZE,
        CKV_DIM,
        dtype=torch.bfloat16,
        device=case.device,
    )
    selection_tables = torch.arange(
        batch * selection_blocks_per_row,
        dtype=torch.int32,
        device=case.device,
    ).view(batch, selection_blocks_per_row)
    selection_status = torch.full(
        (case.cache_slots.shape[0], 1, 1, TOPK + 1),
        -1,
        dtype=torch.int32,
        device=case.device,
    )

    def li_topk() -> torch.Tensor:
        kwargs = dict(
            query=query,
            key=case.index_cache,
            weights=weights,
            actual_seq_lengths_query=query_lens,
            actual_seq_lengths_key=candidate_lens,
            block_table=index_tables,
            layout_query="TND",
            layout_key="PA_BSND",
            sparse_count=TOPK,
            sparse_mode=3,
        )
        if case.heads == 32:
            result = torch_npu.npu_lightning_indexer(**kwargs)
            return result[0] if isinstance(result, (tuple, list)) else result
        return ascend_ops.npu_lightning_indexer(**kwargs)

    def gs_chain() -> None:
        topk = li_topk()
        ascend_ops.npu_gather_selection_kv_cache(
            selection_kpe,
            selection_ckv,
            selection_tables,
            selection_status,
            req_entries,
            topk.view(batch, 1, 1, TOPK),
            case.dram_kpe,
            case.dram_ckv,
            dram_tables,
            gather_lens,
        )

    def lidu_chain() -> None:
        source, slots, counts, _ = torch.ops.nanovllm_dsa.lidu_decode_update.default(
            query,
            case.index_cache,
            weights,
            req_entries,
            case.cache_slots,
            cache_tokens,
            candidate_lens,
            index_tables,
        )
        torch.ops.nanovllm_dsa.scatter_copy.default(
            case.hbm_kpe,
            case.hbm_ckv,
            case.dram_kpe,
            case.dram_ckv,
            hbm_tables,
            dram_tables,
            source.view(batch, TOPK),
            slots.view(batch, TOPK),
            counts,
        )

    # Populate GS state once; LIDU state is already warm from run_case().
    gs_chain()
    torch.npu.synchronize()
    for _ in range(warmup):
        gs_chain()
    torch.npu.synchronize()
    start = perf_counter()
    for _ in range(iters):
        gs_chain()
    torch.npu.synchronize()
    gs_ms = (perf_counter() - start) * 1000.0 / iters

    for _ in range(warmup):
        lidu_chain()
    torch.npu.synchronize()
    start = perf_counter()
    for _ in range(iters):
        lidu_chain()
    torch.npu.synchronize()
    lidu_ms = (perf_counter() - start) * 1000.0 / iters
    print(
        f"LIDU_GS_COMPARE heads={case.heads} batch={batch} "
        f"li_gs_ms={gs_ms:.6f} lidu_scatter_ms={lidu_ms:.6f} "
        f"speedup={gs_ms / lidu_ms:.4f} warmup={warmup} iters={iters}"
    )


def main() -> None:
    args = parse_args()
    if args.warmup < 0 or args.iters <= 0:
        raise ValueError("--warmup must be >=0 and --iters must be >0.")
    device = torch.device(args.device)
    if device.type != "npu":
        raise ValueError("--device must select one NPU, for example npu:0.")
    heads = _parse_heads(args.heads)
    torch.npu.set_device(device)
    torch.npu.config.allow_internal_format = False
    opapi_path = os.environ.get("NANOVLLM_CUST_OPAPI_LIB", "")
    if not opapi_path or not os.path.isfile(opapi_path):
        raise RuntimeError(
            "Repository-local libcust_opapi.so was not selected; rebuild "
            "with `bash scripts/build_nanovllm_ops.sh`."
        )
    print(f"LIDU_SCATTER_OPAPI path={opapi_path} local=1")
    print(
        f"LIDU_SCATTER_CONFIG device={device} heads={list(heads)} "
        f"cache_tiers={list(CACHE_TIERS)} candidate_lens={list(CANDIDATE_LENS)} "
        f"seed={args.seed}"
    )
    for head_count in heads:
        case = make_case(head_count, device, args.seed)
        run_case(case, args.warmup, args.iters)
        if not args.skip_gs_compare:
            compare_with_gs(case, args.warmup, args.iters)
        del case
        torch.npu.empty_cache()
    print("LIDU_SCATTER_UT_OK")


if __name__ == "__main__":
    main()
