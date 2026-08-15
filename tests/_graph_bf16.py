"""Shared BF16 non-MTP offload-chain graph test."""

from __future__ import annotations

import argparse
import math
import statistics
from typing import Literal

import torch

import nanovllm_dsa_a5
import torch_npu  # type: ignore  # noqa: F401

from _utils import physical_token_rows, require_a5, swapped_from_cpu


BLOCK_SIZE = 128
INDEX_DIM = 128
TOPK = 2048
CKV_DIM = 512
KPE_DIM = 64
SOURCE_LEN = 4096
CACHE_TOKENS = 3072
TAIL_TOKENS = 64
OUTPUT_POISON = -123456789


def parse_args(description: str) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--replays", type=int, default=4)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--allow-non-a5", action="store_true")
    return parser.parse_args()


def _private_table(
    batch: int,
    blocks_per_row: int,
    generator: torch.Generator,
) -> tuple[torch.Tensor, int]:
    table = torch.empty((batch, blocks_per_row), dtype=torch.int32)
    for row in range(batch):
        base = row * blocks_per_row
        table[row] = base + torch.randperm(
            blocks_per_row, generator=generator
        ).to(torch.int32)
    return table, batch * blocks_per_row


def _physical(
    table: torch.Tensor, row: int, tokens: torch.Tensor
) -> torch.Tensor:
    return physical_token_rows(table, row, tokens, BLOCK_SIZE)


def _attention_golden(
    query: torch.Tensor,
    query_rope: torch.Tensor,
    hbm_ckv: torch.Tensor,
    hbm_kpe: torch.Tensor,
    hbm_table_cpu: torch.Tensor,
    slots: torch.Tensor,
    cache_tokens: torch.Tensor,
    actual_kv: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    query_cpu = query.cpu().float()
    query_rope_cpu = query_rope.cpu().float()
    ckv_cpu = hbm_ckv.cpu().float().view(-1, CKV_DIM)
    kpe_cpu = hbm_kpe.cpu().float().view(-1, KPE_DIM)
    slots_cpu = slots.cpu().view(query.size(0), TOPK).to(torch.int64)
    budgets = cache_tokens.cpu().tolist()
    lengths = actual_kv.cpu().tolist()
    outputs: list[torch.Tensor] = []
    for row, (budget, length) in enumerate(zip(budgets, lengths)):
        logical = (
            torch.arange(length, dtype=torch.int64)
            if budget == 0
            else torch.cat(
                (
                    slots_cpu[row],
                    torch.arange(budget, length, dtype=torch.int64),
                )
            )
        )
        physical = _physical(hbm_table_cpu, row, logical)
        key = ckv_cpu[physical]
        key_rope = kpe_cpu[physical]
        scores = (
            query_cpu[row] @ key.T
            + query_rope_cpu[row] @ key_rope.T
        ) * scale
        outputs.append(torch.softmax(scores, dim=-1) @ key)
    return torch.stack(outputs)


def _assert_copy(
    *,
    hbm_ckv: torch.Tensor,
    hbm_kpe: torch.Tensor,
    dram_ckv_cpu: torch.Tensor,
    dram_kpe_cpu: torch.Tensor,
    hbm_table_cpu: torch.Tensor,
    dram_table_cpu: torch.Tensor,
    source_ids: torch.Tensor,
    destination_slots: torch.Tensor,
    miss_counts: torch.Tensor,
) -> None:
    sources = source_ids.cpu().view(source_ids.size(0), TOPK)
    destinations = destination_slots.cpu().view(source_ids.size(0), TOPK)
    hbm_ckv_rows = hbm_ckv.view(-1, CKV_DIM)
    hbm_kpe_rows = hbm_kpe.view(-1, KPE_DIM)
    dram_ckv_rows = dram_ckv_cpu.view(-1, CKV_DIM)
    dram_kpe_rows = dram_kpe_cpu.view(-1, KPE_DIM)
    for row, count in enumerate(miss_counts.cpu().tolist()):
        if count == 0:
            continue
        source = sources[row, :count].to(torch.int64)
        destination = destinations[row, :count].to(torch.int64)
        source_physical = _physical(dram_table_cpu, row, source)
        destination_physical = _physical(hbm_table_cpu, row, destination)
        device_indices = destination_physical.to(hbm_ckv.device)
        if not torch.equal(
            hbm_ckv_rows[device_indices].cpu(), dram_ckv_rows[source_physical]
        ):
            raise AssertionError(f"row {row}: CKV DRAM->HBM copy differs")
        if not torch.equal(
            hbm_kpe_rows[device_indices].cpu(), dram_kpe_rows[source_physical]
        ):
            raise AssertionError(f"row {row}: KPE DRAM->HBM copy differs")


def _run_case(
    *,
    path: Literal["split", "fused"],
    mixed: bool,
    device: torch.device,
    replays: int,
    seed: int,
) -> None:
    torch.manual_seed(seed)
    torch.npu.manual_seed_all(seed)
    generator = torch.Generator().manual_seed(seed)
    batch = 2
    index_heads = 32
    attention_heads = 8
    source_blocks_per_row = SOURCE_LEN // BLOCK_SIZE
    index_table_cpu, index_blocks = _private_table(
        batch, source_blocks_per_row, generator
    )

    # Exactly representable +/-1 scores make both selected 2048-token sets
    # deterministic. The initial cache covers the high half plus 1024 low
    # tokens, while the fixed negative query selects the low half. The cold
    # replay therefore has exactly 1024 misses on every offloaded row.
    index_key_cpu = torch.zeros(
        (index_blocks, BLOCK_SIZE, 1, INDEX_DIM), dtype=torch.bfloat16
    )
    logical = torch.arange(SOURCE_LEN, dtype=torch.int64)
    score = torch.where(logical < TOPK, -1, 1).to(torch.bfloat16)
    for row in range(batch):
        index_key_cpu.view(-1, INDEX_DIM)[
            _physical(index_table_cpu, row, logical), 0
        ] = score

    budgets_cpu = torch.tensor(
        [0 if mixed else CACHE_TOKENS, CACHE_TOKENS], dtype=torch.int32
    )
    candidates_cpu = torch.tensor(
        [TOPK if mixed else SOURCE_LEN, SOURCE_LEN], dtype=torch.int32
    )
    actual_kv_cpu = torch.tensor(
        [TOPK if mixed else CACHE_TOKENS + TAIL_TOKENS,
         CACHE_TOKENS + TAIL_TOKENS],
        dtype=torch.int32,
    )
    req_entries_cpu = torch.tensor([2, 0], dtype=torch.int32)
    pool_cpu = torch.full((4, SOURCE_LEN), -1, dtype=torch.int32)
    high = torch.arange(SOURCE_LEN - TOPK, SOURCE_LEN, dtype=torch.int64)
    cached_extra = torch.arange(CACHE_TOKENS - TOPK, dtype=torch.int64)
    cached_sources = torch.cat((high, cached_extra))
    cached_slots = torch.randperm(CACHE_TOKENS, generator=generator).to(
        torch.int64
    )
    for row, budget in enumerate(budgets_cpu.tolist()):
        if budget:
            pool_cpu[int(req_entries_cpu[row]), cached_sources] = (
                cached_slots.to(torch.int32)
            )

    dram_blocks_per_row = math.ceil(
        (SOURCE_LEN + TAIL_TOKENS) / BLOCK_SIZE
    )
    dram_table_cpu, dram_blocks = _private_table(
        batch, dram_blocks_per_row, generator
    )
    hbm_blocks_per_row = math.ceil(
        int(actual_kv_cpu.max()) / BLOCK_SIZE
    )
    hbm_table_cpu, hbm_blocks = _private_table(
        batch, hbm_blocks_per_row, generator
    )
    dram_ckv_cpu = torch.randn(
        dram_blocks, BLOCK_SIZE, CKV_DIM,
        generator=generator, dtype=torch.float32,
    ).to(torch.bfloat16)
    dram_kpe_cpu = torch.randn(
        dram_blocks, BLOCK_SIZE, KPE_DIM,
        generator=generator, dtype=torch.float32,
    ).to(torch.bfloat16)
    initial_hbm_ckv_cpu = torch.randn(
        hbm_blocks, BLOCK_SIZE, 1, CKV_DIM,
        generator=generator, dtype=torch.float32,
    ).to(torch.bfloat16)
    initial_hbm_kpe_cpu = torch.randn(
        hbm_blocks, BLOCK_SIZE, 1, KPE_DIM,
        generator=generator, dtype=torch.float32,
    ).to(torch.bfloat16)
    hbm_ckv_rows = initial_hbm_ckv_cpu.view(-1, CKV_DIM)
    hbm_kpe_rows = initial_hbm_kpe_cpu.view(-1, KPE_DIM)
    dram_ckv_rows = dram_ckv_cpu.view(-1, CKV_DIM)
    dram_kpe_rows = dram_kpe_cpu.view(-1, KPE_DIM)
    for row, budget in enumerate(budgets_cpu.tolist()):
        if budget == 0:
            source = torch.arange(int(actual_kv_cpu[row]), dtype=torch.int64)
            destination = source
        else:
            source = cached_sources
            destination = cached_slots
        hbm_physical = _physical(hbm_table_cpu, row, destination)
        dram_physical = _physical(dram_table_cpu, row, source)
        hbm_ckv_rows[hbm_physical] = dram_ckv_rows[dram_physical]
        hbm_kpe_rows[hbm_physical] = dram_kpe_rows[dram_physical]
        if budget:
            tail_source = torch.arange(
                int(candidates_cpu[row]),
                int(candidates_cpu[row]) + TAIL_TOKENS,
                dtype=torch.int64,
            )
            tail_destination = torch.arange(
                budget, budget + TAIL_TOKENS, dtype=torch.int64
            )
            hbm_tail = _physical(hbm_table_cpu, row, tail_destination)
            dram_tail = _physical(dram_table_cpu, row, tail_source)
            hbm_ckv_rows[hbm_tail] = dram_ckv_rows[dram_tail]
            hbm_kpe_rows[hbm_tail] = dram_kpe_rows[dram_tail]

    index_query = torch.zeros(
        (batch, index_heads, INDEX_DIM),
        dtype=torch.bfloat16, device=device,
    )
    index_query[:, 0, 0] = -1
    weights = torch.zeros(
        (batch, index_heads), dtype=torch.bfloat16, device=device
    )
    weights[:, 0] = 1
    index_key = index_key_cpu.to(device)
    req_entries = req_entries_cpu.to(device)
    cache_tokens = budgets_cpu.to(device)
    candidate_lens = candidates_cpu.to(device)
    index_table = index_table_cpu.to(device)
    dram_table = dram_table_cpu.to(device)
    hbm_table = hbm_table_cpu.to(device)
    actual_q = torch.arange(1, batch + 1, dtype=torch.int32, device=device)
    actual_kv = actual_kv_cpu.to(device)
    dram_ckv = swapped_from_cpu(dram_ckv_cpu, device)
    dram_kpe = swapped_from_cpu(dram_kpe_cpu, device)
    initial_hbm_ckv = initial_hbm_ckv_cpu.to(device)
    initial_hbm_kpe = initial_hbm_kpe_cpu.to(device)
    attention_query = torch.randn(
        batch, attention_heads, CKV_DIM,
        generator=None, dtype=torch.bfloat16, device=device,
    )
    attention_query_rope = torch.randn(
        batch, attention_heads, KPE_DIM,
        generator=None, dtype=torch.bfloat16, device=device,
    )
    scale = 1.0 / math.sqrt(CKV_DIM + KPE_DIM)

    def make_state(
        pool_seed: torch.Tensor | None = None,
        ckv_seed: torch.Tensor | None = None,
        kpe_seed: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, ...]:
        pool = (
            pool_cpu.to(device)
            if pool_seed is None
            else pool_seed.clone()
        )
        hbm_ckv = (
            initial_hbm_ckv.clone() if ckv_seed is None else ckv_seed.clone()
        )
        hbm_kpe = (
            initial_hbm_kpe.clone() if kpe_seed is None else kpe_seed.clone()
        )
        source_ids = torch.full(
            (batch, 1, TOPK), OUTPUT_POISON,
            dtype=torch.int32, device=device,
        )
        destination_slots = torch.full_like(source_ids, OUTPUT_POISON)
        miss_counts = torch.full(
            (batch,), OUTPUT_POISON, dtype=torch.int32, device=device
        )
        return (
            pool, hbm_ckv, hbm_kpe,
            source_ids, destination_slots, miss_counts,
        )

    def chain(state: tuple[torch.Tensor, ...]):
        pool, hbm_ckv, hbm_kpe, sources, slots, counts = state
        lidu = torch.ops.nanovllm_dsa.fused_li_manage_out.default(
            index_query, index_key, weights, req_entries, pool,
            cache_tokens, candidate_lens, index_table,
            sources, slots, counts,
        )
        if path == "split":
            cache_alias = torch.ops.nanovllm_dsa.kvcache_scatter_copy.default(
                hbm_kpe.view(hbm_blocks, BLOCK_SIZE, KPE_DIM),
                hbm_ckv.view(hbm_blocks, BLOCK_SIZE, CKV_DIM),
                dram_kpe, dram_ckv, hbm_table, dram_table,
                lidu[0].view(batch, TOPK),
                lidu[1].view(batch, TOPK), lidu[2],
            )
            attention = torch.ops.nanovllm_dsa.sparse_tail_attention.default(
                attention_query, hbm_ckv, hbm_ckv, lidu[1],
                cache_tokens, hbm_table, actual_q, actual_kv,
                attention_query_rope, hbm_kpe, scale,
            )
            return lidu, cache_alias, attention
        fused = torch.ops.nanovllm_dsa.fused_copy_sparse_tail_attention.default(
            attention_query, hbm_ckv, lidu[1], cache_tokens,
            hbm_table, actual_q, actual_kv, attention_query_rope,
            hbm_kpe, dram_kpe, dram_ckv, dram_table,
            lidu[0].view(batch, TOPK), lidu[2], scale,
        )
        return lidu, (fused[1], fused[2]), fused[0]

    # Eager nonzero-miss result is the independent replay oracle.
    initial_pool = pool_cpu.to(device)
    eager_state = make_state()
    eager_lidu, _, eager_attention = chain(eager_state)
    torch.npu.synchronize()
    expected_sources = eager_lidu[0].cpu()
    expected_slots = eager_lidu[1].cpu()
    expected_counts = eager_lidu[2].cpu()
    if bool((expected_counts[budgets_cpu > 0] <= 0).any()):
        raise AssertionError(
            f"long rows require nonzero misses, got {expected_counts.tolist()}"
        )
    _assert_copy(
        hbm_ckv=eager_state[1], hbm_kpe=eager_state[2],
        dram_ckv_cpu=dram_ckv_cpu, dram_kpe_cpu=dram_kpe_cpu,
        hbm_table_cpu=hbm_table_cpu, dram_table_cpu=dram_table_cpu,
        source_ids=eager_lidu[0], destination_slots=eager_lidu[1],
        miss_counts=eager_lidu[2],
    )
    eager_golden = _attention_golden(
        attention_query, attention_query_rope,
        eager_state[1], eager_state[2], hbm_table_cpu,
        eager_lidu[1], cache_tokens, actual_kv, scale,
    )
    torch.testing.assert_close(
        eager_attention.cpu().float(), eager_golden, rtol=0.02, atol=0.04
    )

    # Seed capture with the warmed cache so capture and the first replay are
    # zero-miss; later replays restore the cold state outside the graph.
    graph_state = make_state(eager_state[0], eager_state[1], eager_state[2])
    graph = torch.npu.NPUGraph()
    graph_pool = torch.npu.graph_pool_handle()
    with torch.npu.graph(graph, pool=graph_pool):
        graph_lidu, graph_cache, graph_attention = chain(graph_state)
    torch.npu.synchronize()
    if tuple(t.data_ptr() for t in graph_lidu[:3]) != tuple(
        t.data_ptr() for t in graph_state[3:6]
    ):
        raise AssertionError("captured LIDU metadata is not caller-owned")
    if graph_lidu[3].data_ptr() != graph_state[0].data_ptr():
        raise AssertionError("captured request pool does not alias input")
    if graph_cache[0].data_ptr() != graph_state[2].data_ptr():
        raise AssertionError("captured KPE output does not alias HBM")
    if graph_cache[1].data_ptr() != graph_state[1].data_ptr():
        raise AssertionError("captured CKV output does not alias HBM")
    attention_ptr = graph_attention.data_ptr()

    for tensor in graph_state[3:6]:
        tensor.fill_(OUTPUT_POISON)
    graph.replay()
    torch.npu.synchronize()
    if bool((graph_lidu[2].cpu() != 0).any()):
        raise AssertionError("first replay of warmed cache must be zero-miss")

    samples_us: list[float] = []
    for replay in range(replays):
        graph_state[0].copy_(initial_pool)
        graph_state[1].copy_(initial_hbm_ckv)
        graph_state[2].copy_(initial_hbm_kpe)
        for tensor in graph_state[3:6]:
            tensor.fill_(OUTPUT_POISON)
        start = torch.npu.Event(enable_timing=True)
        end = torch.npu.Event(enable_timing=True)
        start.record()
        graph.replay()
        end.record()
        end.synchronize()
        samples_us.append(start.elapsed_time(end) * 1000.0)
        if not torch.equal(graph_lidu[0].cpu(), expected_sources):
            raise AssertionError(f"replay {replay}: source IDs changed")
        if not torch.equal(graph_lidu[1].cpu(), expected_slots):
            raise AssertionError(f"replay {replay}: destination slots changed")
        if not torch.equal(graph_lidu[2].cpu(), expected_counts):
            raise AssertionError(f"replay {replay}: miss counts changed")
        _assert_copy(
            hbm_ckv=graph_state[1], hbm_kpe=graph_state[2],
            dram_ckv_cpu=dram_ckv_cpu, dram_kpe_cpu=dram_kpe_cpu,
            hbm_table_cpu=hbm_table_cpu, dram_table_cpu=dram_table_cpu,
            source_ids=graph_lidu[0], destination_slots=graph_lidu[1],
            miss_counts=graph_lidu[2],
        )
        golden = _attention_golden(
            attention_query, attention_query_rope,
            graph_state[1], graph_state[2], hbm_table_cpu,
            graph_lidu[1], cache_tokens, actual_kv, scale,
        )
        torch.testing.assert_close(
            graph_attention.cpu().float(), golden, rtol=0.02, atol=0.04
        )
        if graph_attention.data_ptr() != attention_ptr:
            raise AssertionError("Attention replay changed output address")

    print(
        "A5_BF16_OFFLOAD_GRAPH_CHECK "
        f"path={path} case={'mixed' if mixed else 'pure-long'} "
        f"misses={expected_counts.tolist()} replays={replays} "
        f"avg_replay_us={statistics.mean(samples_us):.3f} "
        "capture_zero_miss=1 replay_nonzero_miss=1 "
        "caller_owned_metadata=1 dram_to_hbm_exact=1 "
        "attention_golden=1 stable_addresses=1 ok=1",
        flush=True,
    )


def run(path: Literal["split", "fused"], description: str) -> None:
    args = parse_args(description)
    if args.replays <= 0:
        raise ValueError("--replays must be positive")
    device = torch.device(args.device)
    if device.type != "npu":
        raise ValueError("--device must select an NPU")
    torch.npu.set_device(device)
    torch.npu.config.allow_internal_format = False
    require_a5(device, args.allow_non_a5)
    for index, mixed in enumerate((False, True)):
        _run_case(
            path=path, mixed=mixed, device=device,
            replays=args.replays, seed=args.seed + index * 1000,
        )
    print(
        f"A5_BF16_{path.upper()}_OFFLOAD_GRAPH_UT_OK",
        flush=True,
    )
