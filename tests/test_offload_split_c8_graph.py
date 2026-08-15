#!/usr/bin/env python3
"""Graph test: C8 fused_li_manage -> scatter -> sparse-tail Attention."""

from __future__ import annotations

import argparse
import math
import statistics

import torch

import nanovllm_dsa_a5
import torch_npu  # type: ignore  # noqa: F401

from _c8_lidu_case import make_case
from _utils import physical_token_rows, require_a5, swapped_from_cpu


BLOCK_SIZE = 128
NOPE_DIM = 512
ROPE_DIM = 64
PACKED_DIM = 656
TOPK = 2048
SOURCE_LEN = 4096
CACHE_TOKENS = 3072
TAIL_TOKENS = 64
MAX_TAIL_TOKENS = 256
OUTPUT_POISON = -123456789


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--replays", type=int, default=4)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--allow-non-a5", action="store_true")
    return parser.parse_args()


def private_table(
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


def physical(
    table: torch.Tensor, row: int, tokens: torch.Tensor
) -> torch.Tensor:
    return physical_token_rows(table, row, tokens, BLOCK_SIZE)


def make_packed_bytes(
    blocks: int, generator: torch.Generator
) -> torch.Tensor:
    nope = torch.randint(
        -3, 4, (blocks, BLOCK_SIZE, 1, NOPE_DIM),
        generator=generator, dtype=torch.int16,
    ).float().to(torch.float8_e4m3fn)
    rope = torch.empty(
        (blocks, BLOCK_SIZE, 1, ROPE_DIM), dtype=torch.float32
    ).uniform_(-0.5, 0.5, generator=generator).to(torch.bfloat16)
    scales = torch.empty(
        (blocks, BLOCK_SIZE, 1, NOPE_DIM // BLOCK_SIZE),
        dtype=torch.float32,
    ).uniform_(0.02, 0.08, generator=generator)
    packed = torch.cat(
        (
            nope.contiguous().view(torch.uint8),
            rope.contiguous().view(torch.uint8),
            scales.contiguous().view(torch.uint8),
        ),
        dim=-1,
    )
    if packed.size(-1) != PACKED_DIM:
        raise AssertionError(f"packed C8 row has {packed.size(-1)} bytes")
    return packed.view(torch.int8).contiguous()


def initialize_hbm(
    *,
    dram: torch.Tensor,
    dram_table: torch.Tensor,
    hbm_table: torch.Tensor,
    pool: torch.Tensor,
    req_entries: torch.Tensor,
    budgets: list[int],
    candidates: list[int],
    actual_lengths: list[int],
) -> torch.Tensor:
    hbm = torch.full(
        (int(hbm_table.max()) + 1, BLOCK_SIZE, 1, PACKED_DIM),
        -91, dtype=torch.int8,
    )
    dram_rows = dram.view(-1, PACKED_DIM)
    hbm_rows = hbm.view(-1, PACKED_DIM)
    pool_cpu = pool.cpu()
    req_cpu = req_entries.cpu()
    for row, (budget, candidate, actual) in enumerate(
        zip(budgets, candidates, actual_lengths)
    ):
        if budget == 0:
            sources = torch.arange(actual, dtype=torch.int64)
            destinations = sources
        else:
            state = pool_cpu[int(req_cpu[row]), :candidate]
            sources = (state >= 0).nonzero().flatten().to(torch.int64)
            destinations = state[sources].to(torch.int64)
        hbm_rows[physical(hbm_table, row, destinations)] = dram_rows[
            physical(dram_table, row, sources)
        ]
        if budget and actual > candidate:
            tail_sources = torch.arange(candidate, actual, dtype=torch.int64)
            tail_destinations = torch.arange(
                budget, budget + actual - candidate, dtype=torch.int64
            )
            hbm_rows[physical(hbm_table, row, tail_destinations)] = dram_rows[
                physical(dram_table, row, tail_sources)
            ]
    return hbm


def decode_packed(
    packed: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    raw = packed.contiguous().view(torch.uint8)
    nope = raw[..., :NOPE_DIM].contiguous().view(
        torch.float8_e4m3fn
    ).reshape(*raw.shape[:-1], NOPE_DIM)
    rope = raw[..., NOPE_DIM : NOPE_DIM + ROPE_DIM * 2].contiguous().view(
        torch.bfloat16
    ).reshape(*raw.shape[:-1], ROPE_DIM)
    scales = raw[..., NOPE_DIM + ROPE_DIM * 2 :].contiguous().view(
        torch.float32
    ).reshape(*raw.shape[:-1], NOPE_DIM // BLOCK_SIZE)
    value = (
        nope.float() * scales.repeat_interleave(BLOCK_SIZE, dim=-1)
    ).to(torch.bfloat16)
    key = torch.cat((value, rope), dim=-1)
    return key.float(), value.float()


def attention_golden(
    *,
    query: torch.Tensor,
    hbm: torch.Tensor,
    slots: torch.Tensor,
    block_table_cpu: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    key, value = decode_packed(hbm.cpu())
    flat_key = key.view(-1, NOPE_DIM + ROPE_DIM)
    flat_value = value.view(-1, NOPE_DIM)
    query_cpu = query.cpu().float()
    slots_cpu = slots.cpu()
    outputs: list[torch.Tensor] = []
    for row in range(query.size(0)):
        active = slots_cpu[row, 0]
        active = active[active >= 0].to(torch.int64)
        physical_rows = physical(block_table_cpu, row, active)
        selected_key = flat_key[physical_rows]
        selected_value = flat_value[physical_rows]
        scores = query_cpu[row] @ selected_key.T * scale
        probabilities = torch.softmax(scores, dim=-1)
        outputs.append(
            probabilities.to(torch.bfloat16).float() @ selected_value
        )
    return torch.stack(outputs)


def assert_copy(
    *,
    hbm: torch.Tensor,
    dram_cpu: torch.Tensor,
    hbm_table_cpu: torch.Tensor,
    dram_table_cpu: torch.Tensor,
    source_ids: torch.Tensor,
    destination_slots: torch.Tensor,
    miss_counts: torch.Tensor,
) -> None:
    sources = source_ids.cpu().view(source_ids.size(0), TOPK)
    destinations = destination_slots.cpu().view(source_ids.size(0), TOPK)
    hbm_rows = hbm.view(torch.int8).view(-1, PACKED_DIM)
    dram_rows = dram_cpu.view(-1, PACKED_DIM)
    for row, count in enumerate(miss_counts.cpu().tolist()):
        if count == 0:
            continue
        source_physical = physical(
            dram_table_cpu, row, sources[row, :count]
        )
        destination_physical = physical(
            hbm_table_cpu, row, destinations[row, :count]
        )
        if not torch.equal(
            hbm_rows[destination_physical.to(hbm.device)].cpu(),
            dram_rows[source_physical],
        ):
            raise AssertionError(f"row {row}: packed DRAM->HBM bytes differ")


def assert_attention_metadata(
    *,
    slots: torch.Tensor,
    resident_lengths: torch.Tensor,
    topk_slots: torch.Tensor,
    budgets: torch.Tensor,
    candidates: torch.Tensor,
    actual_kv: torch.Tensor,
) -> None:
    slots_cpu = slots.cpu()
    topk_cpu = topk_slots.cpu().view(topk_slots.size(0), TOPK)
    budgets_cpu = budgets.cpu().tolist()
    candidates_cpu = candidates.cpu().tolist()
    actual_cpu = actual_kv.cpu().tolist()
    expected_resident: list[int] = []
    for row, (budget, candidate, actual) in enumerate(
        zip(budgets_cpu, candidates_cpu, actual_cpu)
    ):
        if budget == 0:
            expected = torch.arange(actual, dtype=torch.int32)
            resident = actual
        else:
            tail = actual - candidate
            expected = torch.cat(
                (
                    topk_cpu[row],
                    torch.arange(
                        budget, budget + tail, dtype=torch.int32
                    ),
                )
            )
            resident = budget + tail
        actual_slots = slots_cpu[row, 0]
        if not torch.equal(actual_slots[: expected.numel()], expected):
            raise AssertionError(f"row {row}: sparse+tail slots differ")
        if bool((actual_slots[expected.numel() :] != -1).any()):
            raise AssertionError(f"row {row}: slot padding is not -1")
        expected_resident.append(resident)
    if resident_lengths.cpu().tolist() != expected_resident:
        raise AssertionError(
            "resident lengths differ: "
            f"actual={resident_lengths.cpu().tolist()} "
            f"expected={expected_resident}"
        )


def run_case(
    *,
    mixed: bool,
    device: torch.device,
    replays: int,
    seed: int,
) -> None:
    torch.manual_seed(seed)
    torch.npu.manual_seed_all(seed)
    generator = torch.Generator().manual_seed(seed)
    batch = 2
    budgets = [0 if mixed else CACHE_TOKENS, CACHE_TOKENS]
    candidates = [TOPK if mixed else SOURCE_LEN, SOURCE_LEN]
    actual_lengths = [candidate + TAIL_TOKENS for candidate in candidates]
    lidu_case = make_case(
        device=device, batch=batch, source_len=SOURCE_LEN, heads=32,
        budgets=budgets, miss_range=(256, 512), pool_extra=3,
        seed=seed, candidate_lens_cpu=candidates,
    )

    dram_blocks_per_row = math.ceil(
        (SOURCE_LEN + TAIL_TOKENS) / BLOCK_SIZE
    )
    dram_table_cpu, dram_blocks = private_table(
        batch, dram_blocks_per_row, generator
    )
    hbm_tokens_per_row = max(
        max(budgets) + MAX_TAIL_TOKENS,
        max(
            (actual for budget, actual in zip(budgets, actual_lengths)
             if budget == 0),
            default=0,
        ),
    )
    hbm_table_cpu, _ = private_table(
        batch, math.ceil(hbm_tokens_per_row / BLOCK_SIZE), generator
    )
    dram_cpu = make_packed_bytes(dram_blocks, generator)
    initial_hbm_cpu = initialize_hbm(
        dram=dram_cpu, dram_table=dram_table_cpu,
        hbm_table=hbm_table_cpu, pool=lidu_case.initial_pool,
        req_entries=lidu_case.req_entries, budgets=budgets,
        candidates=candidates, actual_lengths=actual_lengths,
    )
    dram = swapped_from_cpu(dram_cpu, device)
    initial_hbm = initial_hbm_cpu.view(torch.float8_e4m3fn).to(device)
    dram_table = dram_table_cpu.to(device)
    hbm_table = hbm_table_cpu.to(device)
    actual_kv = torch.tensor(
        actual_lengths, dtype=torch.int32, device=device
    )
    attention_query = torch.empty(
        (batch, 8, NOPE_DIM + ROPE_DIM),
        dtype=torch.bfloat16, device=device,
    ).uniform_(-0.5, 0.5)
    scale = (NOPE_DIM + ROPE_DIM) ** -0.5

    def make_state(
        hbm_seed: torch.Tensor | None = None,
        pool_seed: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, ...]:
        hbm = initial_hbm.clone() if hbm_seed is None else hbm_seed.clone()
        pool = (
            lidu_case.initial_pool.clone()
            if pool_seed is None else pool_seed.clone()
        )
        sources = torch.full(
            (batch, 1, TOPK), OUTPUT_POISON,
            dtype=torch.int32, device=device,
        )
        destinations = torch.full_like(sources, OUTPUT_POISON)
        counts = torch.full(
            (batch,), OUTPUT_POISON, dtype=torch.int32, device=device
        )
        attention_slots = torch.full(
            (batch, 1, TOPK + MAX_TAIL_TOKENS), OUTPUT_POISON,
            dtype=torch.int32, device=device,
        )
        resident_lengths = torch.full_like(counts, OUTPUT_POISON)
        return (
            hbm, pool, sources, destinations, counts,
            attention_slots, resident_lengths,
        )

    def chain(state: tuple[torch.Tensor, ...]):
        hbm, pool, sources, destinations, counts, slots, lengths = state
        lidu = torch.ops.nanovllm_dsa.fused_li_manage_c8_out.default(
            lidu_case.query, lidu_case.key, lidu_case.weights,
            lidu_case.query_scale, lidu_case.key_scale,
            lidu_case.actual_q, lidu_case.req_entries, pool,
            lidu_case.cache_tokens, lidu_case.candidate_lens,
            lidu_case.block_table, sources, destinations, counts,
        )
        scatter = torch.ops.nanovllm_dsa.kvcache_scatter_copy_c8_out.default(
            hbm.view(torch.int8), dram, hbm_table, dram_table,
            lidu[0], lidu[1], lidu[2], lidu_case.cache_tokens,
            lidu_case.candidate_lens, actual_kv, MAX_TAIL_TOKENS,
            slots, lengths,
        )
        attention = torch.ops.nanovllm_dsa.sparse_tail_attention_c8.default(
            attention_query, hbm, scatter[1], hbm_table,
            lidu_case.actual_q, scatter[2], scale,
        )
        return lidu, scatter, attention

    eager_state = make_state()
    eager_lidu, eager_scatter, eager_attention = chain(eager_state)
    torch.npu.synchronize()
    expected_sources = eager_lidu[0].cpu()
    expected_destinations = eager_lidu[1].cpu()
    expected_counts = eager_lidu[2].cpu()
    long_rows = lidu_case.cache_tokens.cpu() > 0
    if bool((expected_counts[long_rows] <= 0).any()):
        raise AssertionError(
            f"long rows require nonzero misses, got {expected_counts.tolist()}"
        )
    assert_copy(
        hbm=eager_state[0], dram_cpu=dram_cpu,
        hbm_table_cpu=hbm_table_cpu, dram_table_cpu=dram_table_cpu,
        source_ids=eager_lidu[0], destination_slots=eager_lidu[1],
        miss_counts=eager_lidu[2],
    )
    assert_attention_metadata(
        slots=eager_scatter[1], resident_lengths=eager_scatter[2],
        topk_slots=eager_lidu[1], budgets=lidu_case.cache_tokens,
        candidates=lidu_case.candidate_lens, actual_kv=actual_kv,
    )
    eager_golden = attention_golden(
        query=attention_query, hbm=eager_state[0],
        slots=eager_scatter[1], block_table_cpu=hbm_table_cpu,
        scale=scale,
    )
    torch.testing.assert_close(
        eager_attention.cpu().float(), eager_golden, atol=0.08, rtol=0.03
    )

    graph_state = make_state(eager_state[0], eager_state[1])
    graph = torch.npu.NPUGraph()
    graph_pool = torch.npu.graph_pool_handle()
    with torch.npu.graph(graph, pool=graph_pool):
        graph_lidu, graph_scatter, graph_attention = chain(graph_state)
    torch.npu.synchronize()
    if tuple(t.data_ptr() for t in graph_lidu[:3]) != tuple(
        t.data_ptr() for t in graph_state[2:5]
    ):
        raise AssertionError("captured C8 LIDU metadata is not caller-owned")
    if graph_lidu[3].data_ptr() != graph_state[1].data_ptr():
        raise AssertionError("captured C8 request pool does not alias input")
    if graph_scatter[0].data_ptr() != graph_state[0].data_ptr():
        raise AssertionError("captured packed HBM output does not alias input")
    if graph_scatter[1].data_ptr() != graph_state[5].data_ptr():
        raise AssertionError("captured attention slots are not caller-owned")
    if graph_scatter[2].data_ptr() != graph_state[6].data_ptr():
        raise AssertionError("captured resident lengths are not caller-owned")
    attention_ptr = graph_attention.data_ptr()

    for tensor in graph_state[2:7]:
        tensor.fill_(OUTPUT_POISON)
    graph.replay()
    torch.npu.synchronize()
    if bool((graph_lidu[2].cpu() != 0).any()):
        raise AssertionError("first replay of warmed C8 cache must be zero-miss")
    assert_attention_metadata(
        slots=graph_scatter[1], resident_lengths=graph_scatter[2],
        topk_slots=graph_lidu[1], budgets=lidu_case.cache_tokens,
        candidates=lidu_case.candidate_lens, actual_kv=actual_kv,
    )

    samples_us: list[float] = []
    for replay in range(replays):
        graph_state[0].copy_(initial_hbm)
        graph_state[1].copy_(lidu_case.initial_pool)
        for tensor in graph_state[2:7]:
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
        if not torch.equal(graph_lidu[1].cpu(), expected_destinations):
            raise AssertionError(f"replay {replay}: destination slots changed")
        if not torch.equal(graph_lidu[2].cpu(), expected_counts):
            raise AssertionError(f"replay {replay}: miss counts changed")
        assert_copy(
            hbm=graph_state[0], dram_cpu=dram_cpu,
            hbm_table_cpu=hbm_table_cpu, dram_table_cpu=dram_table_cpu,
            source_ids=graph_lidu[0], destination_slots=graph_lidu[1],
            miss_counts=graph_lidu[2],
        )
        assert_attention_metadata(
            slots=graph_scatter[1], resident_lengths=graph_scatter[2],
            topk_slots=graph_lidu[1], budgets=lidu_case.cache_tokens,
            candidates=lidu_case.candidate_lens, actual_kv=actual_kv,
        )
        golden = attention_golden(
            query=attention_query, hbm=graph_state[0],
            slots=graph_scatter[1], block_table_cpu=hbm_table_cpu,
            scale=scale,
        )
        torch.testing.assert_close(
            graph_attention.cpu().float(), golden, atol=0.08, rtol=0.03
        )
        if graph_attention.data_ptr() != attention_ptr:
            raise AssertionError("C8 Attention replay changed output address")

    print(
        "A5_C8_SPLIT_OFFLOAD_GRAPH_CHECK "
        f"case={'mixed' if mixed else 'pure-long'} "
        f"misses={expected_counts.tolist()} replays={replays} "
        f"avg_replay_us={statistics.mean(samples_us):.3f} "
        "capture_zero_miss=1 replay_nonzero_miss=1 "
        "caller_owned_metadata=1 dram_to_hbm_exact=1 "
        "attention_golden=1 stable_addresses=1 ok=1",
        flush=True,
    )


def main() -> None:
    args = parse_args()
    if args.replays <= 0:
        raise ValueError("--replays must be positive")
    device = torch.device(args.device)
    if device.type != "npu":
        raise ValueError("--device must select an NPU")
    torch.npu.set_device(device)
    torch.npu.config.allow_internal_format = False
    require_a5(device, args.allow_non_a5)
    for index, mixed in enumerate((False, True)):
        run_case(
            mixed=mixed, device=device, replays=args.replays,
            seed=args.seed + index * 1000,
        )
    print("A5_C8_SPLIT_OFFLOAD_GRAPH_UT_OK", flush=True)


if __name__ == "__main__":
    main()
