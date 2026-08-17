from __future__ import annotations

from dataclasses import dataclass

import torch


BLOCK_SIZE = 128
NOPE_DIM = 512
ROPE_DIM = 64
QUERY_DIM = NOPE_DIM + ROPE_DIM
TILE_SIZE = 128
SCALE_COUNT = NOPE_DIM // TILE_SIZE
PACKED_DIM = 656
TOPK = 2048


@dataclass
class StagedC8Case:
    query_cpu: torch.Tensor
    packed_cpu: torch.Tensor
    nope_cpu: torch.Tensor
    rope_cpu: torch.Tensor
    scales_cpu: torch.Tensor
    block_table_cpu: torch.Tensor
    actual_q_cpu: torch.Tensor
    resident_lengths_cpu: torch.Tensor
    cache_tokens_cpu: torch.Tensor
    topk_slots_cpu: torch.Tensor
    miss_counts_cpu: torch.Tensor
    query: torch.Tensor
    packed: torch.Tensor
    block_table: torch.Tensor
    actual_q: torch.Tensor
    resident_lengths: torch.Tensor
    cache_tokens: torch.Tensor
    topk_slots: torch.Tensor
    miss_counts: torch.Tensor
    scale: float


def _pack_cache(
    nope: torch.Tensor,
    rope: torch.Tensor,
    scales: torch.Tensor,
) -> torch.Tensor:
    packed_bytes = torch.cat(
        (
            nope.contiguous().view(torch.uint8),
            rope.contiguous().view(torch.uint8),
            scales.contiguous().view(torch.uint8),
        ),
        dim=-1,
    )
    if packed_bytes.shape[-1] != PACKED_DIM:
        raise AssertionError("invalid packed C8 row width")
    return packed_bytes.view(torch.float8_e4m3fn)


def make_case(
    *,
    device: torch.device,
    query_counts: tuple[int, ...] = (3,),
    heads: int = 4,
    cache_tokens: tuple[int, ...] = (2048,),
    final_tail_tokens: tuple[int, ...] = (5,),
    miss_counts: tuple[int, ...] = (0, 37, 2048),
    query_dtype: torch.dtype = torch.bfloat16,
    seed: int = 17,
) -> StagedC8Case:
    batch = len(query_counts)
    if not (
        len(cache_tokens) == batch
        and len(final_tail_tokens) == batch
        and sum(query_counts) == len(miss_counts)
    ):
        raise ValueError("staged C8 case metadata lengths are inconsistent")
    packed_queries = sum(query_counts)
    generator = torch.Generator().manual_seed(seed)
    torch.manual_seed(seed)
    resident_lengths_values = [
        budget + tail
        for budget, tail in zip(cache_tokens, final_tail_tokens)
    ]
    blocks_per_row = max(
        (length + BLOCK_SIZE - 1) // BLOCK_SIZE
        for length in resident_lengths_values
    )
    physical_blocks = batch * blocks_per_row
    block_table_cpu = torch.randperm(
        physical_blocks,
        generator=generator,
        dtype=torch.int64,
    ).reshape(batch, blocks_per_row).to(torch.int32)

    nope_cpu = torch.randint(
        -3,
        4,
        (physical_blocks, BLOCK_SIZE, 1, NOPE_DIM),
        generator=generator,
        dtype=torch.int16,
    ).float().to(torch.float8_e4m3fn)
    rope_cpu = torch.empty(
        (physical_blocks, BLOCK_SIZE, 1, ROPE_DIM),
        dtype=torch.float32,
    ).uniform_(-0.5, 0.5, generator=generator).to(torch.bfloat16)
    scales_cpu = torch.empty(
        (physical_blocks, BLOCK_SIZE, 1, SCALE_COUNT),
        dtype=torch.float32,
    ).uniform_(0.02, 0.08, generator=generator)
    packed_cpu = _pack_cache(nope_cpu, rope_cpu, scales_cpu)

    query_cpu = torch.empty(
        (packed_queries, heads, QUERY_DIM),
        dtype=torch.float32,
    ).uniform_(-0.5, 0.5, generator=generator).to(query_dtype)
    topk_slots_cpu = torch.full(
        (packed_queries, 1, TOPK),
        -1,
        dtype=torch.int32,
    )
    row = 0
    for request, query_count in enumerate(query_counts):
        budget = cache_tokens[request]
        for _ in range(query_count):
            if budget:
                topk_slots_cpu[row, 0] = torch.randperm(
                    budget,
                    generator=generator,
                )[:TOPK].to(torch.int32)
            row += 1

    actual_q_cpu = torch.tensor(query_counts, dtype=torch.int32).cumsum(
        0, dtype=torch.int32
    )
    resident_lengths_cpu = torch.tensor(
        resident_lengths_values,
        dtype=torch.int32,
    )
    cache_tokens_cpu = torch.tensor(cache_tokens, dtype=torch.int32)
    miss_counts_cpu = torch.tensor(miss_counts, dtype=torch.int32)
    return StagedC8Case(
        query_cpu=query_cpu,
        packed_cpu=packed_cpu,
        nope_cpu=nope_cpu,
        rope_cpu=rope_cpu,
        scales_cpu=scales_cpu,
        block_table_cpu=block_table_cpu,
        actual_q_cpu=actual_q_cpu,
        resident_lengths_cpu=resident_lengths_cpu,
        cache_tokens_cpu=cache_tokens_cpu,
        topk_slots_cpu=topk_slots_cpu,
        miss_counts_cpu=miss_counts_cpu,
        query=query_cpu.to(device),
        packed=packed_cpu.to(device),
        block_table=block_table_cpu.to(device),
        actual_q=actual_q_cpu.to(device),
        resident_lengths=resident_lengths_cpu.to(device),
        cache_tokens=cache_tokens_cpu.to(device),
        topk_slots=topk_slots_cpu.to(device),
        miss_counts=miss_counts_cpu.to(device),
        scale=QUERY_DIM**-0.5,
    )


def _request_for_row(case: StagedC8Case, row: int) -> tuple[int, int, int]:
    begin = 0
    for request, end_value in enumerate(case.actual_q_cpu.tolist()):
        end = int(end_value)
        if row < end:
            return request, row - begin, end - begin
        begin = end
    raise IndexError(row)


def selected_slots(
    case: StagedC8Case,
    row: int,
    stage: str,
) -> torch.Tensor:
    request, query_index, query_count = _request_for_row(case, row)
    budget = int(case.cache_tokens_cpu[request])
    resident = int(case.resident_lengths_cpu[request])
    visible_resident = resident - (query_count - 1 - query_index)
    miss_count = int(case.miss_counts_cpu[row])
    if budget == 0:
        if stage == "stage2":
            return torch.empty(0, dtype=torch.int64)
        return torch.arange(visible_resident, dtype=torch.int64)
    topk = case.topk_slots_cpu[row, 0].to(torch.int64)
    tail = torch.arange(
        budget,
        visible_resident,
        dtype=torch.int64,
    )
    if stage == "stage1":
        return torch.cat((topk[miss_count:], tail))
    if stage == "stage2":
        return topk[:miss_count]
    if stage == "full":
        return torch.cat((topk, tail))
    raise ValueError(stage)


def cpu_state(
    case: StagedC8Case,
    stage: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    scales = case.scales_cpu.repeat_interleave(TILE_SIZE, dim=-1)
    nope = case.nope_cpu.float() * scales
    value = nope.to(torch.bfloat16).float()
    key = torch.cat(
        (nope.to(torch.bfloat16), case.rope_cpu),
        dim=-1,
    ).float()
    query = case.query_cpu.float()
    packed_queries, heads, _ = query.shape
    partial = torch.zeros(
        (packed_queries, heads, NOPE_DIM),
        dtype=torch.float32,
    )
    maximum = torch.full(
        (1, packed_queries, heads),
        -torch.inf,
        dtype=torch.float32,
    )
    denominator = torch.zeros_like(maximum)
    normalized = torch.zeros_like(partial)
    block_table = case.block_table_cpu.to(torch.int64)
    for row in range(packed_queries):
        request, _, _ = _request_for_row(case, row)
        slots = selected_slots(case, row, stage)
        if slots.numel() == 0:
            continue
        physical = block_table[request, slots // BLOCK_SIZE]
        offsets = slots % BLOCK_SIZE
        selected_key = key[physical, offsets, 0]
        selected_value = value[physical, offsets, 0]
        scores = query[row] @ selected_key.T * case.scale
        row_max = scores.max(dim=-1).values
        weights = torch.exp(scores - row_max.unsqueeze(-1))
        row_sum = weights.sum(dim=-1)
        # Match the current C8 baseline's BF16 probability/value path.
        row_output = (
            (weights / row_sum.unsqueeze(-1)).to(torch.bfloat16).float()
            @ selected_value
        )
        normalized[row] = row_output
        partial[row] = row_output * row_sum.unsqueeze(-1)
        maximum[0, row] = row_max
        denominator[0, row] = row_sum
    return partial, maximum, denominator, normalized


def full_attention_slots(case: StagedC8Case) -> torch.Tensor:
    max_tail = max(
        int(length - budget)
        for length, budget in zip(
            case.resident_lengths_cpu,
            case.cache_tokens_cpu,
        )
    )
    capacity = TOPK + max_tail
    result = torch.full(
        (case.query_cpu.shape[0], 1, capacity),
        -1,
        dtype=torch.int32,
    )
    for row in range(case.query_cpu.shape[0]):
        request, _, _ = _request_for_row(case, row)
        budget = int(case.cache_tokens_cpu[request])
        if budget == 0:
            slots = selected_slots(case, row, "full").to(torch.int32)
            result[row, 0, : slots.numel()] = slots
            continue
        result[row, 0, :TOPK] = case.topk_slots_cpu[row, 0]
        final_tail = int(
            case.resident_lengths_cpu[request]
            - case.cache_tokens_cpu[request]
        )
        result[row, 0, TOPK : TOPK + final_tail] = torch.arange(
            budget,
            budget + final_tail,
            dtype=torch.int32,
        )
    return result


def error_metrics(
    actual: torch.Tensor,
    expected: torch.Tensor,
) -> tuple[float, float, float]:
    absolute = (actual.float() - expected.float()).abs()
    relative = absolute / expected.float().abs().clamp_min(1e-6)
    return (
        float(absolute.max()),
        float(relative.max()),
        float(absolute.mean()),
    )
