"""C8 LightningIndexer reference inputs shared by operator and graph tests."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

import torch
import torch_npu  # type: ignore

from _lidu_utils import TOPK, build_pool, feasible_miss


BLOCK_SIZE = 128
HEAD_DIM = 128


@dataclass
class C8Case:
    query: torch.Tensor
    key: torch.Tensor
    weights: torch.Tensor
    query_scale: torch.Tensor
    key_scale: torch.Tensor
    actual_q: torch.Tensor
    req_entries: torch.Tensor
    req_entries_cpu: torch.Tensor
    cache_tokens: torch.Tensor
    candidate_lens: torch.Tensor
    block_table: torch.Tensor
    native_topk: torch.Tensor
    initial_pool: torch.Tensor
    target_misses: list[int]
    source_capacity: int


def quantize_fp8(tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    quantized, scale = torch_npu.npu_dynamic_quant(
        tensor, dst_type=torch.float8_e4m3fn
    )
    return (
        quantized.contiguous(),
        scale.view(tensor.shape[:-1]).to(torch.float32).contiguous(),
    )


def normalized_hadamard_128(
    *, dtype: torch.dtype, device: torch.device
) -> torch.Tensor:
    """Match the normalized 128x128 Hadamard used by official GLM C8 LI."""

    matrix = torch.ones((1, 1), dtype=torch.float32)
    while matrix.size(0) < HEAD_DIM:
        top = torch.cat((matrix, matrix), dim=1)
        bottom = torch.cat((matrix, -matrix), dim=1)
        matrix = torch.cat((top, bottom), dim=0)
    return (matrix / math.sqrt(HEAD_DIM)).to(dtype=dtype, device=device)


def official_c8_lightning_indexer(
    query: torch.Tensor,
    key: torch.Tensor,
    weights: torch.Tensor,
    query_scale: torch.Tensor,
    key_scale: torch.Tensor,
    actual_q: torch.Tensor,
    candidate_lens: torch.Tensor,
    block_table: torch.Tensor,
) -> torch.Tensor:
    """Call the official A5 C8 LightningIndexer used by vLLM-Ascend."""

    op = getattr(torch_npu, "npu_quant_lightning_indexer", None)
    if op is None:
        namespace = getattr(torch.ops, "_C_ascend", None)
        op = (
            getattr(namespace, "npu_lightning_indexer_quant", None)
            if namespace is not None
            else None
        )
    if op is None:
        raise RuntimeError("official A5 C8 LightningIndexer is not registered")
    output = op(
        query=query,
        key=key,
        weights=weights,
        query_dequant_scale=query_scale,
        key_dequant_scale=key_scale,
        actual_seq_lengths_query=actual_q,
        actual_seq_lengths_key=candidate_lens,
        block_table=block_table,
        query_quant_mode=0,
        key_quant_mode=0,
        layout_query="TND",
        layout_key="PA_BSND",
        sparse_count=TOPK,
        sparse_mode=3,
    )
    topk = output[0] if isinstance(output, tuple) else output
    if not isinstance(topk, torch.Tensor) or topk.dtype != torch.int32:
        raise RuntimeError("official A5 C8 LI returned no int32 top-k tensor")
    if topk.numel() != query.size(0) * TOPK:
        raise RuntimeError(
            f"official A5 C8 LI returned unexpected shape {tuple(topk.shape)}"
        )
    return topk.reshape(query.size(0), TOPK).contiguous()


def make_case(
    device: torch.device,
    batch: int,
    source_len: int,
    heads: int,
    budgets: list[int],
    miss_range: tuple[int, int],
    pool_extra: int,
    seed: int,
    candidate_lens_cpu: list[int] | None = None,
) -> C8Case:
    if len(budgets) != batch:
        raise ValueError("budget list must match batch")
    torch.manual_seed(seed)
    torch.npu.manual_seed_all(seed)
    blocks = source_len // BLOCK_SIZE
    block_table_cpu = torch.stack(
        [
            torch.randperm(blocks, dtype=torch.int64).to(torch.int32)
            for _ in range(batch)
        ]
    )
    query_fp = torch.empty(
        (batch, heads, HEAD_DIM), dtype=torch.bfloat16, device=device
    ).uniform_(-1, 1)
    key_fp = torch.empty(
        (blocks, BLOCK_SIZE, 1, HEAD_DIM),
        dtype=torch.bfloat16,
        device=device,
    ).uniform_(-1, 1)
    hadamard = normalized_hadamard_128(dtype=query_fp.dtype, device=device)
    query_fp = torch.matmul(query_fp, hadamard)
    key_fp = torch.matmul(key_fp, hadamard)
    query, query_scale = quantize_fp8(query_fp)
    key, key_scale = quantize_fp8(key_fp)
    weights = torch.empty(
        (batch, heads), dtype=torch.bfloat16, device=device
    ).uniform_(0.01, 1.0).contiguous()

    if candidate_lens_cpu is None:
        candidate_lens_cpu = [source_len] * batch
    if len(candidate_lens_cpu) != batch:
        raise ValueError("candidate length list must match batch")
    if any(length < TOPK or length > source_len for length in candidate_lens_cpu):
        raise ValueError("candidate lengths must be in [2048,source_capacity]")
    candidate_lens = torch.tensor(
        candidate_lens_cpu, dtype=torch.int32, device=device
    )
    actual_q = torch.arange(1, batch + 1, dtype=torch.int32, device=device)
    block_table = block_table_cpu.to(device)
    native_topk = official_c8_lightning_indexer(
        query,
        key,
        weights,
        query_scale,
        key_scale,
        actual_q,
        candidate_lens,
        block_table,
    )
    torch.npu.synchronize()

    rng = random.Random(seed + 1)
    target_misses: list[int] = []
    for candidate_len, budget in zip(candidate_lens_cpu, budgets):
        if budget == 0:
            target_misses.append(0)
            continue
        feasible = [
            miss
            for miss in range(miss_range[0], miss_range[1] + 1)
            if feasible_miss(candidate_len, budget, miss)
        ]
        if not feasible:
            raise ValueError(
                f"no feasible miss in {miss_range} for "
                f"candidate_len={candidate_len}, C={budget}"
            )
        target_misses.append(rng.choice(feasible))
    if batch > 1:
        if budgets[0] > 0 and feasible_miss(
            candidate_lens_cpu[0], budgets[0], miss_range[0]
        ):
            target_misses[0] = miss_range[0]
        if budgets[1] > 0 and feasible_miss(
            candidate_lens_cpu[1], budgets[1], miss_range[1]
        ):
            target_misses[1] = miss_range[1]

    pool_size = batch + pool_extra
    req_entries_cpu = torch.randperm(pool_size, dtype=torch.int64)[:batch].to(
        torch.int32
    )
    initial_pool = build_pool(
        native_topk,
        source_len,
        candidate_lens_cpu,
        budgets,
        target_misses,
        req_entries_cpu,
        pool_size,
        seed + 2,
    ).to(device)
    return C8Case(
        query=query,
        key=key,
        weights=weights,
        query_scale=query_scale,
        key_scale=key_scale,
        actual_q=actual_q,
        req_entries=req_entries_cpu.to(device),
        req_entries_cpu=req_entries_cpu,
        cache_tokens=torch.tensor(budgets, dtype=torch.int32, device=device),
        candidate_lens=candidate_lens,
        block_table=block_table,
        native_topk=native_topk,
        initial_pool=initial_pool,
        target_misses=target_misses,
        source_capacity=source_len,
    )
