from __future__ import annotations

import torch

from ._constants import BLOCK_SIZE, SPARSE_COUNT
from .fused_li_manage_c8 import _native_quant_lightning_indexer


MAX_QUERIES_PER_REQUEST = 4
MIN_QUERIES_PER_REQUEST = 2
UNION_CAPACITY = MAX_QUERIES_PER_REQUEST * SPARSE_COUNT

_cache_update = (
    torch.ops.nanovllm_dsa._fused_li_manage_mtp_c8_cache_update
)
_cache_update_out = (
    torch.ops.nanovllm_dsa._fused_li_manage_mtp_c8_cache_update_out
)


def _check_inputs(
    query: torch.Tensor,
    key: torch.Tensor,
    weights: torch.Tensor,
    query_dequant_scale: torch.Tensor,
    key_dequant_scale: torch.Tensor,
    actual_seq_lengths_query: torch.Tensor,
    req_pool_entries: torch.Tensor,
    cache_slots_pool: torch.Tensor,
    cache_tokens: torch.Tensor,
    candidate_lens: torch.Tensor,
    block_table: torch.Tensor,
) -> None:
    if (
        query.ndim != 3
        or query.shape[0] <= 0
        or query.shape[1] not in (32, 64)
        or query.shape[2] != 128
    ):
        raise ValueError("C8 MTP LIDU query must be packed [T,32|64,128]")
    packed_queries, heads, _ = query.shape
    if actual_seq_lengths_query.ndim != 1:
        raise ValueError(
            "C8 MTP LIDU actual_seq_lengths_query must be cumulative int32 [B]"
        )
    batch = actual_seq_lengths_query.shape[0]
    if batch <= 0 or not (
        MIN_QUERIES_PER_REQUEST * batch
        <= packed_queries
        <= MAX_QUERIES_PER_REQUEST * batch
    ):
        raise ValueError("C8 MTP LIDU requires 2 <= Q_b <= 4 for every request")
    if (
        key.ndim != 4
        or key.shape[0] <= 0
        or tuple(key.shape[1:]) != (BLOCK_SIZE, 1, 128)
    ):
        raise ValueError("C8 MTP LIDU key must be [blocks,128,1,128]")
    fp8 = getattr(torch, "float8_e4m3fn", None)
    if fp8 is None or query.dtype != fp8 or key.dtype != fp8:
        raise TypeError("A5 C8 MTP LIDU query/key must be float8_e4m3fn")
    if (
        weights.shape != (packed_queries, heads)
        or weights.dtype != torch.bfloat16
        or query_dequant_scale.shape != (packed_queries, heads)
        or query_dequant_scale.dtype != torch.float32
    ):
        raise ValueError(
            "C8 MTP LIDU weights must be bf16 [T,N] and "
            "query_dequant_scale must be fp32 [T,N]"
        )
    if (
        key_dequant_scale.shape != key.shape[:-1]
        or key_dequant_scale.dtype != torch.float32
    ):
        raise ValueError(
            "C8 MTP LIDU key_dequant_scale must be fp32 [blocks,128,1]"
        )
    if (
        req_pool_entries.shape != (batch,)
        or cache_tokens.shape != (batch,)
        or candidate_lens.shape != (batch,)
        or block_table.ndim != 2
        or block_table.shape[0] != batch
        or cache_slots_pool.ndim != 2
        or cache_slots_pool.shape[0] <= 0
        or block_table.shape[1] * BLOCK_SIZE
        != cache_slots_pool.shape[1]
    ):
        raise ValueError("C8 MTP LIDU metadata shapes are inconsistent")
    integer_tensors = (
        actual_seq_lengths_query,
        req_pool_entries,
        cache_slots_pool,
        cache_tokens,
        candidate_lens,
        block_table,
    )
    if any(tensor.dtype != torch.int32 for tensor in integer_tensors):
        raise TypeError("C8 MTP LIDU metadata and cache state must be int32")
    tensors = (
        key,
        weights,
        query_dequant_scale,
        key_dequant_scale,
        *integer_tensors,
    )
    if any(tensor.device != query.device for tensor in tensors):
        raise ValueError("C8 MTP LIDU tensors must be on one device")
    if not query.is_contiguous() or any(
        not tensor.is_contiguous() for tensor in tensors
    ):
        raise ValueError("C8 MTP LIDU tensors must be contiguous")


def _check_outputs(
    query: torch.Tensor,
    actual_seq_lengths_query: torch.Tensor,
    topk_destination_slots: torch.Tensor,
    miss_source_ids: torch.Tensor,
    miss_destination_slots: torch.Tensor,
    miss_counts: torch.Tensor,
) -> None:
    batch = actual_seq_lengths_query.shape[0]
    expected_topk = (query.shape[0], 1, SPARSE_COUNT)
    expected_miss = (batch, UNION_CAPACITY)
    if topk_destination_slots.shape != expected_topk:
        raise ValueError(
            f"topk_destination_slots must have shape {expected_topk}"
        )
    if (
        miss_source_ids.shape != expected_miss
        or miss_destination_slots.shape != expected_miss
        or miss_counts.shape != (batch,)
    ):
        raise ValueError(
            "C8 MTP LIDU miss buffers must be [B,8192] and [B]"
        )
    outputs = (
        topk_destination_slots,
        miss_source_ids,
        miss_destination_slots,
        miss_counts,
    )
    if any(
        tensor.dtype != torch.int32
        or tensor.device != query.device
        or not tensor.is_contiguous()
        for tensor in outputs
    ):
        raise TypeError(
            "C8 MTP LIDU outputs must be contiguous int32 tensors on the query device"
        )


def _impl(
    query: torch.Tensor,
    key: torch.Tensor,
    weights: torch.Tensor,
    query_dequant_scale: torch.Tensor,
    key_dequant_scale: torch.Tensor,
    actual_seq_lengths_query: torch.Tensor,
    req_pool_entries: torch.Tensor,
    cache_slots_pool: torch.Tensor,
    cache_tokens: torch.Tensor,
    candidate_lens: torch.Tensor,
    block_table: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    _check_inputs(
        query,
        key,
        weights,
        query_dequant_scale,
        key_dequant_scale,
        actual_seq_lengths_query,
        req_pool_entries,
        cache_slots_pool,
        cache_tokens,
        candidate_lens,
        block_table,
    )
    topk = _native_quant_lightning_indexer(
        query,
        key,
        weights,
        query_dequant_scale,
        key_dequant_scale,
        actual_seq_lengths_query,
        candidate_lens,
        block_table,
    )
    return _cache_update(
        topk,
        actual_seq_lengths_query,
        req_pool_entries,
        cache_slots_pool,
        cache_tokens,
        candidate_lens,
    )


def _out_impl(
    query: torch.Tensor,
    key: torch.Tensor,
    weights: torch.Tensor,
    query_dequant_scale: torch.Tensor,
    key_dequant_scale: torch.Tensor,
    actual_seq_lengths_query: torch.Tensor,
    req_pool_entries: torch.Tensor,
    cache_slots_pool: torch.Tensor,
    cache_tokens: torch.Tensor,
    candidate_lens: torch.Tensor,
    block_table: torch.Tensor,
    topk_destination_slots: torch.Tensor,
    miss_source_ids: torch.Tensor,
    miss_destination_slots: torch.Tensor,
    miss_counts: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    _check_inputs(
        query,
        key,
        weights,
        query_dequant_scale,
        key_dequant_scale,
        actual_seq_lengths_query,
        req_pool_entries,
        cache_slots_pool,
        cache_tokens,
        candidate_lens,
        block_table,
    )
    _check_outputs(
        query,
        actual_seq_lengths_query,
        topk_destination_slots,
        miss_source_ids,
        miss_destination_slots,
        miss_counts,
    )
    topk = _native_quant_lightning_indexer(
        query,
        key,
        weights,
        query_dequant_scale,
        key_dequant_scale,
        actual_seq_lengths_query,
        candidate_lens,
        block_table,
    )
    return _cache_update_out(
        topk,
        actual_seq_lengths_query,
        req_pool_entries,
        cache_slots_pool,
        cache_tokens,
        candidate_lens,
        topk_destination_slots,
        miss_source_ids,
        miss_destination_slots,
        miss_counts,
    )


def _meta(
    query: torch.Tensor,
    key: torch.Tensor,
    weights: torch.Tensor,
    query_dequant_scale: torch.Tensor,
    key_dequant_scale: torch.Tensor,
    actual_seq_lengths_query: torch.Tensor,
    req_pool_entries: torch.Tensor,
    cache_slots_pool: torch.Tensor,
    cache_tokens: torch.Tensor,
    candidate_lens: torch.Tensor,
    block_table: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    del (
        key,
        weights,
        query_dequant_scale,
        key_dequant_scale,
        req_pool_entries,
        cache_tokens,
        candidate_lens,
        block_table,
    )
    options = {"dtype": torch.int32, "device": query.device}
    batch = actual_seq_lengths_query.shape[0]
    return (
        torch.empty((query.shape[0], 1, SPARSE_COUNT), **options),
        torch.empty((batch, UNION_CAPACITY), **options),
        torch.empty((batch, UNION_CAPACITY), **options),
        torch.empty((batch,), **options),
        cache_slots_pool,
    )


def _out_meta(
    query: torch.Tensor,
    key: torch.Tensor,
    weights: torch.Tensor,
    query_dequant_scale: torch.Tensor,
    key_dequant_scale: torch.Tensor,
    actual_seq_lengths_query: torch.Tensor,
    req_pool_entries: torch.Tensor,
    cache_slots_pool: torch.Tensor,
    cache_tokens: torch.Tensor,
    candidate_lens: torch.Tensor,
    block_table: torch.Tensor,
    topk_destination_slots: torch.Tensor,
    miss_source_ids: torch.Tensor,
    miss_destination_slots: torch.Tensor,
    miss_counts: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    del (
        query,
        key,
        weights,
        query_dequant_scale,
        key_dequant_scale,
        actual_seq_lengths_query,
        req_pool_entries,
        cache_tokens,
        candidate_lens,
        block_table,
    )
    return (
        topk_destination_slots,
        miss_source_ids,
        miss_destination_slots,
        miss_counts,
        cache_slots_pool,
    )


_NPU_IMPL = torch.library.Library("nanovllm_dsa", "IMPL", "PrivateUse1")
_NPU_IMPL.impl("fused_li_manage_mtp_c8", _impl)
_NPU_IMPL.impl("fused_li_manage_mtp_c8_out", _out_impl)
_META_IMPL = torch.library.Library("nanovllm_dsa", "IMPL", "Meta")
_META_IMPL.impl("fused_li_manage_mtp_c8", _meta)
_META_IMPL.impl("fused_li_manage_mtp_c8_out", _out_meta)

fused_li_manage_mtp_c8 = torch.ops.nanovllm_dsa.fused_li_manage_mtp_c8
fused_li_manage_mtp_c8_out = torch.ops.nanovllm_dsa.fused_li_manage_mtp_c8_out
