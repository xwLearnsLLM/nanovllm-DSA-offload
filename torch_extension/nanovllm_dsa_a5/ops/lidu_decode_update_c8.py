from __future__ import annotations

import torch
import torch_npu

from ._constants import BLOCK_SIZE, SPARSE_COUNT


lidu_cache_update = torch.ops.nanovllm_dsa.lidu_cache_update
lidu_cache_update_out = torch.ops.nanovllm_dsa.lidu_cache_update_out


def _check_c8_lidu_inputs(
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
        raise ValueError("C8 LIDU query must be [B,32|64,128]")
    batch, heads, _ = query.shape
    if (
        key.ndim != 4
        or key.shape[0] <= 0
        or tuple(key.shape[1:]) != (BLOCK_SIZE, 1, 128)
    ):
        raise ValueError("C8 LIDU key must be [blocks,128,1,128]")
    fp8 = getattr(torch, "float8_e4m3fn", None)
    if fp8 is None or query.dtype != fp8 or key.dtype != fp8:
        raise TypeError("A5 C8 LIDU query/key must be float8_e4m3fn")
    if (
        weights.shape != (batch, heads)
        or query_dequant_scale.shape != (batch, heads)
        or weights.dtype != torch.bfloat16
        or query_dequant_scale.dtype != torch.float32
    ):
        raise ValueError(
            "C8 LIDU weights must be bfloat16 [B,N] and "
            "query_dequant_scale must be float32 [B,N]"
        )
    if (
        key_dequant_scale.ndim != 3
        or key_dequant_scale.shape != key.shape[:-1]
        or key_dequant_scale.dtype != torch.float32
    ):
        raise ValueError(
            "C8 LIDU key_dequant_scale must be float32 [blocks,128,1]"
        )
    if (
        actual_seq_lengths_query.shape != (batch,)
        or req_pool_entries.shape != (batch,)
        or cache_tokens.shape != (batch,)
        or candidate_lens.shape != (batch,)
        or block_table.ndim != 2
        or block_table.shape[0] != batch
        or cache_slots_pool.ndim != 2
        or cache_slots_pool.shape[0] <= 0
        or block_table.shape[1] * BLOCK_SIZE
        != cache_slots_pool.shape[1]
    ):
        raise ValueError("C8 LIDU metadata shapes are inconsistent")
    for name, tensor in (
        ("actual_seq_lengths_query", actual_seq_lengths_query),
        ("req_pool_entries", req_pool_entries),
        ("cache_slots_pool", cache_slots_pool),
        ("cache_tokens", cache_tokens),
        ("candidate_lens", candidate_lens),
        ("block_table", block_table),
    ):
        if tensor.dtype != torch.int32:
            raise TypeError(f"C8 LIDU {name} must be int32")
    tensors = (
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
    if any(tensor.device != query.device for tensor in tensors):
        raise ValueError("C8 LIDU tensors must be on one device")
    if not query.is_contiguous() or any(
        not tensor.is_contiguous() for tensor in tensors
    ):
        raise ValueError("C8 LIDU tensors must be contiguous")


def _native_quant_lightning_indexer(
    query: torch.Tensor,
    key: torch.Tensor,
    weights: torch.Tensor,
    query_dequant_scale: torch.Tensor,
    key_dequant_scale: torch.Tensor,
    actual_seq_lengths_query: torch.Tensor,
    candidate_lens: torch.Tensor,
    block_table: torch.Tensor,
) -> torch.Tensor:
    # Exact A5 C8 call used by vLLM-Ascend's A5DeviceAdaptor. The caller owns
    # GLM preprocessing: RoPE, normalized 128x128 Hadamard, then FP8 E4M3
    # dynamic quantization and its FP32 dequant scales.
    op = getattr(torch_npu, "npu_quant_lightning_indexer", None)
    if op is None:
        namespace = getattr(torch.ops, "_C_ascend", None)
        op = (
            getattr(namespace, "npu_lightning_indexer_quant", None)
            if namespace is not None
            else None
        )
    if op is None:
        raise RuntimeError("A5 Quant LightningIndexer is not registered")
    result = op(
        query=query,
        key=key,
        weights=weights,
        query_dequant_scale=query_dequant_scale,
        key_dequant_scale=key_dequant_scale,
        actual_seq_lengths_query=actual_seq_lengths_query,
        actual_seq_lengths_key=candidate_lens,
        block_table=block_table,
        query_quant_mode=0,
        key_quant_mode=0,
        layout_query="TND",
        layout_key="PA_BSND",
        sparse_count=SPARSE_COUNT,
        sparse_mode=3,
    )
    topk = result[0] if isinstance(result, tuple) else result
    if not isinstance(topk, torch.Tensor) or topk.dtype != torch.int32:
        raise RuntimeError(
            "official A5 C8 LightningIndexer returned no int32 top-k tensor"
        )
    if topk.numel() != query.shape[0] * SPARSE_COUNT:
        raise RuntimeError(
            "official A5 C8 LightningIndexer returned an unexpected shape: "
            f"{tuple(topk.shape)}"
        )
    return topk.reshape(query.shape[0], 1, SPARSE_COUNT).contiguous()


def _lidu_decode_update_c8_impl(
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
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    _check_c8_lidu_inputs(
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
    return lidu_cache_update(
        topk,
        req_pool_entries,
        cache_slots_pool,
        cache_tokens,
        candidate_lens,
    )


def _lidu_decode_update_c8_out_impl(
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
    source_ids: torch.Tensor,
    destination_slots: torch.Tensor,
    miss_counts: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    _check_c8_lidu_inputs(
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
    return lidu_cache_update_out(
        topk,
        req_pool_entries,
        cache_slots_pool,
        cache_tokens,
        candidate_lens,
        source_ids,
        destination_slots,
        miss_counts,
    )


def _lidu_decode_update_c8_meta(
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
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    del (
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
    options = {"dtype": torch.int32, "device": query.device}
    return (
        torch.empty((query.shape[0], 1, SPARSE_COUNT), **options),
        torch.empty((query.shape[0], 1, SPARSE_COUNT), **options),
        torch.empty((query.shape[0],), **options),
        cache_slots_pool,
    )


def _lidu_decode_update_c8_out_meta(
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
    source_ids: torch.Tensor,
    destination_slots: torch.Tensor,
    miss_counts: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
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
    return source_ids, destination_slots, miss_counts, cache_slots_pool


_C8_LIDU_NPU_IMPL = torch.library.Library(
    "nanovllm_dsa", "IMPL", "PrivateUse1"
)
_C8_LIDU_NPU_IMPL.impl(
    "lidu_decode_update_c8", _lidu_decode_update_c8_impl
)
_C8_LIDU_NPU_IMPL.impl(
    "lidu_decode_update_c8_out", _lidu_decode_update_c8_out_impl
)
_C8_LIDU_META_IMPL = torch.library.Library("nanovllm_dsa", "IMPL", "Meta")
_C8_LIDU_META_IMPL.impl(
    "lidu_decode_update_c8", _lidu_decode_update_c8_meta
)
_C8_LIDU_META_IMPL.impl(
    "lidu_decode_update_c8_out", _lidu_decode_update_c8_out_meta
)

lidu_decode_update_c8 = torch.ops.nanovllm_dsa.lidu_decode_update_c8
lidu_decode_update_c8_out = torch.ops.nanovllm_dsa.lidu_decode_update_c8_out
