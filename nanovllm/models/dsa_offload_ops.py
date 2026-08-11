from __future__ import annotations

import torch


LIDU_TOPK = 2048
LIDU_MTP_QUERY_COUNT = 4
LIDU_MTP_UNION_CAPACITY = LIDU_TOPK * LIDU_MTP_QUERY_COUNT
_INIT_SCORE_CHUNK_TOKENS = 16 * 1024


def fused_li_manage(
    query: torch.Tensor,
    index_weights: torch.Tensor,
    index_key_cache: torch.Tensor,
    index_block_table: torch.Tensor,
    num_candidate_tokens: torch.Tensor,
    num_cache_tokens: torch.Tensor,
    req_pool_entries: torch.Tensor,
    cache_slots_pool: torch.Tensor,
    topk_src_ids: torch.Tensor,
    topk_dst_slots: torch.Tensor,
    miss_counts: torch.Tensor,
) -> None:
    """Run single-query LIM into caller-owned buffers."""

    torch.ops.nanovllm_dsa.fused_li_manage.default(
        query,
        index_weights,
        index_key_cache,
        index_block_table,
        num_candidate_tokens,
        num_cache_tokens,
        req_pool_entries,
        cache_slots_pool,
        topk_src_ids,
        topk_dst_slots,
        miss_counts,
    )


def fused_li_manage_mtp(
    query: torch.Tensor,
    index_weights: torch.Tensor,
    index_key_cache: torch.Tensor,
    index_block_table: torch.Tensor,
    num_candidate_tokens: torch.Tensor,
    num_cache_tokens: torch.Tensor,
    req_pool_entries: torch.Tensor,
    cache_slots_pool: torch.Tensor,
    topk_src_ids: torch.Tensor,
    topk_dst_slots: torch.Tensor,
    miss_src_ids: torch.Tensor,
    miss_dst_slots: torch.Tensor,
    miss_counts: torch.Tensor,
) -> None:
    """Run fixed-width GLM MTP3 LIM into caller-owned buffers."""

    torch.ops.nanovllm_dsa.fused_li_manage_mtp.default(
        query,
        index_weights,
        index_key_cache,
        index_block_table,
        num_candidate_tokens,
        num_cache_tokens,
        req_pool_entries,
        cache_slots_pool,
        topk_src_ids,
        topk_dst_slots,
        miss_src_ids,
        miss_dst_slots,
        miss_counts,
    )


def scatter_copy(
    src_ids: torch.Tensor,
    dst_slots: torch.Tensor,
    copy_counts: torch.Tensor,
    hbm_block_table: torch.Tensor,
    dram_block_table: torch.Tensor,
    hbm_k_rope: torch.Tensor,
    hbm_kv_cache: torch.Tensor,
    dram_k_rope: torch.Tensor,
    dram_kv_cache: torch.Tensor,
) -> None:
    """Copy selected CKV/KPE tokens with the bundled SCATTER operator."""

    torch.ops.nanovllm_dsa.scatter_copy.default(
        src_ids,
        dst_slots,
        copy_counts,
        hbm_block_table,
        dram_block_table,
        hbm_k_rope,
        hbm_kv_cache,
        dram_k_rope,
        dram_kv_cache,
    )


def sparse_tail_attention(
    query_rope: torch.Tensor,
    query: torch.Tensor,
    actual_seq_lengths_query: torch.Tensor,
    actual_seq_lengths_kv: torch.Tensor,
    num_cache_tokens: torch.Tensor,
    topk_dst_slots: torch.Tensor,
    hbm_block_table: torch.Tensor,
    hbm_k_rope: torch.Tensor,
    hbm_kv_cache: torch.Tensor,
    scale_value: float,
    attention_out: torch.Tensor,
) -> None:
    """Attend to cached top-2048 slots plus the complete dense tail."""

    torch.ops.nanovllm_dsa.sparse_tail_attention.default(
        query_rope,
        query,
        actual_seq_lengths_query,
        actual_seq_lengths_kv,
        num_cache_tokens,
        topk_dst_slots,
        hbm_block_table,
        hbm_k_rope,
        hbm_kv_cache,
        float(scale_value),
        attention_out,
    )


def sparse_tail_attention_mtp(
    query_rope: torch.Tensor,
    query: torch.Tensor,
    actual_seq_lengths_query: torch.Tensor,
    actual_seq_lengths_kv: torch.Tensor,
    num_cache_tokens: torch.Tensor,
    topk_dst_slots: torch.Tensor,
    hbm_block_table: torch.Tensor,
    hbm_k_rope: torch.Tensor,
    hbm_kv_cache: torch.Tensor,
    scale_value: float,
    attention_out: torch.Tensor,
) -> None:
    """MTP3 Attention over each query's top-2048 plus its causal dense tail."""

    torch.ops.nanovllm_dsa.sparse_tail_attention_mtp.default(
        query_rope,
        query,
        actual_seq_lengths_query,
        actual_seq_lengths_kv,
        num_cache_tokens,
        topk_dst_slots,
        hbm_block_table,
        hbm_k_rope,
        hbm_kv_cache,
        float(scale_value),
        attention_out,
    )


def fused_copy_sfa(
    query_rope: torch.Tensor,
    query: torch.Tensor,
    actual_seq_lengths_query: torch.Tensor,
    actual_seq_lengths_kv: torch.Tensor,
    num_cache_tokens: torch.Tensor,
    topk_dst_slots: torch.Tensor,
    topk_src_ids: torch.Tensor,
    miss_counts: torch.Tensor,
    hbm_block_table: torch.Tensor,
    dram_block_table: torch.Tensor,
    hbm_k_rope: torch.Tensor,
    hbm_kv_cache: torch.Tensor,
    dram_k_rope: torch.Tensor,
    dram_kv_cache: torch.Tensor,
    scale_value: float,
    attention_out: torch.Tensor,
) -> None:
    """Copy LIM misses and attend to top-2048 plus tail in one operator."""

    torch.ops.nanovllm_dsa.fused_copy_sfa.default(
        query_rope,
        query,
        actual_seq_lengths_query,
        actual_seq_lengths_kv,
        num_cache_tokens,
        topk_dst_slots,
        topk_src_ids,
        miss_counts,
        hbm_block_table,
        dram_block_table,
        hbm_k_rope,
        hbm_kv_cache,
        dram_k_rope,
        dram_kv_cache,
        float(scale_value),
        attention_out,
    )


def fused_copy_sfa_mtp(
    query_rope: torch.Tensor,
    query: torch.Tensor,
    actual_seq_lengths_query: torch.Tensor,
    actual_seq_lengths_kv: torch.Tensor,
    num_cache_tokens: torch.Tensor,
    topk_dst_slots: torch.Tensor,
    topk_src_ids: torch.Tensor,
    miss_src_ids: torch.Tensor,
    miss_dst_slots: torch.Tensor,
    miss_counts: torch.Tensor,
    hbm_block_table: torch.Tensor,
    dram_block_table: torch.Tensor,
    hbm_k_rope: torch.Tensor,
    hbm_kv_cache: torch.Tensor,
    dram_k_rope: torch.Tensor,
    dram_kv_cache: torch.Tensor,
    scale_value: float,
    attention_out: torch.Tensor,
) -> None:
    """Gather MTP3 misses and write sparse Attention to ``attention_out``.

    The cache tensors are mutated in place.  They are intentionally not
    returned as alias outputs; the caller owns both cache storage and the
    fixed output buffer used by eager execution and full-decode-only graphs.
    """

    torch.ops.nanovllm_dsa.fused_copy_sfa_mtp.default(
        query_rope,
        query,
        actual_seq_lengths_query,
        actual_seq_lengths_kv,
        num_cache_tokens,
        topk_dst_slots,
        topk_src_ids,
        miss_src_ids,
        miss_dst_slots,
        miss_counts,
        hbm_block_table,
        dram_block_table,
        hbm_k_rope,
        hbm_kv_cache,
        dram_k_rope,
        dram_kv_cache,
        float(scale_value),
        attention_out,
    )


@torch.inference_mode()
def initialize_lidu_row(
    *,
    query: torch.Tensor,
    weights: torch.Tensor,
    index_cache: torch.Tensor,
    index_block_table: torch.Tensor,
    candidate_len: int,
    cache_tokens: int,
    cache_slots_row: torch.Tensor,
    hbm_kpe: torch.Tensor,
    hbm_ckv: torch.Tensor,
    dram_kpe: torch.Tensor,
    dram_ckv: torch.Tensor,
    hbm_block_table: torch.Tensor,
    dram_block_table: torch.Tensor,
    block_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Slow-path top-C initialization for one request and one layer.

    This runs only on the first decode.  Keeping it in ordinary PyTorch makes
    the stable custom operator smaller and fixes its output capacity at 2048.
    """

    candidate_len = int(candidate_len)
    cache_tokens = int(cache_tokens)
    if cache_tokens <= 0:
        return hbm_kpe, hbm_ckv
    if cache_tokens < LIDU_TOPK:
        raise ValueError(
            f"LIDU nonzero cache budget must be >= {LIDU_TOPK}, got "
            f"{cache_tokens}."
        )
    if candidate_len < cache_tokens:
        raise ValueError(
            f"LIDU cache budget {cache_tokens} exceeds source length "
            f"{candidate_len}."
        )
    if candidate_len % int(block_size):
        raise ValueError("LIDU source must consist only of complete blocks.")

    query = query.reshape(query.shape[-2], query.shape[-1]).float()
    weights = weights.reshape(-1).float()
    score_chunk_tokens = max(
        int(block_size),
        (_INIT_SCORE_CHUNK_TOKENS // int(block_size)) * int(block_size),
    )
    best_scores = None
    best_source_ids = None
    for token_start in range(0, candidate_len, score_chunk_tokens):
        token_end = min(candidate_len, token_start + score_chunk_tokens)
        block_start = token_start // int(block_size)
        block_end = token_end // int(block_size)
        physical_blocks = index_block_table[block_start:block_end].to(
            torch.long
        )
        keys = index_cache.index_select(0, physical_blocks).reshape(
            token_end - token_start, -1
        )
        scores = torch.matmul(query, keys.float().transpose(0, 1))
        scores = torch.sum(scores * weights.unsqueeze(1), dim=0)
        source_ids = torch.arange(
            token_start,
            token_end,
            dtype=torch.int64,
            device=scores.device,
        )
        if best_scores is not None:
            scores = torch.cat((best_scores, scores))
            source_ids = torch.cat((best_source_ids, source_ids))
        keep = min(cache_tokens, int(scores.numel()))
        best_scores, keep_indices = torch.topk(
            scores,
            k=keep,
            dim=0,
            largest=True,
            sorted=False,
        )
        best_source_ids = source_ids.index_select(0, keep_indices)

    if best_source_ids is None or int(best_source_ids.numel()) != cache_tokens:
        raise RuntimeError("LIDU top-C initialization produced an invalid result.")
    source_ids = best_source_ids.to(torch.int32)
    destination_slots = torch.arange(
        cache_tokens,
        dtype=torch.int32,
        device=source_ids.device,
    )

    cache_slots_row.fill_(-1)
    cache_slots_row.index_copy_(
        0,
        source_ids.to(torch.long),
        destination_slots,
    )
    copy_counts = torch.tensor(
        [cache_tokens], dtype=torch.int32, device=source_ids.device
    )
    scatter_copy(
        source_ids.unsqueeze(0),
        destination_slots.unsqueeze(0),
        copy_counts,
        hbm_block_table.unsqueeze(0),
        dram_block_table.unsqueeze(0),
        hbm_kpe,
        hbm_ckv,
        dram_kpe,
        dram_ckv,
    )
    return hbm_kpe, hbm_ckv


@torch.inference_mode()
def initialize_lidu_row_shared(
    *,
    cache_slots_row: torch.Tensor,
    cache_tokens: int,
    hbm_kpe: torch.Tensor,
    hbm_ckv: torch.Tensor,
    dram_kpe: torch.Tensor,
    dram_ckv: torch.Tensor,
    hbm_block_table: torch.Tensor,
    dram_block_table: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Copy DRAM KV to HBM for a shared IndexShare layer.

    The owner (full) layer has already run :func:`initialize_lidu_row`
    and filled ``cache_slots_row`` with the source-to-destination mapping.
    This function extracts that mapping and copies this layer's own DRAM
    KV payload to the same HBM destination slots.
    """

    cache_tokens = int(cache_tokens)
    if cache_tokens <= 0:
        return hbm_kpe, hbm_ckv

    valid_mask = cache_slots_row >= 0
    source_ids = torch.nonzero(valid_mask, as_tuple=True)[0].to(torch.int32)
    destination_slots = cache_slots_row[source_ids.long()].to(torch.int32)

    if source_ids.numel() != cache_tokens:
        raise RuntimeError(
            "Shared layer LIDU initialization found "
            f"{source_ids.numel()} cached tokens, expected {cache_tokens}."
        )

    copy_counts = torch.tensor(
        [cache_tokens], dtype=torch.int32, device=source_ids.device
    )
    scatter_copy(
        source_ids.unsqueeze(0),
        destination_slots.unsqueeze(0),
        copy_counts,
        hbm_block_table.unsqueeze(0),
        dram_block_table.unsqueeze(0),
        hbm_kpe,
        hbm_ckv,
        dram_kpe,
        dram_ckv,
    )
    return hbm_kpe, hbm_ckv
