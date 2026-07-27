from __future__ import annotations

import torch


LIDU_TOPK = 2048
_INIT_SCORE_CHUNK_TOKENS = 16 * 1024


def lidu_decode_update(
    query: torch.Tensor,
    key: torch.Tensor,
    weights: torch.Tensor,
    req_pool_entries: torch.Tensor,
    cache_slots_pool: torch.Tensor,
    cache_tokens: torch.Tensor,
    candidate_lens: torch.Tensor,
    block_table: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run the repository-bundled stable LIDU operator.

    The first ``miss_counts[b]`` source IDs and destination slots describe
    SCATTER copies.  The complete destination row contains the 2048 HBM slots
    selected for attention after the cache update.  The final return value
    aliases ``cache_slots_pool`` so graph compilers can see mutable state.
    """

    return torch.ops.nanovllm_dsa.lidu_decode_update.default(
        query,
        key,
        weights,
        req_pool_entries,
        cache_slots_pool,
        cache_tokens,
        candidate_lens,
        block_table,
    )


def lidu_decode_update_out(
    query: torch.Tensor,
    key: torch.Tensor,
    weights: torch.Tensor,
    req_pool_entries: torch.Tensor,
    cache_slots_pool: torch.Tensor,
    cache_tokens: torch.Tensor,
    candidate_lens: torch.Tensor,
    block_table: torch.Tensor,
    source_ids: torch.Tensor,
    destination_slots: torch.Tensor,
    miss_counts: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run LIDU into graph-stable, caller-owned output buffers."""

    return torch.ops.nanovllm_dsa.lidu_decode_update_out.default(
        query,
        key,
        weights,
        req_pool_entries,
        cache_slots_pool,
        cache_tokens,
        candidate_lens,
        block_table,
        source_ids,
        destination_slots,
        miss_counts,
    )


def scatter_copy(
    hbm_kpe: torch.Tensor,
    hbm_ckv: torch.Tensor,
    dram_kpe: torch.Tensor,
    dram_ckv: torch.Tensor,
    hbm_block_table: torch.Tensor,
    dram_block_table: torch.Tensor,
    source_token_ids: torch.Tensor,
    destination_slots: torch.Tensor,
    copy_counts: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Copy selected CKV/KPE tokens with the bundled SCATTER operator."""

    return torch.ops.nanovllm_dsa.scatter_copy.default(
        hbm_kpe,
        hbm_ckv,
        dram_kpe,
        dram_ckv,
        hbm_block_table,
        dram_block_table,
        source_token_ids,
        destination_slots,
        copy_counts,
    )


def sparse_and_tail_attention(
    query: torch.Tensor,
    latent_kv_cache: torch.Tensor,
    sparse_slots: torch.Tensor,
    cache_tokens: torch.Tensor,
    block_table: torch.Tensor,
    actual_seq_lengths_query: torch.Tensor,
    actual_seq_lengths_kv: torch.Tensor,
    query_rope: torch.Tensor,
    key_rope: torch.Tensor,
    scale_value: float,
) -> torch.Tensor:
    """Attend to cached top-2048 slots plus the complete dense tail."""

    return torch.ops.nanovllm_dsa.sparse_and_tail_attention.default(
        query,
        latent_kv_cache,
        latent_kv_cache,
        sparse_slots,
        cache_tokens,
        block_table,
        actual_seq_lengths_query,
        actual_seq_lengths_kv,
        query_rope,
        key_rope,
        float(scale_value),
    )


def sparse_and_tail_attention_and_scatter_copy(
    query: torch.Tensor,
    hbm_ckv: torch.Tensor,
    sparse_slots: torch.Tensor,
    cache_tokens: torch.Tensor,
    hbm_block_table: torch.Tensor,
    actual_seq_lengths_query: torch.Tensor,
    actual_seq_lengths_kv: torch.Tensor,
    query_rope: torch.Tensor,
    hbm_kpe: torch.Tensor,
    dram_kpe: torch.Tensor,
    dram_ckv: torch.Tensor,
    dram_block_table: torch.Tensor,
    source_token_ids: torch.Tensor,
    copy_counts: torch.Tensor,
    scale_value: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Copy LIDU misses and attend to top-2048 plus tail in one operator."""

    return (
        torch.ops.nanovllm_dsa
        .sparse_and_tail_attention_and_scatter_copy.default(
            query,
            hbm_ckv,
            sparse_slots,
            cache_tokens,
            hbm_block_table,
            actual_seq_lengths_query,
            actual_seq_lengths_kv,
            query_rope,
            hbm_kpe,
            dram_kpe,
            dram_ckv,
            dram_block_table,
            source_token_ids,
            copy_counts,
            float(scale_value),
        )
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
    return scatter_copy(
        hbm_kpe,
        hbm_ckv,
        dram_kpe,
        dram_ckv,
        hbm_block_table.unsqueeze(0),
        dram_block_table.unsqueeze(0),
        source_ids.unsqueeze(0),
        destination_slots.unsqueeze(0),
        copy_counts,
    )
