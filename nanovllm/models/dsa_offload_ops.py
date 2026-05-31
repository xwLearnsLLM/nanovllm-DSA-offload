from __future__ import annotations

import torch

import nanovllm.ops as ascend_ops


def dsa_indexer_score(
    query_index: torch.Tensor,
    index_cache: torch.Tensor,
    index_weights: torch.Tensor,
    index_block_table: torch.Tensor,
    candidate_lens: torch.Tensor,
    score_out: torch.Tensor,
    *,
    actual_seq_lengths_query: torch.Tensor | None,
) -> None:
    """Score DSA candidates with the Ascend qk_score op."""
    score_out.fill_(-float("inf"))
    block_size = int(index_cache.shape[1])
    score_capacity = int(score_out.shape[1])
    if score_capacity <= 0:
        return

    block_count = (score_capacity + block_size - 1) // block_size
    scores = ascend_ops.npu_qk_score(
        query_index.contiguous(),
        index_cache,
        index_weights.contiguous(),
        actual_seq_lengths_query.contiguous(),
        candidate_lens.contiguous(),
        index_block_table[:, :block_count].contiguous(),
        "TND",
        "PA_BSND",
    )
    copy_len = min(score_capacity, int(scores.shape[-1]))
    score_out[:, :copy_len].copy_(scores[:, 0, :copy_len].to(score_out.dtype))


def dsa_index_update_torch(
    score: torch.Tensor,
    hbm_cached_tokens_pool: torch.Tensor,
    promote_idx: torch.Tensor,
    demote_idx: torch.Tensor,
    copy_counts: torch.Tensor,
    candidate_lens: torch.Tensor,
    selected_lens: torch.Tensor,
    req_pool_entries: torch.Tensor,
    max_copy_tokens: int,
) -> None:
    """PyTorch prototype for fixed-Tx sparse budget update."""
    promote_idx.zero_()
    demote_idx.zero_()
    copy_counts.zero_()
    bs = int(score.shape[0])
    for b in range(bs):
        candidate_len = int(candidate_lens[b].item())
        selected_len = int(selected_lens[b].item())
        pool_entry = int(req_pool_entries[b].item())
        if candidate_len <= 0 or selected_len <= 0 or max_copy_tokens <= 0:
            continue

        selected_tokens = hbm_cached_tokens_pool[
            pool_entry,
            :selected_len,
        ].to(torch.long)
        valid_selected = selected_tokens[
            (selected_tokens >= 0) & (selected_tokens < candidate_len)
        ]
        uncached_mask = torch.ones(
            candidate_len,
            dtype=torch.bool,
            device=score.device,
        )
        if valid_selected.numel() > 0:
            uncached_mask[valid_selected] = False
        available_uncached = int(uncached_mask.sum().item())
        copy_count = min(int(max_copy_tokens), selected_len, available_uncached)
        if copy_count <= 0:
            continue

        promote_scores = score[b, :candidate_len].float().clone()
        promote_scores[~uncached_mask] = -float("inf")
        promote_tokens = torch.topk(
            promote_scores,
            k=copy_count,
            largest=True,
        ).indices.to(torch.int32)

        safe_selected_tokens = selected_tokens.clamp(0, candidate_len - 1)
        selected_scores = score[b].index_select(0, safe_selected_tokens).float()
        selected_scores[
            (selected_tokens < 0) | (selected_tokens >= candidate_len)
        ] = -float("inf")
        demote_slots = torch.topk(
            selected_scores,
            k=copy_count,
            largest=False,
        ).indices.to(torch.int32)

        promote_idx[b, :copy_count] = promote_tokens
        demote_idx[b, :copy_count] = demote_slots
        copy_counts[b] = copy_count
        hbm_cached_tokens_pool[
            pool_entry,
            demote_slots.to(torch.long),
        ] = promote_tokens


def dsa_index_update(
    score: torch.Tensor,
    hbm_cached_tokens_pool: torch.Tensor,
    promote_idx: torch.Tensor,
    demote_idx: torch.Tensor,
    copy_counts: torch.Tensor,
    candidate_lens: torch.Tensor,
    selected_lens: torch.Tensor,
    req_pool_entries: torch.Tensor,
    max_copy_tokens: int,
) -> None:
    """Update sparse HBM budget with the PyTorch prototype.

    Keep this function as the stable framework-facing interface. The current
    implementation is intentionally the PyTorch prototype.
    """
    dsa_index_update_torch(
        score,
        hbm_cached_tokens_pool,
        promote_idx,
        demote_idx,
        copy_counts,
        candidate_lens,
        selected_lens,
        req_pool_entries,
        int(max_copy_tokens),
    )


def dsa_scatter_h2d(
    promote_idx: torch.Tensor,
    demote_idx: torch.Tensor,
    copy_counts: torch.Tensor,
    hbm_block_table: torch.Tensor,
    dram_block_table: torch.Tensor,
    hbm_ckv_cache: torch.Tensor,
    hbm_kpe_cache: torch.Tensor,
    dram_ckv_cache: torch.Tensor,
    dram_kpe_cache: torch.Tensor,
) -> None:
    block_size = int(hbm_ckv_cache.shape[1])
    ascend_ops.paged_scatter_copy_h2d(
        hbm_kpe_cache,
        hbm_ckv_cache,
        dram_kpe_cache,
        dram_ckv_cache,
        hbm_block_table,
        dram_block_table,
        demote_idx,
        promote_idx,
        copy_counts,
        block_size,
    )
