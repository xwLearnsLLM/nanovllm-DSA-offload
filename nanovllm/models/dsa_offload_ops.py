from __future__ import annotations

import os

import torch

import nanovllm.ops as ascend_ops

_DSA_INDEX_UPDATE_CANN_MAX_K = 128


def _env_flag(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() not in ("0", "false", "off", "no", "")


_DSA_QK_SCORE_BF16_OUT = _env_flag("NANOVLLM_DSA_QK_SCORE_BF16_OUT", True)
_DSA_INDEX_UPDATE_USE_CANN = _env_flag("NANOVLLM_DSA_INDEX_UPDATE_USE_CANN", True)


def dsa_indexer_score(
    query_index: torch.Tensor,
    index_cache: torch.Tensor,
    index_weights: torch.Tensor,
    index_block_table: torch.Tensor,
    candidate_lens: torch.Tensor,
    score_out: torch.Tensor,
    *,
    actual_seq_lengths_query: torch.Tensor | None,
    block_count: int | None = None,
) -> None:
    """Score DSA candidates with the Ascend qk_score op."""
    block_size = int(index_cache.shape[1])
    score_capacity = int(score_out.shape[1])
    if score_capacity <= 0:
        return

    block_count = int(block_count) if block_count is not None else (score_capacity + block_size - 1) // block_size
    score_count = block_count * block_size
    if score_count > score_capacity:
        raise RuntimeError(f"score_out capacity {score_capacity} is smaller than qk_score logical length {score_count}.")
    
    if _DSA_QK_SCORE_BF16_OUT:
        ascend_ops.npu_qk_score_bf16_out(
            query_index,
            index_cache,
            index_weights,
            actual_seq_lengths_query,
            candidate_lens,
            index_block_table,
            block_count,
            score_out,
            "TND",
            "PA_BSND",
        )
    else: 
        scores = ascend_ops.npu_qk_score(
            query_index,
            index_cache,
            index_weights,
            actual_seq_lengths_query,
            candidate_lens,
            index_block_table[:, :block_count].contiguous(),
            "TND",
            "PA_BSND",
        )
        copy_len = min(score_count, int(scores.shape[-1]))
        score_out[:, :copy_len].copy_(scores[:, 0, :copy_len])


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


def _dsa_index_update_cann(
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
    """Run the CANN dsa_update_index op behind the framework-facing interface."""
    promote_idx.zero_()
    demote_idx.zero_()
    copy_counts.zero_()

    bs = int(score.shape[0])
    k = int(max_copy_tokens)
    if bs <= 0 or k <= 0:
        return
    if k > _DSA_INDEX_UPDATE_CANN_MAX_K:
        raise RuntimeError(
            "CANN dsa_update_index supports at most "
            f"{_DSA_INDEX_UPDATE_CANN_MAX_K} copy tokens, got {k}. "
            "Set NANOVLLM_DSA_INDEX_UPDATE_USE_CANN=0 to use the torch path, "
            "or lower NANOVLLM_DSA_OFFLOAD_FIXED_TX."
        )
    if not hasattr(ascend_ops, "dsa_update_index"):
        raise RuntimeError(
            "ascend_ops.dsa_update_index is unavailable. Rebuild with "
            "`SOC_VERSION=ascend910_9391 PYTHONPATH=$PWD:$PYTHONPATH bash scripts/build_nanovllm_ops.sh`."
        )

    pool_entries = req_pool_entries.to(torch.long)
    selected_idx = hbm_cached_tokens_pool[pool_entries].contiguous()
    score_contig = score.contiguous()
    candidate_lens_contig = candidate_lens.contiguous()
    selected_lens_contig = selected_lens.contiguous()
    promote_k = promote_idx[:, :k].contiguous()
    demote_k = demote_idx[:, :k].contiguous()

    # copy_counts must keep the torch prototype semantics. This matters for
    # mixed short/long batches: a short sequence may already cache all candidate
    # tokens, in which case the CANN op may still emit padded pairs but scatter
    # must copy zero tokens for that row.
    max_selected = int(selected_idx.shape[1])
    col_range = torch.arange(max_selected, device=score.device).unsqueeze(0)
    within_len = col_range < selected_lens_contig.unsqueeze(1)
    valid_id = (selected_idx >= 0) & (selected_idx < candidate_lens_contig.unsqueeze(1))
    selected_valid = within_len & valid_id
    actual_selected = selected_valid.sum(dim=1).to(candidate_lens_contig.dtype)
    available = candidate_lens_contig - actual_selected
    copy_count = torch.clamp(
        torch.minimum(
            torch.full_like(selected_lens_contig, k),
            torch.minimum(selected_lens_contig, available),
        ),
        min=0,
    )
    copy_counts.copy_(copy_count)

    ascend_ops.dsa_update_index(
        score_contig,
        selected_idx,
        candidate_lens_contig,
        selected_lens_contig,
        k,
        promote_k,
        demote_k,
    )

    if not score.is_contiguous():
        score.copy_(score_contig)
    promote_idx[:, :k].copy_(promote_k)
    demote_idx[:, :k].copy_(demote_k)

    k_range = torch.arange(k, device=score.device).unsqueeze(0)
    valid_k = k_range < copy_count.unsqueeze(1)
    flat_pool_rows = pool_entries.unsqueeze(1).expand(-1, k)[valid_k]
    flat_slots = demote_idx[:, :k][valid_k].to(torch.long)
    flat_vals = promote_idx[:, :k][valid_k].to(torch.int32)
    hbm_cached_tokens_pool[flat_pool_rows, flat_slots] = flat_vals


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
    """Update sparse HBM budget.

    By default this uses the CANN op. Set
    NANOVLLM_DSA_INDEX_UPDATE_USE_CANN=0 to force the PyTorch prototype.
    """
    if _DSA_INDEX_UPDATE_USE_CANN:
        _dsa_index_update_cann(
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
        return

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
