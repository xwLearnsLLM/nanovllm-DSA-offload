from __future__ import annotations

import os

import torch

from nanovllm.models.dsa_index_update_real import (
    dsa_index_update_real,
    is_available as is_dsa_index_update_real_available,
)


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _flatten_index_cache(index_cache: torch.Tensor) -> torch.Tensor:
    return index_cache.view(-1, index_cache.shape[-1])


def _flatten_kv_cache(cache: torch.Tensor) -> torch.Tensor:
    return cache.view(-1, cache.shape[-1])


def dsa_indexer_score(
    query_index: torch.Tensor,
    index_cache: torch.Tensor,
    index_weights: torch.Tensor,
    index_block_table: torch.Tensor,
    candidate_lens: torch.Tensor,
    score_out: torch.Tensor,
) -> None:
    """PyTorch prototype for the full-candidate DSA score operator."""
    score_out.fill_(-float("inf"))
    block_size = int(index_cache.shape[1])
    flat_index = _flatten_index_cache(index_cache)
    bs = int(query_index.shape[0])
    local_offsets = torch.arange(block_size, device=query_index.device)
    for b in range(bs):
        candidate_len = int(candidate_lens[b].item())
        if candidate_len <= 0:
            continue
        num_blocks = (candidate_len + block_size - 1) // block_size
        blocks = index_block_table[b, :num_blocks].to(torch.long)
        slots = (blocks[:, None] * block_size + local_offsets[None, :]).reshape(-1)
        keys = flat_index.index_select(0, slots)[:candidate_len]
        scores = torch.einsum(
            "hd,td->ht",
            query_index[b].float(),
            keys.float(),
        )
        weights = index_weights[b].float().view(-1, 1)
        score_out[b, :candidate_len] = (scores * weights).sum(dim=0).to(
            score_out.dtype,
        )


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
    """Dispatch to the Ascend op when it is built, otherwise use the prototype."""
    if (
        not _env_flag("NANOVLLM_DSA_INDEX_UPDATE_FORCE_TORCH")
        and is_dsa_index_update_real_available()
        and score.device.type == "npu"
        and score.dtype == torch.bfloat16
        and int(max_copy_tokens) <= 128
        and score.is_contiguous()
        and hbm_cached_tokens_pool.is_contiguous()
        and promote_idx.is_contiguous()
        and demote_idx.is_contiguous()
        and copy_counts.is_contiguous()
        and candidate_lens.is_contiguous()
        and selected_lens.is_contiguous()
        and req_pool_entries.is_contiguous()
    ):
        dsa_index_update_real(
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
    """PyTorch prototype for DRAM-to-HBM sparse slot refill."""
    block_size = int(hbm_ckv_cache.shape[2])
    bs = int(promote_idx.shape[0])
    flat_hbm_ckv = _flatten_kv_cache(hbm_ckv_cache)
    flat_hbm_kpe = _flatten_kv_cache(hbm_kpe_cache)
    flat_dram_ckv = _flatten_kv_cache(dram_ckv_cache)
    flat_dram_kpe = _flatten_kv_cache(dram_kpe_cache)
    device = hbm_ckv_cache.device
    for b in range(bs):
        copy_count = int(copy_counts[b].item())
        for i in range(copy_count):
            t = int(promote_idx[b, i].item())
            s = int(demote_idx[b, i].item())
            dram_block = int(dram_block_table[b, t // block_size].item())
            assert dram_block > 0, f"invalid dram block={dram_block}"
            hbm_block = int(hbm_block_table[b, s // block_size].item())
            dram_slot = dram_block * block_size + (t % block_size)
            hbm_slot = hbm_block * block_size + (s % block_size)
            flat_hbm_ckv[hbm_slot].copy_(
                flat_dram_ckv[dram_slot].to(device=device, non_blocking=True),
            )
            flat_hbm_kpe[hbm_slot].copy_(
                flat_dram_kpe[dram_slot].to(device=device, non_blocking=True),
            )
