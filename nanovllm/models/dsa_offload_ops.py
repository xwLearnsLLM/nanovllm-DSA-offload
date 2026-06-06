from __future__ import annotations

import torch

import nanovllm.ops as ascend_ops


def dsa_lightning_indexer(
    query_index: torch.Tensor,
    index_cache: torch.Tensor,
    index_weights: torch.Tensor,
    index_block_table: torch.Tensor,
    candidate_lens: torch.Tensor,
    *,
    actual_seq_lengths_query: torch.Tensor,
    sparse_count: int,
) -> torch.Tensor:
    """Return top sparse token ids with the vLLM-Ascend LightningIndexer op."""
    return ascend_ops.npu_lightning_indexer(
        query=query_index,
        key=index_cache,
        weights=index_weights,
        actual_seq_lengths_query=actual_seq_lengths_query,
        actual_seq_lengths_key=candidate_lens,
        block_table=index_block_table,
        layout_query="TND",
        layout_key="PA_BSND",
        sparse_count=int(sparse_count),
        sparse_mode=3,
    )


def dsa_gather_selection_kv_cache(
    *,
    selection_k_rope: torch.Tensor,
    selection_kv_cache: torch.Tensor,
    selection_kv_block_table: torch.Tensor,
    selection_kv_block_status: torch.Tensor,
    selection_topk_indices: torch.Tensor,
    full_k_rope: torch.Tensor,
    full_kv_cache: torch.Tensor,
    full_kv_block_table: torch.Tensor,
    full_kv_actual_seq: torch.Tensor,
    full_q_actual_seq: torch.Tensor,
) -> torch.Tensor:
    """Gather top2048 full KV tokens into the HBM sparse budget cache."""
    return ascend_ops.npu_gather_selection_kv_cache(
        selection_k_rope,
        selection_kv_cache,
        selection_kv_block_table,
        selection_kv_block_status,
        selection_topk_indices,
        full_k_rope,
        full_kv_cache,
        full_kv_block_table,
        full_kv_actual_seq,
        full_q_actual_seq,
        selection_topk_block_size=1,
    )
