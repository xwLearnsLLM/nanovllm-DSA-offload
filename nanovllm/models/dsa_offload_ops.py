from __future__ import annotations

import importlib

import torch

import nanovllm.ops as ascend_ops

_GATHER_SELECTION_LOADED = False


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


def _ensure_gather_selection_loaded() -> None:
    global _GATHER_SELECTION_LOADED
    if _GATHER_SELECTION_LOADED:
        return
    try:
        importlib.import_module("gather_selection_custom_ops")
    except ImportError as exc:
        raise RuntimeError(
            "gather_selection_custom_ops is not importable. Build/install "
            "D:\\vLLM-ascend\\ops_gather_selection_kv_cache first, then run "
            "nano-vllm with that package on PYTHONPATH."
        ) from exc
    if not hasattr(torch.ops, "custom") or not hasattr(torch.ops.custom, "npu_gather_selection_kv_cache"):
        raise RuntimeError(
            "torch.ops.custom.npu_gather_selection_kv_cache is unavailable. "
            "Check that the gather_selection custom op package and OPP vendor "
            "library are installed for the current CANN environment."
        )
    _GATHER_SELECTION_LOADED = True


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
    _ensure_gather_selection_loaded()
    return torch.ops.custom.npu_gather_selection_kv_cache(
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
