from __future__ import annotations

import torch
import torch_npu

from ._constants import SPARSE_COUNT


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
    """Native C8 LI helper retained only for MTP-C8 and test goldens."""
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


# Public non-MTP C8 paths are registered directly by the C++ extension and
# each call now launches exactly one repository-local MIX kernel.
fused_li_manage_c8 = torch.ops.nanovllm_dsa.fused_li_manage_c8
fused_li_manage_c8_out = torch.ops.nanovllm_dsa.fused_li_manage_c8_out


__all__ = ["fused_li_manage_c8", "fused_li_manage_c8_out"]
