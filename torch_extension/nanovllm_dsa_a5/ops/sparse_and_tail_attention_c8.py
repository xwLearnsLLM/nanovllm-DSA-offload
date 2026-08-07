from __future__ import annotations

import torch
import torch_npu

from ._constants import SPARSE_COUNT
from .scatter_copy_c8 import _packed_byte_view


def _native_qsfa():
    namespace = getattr(torch.ops, "_C_ascend", None)
    op = (
        getattr(namespace, "npu_kv_quant_sparse_flash_attention", None)
        if namespace is not None
        else None
    )
    if op is not None:
        return op, True
    op = getattr(torch_npu, "npu_kv_quant_sparse_flash_attention", None)
    if op is None:
        raise RuntimeError(
            "A5 npu_kv_quant_sparse_flash_attention is not registered"
        )
    return op, False


def _check_c8_attention_inputs(
    query: torch.Tensor,
    packed_kv: torch.Tensor,
    sparse_and_tail_slots: torch.Tensor,
    block_table: torch.Tensor,
    actual_seq_lengths_query: torch.Tensor,
    resident_seq_lengths: torch.Tensor,
) -> None:
    if query.ndim != 3 or query.shape[-1] != 576:
        raise ValueError(
            "C8 QSFA query must be TND [T,Q_HEAD,576], "
            f"got {tuple(query.shape)}"
        )
    if not 1 <= query.shape[1] <= 64:
        raise ValueError("C8 QSFA supports 1 <= Q_HEAD <= 64")
    if query.dtype not in (torch.bfloat16, torch.float16):
        raise TypeError("C8 QSFA query must be bf16 or fp16")
    _packed_byte_view(packed_kv)
    if (
        sparse_and_tail_slots.ndim != 3
        or sparse_and_tail_slots.shape[0] != query.shape[0]
        or sparse_and_tail_slots.shape[1] != 1
        or sparse_and_tail_slots.shape[2] < SPARSE_COUNT
        or sparse_and_tail_slots.dtype != torch.int32
    ):
        raise ValueError(
            "C8 QSFA sparse_and_tail_slots must be int32 "
            "[T,1,2048+max_tail_tokens]"
        )
    batch = actual_seq_lengths_query.numel()
    if (
        actual_seq_lengths_query.ndim != 1
        or resident_seq_lengths.ndim != 1
        or resident_seq_lengths.numel() != batch
        or block_table.ndim != 2
        or block_table.shape[0] != batch
    ):
        raise ValueError("C8 QSFA batch metadata shapes are inconsistent")
    for name, tensor in (
        ("block_table", block_table),
        ("actual_seq_lengths_query", actual_seq_lengths_query),
        ("resident_seq_lengths", resident_seq_lengths),
    ):
        if tensor.dtype != torch.int32:
            raise TypeError(f"C8 QSFA {name} must be int32")


@torch.library.custom_op(
    "nanovllm_dsa::sparse_and_tail_attention_c8",
    mutates_args=(),
)
def _sparse_and_tail_attention_c8(
    query: torch.Tensor,
    packed_kv: torch.Tensor,
    sparse_and_tail_slots: torch.Tensor,
    block_table: torch.Tensor,
    actual_seq_lengths_query: torch.Tensor,
    resident_seq_lengths: torch.Tensor,
    scale_value: float,
) -> torch.Tensor:
    """Call native A5 C8 QSFA with GLM-5.1 packed-MLA attributes."""

    _check_c8_attention_inputs(
        query,
        packed_kv,
        sparse_and_tail_slots,
        block_table,
        actual_seq_lengths_query,
        resident_seq_lengths,
    )
    op, supports_lse = _native_qsfa()
    kwargs = dict(
        query=query,
        key=packed_kv,
        value=packed_kv,
        sparse_indices=sparse_and_tail_slots,
        scale_value=float(scale_value),
        key_quant_mode=2,
        value_quant_mode=2,
        block_table=block_table,
        actual_seq_lengths_query=actual_seq_lengths_query,
        actual_seq_lengths_kv=resident_seq_lengths,
        sparse_block_size=1,
        layout_query="TND",
        layout_kv="PA_BSND",
        sparse_mode=3,
        attention_mode=2,
        quant_scale_repo_mode=1,
        tile_size=128,
        rope_head_dim=64,
    )
    if supports_lse:
        kwargs["return_softmax_lse"] = False
    result = op(**kwargs)
    return result[0] if isinstance(result, tuple) else result


@_sparse_and_tail_attention_c8.register_fake
def _sparse_and_tail_attention_c8_fake(
    query: torch.Tensor,
    packed_kv: torch.Tensor,
    sparse_and_tail_slots: torch.Tensor,
    block_table: torch.Tensor,
    actual_seq_lengths_query: torch.Tensor,
    resident_seq_lengths: torch.Tensor,
    scale_value: float,
) -> torch.Tensor:
    del (
        packed_kv,
        sparse_and_tail_slots,
        block_table,
        actual_seq_lengths_query,
        resident_seq_lengths,
        scale_value,
    )
    if query.ndim != 3 or query.shape[-1] != 576:
        raise ValueError("C8 QSFA query must be [T,Q_HEAD,576]")
    if not 1 <= query.shape[1] <= 64:
        raise ValueError("C8 QSFA supports 1 <= Q_HEAD <= 64")
    return query.new_empty((*query.shape[:-1], 512))


sparse_and_tail_attention_c8 = (
    torch.ops.nanovllm_dsa.sparse_and_tail_attention_c8.default
)
