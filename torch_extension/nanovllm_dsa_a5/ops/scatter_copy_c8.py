from __future__ import annotations

import torch

from ._constants import BLOCK_SIZE, PACKED_KV_DIM


scatter_copy_c8 = torch.ops.nanovllm_dsa.scatter_copy_c8
scatter_copy_c8_out = torch.ops.nanovllm_dsa.scatter_copy_c8_out


def _packed_byte_view(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.ndim != 4 or tuple(tensor.shape[1:]) != (
        BLOCK_SIZE,
        1,
        PACKED_KV_DIM,
    ):
        raise ValueError(
            "packed C8 KV cache must have shape [blocks,128,1,656], "
            f"got {tuple(tensor.shape)}"
        )
    if tensor.element_size() != 1:
        raise TypeError(
            "packed C8 KV cache must use a one-byte dtype, "
            f"got {tensor.dtype}"
        )
    if not tensor.is_contiguous():
        raise ValueError("packed C8 KV cache must be contiguous")
    return tensor if tensor.dtype == torch.int8 else tensor.view(torch.int8)
