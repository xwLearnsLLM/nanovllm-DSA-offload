from __future__ import annotations

import torch

from ._constants import BLOCK_SIZE, PACKED_KV_DIM, SPARSE_COUNT


NOPE_DIM = 512
QUERY_DIM = 576


def _check_common_inputs(
    query: torch.Tensor,
    packed_kv: torch.Tensor,
    actual_seq_lengths_query: torch.Tensor,
    hbm_block_table: torch.Tensor,
    topk_destination_slots: torch.Tensor,
    topk_miss_counts: torch.Tensor,
) -> tuple[int, int, int]:
    if (
        query.ndim != 3
        or query.shape[0] <= 0
        or not 1 <= query.shape[1] <= 64
        or query.shape[2] != QUERY_DIM
    ):
        raise ValueError(
            "C8 staged SFA query must be [T,Q_HEAD,576] with "
            "1 <= Q_HEAD <= 64"
        )
    if query.dtype not in (torch.bfloat16, torch.float16):
        raise TypeError("C8 staged SFA query must be bf16 or fp16")
    if (
        packed_kv.ndim != 4
        or tuple(packed_kv.shape[1:])
        != (BLOCK_SIZE, 1, PACKED_KV_DIM)
        or packed_kv.element_size() != 1
        or not packed_kv.is_contiguous()
    ):
        raise ValueError(
            "C8 staged SFA packed KV must be contiguous one-byte "
            "[blocks,128,1,656]"
        )
    packed_queries, heads, _ = query.shape
    if actual_seq_lengths_query.ndim != 1:
        raise ValueError("actual_seq_lengths_query must be cumulative int32 [B]")
    batch = actual_seq_lengths_query.numel()
    if batch <= 0:
        raise ValueError("C8 staged SFA batch must be positive")
    if (
        hbm_block_table.ndim != 2
        or hbm_block_table.shape[0] != batch
        or hbm_block_table.shape[1] <= 0
        or topk_destination_slots.shape
        != (packed_queries, 1, SPARSE_COUNT)
        or topk_miss_counts.shape != (packed_queries,)
    ):
        raise ValueError("C8 staged SFA metadata shapes are inconsistent")
    for name, tensor in (
        ("actual_seq_lengths_query", actual_seq_lengths_query),
        ("hbm_block_table", hbm_block_table),
        ("topk_destination_slots", topk_destination_slots),
        ("topk_miss_counts", topk_miss_counts),
    ):
        if tensor.dtype != torch.int32:
            raise TypeError(f"C8 staged SFA {name} must be int32")
    tensors = (
        packed_kv,
        actual_seq_lengths_query,
        hbm_block_table,
        topk_destination_slots,
        topk_miss_counts,
    )
    if any(
        tensor.device != query.device or not tensor.is_contiguous()
        for tensor in tensors
    ):
        raise ValueError(
            "C8 staged SFA tensors must be contiguous on one device"
        )
    return packed_queries, heads, batch


def _check_stage1_outputs(
    query: torch.Tensor,
    partial_out: torch.Tensor,
    softmax_max: torch.Tensor,
    softmax_sum: torch.Tensor,
) -> None:
    packed_queries, heads, _ = query.shape
    if (
        partial_out.shape != (packed_queries, heads, NOPE_DIM)
        or softmax_max.shape != (1, packed_queries, heads)
        or softmax_sum.shape != (1, packed_queries, heads)
    ):
        raise ValueError(
            "C8 stage1 outputs must be partial_out [T,N,512] and "
            "softmax_max/softmax_sum [1,T,N]"
        )
    for tensor in (partial_out, softmax_max, softmax_sum):
        if (
            tensor.dtype != torch.float32
            or tensor.device != query.device
            or not tensor.is_contiguous()
        ):
            raise TypeError(
                "C8 stage1 outputs must be contiguous FP32 tensors on "
                "the query device"
            )


def _check_stage2_state(
    query: torch.Tensor,
    partial_out: torch.Tensor,
    softmax_max: torch.Tensor,
    softmax_sum: torch.Tensor,
    attention_out: torch.Tensor,
) -> None:
    _check_stage1_outputs(query, partial_out, softmax_max, softmax_sum)
    if (
        attention_out.shape != (*query.shape[:-1], NOPE_DIM)
        or attention_out.dtype != query.dtype
        or attention_out.device != query.device
        or not attention_out.is_contiguous()
    ):
        raise TypeError(
            "C8 stage2 attention_out must be contiguous [T,N,512] "
            "with the query dtype/device"
        )


def _local_state_out(
    query: torch.Tensor,
    packed_kv: torch.Tensor,
    topk_slots: torch.Tensor,
    hbm_block_table: torch.Tensor,
    actual_seq_lengths_query: torch.Tensor,
    actual_seq_lengths_kv: torch.Tensor,
    miss_counts: torch.Tensor,
    cache_tokens: torch.Tensor,
    scale_value: float,
    partial_out: torch.Tensor,
    softmax_max: torch.Tensor,
    softmax_sum: torch.Tensor,
    kv_dtype: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return torch.ops.nanovllm_dsa._sparse_tail_attention_c8_mtp_stage1_out(
        query,
        packed_kv,
        topk_slots,
        hbm_block_table,
        actual_seq_lengths_query,
        actual_seq_lengths_kv,
        miss_counts,
        cache_tokens,
        float(scale_value),
        partial_out,
        softmax_max,
        softmax_sum,
        kv_dtype,
    )


def _stage1_impl(
    query: torch.Tensor,
    packed_kv: torch.Tensor,
    actual_seq_lengths_query: torch.Tensor,
    resident_seq_lengths: torch.Tensor,
    cache_tokens: torch.Tensor,
    hbm_block_table: torch.Tensor,
    topk_destination_slots: torch.Tensor,
    topk_miss_counts: torch.Tensor,
    scale_value: float,
    partial_out: torch.Tensor,
    softmax_max: torch.Tensor,
    softmax_sum: torch.Tensor,
    kv_dtype: int | None = None,
) -> None:
    _, _, batch = _check_common_inputs(
        query,
        packed_kv,
        actual_seq_lengths_query,
        hbm_block_table,
        topk_destination_slots,
        topk_miss_counts,
    )
    if (
        resident_seq_lengths.shape != (batch,)
        or cache_tokens.shape != (batch,)
    ):
        raise ValueError(
            "resident_seq_lengths and cache_tokens must be int32 [B]"
        )
    for tensor in (resident_seq_lengths, cache_tokens):
        if (
            tensor.dtype != torch.int32
            or tensor.device != query.device
            or not tensor.is_contiguous()
        ):
            raise TypeError(
                "resident_seq_lengths/cache_tokens must be contiguous "
                "int32 tensors on the query device"
            )
    _check_stage1_outputs(query, partial_out, softmax_max, softmax_sum)
    _local_state_out(
        query,
        packed_kv,
        topk_destination_slots,
        hbm_block_table,
        actual_seq_lengths_query,
        resident_seq_lengths,
        topk_miss_counts,
        cache_tokens,
        scale_value,
        partial_out,
        softmax_max,
        softmax_sum,
        kv_dtype,
    )


def _stage2_impl(
    query: torch.Tensor,
    packed_kv: torch.Tensor,
    actual_seq_lengths_query: torch.Tensor,
    resident_seq_lengths: torch.Tensor,
    hbm_block_table: torch.Tensor,
    topk_destination_slots: torch.Tensor,
    topk_miss_counts: torch.Tensor,
    scale_value: float,
    partial_out: torch.Tensor,
    softmax_max: torch.Tensor,
    softmax_sum: torch.Tensor,
    attention_out: torch.Tensor,
    kv_dtype: int | None = None,
) -> None:
    _, _, batch = _check_common_inputs(
        query,
        packed_kv,
        actual_seq_lengths_query,
        hbm_block_table,
        topk_destination_slots,
        topk_miss_counts,
    )
    if (
        resident_seq_lengths.shape != (batch,)
        or resident_seq_lengths.dtype != torch.int32
        or resident_seq_lengths.device != query.device
        or not resident_seq_lengths.is_contiguous()
    ):
        raise TypeError(
            "resident_seq_lengths must be contiguous int32 [B] on the "
            "query device"
        )
    _check_stage2_state(
        query,
        partial_out,
        softmax_max,
        softmax_sum,
        attention_out,
    )
    torch.ops.nanovllm_dsa._sparse_tail_attention_c8_mtp_stage2(
        query,
        packed_kv,
        topk_destination_slots,
        hbm_block_table,
        actual_seq_lengths_query,
        resident_seq_lengths,
        topk_miss_counts,
        resident_seq_lengths,
        float(scale_value),
        partial_out,
        softmax_max,
        softmax_sum,
        attention_out,
        kv_dtype,
    )


def _stage1_meta(
    query: torch.Tensor,
    packed_kv: torch.Tensor,
    actual_seq_lengths_query: torch.Tensor,
    resident_seq_lengths: torch.Tensor,
    cache_tokens: torch.Tensor,
    hbm_block_table: torch.Tensor,
    topk_destination_slots: torch.Tensor,
    topk_miss_counts: torch.Tensor,
    scale_value: float,
    partial_out: torch.Tensor,
    softmax_max: torch.Tensor,
    softmax_sum: torch.Tensor,
    kv_dtype: int | None = None,
) -> None:
    del (
        packed_kv,
        actual_seq_lengths_query,
        resident_seq_lengths,
        cache_tokens,
        hbm_block_table,
        topk_destination_slots,
        topk_miss_counts,
        scale_value,
        kv_dtype,
    )
    _check_stage1_outputs(query, partial_out, softmax_max, softmax_sum)


def _stage2_meta(
    query: torch.Tensor,
    packed_kv: torch.Tensor,
    actual_seq_lengths_query: torch.Tensor,
    resident_seq_lengths: torch.Tensor,
    hbm_block_table: torch.Tensor,
    topk_destination_slots: torch.Tensor,
    topk_miss_counts: torch.Tensor,
    scale_value: float,
    partial_out: torch.Tensor,
    softmax_max: torch.Tensor,
    softmax_sum: torch.Tensor,
    attention_out: torch.Tensor,
    kv_dtype: int | None = None,
) -> None:
    del (
        packed_kv,
        actual_seq_lengths_query,
        resident_seq_lengths,
        hbm_block_table,
        topk_destination_slots,
        topk_miss_counts,
        scale_value,
        kv_dtype,
    )
    _check_stage2_state(
        query,
        partial_out,
        softmax_max,
        softmax_sum,
        attention_out,
    )


def _pml_probe_meta(
    query: torch.Tensor,
    packed_kv: torch.Tensor,
    sparse_indices: torch.Tensor,
    block_table: torch.Tensor,
    actual_q: torch.Tensor,
    actual_kv: torch.Tensor,
    miss_counts: torch.Tensor,
    cache_tokens: torch.Tensor,
    scale_value: float,
    probe_enabled: bool,
    attention_out: torch.Tensor,
    partial_out: torch.Tensor,
    softmax_max: torch.Tensor,
    softmax_sum: torch.Tensor,
    kv_dtype: int | None = None,
) -> None:
    del (
        packed_kv,
        sparse_indices,
        block_table,
        actual_q,
        actual_kv,
        miss_counts,
        cache_tokens,
        scale_value,
        probe_enabled,
        kv_dtype,
    )
    _check_stage2_state(
        query, partial_out, softmax_max, softmax_sum, attention_out
    )


def _tnd_probe_meta(
    query: torch.Tensor,
    packed_kv: torch.Tensor,
    sparse_indices: torch.Tensor,
    block_table: torch.Tensor,
    actual_q: torch.Tensor,
    actual_kv: torch.Tensor,
    miss_counts: torch.Tensor,
    cache_tokens: torch.Tensor,
    scale_value: float,
    probe_enabled: bool,
    attention_out: torch.Tensor,
    kv_dtype: int | None = None,
) -> None:
    del (
        packed_kv,
        sparse_indices,
        block_table,
        actual_q,
        actual_kv,
        miss_counts,
        cache_tokens,
        scale_value,
        probe_enabled,
        kv_dtype,
    )
    if attention_out.shape != (*query.shape[:-1], 512):
        raise ValueError("C8 TND probe attention_out shape is invalid")
    if attention_out.dtype != query.dtype:
        raise TypeError("C8 TND probe attention_out dtype must match query")


torch.library.impl(
    "nanovllm_dsa::sparse_tail_attention_c8_mtp_stage1",
    "PrivateUse1",
)(_stage1_impl)
torch.library.impl(
    "nanovllm_dsa::sparse_tail_attention_c8_mtp_stage1",
    "Meta",
)(_stage1_meta)
torch.library.impl(
    "nanovllm_dsa::sparse_tail_attention_c8_mtp_stage2",
    "PrivateUse1",
)(_stage2_impl)
torch.library.impl(
    "nanovllm_dsa::sparse_tail_attention_c8_mtp_stage2",
    "Meta",
)(_stage2_meta)
torch.library.impl(
    "nanovllm_dsa::_sparse_tail_attention_c8_pml_probe_out",
    "Meta",
)(_pml_probe_meta)
torch.library.impl(
    "nanovllm_dsa::_sparse_tail_attention_c8_tnd_probe_out",
    "Meta",
)(_tnd_probe_meta)


def sparse_tail_attention_c8_mtp_stage1(
    query: torch.Tensor,
    packed_kv: torch.Tensor,
    actual_seq_lengths_query: torch.Tensor,
    resident_seq_lengths: torch.Tensor,
    cache_tokens: torch.Tensor,
    hbm_block_table: torch.Tensor,
    topk_destination_slots: torch.Tensor,
    topk_miss_counts: torch.Tensor,
    scale_value: float,
    partial_out: torch.Tensor,
    softmax_max: torch.Tensor,
    softmax_sum: torch.Tensor,
    kv_dtype: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    torch.ops.nanovllm_dsa.sparse_tail_attention_c8_mtp_stage1.default(
        query,
        packed_kv,
        actual_seq_lengths_query,
        resident_seq_lengths,
        cache_tokens,
        hbm_block_table,
        topk_destination_slots,
        topk_miss_counts,
        float(scale_value),
        partial_out,
        softmax_max,
        softmax_sum,
        kv_dtype,
    )
    return partial_out, softmax_max, softmax_sum


def sparse_tail_attention_c8_mtp_stage2(
    query: torch.Tensor,
    packed_kv: torch.Tensor,
    actual_seq_lengths_query: torch.Tensor,
    resident_seq_lengths: torch.Tensor,
    hbm_block_table: torch.Tensor,
    topk_destination_slots: torch.Tensor,
    topk_miss_counts: torch.Tensor,
    scale_value: float,
    partial_out: torch.Tensor,
    softmax_max: torch.Tensor,
    softmax_sum: torch.Tensor,
    attention_out: torch.Tensor,
    kv_dtype: int | None = None,
) -> torch.Tensor:
    torch.ops.nanovllm_dsa.sparse_tail_attention_c8_mtp_stage2.default(
        query,
        packed_kv,
        actual_seq_lengths_query,
        resident_seq_lengths,
        hbm_block_table,
        topk_destination_slots,
        topk_miss_counts,
        float(scale_value),
        partial_out,
        softmax_max,
        softmax_sum,
        attention_out,
        kv_dtype,
    )
    return attention_out


__all__ = [
    "sparse_tail_attention_c8_mtp_stage1",
    "sparse_tail_attention_c8_mtp_stage2",
]
