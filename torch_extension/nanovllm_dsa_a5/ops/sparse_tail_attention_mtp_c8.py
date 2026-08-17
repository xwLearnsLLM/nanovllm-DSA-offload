from __future__ import annotations

import torch

# The schema, PrivateUse1 implementation and Meta implementation all live in
# the C++ extension. The MTP variant dispatches to aclnnA5SparseTailAttentionMtpC8
# with the same packed-C8 buffers; per-verification-query top-2048 and the
# causal dense tail are encoded in the slots rows and cumulative
# actual_seq_lengths_query by the caller.
sparse_tail_attention_mtp_c8 = (
    torch.ops.nanovllm_dsa.sparse_tail_attention_mtp_c8.default
)

_BLOCK_SIZE = 128
_SPARSE_COUNT = 2048
_MAX_QUERIES_PER_REQUEST = 4
_MIN_QUERIES_PER_REQUEST = 1


def validate_mtp_c8_packing(
    actual_seq_lengths_query: torch.Tensor,
    resident_seq_lengths: torch.Tensor,
    packed_query_rows: int,
    slots_row_width: int,
) -> list[int]:
    """Check the MTP packing contract the C++ binding cannot.

    The binding is deliberately shape-only: reading tensor values there would
    force a synchronous D2H copy, which NPUGraph capture forbids. Call this
    helper once outside capture to fail fast on bad metadata.
    """
    if actual_seq_lengths_query.dim() != 1 or resident_seq_lengths.dim() != 1:
        raise ValueError("seq-length tensors must be 1-D")
    if resident_seq_lengths.numel() != actual_seq_lengths_query.numel():
        raise ValueError(
            "resident_seq_lengths must match actual_seq_lengths_query length"
        )
    totals = actual_seq_lengths_query.cpu().tolist()
    residents = resident_seq_lengths.cpu().tolist()
    query_counts: list[int] = []
    previous = 0
    for request, total in enumerate(totals):
        count = total - previous
        if not (
            _MIN_QUERIES_PER_REQUEST
            <= count
            <= _MAX_QUERIES_PER_REQUEST
        ):
            raise ValueError(
                f"actual_seq_lengths_query must be strictly increasing with "
                f"per-request diffs in [1,4]; request {request} has {count}"
            )
        if residents[request] < count:
            raise ValueError(
                f"resident_seq_lengths[{request}] must cover query_counts="
                f"{count}, got {residents[request]}"
            )
        cache_tokens = residents[request] - count
        if cache_tokens != 0 and (
            cache_tokens < _SPARSE_COUNT or cache_tokens % _BLOCK_SIZE
        ):
            raise ValueError(
                f"cache_tokens must be 0 or block-aligned >= 2048; request "
                f"{request} has {cache_tokens}"
            )
        query_counts.append(count)
        previous = total
    if previous != packed_query_rows:
        raise ValueError(
            f"actual_seq_lengths_query last value {previous} must equal the "
            f"packed query row count T={packed_query_rows}"
        )
    if slots_row_width < _SPARSE_COUNT + max(query_counts):
        raise ValueError(
            f"slots row width {slots_row_width} must cover "
            f"2048 + max(query_counts)={max(query_counts)}"
        )
    return query_counts
