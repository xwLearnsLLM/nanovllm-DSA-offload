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
