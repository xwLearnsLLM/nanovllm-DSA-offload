import torch


fused_copy_sparse_tail_attention = (
    torch.ops.nanovllm_dsa.fused_copy_sparse_tail_attention
)
