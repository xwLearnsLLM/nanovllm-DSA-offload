import torch


sparse_and_tail_attention_and_scatter_copy = (
    torch.ops.nanovllm_dsa.sparse_and_tail_attention_and_scatter_copy
)
sparse_and_tail_attention_and_scatter_copy_mte_pipeline = (
    torch.ops.nanovllm_dsa
    .sparse_and_tail_attention_and_scatter_copy_mte_pipeline
)
