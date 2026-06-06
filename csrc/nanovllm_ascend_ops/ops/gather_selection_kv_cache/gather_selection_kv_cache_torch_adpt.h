/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 *
 * Licensed under CANN Open Software License Agreement Version 2.0.
 */
#ifndef GATHER_SELECTION_KV_CACHE_TORCH_ADPT_H
#define GATHER_SELECTION_KV_CACHE_TORCH_ADPT_H

namespace vllm_ascend {

inline at::Tensor npu_gather_selection_kv_cache(
    const at::Tensor& selection_k_rope,
    const at::Tensor& selection_kv_cache,
    const at::Tensor& selection_kv_block_table,
    const at::Tensor& selection_kv_block_status,
    const at::Tensor& selection_topk_indices,
    const at::Tensor& full_k_rope,
    const at::Tensor& full_kv_cache,
    const at::Tensor& full_kv_block_table,
    const at::Tensor& full_kv_actual_seq,
    const at::Tensor& full_q_actual_seq,
    int64_t selection_topk_block_size)
{
    TORCH_CHECK(selection_k_rope.device().is_privateuseone(), "selection_k_rope must be on NPU.");
    TORCH_CHECK(selection_kv_cache.device().is_privateuseone(), "selection_kv_cache must be on NPU.");
    TORCH_CHECK(selection_kv_block_table.device().is_privateuseone(), "selection_kv_block_table must be on NPU.");
    TORCH_CHECK(selection_kv_block_status.device().is_privateuseone(), "selection_kv_block_status must be on NPU.");
    TORCH_CHECK(selection_topk_indices.device().is_privateuseone(), "selection_topk_indices must be on NPU.");
    TORCH_CHECK(full_kv_block_table.device().is_privateuseone(), "full_kv_block_table must be on NPU.");
    TORCH_CHECK(full_kv_actual_seq.device().is_privateuseone(), "full_kv_actual_seq must be on NPU.");
    TORCH_CHECK(full_q_actual_seq.device().is_privateuseone(), "full_q_actual_seq must be on NPU.");
    TORCH_CHECK(selection_kv_block_table.dim() == 2, "selection_kv_block_table must be [batch, blocks].");
    TORCH_CHECK(selection_kv_block_status.dim() == 4, "selection_kv_block_status must be [batch, 1, 1, topk+1].");
    TORCH_CHECK(selection_topk_indices.dim() == 4, "selection_topk_indices must be [batch, heads, q, topk].");
    TORCH_CHECK(selection_topk_block_size > 0, "selection_topk_block_size must be positive.");

    at::Tensor selection_kv_actual_seq = at::empty(
        {selection_kv_block_table.size(0)},
        selection_topk_indices.options());

    EXEC_NPU_CMD(
        aclnnGatherSelectionKvCache,
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
        selection_topk_block_size,
        selection_kv_actual_seq);

    return selection_kv_actual_seq;
}

}  // namespace vllm_ascend

#endif  // GATHER_SELECTION_KV_CACHE_TORCH_ADPT_H
