#pragma once

#include <cstdint>
#include <torch/torch.h>

#include "common/torch_adapter/op_api_common.h"

namespace vllm_ascend {

inline void dsa_indexer_update(
    at::Tensor& score,                   // [bs, max_candidate]  bf16  in/out
    at::Tensor& selected_idx,            // [bs, max_selected]   int32 in/out
    const at::Tensor& seq_len,           // [bs]                 int32
    const at::Tensor& selected_len,      // [bs]                 int32
    int64_t k,                           // scalar
    at::Tensor& promote_idx,             // [bs, k]              int32 out
    at::Tensor& demote_idx)              // [bs, k]              int32 out
{
    TORCH_CHECK(score.device().is_privateuseone(), "dsa_indexer_update: score must be on NPU");
    TORCH_CHECK(score.scalar_type() == at::kBFloat16 || score.scalar_type() == at::kHalf,
                "dsa_indexer_update: score must be bf16");
    TORCH_CHECK(selected_idx.scalar_type() == at::kInt,
                "dsa_indexer_update: selected_idx must be int32");
    TORCH_CHECK(seq_len.scalar_type() == at::kInt,
                "dsa_indexer_update: seq_len must be int32");
    TORCH_CHECK(selected_len.scalar_type() == at::kInt,
                "dsa_indexer_update: selected_len must be int32");
    TORCH_CHECK(promote_idx.scalar_type() == at::kInt,
                "dsa_indexer_update: promote_idx must be int32");
    TORCH_CHECK(demote_idx.scalar_type() == at::kInt,
                "dsa_indexer_update: demote_idx must be int32");

    TORCH_CHECK(score.dim() == 2, "dsa_indexer_update: score must be 2-D");
    TORCH_CHECK(selected_idx.dim() == 2, "dsa_indexer_update: selected_idx must be 2-D");
    TORCH_CHECK(seq_len.dim() == 1, "dsa_indexer_update: seq_len must be 1-D");
    TORCH_CHECK(selected_len.dim() == 1, "dsa_indexer_update: selected_len must be 1-D");
    TORCH_CHECK(promote_idx.dim() == 2, "dsa_indexer_update: promote_idx must be 2-D");
    TORCH_CHECK(demote_idx.dim() == 2, "dsa_indexer_update: demote_idx must be 2-D");

    int64_t bs = score.size(0);
    TORCH_CHECK(bs > 0, "dsa_indexer_update: batch size must be positive");
    TORCH_CHECK(selected_idx.size(0) == bs &&
                seq_len.size(0) == bs &&
                selected_len.size(0) == bs &&
                promote_idx.size(0) == bs &&
                demote_idx.size(0) == bs,
                "dsa_indexer_update: batch size mismatch");
    TORCH_CHECK(promote_idx.size(1) == k && demote_idx.size(1) == k,
                "dsa_indexer_update: k size mismatch");

    const c10_npu::OptionalNPUGuard npu_guard(score.device());

    at::Tensor score_contig = score.contiguous();
    at::Tensor selected_contig = selected_idx.contiguous();
    at::Tensor seq_len_contig = seq_len.contiguous();
    at::Tensor selected_len_contig = selected_len.contiguous();
    at::Tensor promote_contig = promote_idx.contiguous();
    at::Tensor demote_contig = demote_idx.contiguous();

    EXEC_NPU_CMD(aclnnDsaUpdateIndex,
                 score_contig, selected_contig,
                 seq_len_contig, selected_len_contig,
                 k,
                 promote_contig, demote_contig);

    // Copy results back if input tensors were not contiguous
    if (!score.is_contiguous()) {
        score.copy_(score_contig);
    }
    if (!selected_idx.is_contiguous()) {
        selected_idx.copy_(selected_contig);
    }
    if (!promote_idx.is_contiguous()) {
        promote_idx.copy_(promote_contig);
    }
    if (!demote_idx.is_contiguous()) {
        demote_idx.copy_(demote_contig);
    }
}

}  // namespace vllm_ascend
