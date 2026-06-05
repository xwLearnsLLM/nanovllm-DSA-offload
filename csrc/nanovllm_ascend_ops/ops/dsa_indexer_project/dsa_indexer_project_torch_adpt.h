#ifndef DSA_INDEXER_PROJECT_TORCH_ADPT_H
#define DSA_INDEXER_PROJECT_TORCH_ADPT_H

#include <algorithm>
#include <tuple>

#include <c10/core/DeviceGuard.h>
#include <torch/extension.h>

#include "common/torch_adapter/op_api_common.h"
#include "common/kernels/types.h"

namespace vllm_ascend {

extern void dsa_indexer_project_post_impl(
    AscendType type,
    void* stream,
    void* q_in,
    void* k_in,
    void* weights_in,
    void* cos,
    void* sin,
    void* q_out,
    void* k_out,
    void* weights_out,
    uint32_t num_tokens,
    uint32_t n_head,
    uint32_t head_dim,
    uint32_t rope_dim,
    float score_scale,
    uint32_t block_dim);

inline void CheckIndexerProjectPostTensor(
    const at::Tensor& tensor,
    const char* name,
    at::ScalarType dtype,
    int64_t dim)
{
    TORCH_CHECK(tensor.defined(), name, " must be defined.");
    TORCH_CHECK(tensor.scalar_type() == dtype, name, " dtype mismatch.");
    TORCH_CHECK(tensor.dim() == dim, name, " rank mismatch.");
    TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous.");
    TORCH_CHECK(tensor.is_privateuseone(), name, " must be an NPU tensor.");
}

inline void CheckIndexerProjectPostInputs(
    const at::Tensor& q_in,
    const at::Tensor& k_in,
    const at::Tensor& weights_in,
    const at::Tensor& cos,
    const at::Tensor& sin,
    int64_t rope_dim)
{
    CheckIndexerProjectPostTensor(q_in, "q_in", q_in.scalar_type(), 3);
    CheckIndexerProjectPostTensor(k_in, "k_in", q_in.scalar_type(), 2);
    CheckIndexerProjectPostTensor(weights_in, "weights_in", at::kFloat, 2);
    CheckIndexerProjectPostTensor(cos, "cos", q_in.scalar_type(), 4);
    CheckIndexerProjectPostTensor(sin, "sin", q_in.scalar_type(), 4);

    TORCH_CHECK(q_in.scalar_type() == at::kHalf || q_in.scalar_type() == at::kBFloat16,
        "q_in must be float16 or bfloat16.");
    const int64_t num_tokens = q_in.size(0);
    const int64_t n_head = q_in.size(1);
    const int64_t head_dim = q_in.size(2);
    TORCH_CHECK(num_tokens > 0 && n_head > 0 && head_dim > 0, "q_in must be non-empty.");
    TORCH_CHECK(k_in.size(0) == num_tokens && k_in.size(1) == head_dim, "k_in shape mismatch.");
    TORCH_CHECK(weights_in.size(0) == num_tokens && weights_in.size(1) == n_head, "weights_in shape mismatch.");
    TORCH_CHECK(cos.size(0) == num_tokens && sin.size(0) == num_tokens, "cos/sin token dimension mismatch.");
    TORCH_CHECK(cos.numel() >= num_tokens * rope_dim && sin.numel() >= num_tokens * rope_dim,
        "cos/sin storage is smaller than num_tokens * rope_dim.");
    TORCH_CHECK(rope_dim > 0 && rope_dim <= head_dim && rope_dim % 2 == 0, "invalid rope_dim.");
    TORCH_CHECK(head_dim <= 256 && n_head <= 128, "dsa_indexer_project_post currently supports head_dim <= 256 and n_head <= 128.");
    TORCH_CHECK(head_dim % 16 == 0 && rope_dim % 16 == 0 && n_head % 8 == 0,
        "head_dim, rope_dim and n_head must be aligned for vector DataCopy.");
}

inline void dsa_indexer_project_post_out(
    const at::Tensor& q_in,
    const at::Tensor& k_in,
    const at::Tensor& weights_in,
    const at::Tensor& cos,
    const at::Tensor& sin,
    const at::Tensor& q_out,
    const at::Tensor& k_out,
    const at::Tensor& weights_out,
    double score_scale,
    int64_t rope_dim)
{
    CheckIndexerProjectPostInputs(q_in, k_in, weights_in, cos, sin, rope_dim);
    CheckIndexerProjectPostTensor(q_out, "q_out", q_in.scalar_type(), 3);
    CheckIndexerProjectPostTensor(k_out, "k_out", q_in.scalar_type(), 2);
    CheckIndexerProjectPostTensor(weights_out, "weights_out", q_in.scalar_type(), 2);
    TORCH_CHECK(q_out.sizes() == q_in.sizes(), "q_out shape mismatch.");
    TORCH_CHECK(k_out.sizes() == k_in.sizes(), "k_out shape mismatch.");
    TORCH_CHECK(weights_out.sizes() == weights_in.sizes(), "weights_out shape mismatch.");

    const int64_t num_tokens = q_in.size(0);
    const int64_t n_head = q_in.size(1);
    const int64_t head_dim = q_in.size(2);
    c10::OptionalDeviceGuard device_guard(q_in.device());
    const int64_t work_items = num_tokens * n_head * head_dim + num_tokens * head_dim + num_tokens * n_head;
    const uint32_t block_dim = static_cast<uint32_t>(std::min<int64_t>(48, std::max<int64_t>(1, (work_items + 1023) / 1024)));
    const AscendType type = q_in.scalar_type() == at::kHalf ? AscendType::FP16 : AscendType::BF16;

    void* q_in_ptr = q_in.data_ptr();
    void* k_in_ptr = k_in.data_ptr();
    void* weights_in_ptr = weights_in.data_ptr();
    void* cos_ptr = cos.data_ptr();
    void* sin_ptr = sin.data_ptr();
    void* q_out_ptr = q_out.data_ptr();
    void* k_out_ptr = k_out.data_ptr();
    void* weights_out_ptr = weights_out.data_ptr();
    aclrtStream stream = c10_npu::getCurrentNPUStream().stream(true);

    at_npu::native::OpCommand cmd;
    cmd.Name("dsa_indexer_project_post");
    cmd.SetCustomHandler([type, stream, q_in_ptr, k_in_ptr, weights_in_ptr, cos_ptr, sin_ptr,
                          q_out_ptr, k_out_ptr, weights_out_ptr, num_tokens, n_head, head_dim,
                          rope_dim, score_scale, block_dim]() -> int {
        dsa_indexer_project_post_impl(
            type,
            stream,
            q_in_ptr,
            k_in_ptr,
            weights_in_ptr,
            cos_ptr,
            sin_ptr,
            q_out_ptr,
            k_out_ptr,
            weights_out_ptr,
            static_cast<uint32_t>(num_tokens),
            static_cast<uint32_t>(n_head),
            static_cast<uint32_t>(head_dim),
            static_cast<uint32_t>(rope_dim),
            static_cast<float>(score_scale),
            block_dim);
        return 0;
    });
    cmd.Run();
}

inline std::tuple<at::Tensor, at::Tensor, at::Tensor> dsa_indexer_project_post(
    const at::Tensor& q_in,
    const at::Tensor& k_in,
    const at::Tensor& weights_in,
    const at::Tensor& cos,
    const at::Tensor& sin,
    double score_scale,
    int64_t rope_dim)
{
    CheckIndexerProjectPostInputs(q_in, k_in, weights_in, cos, sin, rope_dim);
    at::Tensor q_out = at::empty_like(q_in);
    at::Tensor k_out = at::empty_like(k_in);
    at::Tensor weights_out = at::empty(weights_in.sizes(), q_in.options());
    dsa_indexer_project_post_out(q_in, k_in, weights_in, cos, sin, q_out, k_out, weights_out, score_scale, rope_dim);
    return std::make_tuple(q_out, k_out, weights_out);
}

} // namespace vllm_ascend

#endif
