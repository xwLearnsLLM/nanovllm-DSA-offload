#ifndef DSA_INDEXER_PROJECT_TORCH_ADPT_H
#define DSA_INDEXER_PROJECT_TORCH_ADPT_H

#include <algorithm>
#include <tuple>

#include <c10/core/DeviceGuard.h>
#include <torch/extension.h>

#include "common/torch_adapter/op_api_common.h"
#include "dsa_indexer_project_types.h"

namespace vllm_ascend {

extern void dsa_indexer_query_rope_impl(
    AscendType type,
    void* stream,
    void* q_inout,
    void* cos,
    void* sin,
    uint32_t num_tokens,
    uint32_t n_head,
    uint32_t head_dim,
    uint32_t rope_dim,
    uint32_t sign_pair_bits,
    uint32_t block_dim);

inline void CheckIndexerTensor(
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

inline void dsa_indexer_query_rope_inplace(
    const at::Tensor& q_inout,
    const at::Tensor& cos,
    const at::Tensor& sin,
    int64_t rope_dim)
{
    CheckIndexerTensor(
        q_inout, "q_inout", q_inout.scalar_type(), 3);
    CheckIndexerTensor(
        cos, "cos", q_inout.scalar_type(), 4);
    CheckIndexerTensor(
        sin, "sin", q_inout.scalar_type(), 4);
    TORCH_CHECK(
        q_inout.scalar_type() == at::kHalf ||
            q_inout.scalar_type() == at::kBFloat16,
        "q_inout must be float16 or bfloat16.");
    TORCH_CHECK(
        q_inout.device() == cos.device() && q_inout.device() == sin.device(),
        "q_inout, cos and sin must be on the same NPU.");

    const int64_t num_tokens = q_inout.size(0);
    const int64_t n_head = q_inout.size(1);
    const int64_t head_dim = q_inout.size(2);
    TORCH_CHECK(
        num_tokens > 0 && n_head > 0 && head_dim > 0,
        "q_inout must be non-empty.");
    TORCH_CHECK(
        cos.size(0) == num_tokens && sin.size(0) == num_tokens,
        "cos/sin token dimension mismatch.");
    TORCH_CHECK(
        cos.numel() >= num_tokens * rope_dim &&
            sin.numel() >= num_tokens * rope_dim,
        "cos/sin storage is smaller than num_tokens * rope_dim.");
    TORCH_CHECK(
        rope_dim > 0 && rope_dim <= head_dim && rope_dim % 16 == 0,
        "rope_dim must be positive, no larger than head_dim, and aligned to 16.");
    TORCH_CHECK(
        head_dim <= 256 && head_dim % 16 == 0 && n_head <= 128,
        "query RoPE requires head_dim <= 256 aligned to 16 and n_head <= 128.");
    const AscendType type =
        q_inout.scalar_type() == at::kHalf ? AscendType::FP16
                                           : AscendType::BF16;
    // Reinterpreted as two adjacent 16-bit values: [-1, +1].
    const uint32_t sign_pair_bits =
        type == AscendType::FP16 ? 0x3c00bc00U : 0x3f80bf80U;
    const int64_t rows = num_tokens * n_head;
    const uint32_t block_dim = static_cast<uint32_t>(
        std::min<int64_t>(48, std::max<int64_t>(1, rows)));

    c10::OptionalDeviceGuard device_guard(q_inout.device());
    void* q_ptr = q_inout.data_ptr();
    void* cos_ptr = cos.data_ptr();
    void* sin_ptr = sin.data_ptr();
    // Keep the custom BMM -> in-place RoPE -> native LIDU chain on the task
    // queue stream. Mixing stream(true) callbacks with RunOpApiV2 here can
    // let the consumer observe the unrotated BMM result.
    aclrtStream stream = c10_npu::getCurrentNPUStream().stream(false);
    UseStreamResIfNeeded(stream, true);
    auto tensor_keepalive = std::make_tuple(q_inout, cos, sin);
    auto acl_call =
        [type, stream, q_ptr, cos_ptr, sin_ptr, num_tokens, n_head,
         head_dim, rope_dim, sign_pair_bits, block_dim,
         tensor_keepalive]() -> int {
            (void)tensor_keepalive;
            UseStreamResIfNeeded(stream, false);
            dsa_indexer_query_rope_impl(
                type,
                stream,
                q_ptr,
                cos_ptr,
                sin_ptr,
                static_cast<uint32_t>(num_tokens),
                static_cast<uint32_t>(n_head),
                static_cast<uint32_t>(head_dim),
                static_cast<uint32_t>(rope_dim),
                sign_pair_bits,
                block_dim);
            return ACL_SUCCESS;
        };
    at_npu::native::OpCommand::RunOpApiV2(
        "dsa_indexer_query_rope_inplace", acl_call);
}

} // namespace vllm_ascend

#endif
