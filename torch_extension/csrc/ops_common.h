#pragma once

#include <initializer_list>

#include <torch/extension.h>
#include <torch/library.h>

#include "op_api_common.h"

namespace nanovllm_dsa_a5_impl {

constexpr int64_t kBlockSize = 128;
constexpr int64_t kKpeDim = 64;
constexpr int64_t kCkvDim = 512;
constexpr int64_t kIndexerDim = 128;
constexpr int64_t kSparseCount = 2048;
constexpr int64_t kPackedKvDim = 656;
constexpr int64_t kMaxSourceCapacity = 1 << 18;

inline void CheckOneDeviceAndContiguous(
    const at::Tensor& reference,
    std::initializer_list<const at::Tensor*> tensors,
    const char* op_name) {
  TORCH_CHECK(reference.device().is_privateuseone(), op_name, " inputs must be on NPU.");
  for (const at::Tensor* tensor : tensors) {
    TORCH_CHECK(
        tensor->device() == reference.device() && tensor->is_contiguous(),
        op_name, " inputs must be contiguous tensors on one NPU.");
  }
}

inline void CheckLiduOutputs(
    const at::Tensor& reference,
    const at::Tensor& source_ids,
    const at::Tensor& destination_slots,
    const at::Tensor& miss_counts) {
  const int64_t batch = reference.size(0);
  TORCH_CHECK(
      source_ids.dim() == 3 && source_ids.size(0) == batch &&
          source_ids.size(1) == 1 && source_ids.size(2) == kSparseCount &&
          destination_slots.sizes() == source_ids.sizes() &&
          miss_counts.dim() == 1 && miss_counts.size(0) == batch,
      "LIDU out buffers must be source/destination [B,1,2048], miss_counts [B].");
  for (const at::Tensor* tensor : {&source_ids, &destination_slots, &miss_counts}) {
    TORCH_CHECK(
        tensor->scalar_type() == at::kInt &&
            tensor->device() == reference.device() && tensor->is_contiguous(),
        "LIDU out buffers must be contiguous int32 tensors on the query NPU.");
  }
}

}  // namespace nanovllm_dsa_a5_impl
