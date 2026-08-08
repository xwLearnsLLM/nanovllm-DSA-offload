#include <tuple>

#include "ops_common.h"

namespace nanovllm_dsa_a5_impl {

namespace {
constexpr int64_t kMtpCacheTokens = 8192;
constexpr int64_t kMtpMaxQueriesPerRequest = 4;
}

void CheckFusedLiManageMtpInputs(
    const at::Tensor& query,
    const at::Tensor& key,
    const at::Tensor& weights,
    const at::Tensor& cache_slots,
    const at::Tensor& actual_seq_lengths_query,
    const at::Tensor& actual_seq_lengths_key,
    const at::Tensor& block_table) {
  TORCH_CHECK(
      query.dim() == 3 && query.size(0) > 0 &&
          (query.size(1) == 32 || query.size(1) == 64) &&
          query.size(2) == kIndexerDim,
      "MTP LIDU query must be packed [T,32|64,128].");
  TORCH_CHECK(
      key.dim() == 4 && key.size(0) > 0 && key.size(1) > 0 &&
          key.size(2) == 1 && key.size(3) == kIndexerDim,
      "MTP LIDU key must be [blocks,block_size,1,128].");
  TORCH_CHECK(
      weights.dim() == 2 && weights.size(0) == query.size(0) &&
          weights.size(1) == query.size(1),
      "MTP LIDU weights must be [T,32|64] and match query.");
  TORCH_CHECK(
      cache_slots.dim() == 2 && cache_slots.size(0) > 0 &&
          cache_slots.size(1) == kMaxSourceCapacity,
      "MTP LIDU cache_slots must be [B,262144].");
  const int64_t batch = cache_slots.size(0);
  TORCH_CHECK(
      query.size(0) >= batch &&
          query.size(0) <= batch * kMtpMaxQueriesPerRequest,
      "MTP LIDU packed T must be in [B,4*B].");
  TORCH_CHECK(
      actual_seq_lengths_query.dim() == 1 &&
          actual_seq_lengths_query.size(0) == batch &&
          actual_seq_lengths_key.dim() == 1 &&
          actual_seq_lengths_key.size(0) == batch &&
          block_table.dim() == 2 && block_table.size(0) == batch &&
          block_table.size(1) > 0,
      "MTP LIDU request metadata must be actual_q/actual_k [B] and block_table [B,max_blocks].");
  TORCH_CHECK(
      query.scalar_type() == at::kBFloat16 || query.scalar_type() == at::kHalf,
      "MTP LIDU floating tensors must be bf16 or fp16.");
  TORCH_CHECK(
      key.scalar_type() == query.scalar_type() &&
          weights.scalar_type() == query.scalar_type(),
      "MTP LIDU query/key/weights dtypes must match.");
  for (const at::Tensor* tensor :
       {&cache_slots, &actual_seq_lengths_query,
        &actual_seq_lengths_key, &block_table}) {
    TORCH_CHECK(tensor->scalar_type() == at::kInt,
                "MTP LIDU metadata and cache_slots must be int32.");
  }
  CheckOneDeviceAndContiguous(
      query,
      {&query, &key, &weights, &cache_slots,
       &actual_seq_lengths_query, &actual_seq_lengths_key, &block_table},
      "MTP LIDU");
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
FusedLiManageMtpNpu(
    const at::Tensor& query,
    const at::Tensor& key,
    const at::Tensor& weights,
    at::Tensor cache_slots,
    const at::Tensor& actual_seq_lengths_query,
    const at::Tensor& actual_seq_lengths_key,
    const at::Tensor& block_table) {
  CheckFusedLiManageMtpInputs(
      query, key, weights, cache_slots, actual_seq_lengths_query,
      actual_seq_lengths_key, block_table);
  auto options = query.options().dtype(at::kInt);
  auto topk_index = at::empty({query.size(0), 1, kSparseCount}, options);
  auto topk_slots = at::empty_like(topk_index);
  auto miss_index = at::empty({cache_slots.size(0), kMtpCacheTokens}, options);
  auto miss_slots = at::empty_like(miss_index);
  auto miss_count = at::empty({cache_slots.size(0)}, options);
  auto keepalive = std::make_tuple(
      query, key, weights, cache_slots, actual_seq_lengths_query,
      actual_seq_lengths_key, block_table, topk_index, topk_slots,
      miss_index, miss_slots, miss_count);
  EXEC_NPU_CMD_ORDERED(
      aclnnA5FusedLiManageMtp,
      keepalive,
      query,
      key,
      weights,
      cache_slots,
      actual_seq_lengths_query,
      actual_seq_lengths_key,
      block_table,
      topk_index,
      topk_slots,
      miss_index,
      miss_slots,
      miss_count);
  return std::make_tuple(
      topk_index, topk_slots, miss_index, miss_slots, miss_count);
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
FusedLiManageMtpMeta(
    const at::Tensor& query,
    const at::Tensor&,
    const at::Tensor&,
    const at::Tensor& cache_slots,
    const at::Tensor&,
    const at::Tensor&,
    const at::Tensor&) {
  auto options = query.options().dtype(at::kInt);
  return std::make_tuple(
      at::empty({query.size(0), 1, kSparseCount}, options),
      at::empty({query.size(0), 1, kSparseCount}, options),
      at::empty({cache_slots.size(0), kMtpCacheTokens}, options),
      at::empty({cache_slots.size(0), kMtpCacheTokens}, options),
      at::empty({cache_slots.size(0)}, options));
}

}  // namespace nanovllm_dsa_a5_impl

TORCH_LIBRARY_IMPL(nanovllm_dsa, PrivateUse1, m) {
  m.impl("fused_li_manage_mtp", &nanovllm_dsa_a5_impl::FusedLiManageMtpNpu);
}

TORCH_LIBRARY_IMPL(nanovllm_dsa, Meta, m) {
  m.impl("fused_li_manage_mtp", &nanovllm_dsa_a5_impl::FusedLiManageMtpMeta);
}
