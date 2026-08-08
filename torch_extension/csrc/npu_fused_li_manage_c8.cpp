#include <tuple>

#include "ops_common.h"

namespace nanovllm_dsa_a5_impl {

void CheckFusedLiManageC8CacheUpdateCommon(
    const at::Tensor& topk_indices,
    const at::Tensor& req_pool_entries,
    const at::Tensor& cache_slots_pool,
    const at::Tensor& cache_tokens,
    const at::Tensor& candidate_lens) {
  TORCH_CHECK(
      topk_indices.dim() == 3 && topk_indices.size(0) > 0 &&
          topk_indices.size(1) == 1 &&
          topk_indices.size(2) == kSparseCount,
      "C8 LIDU topk_indices must be int32 [B,1,2048].");
  const int64_t batch = topk_indices.size(0);
  TORCH_CHECK(
      req_pool_entries.dim() == 1 && req_pool_entries.size(0) == batch &&
          cache_tokens.dim() == 1 && cache_tokens.size(0) == batch &&
          candidate_lens.dim() == 1 && candidate_lens.size(0) == batch,
      "C8 LIDU metadata must be int32 [B].");
  TORCH_CHECK(
      cache_slots_pool.dim() == 2 && cache_slots_pool.size(0) > 0 &&
          cache_slots_pool.size(1) > 0 &&
          cache_slots_pool.size(1) <= kMaxSourceCapacity,
      "C8 LIDU cache_slots_pool must be [pool_size,capacity], capacity <= 2^18.");
  for (const at::Tensor* tensor :
       {&topk_indices, &req_pool_entries, &cache_slots_pool,
        &cache_tokens, &candidate_lens}) {
    TORCH_CHECK(
        tensor->scalar_type() == at::kInt,
        "C8 LIDU state-update tensors must be int32.");
  }
  CheckOneDeviceAndContiguous(
      topk_indices,
      {&topk_indices, &req_pool_entries, &cache_slots_pool,
       &cache_tokens, &candidate_lens},
      "C8 LIDU state update");
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor>
FusedLiManageC8CacheUpdateOutNpu(
    const at::Tensor& topk_indices,
    const at::Tensor& req_pool_entries,
    at::Tensor cache_slots_pool,
    const at::Tensor& cache_tokens,
    const at::Tensor& candidate_lens,
    at::Tensor source_ids,
    at::Tensor destination_slots,
    at::Tensor miss_counts) {
  CheckFusedLiManageC8CacheUpdateCommon(
      topk_indices, req_pool_entries, cache_slots_pool,
      cache_tokens, candidate_lens);
  CheckLiduOutputs(
      topk_indices, source_ids, destination_slots, miss_counts);
  auto keepalive = std::make_tuple(
      topk_indices, req_pool_entries, cache_slots_pool,
      cache_tokens, candidate_lens, source_ids,
      destination_slots, miss_counts);
  EXEC_NPU_CMD_ORDERED(
      aclnnA5FusedLiManageC8CacheUpdate,
      keepalive,
      topk_indices,
      req_pool_entries,
      cache_slots_pool,
      cache_tokens,
      candidate_lens,
      source_ids,
      destination_slots,
      miss_counts,
      cache_slots_pool);
  return std::make_tuple(
      source_ids, destination_slots, miss_counts, cache_slots_pool);
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor>
FusedLiManageC8CacheUpdateNpu(
    const at::Tensor& topk_indices,
    const at::Tensor& req_pool_entries,
    at::Tensor cache_slots_pool,
    const at::Tensor& cache_tokens,
    const at::Tensor& candidate_lens) {
  auto options = topk_indices.options().dtype(at::kInt);
  auto source_ids = at::empty(topk_indices.sizes(), options);
  auto destination_slots = at::empty(topk_indices.sizes(), options);
  auto miss_counts = at::empty({topk_indices.size(0)}, options);
  return FusedLiManageC8CacheUpdateOutNpu(
      topk_indices, req_pool_entries, cache_slots_pool,
      cache_tokens, candidate_lens,
      source_ids, destination_slots, miss_counts);
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor>
FusedLiManageC8CacheUpdateMeta(
    const at::Tensor& topk_indices,
    const at::Tensor&,
    at::Tensor cache_slots_pool,
    const at::Tensor&,
    const at::Tensor&) {
  auto options = topk_indices.options().dtype(at::kInt);
  return std::make_tuple(
      at::empty(topk_indices.sizes(), options),
      at::empty(topk_indices.sizes(), options),
      at::empty({topk_indices.size(0)}, options),
      cache_slots_pool);
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor>
FusedLiManageC8CacheUpdateOutMeta(
    const at::Tensor&,
    const at::Tensor&,
    at::Tensor cache_slots_pool,
    const at::Tensor&,
    const at::Tensor&,
    at::Tensor source_ids,
    at::Tensor destination_slots,
    at::Tensor miss_counts) {
  return std::make_tuple(
      source_ids, destination_slots, miss_counts, cache_slots_pool);
}

}  // namespace nanovllm_dsa_a5_impl

TORCH_LIBRARY_IMPL(nanovllm_dsa, PrivateUse1, m) {
  m.impl("_fused_li_manage_c8_cache_update", &nanovllm_dsa_a5_impl::FusedLiManageC8CacheUpdateNpu);
  m.impl("_fused_li_manage_c8_cache_update_out", &nanovllm_dsa_a5_impl::FusedLiManageC8CacheUpdateOutNpu);
}

TORCH_LIBRARY_IMPL(nanovllm_dsa, Meta, m) {
  m.impl("_fused_li_manage_c8_cache_update", &nanovllm_dsa_a5_impl::FusedLiManageC8CacheUpdateMeta);
  m.impl("_fused_li_manage_c8_cache_update_out", &nanovllm_dsa_a5_impl::FusedLiManageC8CacheUpdateOutMeta);
}
