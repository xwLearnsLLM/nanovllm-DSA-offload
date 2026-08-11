#include <tuple>

#include "ops_common.h"

namespace nanovllm_dsa_a5_impl {

namespace {
constexpr int64_t kMtpMaxQueriesPerRequest = 4;
constexpr int64_t kMtpMinQueriesPerRequest = 2;
constexpr int64_t kMtpUnionCapacity =
    kSparseCount * kMtpMaxQueriesPerRequest;
}

void CheckFusedLiManageMtpC8CacheUpdateCommon(
    const at::Tensor& topk_indices,
    const at::Tensor& actual_seq_lengths_query,
    const at::Tensor& req_pool_entries,
    const at::Tensor& cache_slots_pool,
    const at::Tensor& cache_tokens,
    const at::Tensor& candidate_lens) {
  TORCH_CHECK(
      topk_indices.dim() == 3 && topk_indices.size(0) > 0 &&
          topk_indices.size(1) == 1 &&
          topk_indices.size(2) == kSparseCount,
      "C8 MTP LIDU topk_indices must be int32 [T,1,2048].");
  TORCH_CHECK(
      actual_seq_lengths_query.dim() == 1 &&
          actual_seq_lengths_query.size(0) > 0,
      "C8 MTP LIDU actual_seq_lengths_query must be cumulative int32 [B].");
  const int64_t batch = actual_seq_lengths_query.size(0);
  const int64_t packed_queries = topk_indices.size(0);
  TORCH_CHECK(
      packed_queries >= batch * kMtpMinQueriesPerRequest &&
          packed_queries <= batch * kMtpMaxQueriesPerRequest,
      "C8 MTP LIDU packed T must be in [2*B,4*B].");
  TORCH_CHECK(
      req_pool_entries.dim() == 1 && req_pool_entries.size(0) == batch &&
          cache_tokens.dim() == 1 && cache_tokens.size(0) == batch &&
          candidate_lens.dim() == 1 && candidate_lens.size(0) == batch,
      "C8 MTP LIDU request metadata must be int32 [B].");
  TORCH_CHECK(
      cache_slots_pool.dim() == 2 && cache_slots_pool.size(0) > 0 &&
          cache_slots_pool.size(1) > 0 &&
          cache_slots_pool.size(1) <= kMaxSourceCapacity,
      "C8 MTP LIDU cache_slots_pool must be [pool_size,capacity], "
      "capacity <= 2^18.");
  for (const at::Tensor* tensor :
       {&topk_indices, &actual_seq_lengths_query, &req_pool_entries,
        &cache_slots_pool, &cache_tokens, &candidate_lens}) {
    TORCH_CHECK(
        tensor->scalar_type() == at::kInt,
        "C8 MTP LIDU state-update tensors must be int32.");
  }
  CheckOneDeviceAndContiguous(
      topk_indices,
      {&topk_indices, &actual_seq_lengths_query, &req_pool_entries,
       &cache_slots_pool, &cache_tokens, &candidate_lens},
      "C8 MTP LIDU state update");
}

void CheckFusedLiManageMtpC8Outputs(
    const at::Tensor& topk_indices,
    const at::Tensor& actual_seq_lengths_query,
    const at::Tensor& topk_destination_slots,
    const at::Tensor& miss_source_ids,
    const at::Tensor& miss_destination_slots,
    const at::Tensor& miss_counts) {
  const int64_t batch = actual_seq_lengths_query.size(0);
  TORCH_CHECK(
      topk_destination_slots.sizes() == topk_indices.sizes(),
      "C8 MTP LIDU topk_destination_slots must be int32 [T,1,2048].");
  TORCH_CHECK(
      miss_source_ids.dim() == 2 && miss_source_ids.size(0) == batch &&
          miss_source_ids.size(1) == kMtpUnionCapacity &&
          miss_destination_slots.sizes() == miss_source_ids.sizes() &&
          miss_counts.dim() == 1 && miss_counts.size(0) == batch,
      "C8 MTP LIDU miss buffers must be int32 [B,8192] and miss_counts [B].");
  for (const at::Tensor* tensor :
       {&topk_destination_slots, &miss_source_ids,
        &miss_destination_slots, &miss_counts}) {
    TORCH_CHECK(
        tensor->scalar_type() == at::kInt &&
            tensor->device() == topk_indices.device() &&
            tensor->is_contiguous(),
        "C8 MTP LIDU out buffers must be contiguous int32 tensors on one NPU.");
  }
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
FusedLiManageMtpC8CacheUpdateOutNpu(
    const at::Tensor& topk_indices,
    const at::Tensor& actual_seq_lengths_query,
    const at::Tensor& req_pool_entries,
    at::Tensor cache_slots_pool,
    const at::Tensor& cache_tokens,
    const at::Tensor& candidate_lens,
    at::Tensor topk_destination_slots,
    at::Tensor miss_source_ids,
    at::Tensor miss_destination_slots,
    at::Tensor miss_counts) {
  CheckFusedLiManageMtpC8CacheUpdateCommon(
      topk_indices, actual_seq_lengths_query, req_pool_entries,
      cache_slots_pool, cache_tokens, candidate_lens);
  CheckFusedLiManageMtpC8Outputs(
      topk_indices, actual_seq_lengths_query, topk_destination_slots,
      miss_source_ids, miss_destination_slots, miss_counts);
  auto keepalive = std::make_tuple(
      topk_indices, actual_seq_lengths_query, req_pool_entries,
      cache_slots_pool, cache_tokens, candidate_lens,
      topk_destination_slots, miss_source_ids,
      miss_destination_slots, miss_counts);
  EXEC_NPU_CMD_ORDERED(
      aclnnA5FusedLiManageMtpC8CacheUpdate,
      keepalive,
      topk_indices,
      actual_seq_lengths_query,
      req_pool_entries,
      cache_slots_pool,
      cache_tokens,
      candidate_lens,
      topk_destination_slots,
      miss_source_ids,
      miss_destination_slots,
      miss_counts,
      cache_slots_pool);
  return std::make_tuple(
      topk_destination_slots, miss_source_ids,
      miss_destination_slots, miss_counts, cache_slots_pool);
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
FusedLiManageMtpC8CacheUpdateNpu(
    const at::Tensor& topk_indices,
    const at::Tensor& actual_seq_lengths_query,
    const at::Tensor& req_pool_entries,
    at::Tensor cache_slots_pool,
    const at::Tensor& cache_tokens,
    const at::Tensor& candidate_lens) {
  const int64_t batch = actual_seq_lengths_query.size(0);
  auto options = topk_indices.options().dtype(at::kInt);
  auto topk_destination_slots = at::empty(topk_indices.sizes(), options);
  auto miss_source_ids =
      at::empty({batch, kMtpUnionCapacity}, options);
  auto miss_destination_slots = at::empty_like(miss_source_ids);
  auto miss_counts = at::empty({batch}, options);
  return FusedLiManageMtpC8CacheUpdateOutNpu(
      topk_indices, actual_seq_lengths_query, req_pool_entries,
      cache_slots_pool, cache_tokens, candidate_lens,
      topk_destination_slots, miss_source_ids,
      miss_destination_slots, miss_counts);
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
FusedLiManageMtpC8CacheUpdateMeta(
    const at::Tensor& topk_indices,
    const at::Tensor& actual_seq_lengths_query,
    const at::Tensor&,
    at::Tensor cache_slots_pool,
    const at::Tensor&,
    const at::Tensor&) {
  auto options = topk_indices.options().dtype(at::kInt);
  const int64_t batch = actual_seq_lengths_query.size(0);
  return std::make_tuple(
      at::empty(topk_indices.sizes(), options),
      at::empty({batch, kMtpUnionCapacity}, options),
      at::empty({batch, kMtpUnionCapacity}, options),
      at::empty({batch}, options),
      cache_slots_pool);
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
FusedLiManageMtpC8CacheUpdateOutMeta(
    const at::Tensor&,
    const at::Tensor&,
    const at::Tensor&,
    at::Tensor cache_slots_pool,
    const at::Tensor&,
    const at::Tensor&,
    at::Tensor topk_destination_slots,
    at::Tensor miss_source_ids,
    at::Tensor miss_destination_slots,
    at::Tensor miss_counts) {
  return std::make_tuple(
      topk_destination_slots, miss_source_ids,
      miss_destination_slots, miss_counts, cache_slots_pool);
}

}  // namespace nanovllm_dsa_a5_impl

TORCH_LIBRARY_IMPL(nanovllm_dsa, PrivateUse1, m) {
  m.impl(
      "_fused_li_manage_mtp_c8_cache_update",
      &nanovllm_dsa_a5_impl::FusedLiManageMtpC8CacheUpdateNpu);
  m.impl(
      "_fused_li_manage_mtp_c8_cache_update_out",
      &nanovllm_dsa_a5_impl::FusedLiManageMtpC8CacheUpdateOutNpu);
}

TORCH_LIBRARY_IMPL(nanovllm_dsa, Meta, m) {
  m.impl(
      "_fused_li_manage_mtp_c8_cache_update",
      &nanovllm_dsa_a5_impl::FusedLiManageMtpC8CacheUpdateMeta);
  m.impl(
      "_fused_li_manage_mtp_c8_cache_update_out",
      &nanovllm_dsa_a5_impl::FusedLiManageMtpC8CacheUpdateOutMeta);
}
