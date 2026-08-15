#include <tuple>

#include "ops_common.h"

namespace nanovllm_dsa_a5_impl {

namespace {
constexpr int64_t kMtpMaxQueriesPerRequest = 4;
constexpr int64_t kMtpMinQueriesPerRequest = 2;
constexpr int64_t kMtpUnionCapacity =
    kSparseCount * kMtpMaxQueriesPerRequest;
}

void CheckFusedLiManageMtpC8Common(
    const at::Tensor& query,
    const at::Tensor& key,
    const at::Tensor& weights,
    const at::Tensor& query_dequant_scale,
    const at::Tensor& key_dequant_scale,
    const at::Tensor& actual_seq_lengths_query,
    const at::Tensor& req_pool_entries,
    const at::Tensor& cache_slots_pool,
    const at::Tensor& cache_tokens,
    const at::Tensor& candidate_lens,
    const at::Tensor& block_table) {
  TORCH_CHECK(
      query.dim() == 3 && query.size(0) > 0 &&
          (query.size(1) == 32 || query.size(1) == 64) &&
          query.size(2) == kIndexerDim &&
          query.scalar_type() == at::ScalarType::Float8_e4m3fn,
      "C8 MTP LIDU query must be float8_e4m3fn [T,32|64,128].");
  const int64_t packed_queries = query.size(0);
  const int64_t heads = query.size(1);
  TORCH_CHECK(
      actual_seq_lengths_query.dim() == 1 &&
          actual_seq_lengths_query.size(0) > 0,
      "C8 MTP LIDU actual_seq_lengths_query must be cumulative int32 [B].");
  const int64_t batch = actual_seq_lengths_query.size(0);
  TORCH_CHECK(
      packed_queries >= batch * kMtpMinQueriesPerRequest &&
          packed_queries <= batch * kMtpMaxQueriesPerRequest,
      "C8 MTP LIDU packed T must be in [2*B,4*B].");
  TORCH_CHECK(
      key.dim() == 4 && key.size(0) > 0 &&
          key.size(1) == kBlockSize && key.size(2) == 1 &&
          key.size(3) == kIndexerDim &&
          key.scalar_type() == at::ScalarType::Float8_e4m3fn,
      "C8 MTP LIDU key must be float8_e4m3fn [blocks,128,1,128].");
  TORCH_CHECK(
      weights.dim() == 2 && weights.size(0) == packed_queries &&
          weights.size(1) == heads &&
          weights.scalar_type() == at::kBFloat16 &&
          query_dequant_scale.dim() == 2 &&
          query_dequant_scale.size(0) == packed_queries &&
          query_dequant_scale.size(1) == heads &&
          query_dequant_scale.scalar_type() == at::kFloat,
      "C8 MTP LIDU weights/query scales must be BF16/FP32 [T,N].");
  TORCH_CHECK(
      key_dequant_scale.dim() == 3 &&
          key_dequant_scale.size(0) == key.size(0) &&
          key_dequant_scale.size(1) == kBlockSize &&
          key_dequant_scale.size(2) == 1 &&
          key_dequant_scale.scalar_type() == at::kFloat,
      "C8 MTP LIDU key_dequant_scale must be FP32 [blocks,128,1].");
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
  TORCH_CHECK(
      block_table.dim() == 2 && block_table.size(0) == batch &&
          block_table.size(1) > 0 &&
          block_table.size(1) * kBlockSize == cache_slots_pool.size(1),
      "C8 MTP LIDU block-table capacity must equal cache_slots_pool.shape[1].");
  for (const at::Tensor* tensor : {
           &actual_seq_lengths_query, &req_pool_entries,
           &cache_slots_pool, &cache_tokens, &candidate_lens,
           &block_table}) {
    TORCH_CHECK(
        tensor->scalar_type() == at::kInt,
        "C8 MTP LIDU metadata/state tensors must be int32.");
  }
  CheckOneDeviceAndContiguous(
      query,
      {&query, &key, &weights, &query_dequant_scale,
       &key_dequant_scale, &actual_seq_lengths_query,
       &req_pool_entries, &cache_slots_pool, &cache_tokens,
       &candidate_lens, &block_table},
      "C8 MTP LIDU");
}

void CheckFusedLiManageMtpC8Outputs(
    const at::Tensor& query,
    const at::Tensor& actual_seq_lengths_query,
    const at::Tensor& topk_destination_slots,
    const at::Tensor& miss_source_ids,
    const at::Tensor& miss_destination_slots,
    const at::Tensor& miss_counts) {
  const int64_t batch = actual_seq_lengths_query.size(0);
  TORCH_CHECK(
      topk_destination_slots.dim() == 3 &&
          topk_destination_slots.size(0) == query.size(0) &&
          topk_destination_slots.size(1) == 1 &&
          topk_destination_slots.size(2) == kSparseCount,
      "C8 MTP LIDU topk_destination_slots must be int32 [T,1,2048].");
  TORCH_CHECK(
      miss_source_ids.dim() == 2 && miss_source_ids.size(0) == batch &&
          miss_source_ids.size(1) == kMtpUnionCapacity &&
          miss_destination_slots.sizes() == miss_source_ids.sizes() &&
          miss_counts.dim() == 1 && miss_counts.size(0) == batch,
      "C8 MTP LIDU miss buffers must be int32 [B,8192] and [B].");
  for (const at::Tensor* tensor : {
           &topk_destination_slots, &miss_source_ids,
           &miss_destination_slots, &miss_counts}) {
    TORCH_CHECK(
        tensor->scalar_type() == at::kInt &&
            tensor->device() == query.device() && tensor->is_contiguous(),
        "C8 MTP LIDU outputs must be contiguous int32 on the query NPU.");
  }
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
FusedLiManageMtpC8OutNpu(
    const at::Tensor& query,
    const at::Tensor& key,
    const at::Tensor& weights,
    const at::Tensor& query_dequant_scale,
    const at::Tensor& key_dequant_scale,
    const at::Tensor& actual_seq_lengths_query,
    const at::Tensor& req_pool_entries,
    at::Tensor cache_slots_pool,
    const at::Tensor& cache_tokens,
    const at::Tensor& candidate_lens,
    const at::Tensor& block_table,
    at::Tensor topk_destination_slots,
    at::Tensor miss_source_ids,
    at::Tensor miss_destination_slots,
    at::Tensor miss_counts) {
  CheckFusedLiManageMtpC8Common(
      query, key, weights, query_dequant_scale, key_dequant_scale,
      actual_seq_lengths_query, req_pool_entries, cache_slots_pool,
      cache_tokens, candidate_lens, block_table);
  CheckFusedLiManageMtpC8Outputs(
      query, actual_seq_lengths_query, topk_destination_slots,
      miss_source_ids, miss_destination_slots, miss_counts);
  auto keepalive = std::make_tuple(
      query, key, weights, query_dequant_scale, key_dequant_scale,
      actual_seq_lengths_query, req_pool_entries, cache_slots_pool,
      cache_tokens, candidate_lens, block_table, topk_destination_slots,
      miss_source_ids, miss_destination_slots, miss_counts);
  EXEC_NPU_CMD_ORDERED(
      aclnnA5FusedLiManageMtpC8,
      keepalive,
      query,
      key,
      weights,
      query_dequant_scale,
      key_dequant_scale,
      actual_seq_lengths_query,
      req_pool_entries,
      cache_slots_pool,
      cache_tokens,
      candidate_lens,
      block_table,
      topk_destination_slots,
      miss_source_ids,
      miss_destination_slots,
      miss_counts,
      cache_slots_pool);
  return std::make_tuple(
      topk_destination_slots, miss_source_ids, miss_destination_slots,
      miss_counts, cache_slots_pool);
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
FusedLiManageMtpC8Npu(
    const at::Tensor& query,
    const at::Tensor& key,
    const at::Tensor& weights,
    const at::Tensor& query_dequant_scale,
    const at::Tensor& key_dequant_scale,
    const at::Tensor& actual_seq_lengths_query,
    const at::Tensor& req_pool_entries,
    at::Tensor cache_slots_pool,
    const at::Tensor& cache_tokens,
    const at::Tensor& candidate_lens,
    const at::Tensor& block_table) {
  const int64_t batch = actual_seq_lengths_query.size(0);
  auto options = query.options().dtype(at::kInt);
  auto topk_destination_slots =
      at::empty({query.size(0), 1, kSparseCount}, options);
  auto miss_source_ids =
      at::empty({batch, kMtpUnionCapacity}, options);
  auto miss_destination_slots = at::empty_like(miss_source_ids);
  auto miss_counts = at::empty({batch}, options);
  return FusedLiManageMtpC8OutNpu(
      query, key, weights, query_dequant_scale, key_dequant_scale,
      actual_seq_lengths_query, req_pool_entries, cache_slots_pool,
      cache_tokens, candidate_lens, block_table,
      topk_destination_slots, miss_source_ids,
      miss_destination_slots, miss_counts);
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
FusedLiManageMtpC8Meta(
    const at::Tensor& query,
    const at::Tensor&,
    const at::Tensor&,
    const at::Tensor&,
    const at::Tensor&,
    const at::Tensor& actual_seq_lengths_query,
    const at::Tensor&,
    at::Tensor cache_slots_pool,
    const at::Tensor&,
    const at::Tensor&,
    const at::Tensor&) {
  auto options = query.options().dtype(at::kInt);
  const int64_t batch = actual_seq_lengths_query.size(0);
  return std::make_tuple(
      at::empty({query.size(0), 1, kSparseCount}, options),
      at::empty({batch, kMtpUnionCapacity}, options),
      at::empty({batch, kMtpUnionCapacity}, options),
      at::empty({batch}, options),
      cache_slots_pool);
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
FusedLiManageMtpC8OutMeta(
    const at::Tensor&,
    const at::Tensor&,
    const at::Tensor&,
    const at::Tensor&,
    const at::Tensor&,
    const at::Tensor&,
    const at::Tensor&,
    at::Tensor cache_slots_pool,
    const at::Tensor&,
    const at::Tensor&,
    const at::Tensor&,
    at::Tensor topk_destination_slots,
    at::Tensor miss_source_ids,
    at::Tensor miss_destination_slots,
    at::Tensor miss_counts) {
  return std::make_tuple(
      topk_destination_slots, miss_source_ids, miss_destination_slots,
      miss_counts, cache_slots_pool);
}

}  // namespace nanovllm_dsa_a5_impl

TORCH_LIBRARY_IMPL(nanovllm_dsa, PrivateUse1, m) {
  m.impl(
      "fused_li_manage_mtp_c8",
      &nanovllm_dsa_a5_impl::FusedLiManageMtpC8Npu);
  m.impl(
      "fused_li_manage_mtp_c8_out",
      &nanovllm_dsa_a5_impl::FusedLiManageMtpC8OutNpu);
}

TORCH_LIBRARY_IMPL(nanovllm_dsa, Meta, m) {
  m.impl(
      "fused_li_manage_mtp_c8",
      &nanovllm_dsa_a5_impl::FusedLiManageMtpC8Meta);
  m.impl(
      "fused_li_manage_mtp_c8_out",
      &nanovllm_dsa_a5_impl::FusedLiManageMtpC8OutMeta);
}
