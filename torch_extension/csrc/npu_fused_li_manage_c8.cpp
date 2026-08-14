#include <tuple>

#include "ops_common.h"

namespace nanovllm_dsa_a5_impl {

void CheckFusedLiManageC8Common(
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
      "C8 LIDU query must be float8_e4m3fn [B,32|64,128].");
  const int64_t batch = query.size(0);
  const int64_t heads = query.size(1);
  TORCH_CHECK(
      key.dim() == 4 && key.size(0) > 0 &&
          key.size(1) == kBlockSize && key.size(2) == 1 &&
          key.size(3) == kIndexerDim &&
          key.scalar_type() == at::ScalarType::Float8_e4m3fn,
      "C8 LIDU key must be float8_e4m3fn [blocks,128,1,128].");
  TORCH_CHECK(
      weights.dim() == 2 && weights.size(0) == batch &&
          weights.size(1) == heads &&
          weights.scalar_type() == at::kBFloat16 &&
          query_dequant_scale.dim() == 2 &&
          query_dequant_scale.size(0) == batch &&
          query_dequant_scale.size(1) == heads &&
          query_dequant_scale.scalar_type() == at::kFloat,
      "C8 LIDU weights/scales must be BF16/FP32 [B,N].");
  TORCH_CHECK(
      key_dequant_scale.dim() == 3 &&
          key_dequant_scale.size(0) == key.size(0) &&
          key_dequant_scale.size(1) == kBlockSize &&
          key_dequant_scale.size(2) == 1 &&
          key_dequant_scale.scalar_type() == at::kFloat,
      "C8 LIDU key_dequant_scale must be FP32 [blocks,128,1].");
  TORCH_CHECK(
      actual_seq_lengths_query.dim() == 1 &&
          actual_seq_lengths_query.size(0) == batch &&
          req_pool_entries.dim() == 1 &&
          req_pool_entries.size(0) == batch &&
          cache_tokens.dim() == 1 && cache_tokens.size(0) == batch &&
          candidate_lens.dim() == 1 && candidate_lens.size(0) == batch,
      "C8 LIDU request metadata must be int32[B].");
  TORCH_CHECK(
      cache_slots_pool.dim() == 2 && cache_slots_pool.size(0) > 0 &&
          cache_slots_pool.size(1) > 0 &&
          cache_slots_pool.size(1) <= kMaxSourceCapacity,
      "C8 LIDU cache_slots_pool must be [pool_size,capacity], capacity <= 2^18.");
  TORCH_CHECK(
      block_table.dim() == 2 && block_table.size(0) == batch &&
          block_table.size(1) > 0 &&
          block_table.size(1) * kBlockSize == cache_slots_pool.size(1),
      "C8 LIDU block-table capacity must equal cache_slots_pool.shape[1].");
  for (const at::Tensor* tensor : {
           &actual_seq_lengths_query, &req_pool_entries,
           &cache_slots_pool, &cache_tokens, &candidate_lens,
           &block_table}) {
    TORCH_CHECK(
        tensor->scalar_type() == at::kInt,
        "C8 LIDU metadata/state tensors must be int32.");
  }
  CheckOneDeviceAndContiguous(
      query,
      {&query, &key, &weights, &query_dequant_scale,
       &key_dequant_scale, &actual_seq_lengths_query,
       &req_pool_entries, &cache_slots_pool, &cache_tokens,
       &candidate_lens, &block_table},
      "C8 LIDU");
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor>
FusedLiManageC8OutNpu(
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
    at::Tensor source_ids,
    at::Tensor destination_slots,
    at::Tensor miss_counts) {
  CheckFusedLiManageC8Common(
      query, key, weights, query_dequant_scale, key_dequant_scale,
      actual_seq_lengths_query, req_pool_entries, cache_slots_pool,
      cache_tokens, candidate_lens, block_table);
  CheckLiduOutputs(query, source_ids, destination_slots, miss_counts);
  auto keepalive = std::make_tuple(
      query, key, weights, query_dequant_scale, key_dequant_scale,
      actual_seq_lengths_query, req_pool_entries, cache_slots_pool,
      cache_tokens, candidate_lens, block_table, source_ids,
      destination_slots, miss_counts);
  EXEC_NPU_CMD_ORDERED(
      aclnnA5FusedLiManageC8,
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
      source_ids,
      destination_slots,
      miss_counts,
      cache_slots_pool);
  return std::make_tuple(
      source_ids, destination_slots, miss_counts, cache_slots_pool);
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor>
FusedLiManageC8Npu(
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
  auto options = query.options().dtype(at::kInt);
  auto source_ids = at::empty({query.size(0), 1, kSparseCount}, options);
  auto destination_slots = at::empty_like(source_ids);
  auto miss_counts = at::empty({query.size(0)}, options);
  return FusedLiManageC8OutNpu(
      query, key, weights, query_dequant_scale, key_dequant_scale,
      actual_seq_lengths_query, req_pool_entries, cache_slots_pool,
      cache_tokens, candidate_lens, block_table,
      source_ids, destination_slots, miss_counts);
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor>
FusedLiManageC8Meta(
    const at::Tensor& query,
    const at::Tensor&,
    const at::Tensor&,
    const at::Tensor&,
    const at::Tensor&,
    const at::Tensor&,
    const at::Tensor&,
    at::Tensor cache_slots_pool,
    const at::Tensor&,
    const at::Tensor&,
    const at::Tensor&) {
  auto options = query.options().dtype(at::kInt);
  return std::make_tuple(
      at::empty({query.size(0), 1, kSparseCount}, options),
      at::empty({query.size(0), 1, kSparseCount}, options),
      at::empty({query.size(0)}, options),
      cache_slots_pool);
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor>
FusedLiManageC8OutMeta(
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
    at::Tensor source_ids,
    at::Tensor destination_slots,
    at::Tensor miss_counts) {
  return std::make_tuple(
      source_ids, destination_slots, miss_counts, cache_slots_pool);
}
}  // namespace nanovllm_dsa_a5_impl

TORCH_LIBRARY_IMPL(nanovllm_dsa, PrivateUse1, m) {
  m.impl("fused_li_manage_c8", &nanovllm_dsa_a5_impl::FusedLiManageC8Npu);
  m.impl("fused_li_manage_c8_out", &nanovllm_dsa_a5_impl::FusedLiManageC8OutNpu);
}

TORCH_LIBRARY_IMPL(nanovllm_dsa, Meta, m) {
  m.impl("fused_li_manage_c8", &nanovllm_dsa_a5_impl::FusedLiManageC8Meta);
  m.impl("fused_li_manage_c8_out", &nanovllm_dsa_a5_impl::FusedLiManageC8OutMeta);
}
