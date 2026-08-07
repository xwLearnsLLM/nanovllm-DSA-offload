#include <tuple>

#include "ops_common.h"

namespace nanovllm_dsa_a5_impl {

void CheckLiduCommon(
    const at::Tensor& query,
    const at::Tensor& key,
    const at::Tensor& weights,
    const at::Tensor& req_pool_entries,
    const at::Tensor& cache_slots_pool,
    const at::Tensor& cache_tokens,
    const at::Tensor& candidate_lens,
    const at::Tensor& block_table) {
  TORCH_CHECK(
      query.dim() == 3 && query.size(0) > 0 &&
          (query.size(1) == 32 || query.size(1) == 64) &&
          query.size(2) == kIndexerDim,
      "LIDU query must be [B,32|64,128].");
  const int64_t batch = query.size(0);
  TORCH_CHECK(
      key.dim() == 4 && key.size(0) > 0 && key.size(1) == kBlockSize &&
          key.size(2) == 1 && key.size(3) == kIndexerDim,
      "LIDU key must be [blocks,128,1,128].");
  TORCH_CHECK(
      weights.dim() == 2 && weights.size(0) == batch &&
          weights.size(1) == query.size(1),
      "LIDU weights must match query [B,N].");
  TORCH_CHECK(
      req_pool_entries.dim() == 1 && req_pool_entries.size(0) == batch &&
          cache_tokens.dim() == 1 && cache_tokens.size(0) == batch &&
          candidate_lens.dim() == 1 && candidate_lens.size(0) == batch,
      "LIDU request metadata must be int32[B].");
  TORCH_CHECK(
      cache_slots_pool.dim() == 2 && cache_slots_pool.size(0) > 0 &&
          cache_slots_pool.size(1) > 0 &&
          cache_slots_pool.size(1) <= kMaxSourceCapacity,
      "LIDU cache_slots_pool must be [pool_size,capacity], capacity <= 2^18.");
  TORCH_CHECK(
      block_table.dim() == 2 && block_table.size(0) == batch &&
          block_table.size(1) > 0 &&
          block_table.size(1) * kBlockSize == cache_slots_pool.size(1),
      "LIDU block_table capacity must equal cache_slots_pool.shape[1].");
  TORCH_CHECK(
      query.scalar_type() == at::kBFloat16 || query.scalar_type() == at::kHalf,
      "LIDU floating tensors must be bf16 or fp16.");
  TORCH_CHECK(
      key.scalar_type() == query.scalar_type() &&
          weights.scalar_type() == query.scalar_type(),
      "LIDU query/key/weights dtypes must match.");
  for (const at::Tensor* tensor :
       {&req_pool_entries, &cache_slots_pool, &cache_tokens,
        &candidate_lens, &block_table}) {
    TORCH_CHECK(tensor->scalar_type() == at::kInt, "LIDU metadata must be int32.");
  }
  CheckOneDeviceAndContiguous(
      query,
      {&query, &key, &weights, &req_pool_entries, &cache_slots_pool,
       &cache_tokens, &candidate_lens, &block_table},
      "LIDU");
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor>
LiduDecodeUpdateOutNpu(
    const at::Tensor& query,
    const at::Tensor& key,
    const at::Tensor& weights,
    const at::Tensor& req_pool_entries,
    at::Tensor cache_slots_pool,
    const at::Tensor& cache_tokens,
    const at::Tensor& candidate_lens,
    const at::Tensor& block_table,
    at::Tensor source_ids,
    at::Tensor destination_slots,
    at::Tensor miss_counts) {
  CheckLiduCommon(
      query, key, weights, req_pool_entries, cache_slots_pool,
      cache_tokens, candidate_lens, block_table);
  CheckLiduOutputs(query, source_ids, destination_slots, miss_counts);

  auto keepalive = std::make_tuple(
      query, key, weights, req_pool_entries, cache_slots_pool,
      cache_tokens, candidate_lens, block_table, source_ids,
      destination_slots, miss_counts);
  EXEC_NPU_CMD_ORDERED(
      aclnnLightningIndexerDecodeUpdateA5,
      keepalive,
      query,
      key,
      weights,
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
LiduDecodeUpdateNpu(
    const at::Tensor& query,
    const at::Tensor& key,
    const at::Tensor& weights,
    const at::Tensor& req_pool_entries,
    at::Tensor cache_slots_pool,
    const at::Tensor& cache_tokens,
    const at::Tensor& candidate_lens,
    const at::Tensor& block_table) {
  auto options = query.options().dtype(at::kInt);
  auto source_ids = at::empty({query.size(0), 1, kSparseCount}, options);
  auto destination_slots = at::empty_like(source_ids);
  auto miss_counts = at::empty({query.size(0)}, options);
  return LiduDecodeUpdateOutNpu(
      query, key, weights, req_pool_entries, cache_slots_pool,
      cache_tokens, candidate_lens, block_table,
      source_ids, destination_slots, miss_counts);
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor>
LiduDecodeUpdateMeta(
    const at::Tensor& query,
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
LiduDecodeUpdateOutMeta(
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
  m.impl("lidu_decode_update", &nanovllm_dsa_a5_impl::LiduDecodeUpdateNpu);
  m.impl("lidu_decode_update_out", &nanovllm_dsa_a5_impl::LiduDecodeUpdateOutNpu);
}

TORCH_LIBRARY_IMPL(nanovllm_dsa, Meta, m) {
  m.impl("lidu_decode_update", &nanovllm_dsa_a5_impl::LiduDecodeUpdateMeta);
  m.impl("lidu_decode_update_out", &nanovllm_dsa_a5_impl::LiduDecodeUpdateOutMeta);
}
