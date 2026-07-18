#ifndef NANOVLLM_LIGHTNING_INDEXER_DECODE_UPDATE_TORCH_ADPT_H
#define NANOVLLM_LIGHTNING_INDEXER_DECODE_UPDATE_TORCH_ADPT_H

namespace vllm_ascend {

inline std::tuple<at::Tensor, at::Tensor, at::Tensor>
npu_lightning_indexer_decode_update(
    const at::Tensor& query,
    const at::Tensor& key,
    const at::Tensor& weights,
    const at::Tensor& req_pool_entries,
    at::Tensor cache_slots,
    const at::Tensor& cache_tokens,
    const at::Tensor& actual_seq_lengths_key,
    const at::Tensor& block_table) {
  TORCH_CHECK(query.dim() == 3, "LIDU query must be [B, N, 128].");
  TORCH_CHECK(query.device().is_privateuseone(),
              "LIDU inputs must be on NPU.");
  TORCH_CHECK(query.size(1) == 32 || query.size(1) == 64,
              "LIDU supports 32 or 64 index heads.");
  TORCH_CHECK(query.size(2) == 128, "LIDU head_dim must be 128.");
  TORCH_CHECK(key.dim() == 4 && key.size(1) == 128 && key.size(2) == 1 &&
                  key.size(3) == 128,
              "LIDU key must be [blocks, 128, 1, 128].");
  TORCH_CHECK(weights.dim() == 2 && weights.size(0) == query.size(0) &&
                  weights.size(1) == query.size(1),
              "LIDU weights must be [B, N].");
  TORCH_CHECK(req_pool_entries.dim() == 1 && cache_tokens.dim() == 1 &&
                  actual_seq_lengths_key.dim() == 1,
              "LIDU row metadata must be rank one.");
  TORCH_CHECK(cache_slots.dim() == 2 && block_table.dim() == 2,
              "LIDU cache_slots and block_table must be rank two.");
  TORCH_CHECK(query.size(0) == req_pool_entries.size(0) &&
                  query.size(0) == cache_tokens.size(0) &&
                  query.size(0) == actual_seq_lengths_key.size(0) &&
                  query.size(0) == block_table.size(0),
              "LIDU batch dimensions must match.");
  TORCH_CHECK(cache_slots.size(1) == block_table.size(1) * 128,
              "LIDU request state width must match block-table capacity.");
  TORCH_CHECK(cache_slots.size(0) > 0 && block_table.size(1) > 0,
              "LIDU request pool and block table must be non-empty.");
  TORCH_CHECK(block_table.size(1) <= (1 << 11),
              "LIDU block-table capacity must be <= 2048 blocks.");
  TORCH_CHECK(query.scalar_type() == key.scalar_type() &&
                  query.scalar_type() == weights.scalar_type(),
              "LIDU query/key/weights dtypes must match.");
  TORCH_CHECK(cache_slots.scalar_type() == at::kInt &&
                  req_pool_entries.scalar_type() == at::kInt &&
                  cache_tokens.scalar_type() == at::kInt &&
                  actual_seq_lengths_key.scalar_type() == at::kInt &&
                  block_table.scalar_type() == at::kInt,
              "LIDU metadata must be int32.");
  TORCH_CHECK(query.scalar_type() == at::kHalf ||
                  query.scalar_type() == at::kBFloat16,
              "LIDU query/key/weights must be fp16 or bf16.");
  TORCH_CHECK(query.is_contiguous() && key.is_contiguous() &&
                  weights.is_contiguous() && req_pool_entries.is_contiguous() &&
                  cache_slots.is_contiguous() && cache_tokens.is_contiguous() &&
                  actual_seq_lengths_key.is_contiguous() &&
                  block_table.is_contiguous(),
              "All LIDU inputs must be contiguous.");
  const auto device = query.device();
  TORCH_CHECK(key.device() == device && weights.device() == device &&
                  req_pool_entries.device() == device &&
                  cache_slots.device() == device && cache_tokens.device() == device &&
                  actual_seq_lengths_key.device() == device &&
                  block_table.device() == device,
              "All LIDU inputs must be on the same NPU device.");

  constexpr int64_t kTopK = 2048;
  auto options = query.options().dtype(at::kInt);
  auto source_ids = at::empty({query.size(0), 1, kTopK}, options);
  auto destination_slots = at::empty({query.size(0), 1, kTopK}, options);
  auto miss_counts = at::empty({query.size(0)}, options);
  auto keepalive = std::make_tuple(
      query, key, weights, req_pool_entries, cache_slots, cache_tokens,
      actual_seq_lengths_key, block_table, source_ids, destination_slots,
      miss_counts);
  EXEC_NPU_CMD_ORDERED(
      aclnnLightningIndexerDecodeUpdate,
      keepalive,
      query,
      key,
      weights,
      req_pool_entries,
      cache_slots,
      cache_tokens,
      actual_seq_lengths_key,
      block_table,
      source_ids,
      destination_slots,
      miss_counts,
      cache_slots);
  return std::make_tuple(source_ids, destination_slots, miss_counts);
}

}  // namespace vllm_ascend

#endif
