#ifndef NANOVLLM_FUSED_LI_MANAGE_MTP_TORCH_ADPT_H_
#define NANOVLLM_FUSED_LI_MANAGE_MTP_TORCH_ADPT_H_

namespace vllm_ascend {

inline void npu_fused_li_manage_mtp_out(
    const at::Tensor& query,
    const at::Tensor& key,
    const at::Tensor& weights,
    const at::Tensor& req_pool_entries,
    at::Tensor cache_slots,
    const at::Tensor& cache_tokens,
    const at::Tensor& candidate_lens,
    const at::Tensor& block_table,
    at::Tensor topk_slots,
    at::Tensor topk_source_ids,
    at::Tensor miss_source_ids,
    at::Tensor miss_destination_slots,
    at::Tensor miss_counts) {
  constexpr int64_t kQueryCount = 4;
  constexpr int64_t kTopK = 2048;
  constexpr int64_t kUnionCapacity = 8192;
  TORCH_CHECK(query.dim() == 3 && query.size(1) == 32 &&
                  query.size(2) == 128,
              "MTP-LIDU query must be [4B, 32, 128].");
  TORCH_CHECK(query.device().is_privateuseone(),
              "MTP-LIDU inputs must be on NPU.");
  TORCH_CHECK(req_pool_entries.dim() == 1,
              "MTP-LIDU req_pool_entries must be [B].");
  const int64_t batch_size = req_pool_entries.size(0);
  TORCH_CHECK(batch_size > 0 && query.size(0) == batch_size * kQueryCount,
              "MTP-LIDU requires exactly four packed queries per request.");
  TORCH_CHECK(key.dim() == 4 && key.size(0) > 0 && key.size(1) == 128 &&
                  key.size(2) == 1 && key.size(3) == 128,
              "MTP-LIDU key must be [blocks, 128, 1, 128].");
  TORCH_CHECK(weights.dim() == 2 && weights.size(0) == query.size(0) &&
                  weights.size(1) == 32,
              "MTP-LIDU weights must be [4B, 32].");
  TORCH_CHECK(cache_slots.dim() == 2 && cache_slots.size(0) > 0 &&
                  cache_tokens.dim() == 1 && candidate_lens.dim() == 1 &&
                  block_table.dim() == 2 && block_table.size(1) > 0,
              "MTP-LIDU cache metadata has invalid rank or empty capacity.");
  TORCH_CHECK(cache_tokens.size(0) == batch_size &&
                  candidate_lens.size(0) == batch_size &&
                  block_table.size(0) == batch_size,
              "MTP-LIDU request metadata batch dimensions must match.");
  TORCH_CHECK(cache_slots.size(1) == block_table.size(1) * 128,
              "MTP-LIDU pool width must match block-table capacity.");
  TORCH_CHECK(block_table.size(1) <= (1 << 11),
              "MTP-LIDU source capacity must be <= 2^18 tokens.");
  TORCH_CHECK(topk_slots.dim() == 3 &&
                  topk_slots.size(0) == query.size(0) &&
                  topk_slots.size(1) == 1 && topk_slots.size(2) == kTopK,
              "MTP-LIDU topk_slots must be [4B, 1, 2048].");
  TORCH_CHECK(topk_source_ids.sizes() == topk_slots.sizes(),
              "MTP-LIDU topk_source_ids must match topk_slots.");
  TORCH_CHECK(miss_source_ids.dim() == 2 &&
                  miss_source_ids.size(0) == batch_size &&
                  miss_source_ids.size(1) == kUnionCapacity &&
                  miss_destination_slots.sizes() == miss_source_ids.sizes(),
              "MTP-LIDU miss outputs must be [B, 8192].");
  TORCH_CHECK(miss_counts.dim() == 1 && miss_counts.size(0) == batch_size,
              "MTP-LIDU miss_counts must be [B].");
  TORCH_CHECK(query.scalar_type() == key.scalar_type() &&
                  query.scalar_type() == weights.scalar_type() &&
                  (query.scalar_type() == at::kHalf ||
                   query.scalar_type() == at::kBFloat16),
              "MTP-LIDU query/key/weights must share fp16 or bf16 dtype.");
  TORCH_CHECK(req_pool_entries.scalar_type() == at::kInt &&
                  cache_slots.scalar_type() == at::kInt &&
                  cache_tokens.scalar_type() == at::kInt &&
                  candidate_lens.scalar_type() == at::kInt &&
                  block_table.scalar_type() == at::kInt &&
                  topk_slots.scalar_type() == at::kInt &&
                  topk_source_ids.scalar_type() == at::kInt &&
                  miss_source_ids.scalar_type() == at::kInt &&
                  miss_destination_slots.scalar_type() == at::kInt &&
                  miss_counts.scalar_type() == at::kInt,
              "MTP-LIDU metadata and outputs must be int32.");
  TORCH_CHECK(query.is_contiguous() && key.is_contiguous() &&
                  weights.is_contiguous() && req_pool_entries.is_contiguous() &&
                  cache_slots.is_contiguous() && cache_tokens.is_contiguous() &&
                  candidate_lens.is_contiguous() && block_table.is_contiguous() &&
                  topk_slots.is_contiguous() && topk_source_ids.is_contiguous() &&
                  miss_source_ids.is_contiguous() &&
                  miss_destination_slots.is_contiguous() && miss_counts.is_contiguous(),
              "All MTP-LIDU tensors must be contiguous.");
  const auto device = query.device();
  TORCH_CHECK(key.device() == device && weights.device() == device &&
                  req_pool_entries.device() == device &&
                  cache_slots.device() == device && cache_tokens.device() == device &&
                  candidate_lens.device() == device && block_table.device() == device &&
                  topk_slots.device() == device && topk_source_ids.device() == device &&
                  miss_source_ids.device() == device &&
                  miss_destination_slots.device() == device && miss_counts.device() == device,
              "All MTP-LIDU tensors must be on the same NPU device.");

  auto keepalive = std::make_tuple(
      query, key, weights, req_pool_entries, cache_slots, cache_tokens,
      candidate_lens, block_table, topk_slots, topk_source_ids,
      miss_source_ids,
      miss_destination_slots, miss_counts);
  EXEC_NPU_CMD_ORDERED(
      aclnnNanovllmFusedLiManageMtp,
      keepalive,
      query,
      key,
      weights,
      req_pool_entries,
      cache_slots,
      cache_tokens,
      candidate_lens,
      block_table,
      topk_slots,
      topk_source_ids,
      miss_source_ids,
      miss_destination_slots,
      miss_counts,
      cache_slots);
}

inline std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
npu_fused_li_manage_mtp(
    const at::Tensor& query,
    const at::Tensor& key,
    const at::Tensor& weights,
    const at::Tensor& req_pool_entries,
    at::Tensor cache_slots,
    const at::Tensor& cache_tokens,
    const at::Tensor& candidate_lens,
    const at::Tensor& block_table) {
  constexpr int64_t kTopK = 2048;
  constexpr int64_t kUnionCapacity = 8192;
  const int64_t batch_size = req_pool_entries.size(0);
  auto options = query.options().dtype(at::kInt);
  auto topk_slots = at::empty({query.size(0), 1, kTopK}, options);
  auto topk_source_ids = at::empty_like(topk_slots);
  auto miss_source_ids = at::empty({batch_size, kUnionCapacity}, options);
  auto miss_destination_slots = at::empty_like(miss_source_ids);
  auto miss_counts = at::empty({batch_size}, options);
  npu_fused_li_manage_mtp_out(
      query, key, weights, req_pool_entries, cache_slots, cache_tokens,
      candidate_lens, block_table, topk_slots, topk_source_ids,
      miss_source_ids,
      miss_destination_slots, miss_counts);
  return std::make_tuple(topk_slots, topk_source_ids, miss_source_ids,
                         miss_destination_slots, miss_counts);
}

} // namespace vllm_ascend
#endif
