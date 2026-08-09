#ifndef NANOVLLM_FUSED_LI_MANAGE_TORCH_ADPT_H
#define NANOVLLM_FUSED_LI_MANAGE_TORCH_ADPT_H

namespace vllm_ascend {

inline void npu_fused_li_manage(
    const at::Tensor& query,
    const at::Tensor& index_weights,
    const at::Tensor& index_key_cache,
    const at::Tensor& index_block_table,
    const at::Tensor& num_candidate_tokens,
    const at::Tensor& num_cache_tokens,
    const at::Tensor& req_pool_entries,
    at::Tensor cache_slots_pool,
    at::Tensor topk_src_ids,
    at::Tensor topk_dst_slots,
    at::Tensor miss_counts) {
  TORCH_CHECK(query.dim() == 3, "Fused LI Manage query must be [B, N, 128].");
  TORCH_CHECK(query.device().is_privateuseone(),
              "Fused LI Manage inputs must be on NPU.");
  TORCH_CHECK(query.size(1) == 32 || query.size(1) == 64,
              "Fused LI Manage index head count must be 32 or 64.");
  TORCH_CHECK(query.size(2) == 128, "Fused LI Manage head_dim must be 128.");
  TORCH_CHECK(index_key_cache.dim() == 4 && index_key_cache.size(1) == 128 &&
                  index_key_cache.size(2) == 1 &&
                  index_key_cache.size(3) == 128,
              "Fused LI Manage key must be [blocks, 128, 1, 128].");
  TORCH_CHECK(index_weights.dim() == 2 &&
                  index_weights.size(0) == query.size(0) &&
                  index_weights.size(1) == query.size(1),
              "Fused LI Manage weights must be [B, N].");
  TORCH_CHECK(req_pool_entries.dim() == 1 && num_cache_tokens.dim() == 1 &&
                  num_candidate_tokens.dim() == 1,
              "Fused LI Manage row metadata must be rank one.");
  TORCH_CHECK(cache_slots_pool.dim() == 2 && index_block_table.dim() == 2,
              "Fused LI Manage cache_slots and block_table must be rank two.");
  TORCH_CHECK(topk_src_ids.dim() == 3 && topk_src_ids.size(1) == 1 &&
                  topk_src_ids.size(2) == 2048,
              "Fused LI Manage source_ids must be [B, 1, 2048].");
  TORCH_CHECK(topk_dst_slots.sizes() == topk_src_ids.sizes(),
              "Fused LI Manage destination_slots must match source_ids.");
  TORCH_CHECK(miss_counts.dim() == 1,
              "Fused LI Manage miss_counts must be rank one.");
  TORCH_CHECK(query.size(0) == req_pool_entries.size(0) &&
                  query.size(0) == num_cache_tokens.size(0) &&
                  query.size(0) == num_candidate_tokens.size(0) &&
                  query.size(0) == index_block_table.size(0) &&
                  query.size(0) == topk_src_ids.size(0) &&
                  query.size(0) == miss_counts.size(0),
              "Fused LI Manage batch dimensions must match.");
  TORCH_CHECK(cache_slots_pool.size(1) == index_block_table.size(1) * 128,
              "Fused LI Manage request state width must match block-table capacity.");
  TORCH_CHECK(cache_slots_pool.size(0) > 0 && index_block_table.size(1) > 0,
              "Fused LI Manage request pool and block table must be non-empty.");
  TORCH_CHECK(index_block_table.size(1) <= (1 << 14),
              "Fused LI Manage block-table capacity must be <= 16384 blocks.");
  TORCH_CHECK(query.scalar_type() == index_key_cache.scalar_type() &&
                  query.scalar_type() == index_weights.scalar_type(),
              "Fused LI Manage query/key/weights dtypes must match.");
  TORCH_CHECK(cache_slots_pool.scalar_type() == at::kInt &&
                  req_pool_entries.scalar_type() == at::kInt &&
                  num_cache_tokens.scalar_type() == at::kInt &&
                  num_candidate_tokens.scalar_type() == at::kInt &&
                  index_block_table.scalar_type() == at::kInt &&
                  topk_src_ids.scalar_type() == at::kInt &&
                  topk_dst_slots.scalar_type() == at::kInt &&
                  miss_counts.scalar_type() == at::kInt,
              "Fused LI Manage metadata must be int32.");
  TORCH_CHECK(query.scalar_type() == at::kHalf ||
                  query.scalar_type() == at::kBFloat16,
              "Fused LI Manage query/key/weights must be fp16 or bf16.");
  TORCH_CHECK(query.is_contiguous() && index_key_cache.is_contiguous() &&
                  index_weights.is_contiguous() &&
                  req_pool_entries.is_contiguous() &&
                  cache_slots_pool.is_contiguous() &&
                  num_cache_tokens.is_contiguous() &&
                  num_candidate_tokens.is_contiguous() &&
                  index_block_table.is_contiguous() &&
                  topk_src_ids.is_contiguous() &&
                  topk_dst_slots.is_contiguous() && miss_counts.is_contiguous(),
              "All Fused LI Manage inputs must be contiguous.");
  const auto device = query.device();
  TORCH_CHECK(index_key_cache.device() == device &&
                  index_weights.device() == device &&
                  req_pool_entries.device() == device &&
                  cache_slots_pool.device() == device &&
                  num_cache_tokens.device() == device &&
                  num_candidate_tokens.device() == device &&
                  index_block_table.device() == device &&
                  topk_src_ids.device() == device &&
                  topk_dst_slots.device() == device && miss_counts.device() == device,
              "All Fused LI Manage inputs must be on the same NPU device.");

  auto keepalive = std::make_tuple(
      query, index_key_cache, index_weights, req_pool_entries,
      cache_slots_pool, num_cache_tokens, num_candidate_tokens,
      index_block_table, topk_src_ids, topk_dst_slots,
      miss_counts);
  EXEC_NPU_CMD_ORDERED(
      aclnnNanovllmFusedLiManage,
      keepalive,
      query,
      index_key_cache,
      index_weights,
      req_pool_entries,
      cache_slots_pool,
      num_cache_tokens,
      num_candidate_tokens,
      index_block_table,
      topk_src_ids,
      topk_dst_slots,
      miss_counts,
      cache_slots_pool);
}

}  // namespace vllm_ascend

#endif // NANOVLLM_FUSED_LI_MANAGE_TORCH_ADPT_H
