#ifndef NANOVLLM_FUSED_LI_MANAGE_MTP_TORCH_ADPT_H_
#define NANOVLLM_FUSED_LI_MANAGE_MTP_TORCH_ADPT_H_
#include <array>
namespace vllm_ascend {
inline void npu_fused_li_manage_mtp(
    const at::Tensor& weights, const at::Tensor& query_scale,
    const at::Tensor& query, const at::Tensor& key_scale,
    const at::Tensor& key, const at::Tensor& block_table,
    const at::Tensor& actual_query, const at::Tensor& actual_key,
    const at::Tensor& offload_key, const at::Tensor& req_valid,
    const at::Tensor& req_pool, at::Tensor cache_state,
    at::Tensor cache_slots, at::Tensor topk_src, at::Tensor topk_dst,
    at::Tensor topk_miss_counts,
    at::Tensor miss_src, at::Tensor miss_dst, at::Tensor miss_counts) {
  constexpr int64_t K = 2048, U = 8192;
  TORCH_CHECK(query.dim() == 3 && (query.size(1) == 32 || query.size(1) == 64) && query.size(2) == 128,
              "query must be [T,H,128], H=32 or 64");
  TORCH_CHECK(query.device().is_privateuseone(), "MTP LIM inputs must be on NPU");
  TORCH_CHECK(query.scalar_type() == at::kHalf || query.scalar_type() == at::kBFloat16,
              "FP8 Q/K is reserved but not implemented");
  TORCH_CHECK(weights.scalar_type() == query.scalar_type() && key.scalar_type() == query.scalar_type(),
              "query/key/weights dtype must match");
  TORCH_CHECK(req_pool.dim() == 1 && req_pool.numel() > 0, "req_pool_entries must be [B]");
  int64_t b = req_pool.numel(), t = query.size(0);
  TORCH_CHECK(t >= b && t <= 4 * b, "requires B<=T<=4B");
  TORCH_CHECK(weights.dim() == 2 && weights.size(0) == t && weights.size(1) == query.size(1),
              "weights must be [T,H]");
  TORCH_CHECK(key.dim() == 4 && key.size(1) == 128 && key.size(2) == 1 && key.size(3) == 128,
              "key must be [blocks,128,1,128]");
  TORCH_CHECK(block_table.dim() == 2 && block_table.size(0) == b, "block_table must be [B,max_blocks]");
  for (const auto* x : {&actual_query, &actual_key, &offload_key, &req_valid})
    TORCH_CHECK(x->dim() == 1 && x->numel() == b && x->scalar_type() == at::kInt,
                "request lengths/valid must be int32[B]");
  TORCH_CHECK(cache_state.dim() == 1 && cache_slots.dim() == 2 && cache_state.size(0) == cache_slots.size(0),
              "cache state/pool mismatch");
  const std::array<const at::Tensor*, 10> int_tensors = {
      &block_table, &req_pool, &cache_state, &cache_slots, &topk_src,
      &topk_dst, &topk_miss_counts, &miss_src, &miss_dst, &miss_counts};
  for (const auto* x : int_tensors)
    TORCH_CHECK(x->scalar_type() == at::kInt, "metadata and outputs must be int32");
  TORCH_CHECK(query_scale.scalar_type() == at::kFloat && key_scale.scalar_type() == at::kFloat,
              "dequant scales must be fp32");
  TORCH_CHECK(query_scale.dim() == 2 && query_scale.size(0) == t &&
                  query_scale.size(1) == query.size(1),
              "query_dequant_scale must be [T,H]");
  TORCH_CHECK(key_scale.dim() == 3 && key_scale.size(0) == key.size(0) &&
                  key_scale.size(1) == 128 && key_scale.size(2) == 1,
              "index_key_dequant_scale must be [blocks,128,1]");
  const auto device = query.device();
  const std::array<const at::Tensor*, 18> device_tensors = {
      &weights, &query_scale, &key_scale, &key, &block_table, &actual_query,
      &actual_key, &offload_key, &req_valid, &req_pool, &cache_state,
      &cache_slots, &topk_src, &topk_dst, &topk_miss_counts, &miss_src,
      &miss_dst, &miss_counts};
  for (const auto* x : device_tensors)
    TORCH_CHECK(x->device() == device, "all tensors must be on the same NPU");
  TORCH_CHECK(topk_src.sizes() == at::IntArrayRef({t, 1, K}) && topk_dst.sizes() == topk_src.sizes(),
              "topk outputs must be [T,1,2048]");
  TORCH_CHECK(topk_miss_counts.sizes() == at::IntArrayRef({t}),
              "topk_miss_counts must be [T]");
  TORCH_CHECK(miss_src.sizes() == at::IntArrayRef({b, U}) && miss_dst.sizes() == miss_src.sizes() &&
                  miss_counts.sizes() == at::IntArrayRef({b}), "miss outputs have invalid shape");
  TORCH_CHECK(query.is_contiguous() && weights.is_contiguous() && key.is_contiguous() &&
                  block_table.is_contiguous() && actual_query.is_contiguous() && actual_key.is_contiguous() &&
                  offload_key.is_contiguous() && req_valid.is_contiguous() && req_pool.is_contiguous() &&
                  cache_state.is_contiguous() && cache_slots.is_contiguous() && topk_src.is_contiguous() &&
                  topk_dst.is_contiguous() && topk_miss_counts.is_contiguous() &&
                  miss_src.is_contiguous() && miss_dst.is_contiguous() &&
                  miss_counts.is_contiguous(), "all tensors must be contiguous");
  auto keepalive = std::make_tuple(weights, query_scale, query, key_scale, key, block_table,
      actual_query, actual_key, offload_key, req_valid, req_pool, cache_state, cache_slots,
      topk_src, topk_dst, topk_miss_counts, miss_src, miss_dst, miss_counts);
  EXEC_NPU_CMD_ORDERED(aclnnNanovllmFusedLiManageMtp, keepalive, weights, query_scale,
      query, key_scale, key, block_table, actual_query, actual_key, offload_key, req_valid,
      req_pool, cache_state, cache_slots, topk_src, topk_dst, topk_miss_counts,
      miss_src, miss_dst,
      miss_counts, cache_state, cache_slots);
}
} // namespace vllm_ascend
#endif
