#ifndef NANOVLLM_SPARSE_TAIL_ATTENTION_TORCH_ADPT_H
#define NANOVLLM_SPARSE_TAIL_ATTENTION_TORCH_ADPT_H

namespace vllm_ascend {

inline void npu_sparse_tail_attention(
    const at::Tensor& query_rope,
    const at::Tensor& query,
    const at::Tensor& actual_seq_lengths_query,
    const at::Tensor& actual_seq_lengths_kv,
    const at::Tensor& num_cache_tokens,
    const at::Tensor& topk_dst_slots,
    const at::Tensor& hbm_block_table,
    const at::Tensor& hbm_k_rope,
    const at::Tensor& hbm_kv_cache,
    double scale_value,
    at::Tensor attention_out) {
  TORCH_CHECK(query.device().is_privateuseone(),
              "Sparse-tail Attention inputs must be on NPU.");
  TORCH_CHECK(query.dim() == 3 && query.size(2) == 512,
              "query must be [B, N, 512].");
  const auto num_heads = query.size(1);
  TORCH_CHECK(num_heads > 0 && num_heads <= 128 &&
                  (num_heads & (num_heads - 1)) == 0,
              "query local head count must be a power of two in [1, 128].");
  TORCH_CHECK(hbm_kv_cache.dim() == 4 && hbm_kv_cache.size(1) == 128 &&
                  hbm_kv_cache.size(2) == 1 &&
                  hbm_kv_cache.size(3) == 512,
              "key must be [blocks, 128, 1, 512].");
  TORCH_CHECK(query_rope.dim() == 3 &&
                  query_rope.size(0) == query.size(0) &&
                  query_rope.size(1) == query.size(1) &&
                  query_rope.size(2) == 64,
              "query_rope must be [B, N, 64].");
  TORCH_CHECK(hbm_k_rope.dim() == 4 &&
                  hbm_k_rope.size(0) == hbm_kv_cache.size(0) &&
                  hbm_k_rope.size(1) == 128 && hbm_k_rope.size(2) == 1 &&
                  hbm_k_rope.size(3) == 64,
              "key_rope must be [blocks, 128, 1, 64].");
  TORCH_CHECK(topk_dst_slots.dim() == 3 &&
                  topk_dst_slots.size(0) == query.size(0) &&
                  topk_dst_slots.size(1) == 1 &&
                  topk_dst_slots.size(2) == 2048,
              "sparse_slots must be [B, 1, 2048].");
  TORCH_CHECK(num_cache_tokens.dim() == 1 &&
                  num_cache_tokens.size(0) == query.size(0),
              "cache_tokens must be [B].");
  TORCH_CHECK(hbm_block_table.dim() == 2 &&
                  hbm_block_table.size(0) == query.size(0) &&
                  hbm_block_table.size(1) > 0,
              "block_table must be [B, max_blocks] with max_blocks > 0.");
  TORCH_CHECK(actual_seq_lengths_query.dim() == 1 &&
                  actual_seq_lengths_query.size(0) == query.size(0) &&
                  actual_seq_lengths_kv.dim() == 1 &&
                  actual_seq_lengths_kv.size(0) == query.size(0),
              "actual sequence lengths must be [B].");
  TORCH_CHECK(attention_out.sizes() == query.sizes(),
              "attention_out must have the same shape as query.");

  const auto dtype = query.scalar_type();
  TORCH_CHECK(dtype == at::kHalf || dtype == at::kBFloat16,
              "Sparse-tail Attention supports fp16 or bf16.");
  TORCH_CHECK(hbm_kv_cache.scalar_type() == dtype &&
                  query_rope.scalar_type() == dtype &&
                  hbm_k_rope.scalar_type() == dtype &&
                  attention_out.scalar_type() == dtype,
              "All floating-point inputs must have the same dtype.");
  TORCH_CHECK(topk_dst_slots.scalar_type() == at::kInt &&
                  num_cache_tokens.scalar_type() == at::kInt &&
                  hbm_block_table.scalar_type() == at::kInt &&
                  actual_seq_lengths_query.scalar_type() == at::kInt &&
                  actual_seq_lengths_kv.scalar_type() == at::kInt,
              "Sparse slots, cache metadata, tables and lengths must be int32.");

  const auto device = query.device();
  TORCH_CHECK(hbm_kv_cache.device() == device &&
                  topk_dst_slots.device() == device &&
                  num_cache_tokens.device() == device &&
                  hbm_block_table.device() == device &&
                  actual_seq_lengths_query.device() == device &&
                  actual_seq_lengths_kv.device() == device &&
                  query_rope.device() == device &&
                  hbm_k_rope.device() == device &&
                  attention_out.device() == device,
              "All Sparse-tail Attention inputs must be on the same NPU.");
  TORCH_CHECK(query.is_contiguous() && hbm_kv_cache.is_contiguous() &&
                  topk_dst_slots.is_contiguous() &&
                  num_cache_tokens.is_contiguous() &&
                  hbm_block_table.is_contiguous() &&
                  actual_seq_lengths_query.is_contiguous() &&
                  actual_seq_lengths_kv.is_contiguous() &&
                  query_rope.is_contiguous() && hbm_k_rope.is_contiguous() &&
                  attention_out.is_contiguous(),
              "All Sparse-tail Attention inputs must be contiguous.");

  std::string query_layout = "TND";
  std::string kv_layout = "PA_BSND";
  char* query_layout_ptr = const_cast<char*>(query_layout.c_str());
  char* kv_layout_ptr = const_cast<char*>(kv_layout.c_str());
  constexpr int64_t kSparseBlockSize = 1;
  constexpr int64_t kSparseMode = 3;
  auto keepalive = std::make_tuple(
      query_rope, query, actual_seq_lengths_query, actual_seq_lengths_kv,
      num_cache_tokens, topk_dst_slots, hbm_block_table, hbm_k_rope,
      hbm_kv_cache, attention_out);
  EXEC_NPU_CMD_ORDERED(
      aclnnNanovllmSparseTailAttention,
      keepalive,
      query,
      hbm_kv_cache,
      hbm_kv_cache,
      topk_dst_slots,
      num_cache_tokens,
      hbm_block_table,
      actual_seq_lengths_query,
      actual_seq_lengths_kv,
      query_rope,
      hbm_k_rope,
      scale_value,
      kSparseBlockSize,
      query_layout_ptr,
      kv_layout_ptr,
      kSparseMode,
      attention_out);
}

}  // namespace vllm_ascend

#endif
