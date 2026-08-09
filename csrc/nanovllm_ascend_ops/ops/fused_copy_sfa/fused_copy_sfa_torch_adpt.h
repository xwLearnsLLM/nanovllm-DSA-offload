#ifndef NANOVLLM_FUSED_COPY_SFA_TORCH_ADPT_H
#define NANOVLLM_FUSED_COPY_SFA_TORCH_ADPT_H

namespace vllm_ascend {

inline void
npu_fused_copy_sfa(
    const at::Tensor& query_rope,
    const at::Tensor& query,
    const at::Tensor& actual_seq_lengths_query,
    const at::Tensor& actual_seq_lengths_kv,
    const at::Tensor& num_cache_tokens,
    const at::Tensor& topk_dst_slots,
    const at::Tensor& topk_src_ids,
    const at::Tensor& miss_counts,
    const at::Tensor& hbm_block_table,
    const at::Tensor& dram_block_table,
    at::Tensor hbm_k_rope,
    at::Tensor hbm_kv_cache,
    const at::Tensor& dram_k_rope,
    const at::Tensor& dram_kv_cache,
    double scale_value,
    at::Tensor attention_out) {
  TORCH_CHECK(query.device().is_privateuseone(),
              "Fused Attention+SCATTER inputs must be on NPU.");
  TORCH_CHECK(query.dim() == 3 && query.size(0) >= 1 &&
                  query.size(1) >= 1 && query.size(1) <= 128 &&
                  query.size(2) == 512,
              "query must be [B, N, 512] with B >= 1 and "
              "1 <= N <= 128.");
  const auto batch_size = query.size(0);
  TORCH_CHECK(hbm_kv_cache.dim() == 4 &&
                  hbm_kv_cache.size(1) == 128 &&
                  hbm_kv_cache.size(2) == 1 &&
                  hbm_kv_cache.size(3) == 512,
              "HBM CKV must be [blocks, 128, 1, 512].");
  TORCH_CHECK(hbm_k_rope.dim() == 4 &&
                  hbm_k_rope.size(0) == hbm_kv_cache.size(0) &&
                  hbm_k_rope.size(1) == 128 &&
                  hbm_k_rope.size(2) == 1 &&
                  hbm_k_rope.size(3) == 64,
              "HBM KPE must be [blocks, 128, 1, 64].");
  TORCH_CHECK(dram_kv_cache.dim() == 3 &&
                  dram_kv_cache.size(1) == 128 &&
                  dram_kv_cache.size(2) == 512,
              "DRAM CKV must be [blocks, 128, 512].");
  TORCH_CHECK(dram_k_rope.dim() == 3 &&
                  dram_k_rope.size(0) == dram_kv_cache.size(0) &&
                  dram_k_rope.size(1) == 128 &&
                  dram_k_rope.size(2) == 64,
              "DRAM KPE must be [blocks, 128, 64].");
  TORCH_CHECK(topk_dst_slots.dim() == 3 &&
                  topk_dst_slots.size(0) == batch_size &&
                  topk_dst_slots.size(1) == 1 &&
                  topk_dst_slots.size(2) == 2048,
              "sparse_slots must be [B, 1, 2048].");
  TORCH_CHECK(num_cache_tokens.dim() == 1 &&
                  num_cache_tokens.size(0) == batch_size &&
                  hbm_block_table.dim() == 2 &&
                  hbm_block_table.size(0) == batch_size &&
                  actual_seq_lengths_query.dim() == 1 &&
                  actual_seq_lengths_query.size(0) == batch_size &&
                  actual_seq_lengths_kv.dim() == 1 &&
                  actual_seq_lengths_kv.size(0) == batch_size &&
                  dram_block_table.dim() == 2 &&
                  dram_block_table.size(0) == batch_size &&
                  topk_src_ids.dim() == 2 &&
                  topk_src_ids.size(0) == batch_size &&
                  topk_src_ids.size(1) == 2048 &&
                  miss_counts.dim() == 1 &&
                  miss_counts.size(0) == batch_size,
              "Fused Attention+SCATTER metadata shapes are invalid.");
  TORCH_CHECK(query_rope.dim() == 3 &&
                  query_rope.size(0) == batch_size &&
                  query_rope.size(1) == query.size(1) &&
                  query_rope.size(2) == 64,
              "query_rope must be [B, N, 64].");
  TORCH_CHECK(attention_out.sizes() == query.sizes(),
              "attention_out must have the query shape.");

  const auto dtype = query.scalar_type();
  TORCH_CHECK(dtype == at::kHalf || dtype == at::kBFloat16,
              "Fused Attention+SCATTER supports fp16 or bf16.");
  for (const at::Tensor* tensor :
       std::array<const at::Tensor*, 6>{
           &hbm_kv_cache, &query_rope, &hbm_k_rope, &dram_k_rope,
           &dram_kv_cache, &attention_out}) {
    TORCH_CHECK(tensor->scalar_type() == dtype,
                "All fused floating-point inputs must share one dtype.");
  }
  for (const at::Tensor* tensor :
       std::array<const at::Tensor*, 8>{
           &topk_dst_slots, &num_cache_tokens, &hbm_block_table,
           &actual_seq_lengths_query, &actual_seq_lengths_kv,
           &dram_block_table, &topk_src_ids, &miss_counts}) {
    TORCH_CHECK(tensor->scalar_type() == at::kInt,
                "All fused metadata inputs must be int32.");
  }
  const auto device = query.device();
  for (const at::Tensor* tensor :
       std::array<const at::Tensor*, 15>{
           &query, &hbm_kv_cache, &topk_dst_slots, &num_cache_tokens,
           &hbm_block_table, &actual_seq_lengths_query,
           &actual_seq_lengths_kv, &query_rope, &hbm_k_rope,
           &dram_k_rope, &dram_kv_cache, &dram_block_table,
           &topk_src_ids, &miss_counts, &attention_out}) {
    TORCH_CHECK(tensor->device() == device,
                "All fused inputs must be on the same NPU.");
    TORCH_CHECK(tensor->is_contiguous(),
                "All fused inputs must be contiguous.");
  }

  std::string query_layout = "TND";
  std::string kv_layout = "PA_BSND";
  char* query_layout_ptr = const_cast<char*>(query_layout.c_str());
  char* kv_layout_ptr = const_cast<char*>(kv_layout.c_str());
  constexpr int64_t kSparseBlockSize = 1;
  constexpr int64_t kSparseMode = 3;
  auto keepalive = std::make_tuple(
      query_rope, query, actual_seq_lengths_query, actual_seq_lengths_kv,
      num_cache_tokens, topk_dst_slots, topk_src_ids, miss_counts,
      hbm_block_table, dram_block_table,
      hbm_k_rope, dram_k_rope, dram_kv_cache,
      hbm_kv_cache, attention_out);
  EXEC_NPU_CMD_ORDERED(
      aclnnNanovllmFusedCopySfa,
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
      dram_k_rope,
      dram_kv_cache,
      dram_block_table,
      topk_src_ids,
      miss_counts,
      scale_value,
      kSparseBlockSize,
      query_layout_ptr,
      kv_layout_ptr,
      kSparseMode,
      attention_out);
}

}  // namespace vllm_ascend

#endif
