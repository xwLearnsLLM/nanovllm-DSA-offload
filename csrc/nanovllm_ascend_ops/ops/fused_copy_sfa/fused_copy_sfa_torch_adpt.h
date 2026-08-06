#ifndef NANOVLLM_FUSED_COPY_SFA_TORCH_ADPT_H
#define NANOVLLM_FUSED_COPY_SFA_TORCH_ADPT_H

namespace vllm_ascend {

inline std::tuple<at::Tensor, at::Tensor, at::Tensor>
npu_fused_copy_sfa(
    const at::Tensor& query,
    at::Tensor hbm_kv_cache,
    const at::Tensor& sparse_slots,
    const at::Tensor& cache_tokens,
    const at::Tensor& hbm_block_table,
    const at::Tensor& actual_seq_lengths_query,
    const at::Tensor& actual_seq_lengths_kv,
    const at::Tensor& query_rope,
    at::Tensor hbm_k_rope,
    const at::Tensor& dram_k_rope,
    const at::Tensor& dram_kv_cache,
    const at::Tensor& dram_block_table,
    const at::Tensor& source_token_ids,
    const at::Tensor& copy_counts,
    double scale_value) {
  TORCH_CHECK(query.device().is_privateuseone(),
              "Fused Attention+SCATTER inputs must be on NPU.");
  TORCH_CHECK(query.dim() == 3 && query.size(0) >= 1 &&
                  query.size(0) <= 24 && query.size(1) >= 1 &&
                  query.size(1) <= 128 && query.size(2) == 512,
              "query must be [B, N, 512] with 1 <= B <= 24 and "
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
  TORCH_CHECK(sparse_slots.dim() == 3 &&
                  sparse_slots.size(0) == batch_size &&
                  sparse_slots.size(1) == 1 &&
                  sparse_slots.size(2) == 2048,
              "sparse_slots must be [B, 1, 2048].");
  TORCH_CHECK(cache_tokens.dim() == 1 &&
                  cache_tokens.size(0) == batch_size &&
                  hbm_block_table.dim() == 2 &&
                  hbm_block_table.size(0) == batch_size &&
                  actual_seq_lengths_query.dim() == 1 &&
                  actual_seq_lengths_query.size(0) == batch_size &&
                  actual_seq_lengths_kv.dim() == 1 &&
                  actual_seq_lengths_kv.size(0) == batch_size &&
                  dram_block_table.dim() == 2 &&
                  dram_block_table.size(0) == batch_size &&
                  source_token_ids.dim() == 2 &&
                  source_token_ids.size(0) == batch_size &&
                  source_token_ids.size(1) == 2048 &&
                  copy_counts.dim() == 1 &&
                  copy_counts.size(0) == batch_size,
              "Fused Attention+SCATTER metadata shapes are invalid.");
  TORCH_CHECK(query_rope.dim() == 3 &&
                  query_rope.size(0) == batch_size &&
                  query_rope.size(1) == query.size(1) &&
                  query_rope.size(2) == 64,
              "query_rope must be [B, N, 64].");

  const auto dtype = query.scalar_type();
  TORCH_CHECK(dtype == at::kHalf || dtype == at::kBFloat16,
              "Fused Attention+SCATTER supports fp16 or bf16.");
  for (const at::Tensor* tensor :
       std::array<const at::Tensor*, 5>{
           &hbm_kv_cache, &query_rope, &hbm_k_rope, &dram_k_rope,
           &dram_kv_cache}) {
    TORCH_CHECK(tensor->scalar_type() == dtype,
                "All fused floating-point inputs must share one dtype.");
  }
  for (const at::Tensor* tensor :
       std::array<const at::Tensor*, 8>{
           &sparse_slots, &cache_tokens, &hbm_block_table,
           &actual_seq_lengths_query, &actual_seq_lengths_kv,
           &dram_block_table, &source_token_ids, &copy_counts}) {
    TORCH_CHECK(tensor->scalar_type() == at::kInt,
                "All fused metadata inputs must be int32.");
  }
  const auto device = query.device();
  for (const at::Tensor* tensor :
       std::array<const at::Tensor*, 14>{
           &query, &hbm_kv_cache, &sparse_slots, &cache_tokens,
           &hbm_block_table, &actual_seq_lengths_query,
           &actual_seq_lengths_kv, &query_rope, &hbm_k_rope,
           &dram_k_rope, &dram_kv_cache, &dram_block_table,
           &source_token_ids, &copy_counts}) {
    TORCH_CHECK(tensor->device() == device,
                "All fused inputs must be on the same NPU.");
    TORCH_CHECK(tensor->is_contiguous(),
                "All fused inputs must be contiguous.");
  }

  auto output = at::empty_like(query);
  std::string query_layout = "TND";
  std::string kv_layout = "PA_BSND";
  char* query_layout_ptr = const_cast<char*>(query_layout.c_str());
  char* kv_layout_ptr = const_cast<char*>(kv_layout.c_str());
  constexpr int64_t kSparseBlockSize = 1;
  constexpr int64_t kSparseMode = 3;
  auto keepalive = std::make_tuple(
      query, hbm_kv_cache, sparse_slots, cache_tokens, hbm_block_table,
      actual_seq_lengths_query, actual_seq_lengths_kv, query_rope,
      hbm_k_rope, dram_k_rope, dram_kv_cache, dram_block_table,
      source_token_ids, copy_counts, output);
  EXEC_NPU_CMD_ORDERED(
      aclnnNanovllmFusedCopySfa,
      keepalive,
      query,
      hbm_kv_cache,
      hbm_kv_cache,
      sparse_slots,
      cache_tokens,
      hbm_block_table,
      actual_seq_lengths_query,
      actual_seq_lengths_kv,
      query_rope,
      hbm_k_rope,
      dram_k_rope,
      dram_kv_cache,
      dram_block_table,
      source_token_ids,
      copy_counts,
      scale_value,
      kSparseBlockSize,
      query_layout_ptr,
      kv_layout_ptr,
      kSparseMode,
      output);
  return std::make_tuple(output, hbm_k_rope, hbm_kv_cache);
}

}  // namespace vllm_ascend

#endif
