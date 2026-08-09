#ifndef NANOVLLM_FUSED_COPY_SFA_MTP_TORCH_ADPT_H
#define NANOVLLM_FUSED_COPY_SFA_MTP_TORCH_ADPT_H

namespace vllm_ascend {

// Functional MTP3 implementation of the fused-copy Attention interface.
//
// Keep this interface graph-stable and caller-owned from its first version:
// the HBM caches are mutated in place and attention_out is supplied by the
// caller.  The implementation currently composes the repository-local
// SCATTER and MTP sparse Attention commands on the same stream.  A later
// source-aware single-kernel implementation can replace this body without
// changing the torch.library schema or the nanovllm call site.
inline void npu_fused_copy_sfa_mtp_out(
    const at::Tensor& query_rope,
    const at::Tensor& query,
    const at::Tensor& actual_seq_lengths_query,
    const at::Tensor& actual_seq_lengths_kv,
    const at::Tensor& cache_tokens,
    const at::Tensor& topk_slots,
    const at::Tensor& miss_source_ids,
    const at::Tensor& miss_destination_slots,
    const at::Tensor& miss_counts,
    const at::Tensor& hbm_block_table,
    const at::Tensor& dram_block_table,
    at::Tensor hbm_k_rope,
    at::Tensor hbm_kv_cache,
    const at::Tensor& dram_k_rope,
    const at::Tensor& dram_kv_cache,
    double scale_value,
    at::Tensor attention_out) {
  constexpr int64_t kBlockSize = 128;
  constexpr int64_t kQueryCount = 4;
  constexpr int64_t kTopK = 2048;
  constexpr int64_t kUnionCapacity = kQueryCount * kTopK;
  constexpr int64_t kCkvDim = 512;
  constexpr int64_t kKpeDim = 64;

  TORCH_CHECK(query.device().is_privateuseone(),
              "Fused MTP copy+Attention inputs must be on NPU.");
  TORCH_CHECK(query.dim() == 3 && query.size(2) == kCkvDim,
              "Fused MTP query must be [4B, N, 512].");
  TORCH_CHECK(cache_tokens.dim() == 1 && cache_tokens.size(0) > 0,
              "Fused MTP cache_tokens must be [B] with B > 0.");
  const int64_t batch_size = cache_tokens.size(0);
  TORCH_CHECK(query.size(0) == batch_size * kQueryCount,
              "Fused MTP copy+Attention requires four queries per request.");
  TORCH_CHECK(query_rope.dim() == 3 &&
                  query_rope.size(0) == query.size(0) &&
                  query_rope.size(1) == query.size(1) &&
                  query_rope.size(2) == kKpeDim,
              "Fused MTP query_rope must be [4B, N, 64].");
  TORCH_CHECK(topk_slots.dim() == 3 &&
                  topk_slots.size(0) == query.size(0) &&
                  topk_slots.size(1) == 1 &&
                  topk_slots.size(2) == kTopK,
              "Fused MTP topk_slots must be [4B, 1, 2048].");
  TORCH_CHECK(miss_source_ids.dim() == 2 &&
                  miss_source_ids.size(0) == batch_size &&
                  miss_source_ids.size(1) == kUnionCapacity &&
                  miss_destination_slots.sizes() ==
                      miss_source_ids.sizes(),
              "Fused MTP miss source/destination buffers must be [B, 8192].");
  TORCH_CHECK(miss_counts.dim() == 1 &&
                  miss_counts.size(0) == batch_size,
              "Fused MTP miss_counts must be [B].");
  TORCH_CHECK(hbm_block_table.dim() == 2 &&
                  hbm_block_table.size(0) == batch_size &&
                  hbm_block_table.size(1) > 0 &&
                  dram_block_table.dim() == 2 &&
                  dram_block_table.size(0) == batch_size &&
                  dram_block_table.size(1) > 0,
              "Fused MTP block tables must be [B, max_blocks].");
  TORCH_CHECK(actual_seq_lengths_query.dim() == 1 &&
                  actual_seq_lengths_query.size(0) == batch_size &&
                  actual_seq_lengths_kv.dim() == 1 &&
                  actual_seq_lengths_kv.size(0) == batch_size,
              "Fused MTP actual sequence lengths must be [B].");

  TORCH_CHECK(hbm_k_rope.dim() == 4 &&
                  hbm_k_rope.size(1) == kBlockSize &&
                  hbm_k_rope.size(2) == 1 &&
                  hbm_k_rope.size(3) == kKpeDim,
              "Fused MTP HBM KPE must be [blocks, 128, 1, 64].");
  TORCH_CHECK(hbm_kv_cache.dim() == 4 &&
                  hbm_kv_cache.size(0) == hbm_k_rope.size(0) &&
                  hbm_kv_cache.size(1) == kBlockSize &&
                  hbm_kv_cache.size(2) == 1 &&
                  hbm_kv_cache.size(3) == kCkvDim,
              "Fused MTP HBM CKV must be [blocks, 128, 1, 512].");
  TORCH_CHECK(dram_k_rope.dim() == 3 &&
                  dram_k_rope.size(1) == kBlockSize &&
                  dram_k_rope.size(2) == kKpeDim,
              "Fused MTP DRAM KPE must be [blocks, 128, 64].");
  TORCH_CHECK(dram_kv_cache.dim() == 3 &&
                  dram_kv_cache.size(0) == dram_k_rope.size(0) &&
                  dram_kv_cache.size(1) == kBlockSize &&
                  dram_kv_cache.size(2) == kCkvDim,
              "Fused MTP DRAM CKV must be [blocks, 128, 512].");
  TORCH_CHECK(attention_out.sizes() == query.sizes(),
              "Fused MTP attention_out must have the query shape.");

  const auto dtype = query.scalar_type();
  TORCH_CHECK(dtype == at::kHalf || dtype == at::kBFloat16,
              "Fused MTP copy+Attention supports fp16 or bf16.");
  for (const at::Tensor* tensor :
       std::array<const at::Tensor*, 7>{
           &query_rope, &hbm_k_rope, &hbm_kv_cache, &dram_k_rope,
           &dram_kv_cache, &attention_out, &query}) {
    TORCH_CHECK(tensor->scalar_type() == dtype,
                "All fused MTP floating-point tensors must share one dtype.");
  }
  for (const at::Tensor* tensor :
       std::array<const at::Tensor*, 9>{
           &actual_seq_lengths_query, &actual_seq_lengths_kv,
           &cache_tokens, &topk_slots, &miss_source_ids,
           &miss_destination_slots, &miss_counts, &hbm_block_table,
           &dram_block_table}) {
    TORCH_CHECK(tensor->scalar_type() == at::kInt,
                "All fused MTP metadata tensors must be int32.");
  }

  const auto device = query.device();
  for (const at::Tensor* tensor :
       std::array<const at::Tensor*, 16>{
           &query_rope, &query, &actual_seq_lengths_query,
           &actual_seq_lengths_kv, &cache_tokens, &topk_slots,
           &miss_source_ids, &miss_destination_slots, &miss_counts,
           &hbm_block_table, &dram_block_table, &hbm_k_rope,
           &hbm_kv_cache, &dram_k_rope, &dram_kv_cache,
           &attention_out}) {
    TORCH_CHECK(tensor->device() == device,
                "All fused MTP tensors must be on the same NPU.");
    TORCH_CHECK(tensor->is_contiguous(),
                "All fused MTP tensors must be contiguous.");
  }

  // SCATTER's stable ABI uses [blocks, 128, dim], while MTP Attention uses
  // PA_BSND [blocks, 128, 1, dim].  These views share storage and enqueue no
  // device work.
  auto flat_hbm_k_rope = hbm_k_rope.view(
      {hbm_k_rope.size(0), kBlockSize, kKpeDim});
  auto flat_hbm_kv_cache = hbm_kv_cache.view(
      {hbm_kv_cache.size(0), kBlockSize, kCkvDim});
  npu_kvcache_scatter_copy(
      flat_hbm_k_rope, flat_hbm_kv_cache, dram_k_rope, dram_kv_cache,
      hbm_block_table, dram_block_table, miss_source_ids,
      miss_destination_slots, miss_counts);
  npu_sparse_tail_attention_mtp_out(
      query, hbm_kv_cache, hbm_kv_cache, topk_slots, cache_tokens,
      hbm_block_table, actual_seq_lengths_query, actual_seq_lengths_kv,
      query_rope, hbm_k_rope, scale_value, attention_out);
}

}  // namespace vllm_ascend

#endif
