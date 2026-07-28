#ifndef NANOVLLM_SPARSE_AND_TAIL_ATTENTION_TORCH_ADPT_H
#define NANOVLLM_SPARSE_AND_TAIL_ATTENTION_TORCH_ADPT_H

namespace vllm_ascend {

inline at::Tensor npu_sparse_and_tail_attention(
    const at::Tensor& query,
    const at::Tensor& key,
    const at::Tensor& value,
    const at::Tensor& sparse_slots,
    const at::Tensor& cache_tokens,
    const at::Tensor& block_table,
    const at::Tensor& actual_seq_lengths_query,
    const at::Tensor& actual_seq_lengths_kv,
    const at::Tensor& query_rope,
    const at::Tensor& key_rope,
    double scale_value) {
  TORCH_CHECK(query.device().is_privateuseone(),
              "Sparse-and-tail Attention inputs must be on NPU.");
  TORCH_CHECK(query.dim() == 3 && query.size(2) == 512,
              "query must be [B, N, 512].");
  const auto num_heads = query.size(1);
  TORCH_CHECK(num_heads > 0 && num_heads <= 128 &&
                  (num_heads & (num_heads - 1)) == 0,
              "query local head count must be a power of two in [1, 128].");
  TORCH_CHECK(key.dim() == 4 && key.size(1) == 128 &&
                  key.size(2) == 1 && key.size(3) == 512,
              "key must be [blocks, 128, 1, 512].");
  TORCH_CHECK(value.sizes() == key.sizes(),
              "value must have the same shape as key.");
  TORCH_CHECK(value.is_same(key),
              "MLA sparse-and-tail Attention requires key and value to "
              "alias the same latent KV cache tensor.");
  TORCH_CHECK(query_rope.dim() == 3 &&
                  query_rope.size(0) == query.size(0) &&
                  query_rope.size(1) == query.size(1) &&
                  query_rope.size(2) == 64,
              "query_rope must be [B, N, 64].");
  TORCH_CHECK(key_rope.dim() == 4 &&
                  key_rope.size(0) == key.size(0) &&
                  key_rope.size(1) == 128 && key_rope.size(2) == 1 &&
                  key_rope.size(3) == 64,
              "key_rope must be [blocks, 128, 1, 64].");
  TORCH_CHECK(sparse_slots.dim() == 3 &&
                  sparse_slots.size(0) == query.size(0) &&
                  sparse_slots.size(1) == 1 && sparse_slots.size(2) == 2048,
              "sparse_slots must be [B, 1, 2048].");
  TORCH_CHECK(cache_tokens.dim() == 1 &&
                  cache_tokens.size(0) == query.size(0),
              "cache_tokens must be [B].");
  TORCH_CHECK(block_table.dim() == 2 &&
                  block_table.size(0) == query.size(0) &&
                  block_table.size(1) > 0,
              "block_table must be [B, max_blocks] with max_blocks > 0.");
  TORCH_CHECK(actual_seq_lengths_query.dim() == 1 &&
                  actual_seq_lengths_query.size(0) == query.size(0) &&
                  actual_seq_lengths_kv.dim() == 1 &&
                  actual_seq_lengths_kv.size(0) == query.size(0),
              "actual sequence lengths must be [B].");

  const auto dtype = query.scalar_type();
  TORCH_CHECK(dtype == at::kHalf || dtype == at::kBFloat16,
              "Sparse-and-tail Attention supports fp16 or bf16.");
  TORCH_CHECK(key.scalar_type() == dtype && value.scalar_type() == dtype &&
                  query_rope.scalar_type() == dtype &&
                  key_rope.scalar_type() == dtype,
              "All floating-point inputs must have the same dtype.");
  TORCH_CHECK(sparse_slots.scalar_type() == at::kInt &&
                  cache_tokens.scalar_type() == at::kInt &&
                  block_table.scalar_type() == at::kInt &&
                  actual_seq_lengths_query.scalar_type() == at::kInt &&
                  actual_seq_lengths_kv.scalar_type() == at::kInt,
              "Sparse slots, cache metadata, tables and lengths must be int32.");

  const auto device = query.device();
  TORCH_CHECK(key.device() == device && value.device() == device &&
                  sparse_slots.device() == device &&
                  cache_tokens.device() == device &&
                  block_table.device() == device &&
                  actual_seq_lengths_query.device() == device &&
                  actual_seq_lengths_kv.device() == device &&
                  query_rope.device() == device && key_rope.device() == device,
              "All Sparse-and-tail Attention inputs must be on the same NPU.");
  TORCH_CHECK(query.is_contiguous() && key.is_contiguous() &&
                  value.is_contiguous() && sparse_slots.is_contiguous() &&
                  cache_tokens.is_contiguous() && block_table.is_contiguous() &&
                  actual_seq_lengths_query.is_contiguous() &&
                  actual_seq_lengths_kv.is_contiguous() &&
                  query_rope.is_contiguous() && key_rope.is_contiguous(),
              "All Sparse-and-tail Attention inputs must be contiguous.");

  auto output = at::empty_like(query);
  std::string query_layout = "TND";
  std::string kv_layout = "PA_BSND";
  char* query_layout_ptr = const_cast<char*>(query_layout.c_str());
  char* kv_layout_ptr = const_cast<char*>(kv_layout.c_str());
  constexpr int64_t kSparseBlockSize = 1;
  constexpr int64_t kSparseMode = 3;
  auto keepalive = std::make_tuple(
      query, key, value, sparse_slots, cache_tokens, block_table,
      actual_seq_lengths_query, actual_seq_lengths_kv, query_rope, key_rope,
      output);
  EXEC_NPU_CMD_ORDERED(
      aclnnNanovllmSparseAndTailAttention,
      keepalive,
      query,
      key,
      value,
      sparse_slots,
      cache_tokens,
      block_table,
      actual_seq_lengths_query,
      actual_seq_lengths_kv,
      query_rope,
      key_rope,
      scale_value,
      kSparseBlockSize,
      query_layout_ptr,
      kv_layout_ptr,
      kSparseMode,
      output);
  return output;
}

}  // namespace vllm_ascend

#endif
