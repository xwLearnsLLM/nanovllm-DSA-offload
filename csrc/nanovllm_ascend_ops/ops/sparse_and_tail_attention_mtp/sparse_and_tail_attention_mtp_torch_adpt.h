#ifndef NANOVLLM_SPARSE_AND_TAIL_ATTENTION_MTP_TORCH_ADPT_H
#define NANOVLLM_SPARSE_AND_TAIL_ATTENTION_MTP_TORCH_ADPT_H

namespace vllm_ascend {

inline void npu_sparse_and_tail_attention_mtp_out(
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
    double scale_value,
    at::Tensor attention_out) {
  constexpr int64_t kQueryCount = 4;
  constexpr int64_t kTopK = 2048;
  TORCH_CHECK(query.device().is_privateuseone(),
              "MTP sparse-and-tail Attention inputs must be on NPU.");
  TORCH_CHECK(query.dim() == 3 && query.size(2) == 512,
              "MTP query must be [4B, N, 512].");
  const auto num_heads = query.size(1);
  TORCH_CHECK(num_heads > 0 && num_heads <= 128 &&
                  (num_heads & (num_heads - 1)) == 0,
              "MTP query local head count must be a power of two in [1, 128].");
  TORCH_CHECK(cache_tokens.dim() == 1 && cache_tokens.size(0) > 0,
              "MTP cache_tokens must be [B] with B > 0.");
  const auto batch_size = cache_tokens.size(0);
  TORCH_CHECK(query.size(0) == batch_size * kQueryCount,
              "MTP Attention requires exactly four packed queries per request.");
  TORCH_CHECK(key.dim() == 4 && key.size(1) == 128 &&
                  key.size(2) == 1 && key.size(3) == 512,
              "MTP key must be [blocks, 128, 1, 512].");
  TORCH_CHECK(value.sizes() == key.sizes() && value.is_same(key),
              "MTP MLA requires value to alias the latent key cache.");
  TORCH_CHECK(query_rope.dim() == 3 &&
                  query_rope.size(0) == query.size(0) &&
                  query_rope.size(1) == num_heads &&
                  query_rope.size(2) == 64,
              "MTP query_rope must be [4B, N, 64].");
  TORCH_CHECK(key_rope.dim() == 4 &&
                  key_rope.size(0) == key.size(0) &&
                  key_rope.size(1) == 128 && key_rope.size(2) == 1 &&
                  key_rope.size(3) == 64,
              "MTP key_rope must be [blocks, 128, 1, 64].");
  TORCH_CHECK(sparse_slots.dim() == 3 &&
                  sparse_slots.size(0) == query.size(0) &&
                  sparse_slots.size(1) == 1 &&
                  sparse_slots.size(2) == kTopK,
              "MTP sparse_slots must be [4B, 1, 2048].");
  TORCH_CHECK(block_table.dim() == 2 &&
                  block_table.size(0) == batch_size &&
                  block_table.size(1) > 0,
              "MTP block_table must be [B, max_blocks].");
  TORCH_CHECK(actual_seq_lengths_query.dim() == 1 &&
                  actual_seq_lengths_query.size(0) == batch_size &&
                  actual_seq_lengths_kv.dim() == 1 &&
                  actual_seq_lengths_kv.size(0) == batch_size,
              "MTP actual sequence lengths must be [B].");
  TORCH_CHECK(attention_out.sizes() == query.sizes(),
              "MTP attention_out must have the same shape as query.");

  const auto dtype = query.scalar_type();
  TORCH_CHECK(dtype == at::kHalf || dtype == at::kBFloat16,
              "MTP sparse-and-tail Attention supports fp16 or bf16.");
  TORCH_CHECK(key.scalar_type() == dtype && value.scalar_type() == dtype &&
                  query_rope.scalar_type() == dtype &&
                  key_rope.scalar_type() == dtype &&
                  attention_out.scalar_type() == dtype,
              "All MTP floating-point tensors must have the same dtype.");
  TORCH_CHECK(sparse_slots.scalar_type() == at::kInt &&
                  cache_tokens.scalar_type() == at::kInt &&
                  block_table.scalar_type() == at::kInt &&
                  actual_seq_lengths_query.scalar_type() == at::kInt &&
                  actual_seq_lengths_kv.scalar_type() == at::kInt,
              "MTP sparse slots, tables and lengths must be int32.");

  const auto device = query.device();
  TORCH_CHECK(key.device() == device && value.device() == device &&
                  sparse_slots.device() == device &&
                  cache_tokens.device() == device &&
                  block_table.device() == device &&
                  actual_seq_lengths_query.device() == device &&
                  actual_seq_lengths_kv.device() == device &&
                  query_rope.device() == device && key_rope.device() == device &&
                  attention_out.device() == device,
              "All MTP Attention tensors must be on the same NPU.");
  TORCH_CHECK(query.is_contiguous() && key.is_contiguous() &&
                  value.is_contiguous() && sparse_slots.is_contiguous() &&
                  cache_tokens.is_contiguous() && block_table.is_contiguous() &&
                  actual_seq_lengths_query.is_contiguous() &&
                  actual_seq_lengths_kv.is_contiguous() &&
                  query_rope.is_contiguous() && key_rope.is_contiguous() &&
                  attention_out.is_contiguous(),
              "All MTP Attention tensors must be contiguous.");

  std::string query_layout = "TND";
  std::string kv_layout = "PA_BSND";
  char* query_layout_ptr = const_cast<char*>(query_layout.c_str());
  char* kv_layout_ptr = const_cast<char*>(kv_layout.c_str());
  constexpr int64_t kSparseBlockSize = 1;
  constexpr int64_t kSparseMode = 3;
  auto keepalive = std::make_tuple(
      query, key, value, sparse_slots, cache_tokens, block_table,
      actual_seq_lengths_query, actual_seq_lengths_kv, query_rope, key_rope,
      attention_out);
  EXEC_NPU_CMD_ORDERED(
      aclnnNanovllmSparseAndTailAttentionMtp,
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
      attention_out);
}

inline at::Tensor npu_sparse_and_tail_attention_mtp(
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
  auto attention_out = at::empty_like(query);
  npu_sparse_and_tail_attention_mtp_out(
      query, key, value, sparse_slots, cache_tokens, block_table,
      actual_seq_lengths_query, actual_seq_lengths_kv, query_rope, key_rope,
      scale_value, attention_out);
  return attention_out;
}

} // namespace vllm_ascend

#endif
