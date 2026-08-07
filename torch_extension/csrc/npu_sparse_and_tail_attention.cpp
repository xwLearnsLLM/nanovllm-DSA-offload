#include <limits>
#include <string>
#include <tuple>

#include "ops_common.h"

namespace nanovllm_dsa_a5_impl {

void CheckAttentionInputs(
    const at::Tensor& query,
    const at::Tensor& key,
    const at::Tensor& value,
    const at::Tensor& sparse_slots,
    const at::Tensor& cache_tokens,
    const at::Tensor& block_table,
    const at::Tensor& actual_q,
    const at::Tensor& actual_kv,
    const at::Tensor& query_rope,
    const at::Tensor& key_rope) {
  TORCH_CHECK(
      query.dim() == 3 && query.size(0) > 0 && query.size(2) == kCkvDim &&
          query.size(1) >= 1 && query.size(1) <= 64,
      "SFA query must be [B,N,512] with 1 <= N <= 64.");
  const int64_t batch = query.size(0);
  const int64_t heads = query.size(1);
  TORCH_CHECK(
      key.dim() == 4 && key.size(1) == kBlockSize && key.size(2) == 1 &&
          key.size(3) == kCkvDim && value.sizes() == key.sizes(),
      "SFA key/value must be [blocks,128,1,512].");
  TORCH_CHECK(
      value.data_ptr() == key.data_ptr(),
      "SFA is the MLA path and requires value to alias key storage.");
  TORCH_CHECK(
      key_rope.dim() == 4 && key_rope.size(0) == key.size(0) &&
          key_rope.size(1) == kBlockSize && key_rope.size(2) == 1 &&
          key_rope.size(3) == kKpeDim &&
          query_rope.dim() == 3 && query_rope.size(0) == batch &&
          query_rope.size(1) == heads && query_rope.size(2) == kKpeDim,
      "SFA RoPE tensors must be [blocks,128,1,64] and [B,N,64].");
  TORCH_CHECK(
      sparse_slots.dim() == 3 && sparse_slots.size(0) == batch &&
          sparse_slots.size(1) == 1 && sparse_slots.size(2) == kSparseCount &&
          cache_tokens.dim() == 1 && cache_tokens.size(0) == batch &&
          actual_q.dim() == 1 && actual_q.size(0) == batch &&
          actual_kv.dim() == 1 && actual_kv.size(0) == batch &&
          block_table.dim() == 2 && block_table.size(0) == batch &&
          block_table.size(1) > 0 &&
          block_table.size(1) * kBlockSize <= kMaxSourceCapacity,
      "SFA metadata shapes are invalid.");
  const auto dtype = query.scalar_type();
  TORCH_CHECK(dtype == at::kBFloat16 || dtype == at::kHalf, "SFA supports bf16/fp16.");
  for (const at::Tensor* tensor : {&key, &value, &query_rope, &key_rope}) {
    TORCH_CHECK(tensor->scalar_type() == dtype, "All SFA floating dtypes must match.");
  }
  for (const at::Tensor* tensor :
       {&sparse_slots, &cache_tokens, &block_table, &actual_q, &actual_kv}) {
    TORCH_CHECK(tensor->scalar_type() == at::kInt, "SFA metadata must be int32.");
  }
  CheckOneDeviceAndContiguous(
      query,
      {&query, &key, &value, &sparse_slots, &cache_tokens, &block_table,
       &actual_q, &actual_kv, &query_rope, &key_rope},
      "SFA");
}

at::Tensor SparseAndTailAttentionNpu(
    const at::Tensor& query,
    const at::Tensor& key,
    const at::Tensor& value,
    const at::Tensor& sparse_slots,
    const at::Tensor& cache_tokens,
    const at::Tensor& block_table,
    const at::Tensor& actual_q,
    const at::Tensor& actual_kv,
    const at::Tensor& query_rope,
    const at::Tensor& key_rope,
    double scale_value) {
  CheckAttentionInputs(
      query, key, value, sparse_slots, cache_tokens, block_table,
      actual_q, actual_kv, query_rope, key_rope);
  auto output = at::empty_like(query);
  auto softmax_max = at::empty({1}, query.options().dtype(at::kFloat));
  auto softmax_sum = at::empty({1}, query.options().dtype(at::kFloat));
  std::string query_layout = "TND";
  std::string kv_layout = "PA_BSND";
  char* query_layout_ptr = const_cast<char*>(query_layout.c_str());
  char* kv_layout_ptr = const_cast<char*>(kv_layout.c_str());
  constexpr int64_t kSparseBlockSize = 1;
  constexpr int64_t kSparseMode = 3;
  constexpr int64_t kAllTokens = std::numeric_limits<int64_t>::max();
  constexpr int64_t kAttentionMode = 2;
  constexpr bool kReturnSoftmaxLse = false;
  float scale_value_float = static_cast<float>(scale_value);
  auto keepalive = std::make_tuple(
      query, key, value, sparse_slots, block_table, actual_q, actual_kv,
      query_rope, key_rope, cache_tokens, output, softmax_max, softmax_sum);
  EXEC_NPU_CMD_ORDERED(
      aclnnA5SparseAndTailAttention,
      keepalive,
      query,
      key,
      value,
      sparse_slots,
      block_table,
      actual_q,
      actual_kv,
      query_rope,
      key_rope,
      cache_tokens,
      scale_value_float,
      kSparseBlockSize,
      query_layout_ptr,
      kv_layout_ptr,
      kSparseMode,
      kAllTokens,
      kAllTokens,
      kAttentionMode,
      kReturnSoftmaxLse,
      output,
      softmax_max,
      softmax_sum);
  return output;
}

at::Tensor SparseAndTailAttentionMeta(
    const at::Tensor& query,
    const at::Tensor&,
    const at::Tensor&,
    const at::Tensor&,
    const at::Tensor&,
    const at::Tensor&,
    const at::Tensor&,
    const at::Tensor&,
    const at::Tensor&,
    const at::Tensor&,
    double) {
  TORCH_CHECK(
      query.dim() == 3 && query.size(0) > 0 && query.size(2) == kCkvDim &&
          query.size(1) >= 1 && query.size(1) <= 64,
      "SFA query must be [B,N,512] with 1 <= N <= 64.");
  return at::empty_like(query);
}

}  // namespace nanovllm_dsa_a5_impl

TORCH_LIBRARY_IMPL(nanovllm_dsa, PrivateUse1, m) {
  m.impl("sparse_and_tail_attention", &nanovllm_dsa_a5_impl::SparseAndTailAttentionNpu);
}

TORCH_LIBRARY_IMPL(nanovllm_dsa, Meta, m) {
  m.impl("sparse_and_tail_attention", &nanovllm_dsa_a5_impl::SparseAndTailAttentionMeta);
}
