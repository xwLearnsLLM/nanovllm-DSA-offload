#include <algorithm>
#include <tuple>
#include <vector>

#include "ops_common.h"

namespace nanovllm_dsa_a5_impl {
namespace {

constexpr int64_t kC8QueryDim = kCkvDim + kKpeDim;
constexpr int64_t kMaxQueriesPerRequest = 4;
constexpr int64_t kMinQueriesPerRequest = 1;

// Returns query_counts[b] (the cumulative diff) for every request and validates
// the MTP packing contract: strictly increasing totals, last total == T, and
// per-request counts within [1, 4].
std::vector<int64_t> CheckSparseTailAttentionMtpC8Inputs(
    const at::Tensor& query,
    const at::Tensor& packed_kv,
    const at::Tensor& sparse_and_tail_slots,
    const at::Tensor& block_table,
    const at::Tensor& actual_seq_lengths_query,
    const at::Tensor& resident_seq_lengths) {
  TORCH_CHECK(
      query.dim() == 3 && query.size(0) > 0 &&
          query.size(1) >= 1 && query.size(1) <= 64 &&
          query.size(2) == kC8QueryDim,
      "C8 MTP sparse+tail attention query must be TND [T,Q_HEAD,576] "
      "with 1 <= Q_HEAD <= 64.");
  TORCH_CHECK(
      query.scalar_type() == at::kBFloat16 ||
          query.scalar_type() == at::kHalf,
      "C8 MTP sparse+tail attention query must be BF16 or FP16.");
  TORCH_CHECK(
      packed_kv.dim() == 4 && packed_kv.size(0) > 0 &&
          packed_kv.size(1) == kBlockSize && packed_kv.size(2) == 1 &&
          packed_kv.size(3) == kPackedKvDim && packed_kv.element_size() == 1,
      "C8 MTP sparse+tail attention packed KV must be a one-byte tensor "
      "[blocks,128,1,656].");
  TORCH_CHECK(
      packed_kv.scalar_type() == at::ScalarType::Float8_e4m3fn ||
          packed_kv.scalar_type() == at::kChar,
      "C8 MTP sparse+tail attention packed KV must be FP8 E4M3FN or INT8.");
  TORCH_CHECK(
      sparse_and_tail_slots.dim() == 3 &&
          sparse_and_tail_slots.size(0) == query.size(0) &&
          sparse_and_tail_slots.size(1) == 1 &&
          sparse_and_tail_slots.scalar_type() == at::kInt,
      "C8 MTP sparse+tail slots must be int32 [T,1,2048+max_tail_tokens].");
  const int64_t batch = actual_seq_lengths_query.numel();
  TORCH_CHECK(
      batch > 0 && actual_seq_lengths_query.dim() == 1 &&
          resident_seq_lengths.dim() == 1 &&
          resident_seq_lengths.numel() == batch &&
          block_table.dim() == 2 && block_table.size(0) == batch &&
          block_table.size(1) > 0,
      "C8 MTP sparse+tail attention batch metadata shapes are inconsistent.");
  for (const at::Tensor* tensor : {
           &block_table, &actual_seq_lengths_query,
           &resident_seq_lengths}) {
    TORCH_CHECK(
        tensor->scalar_type() == at::kInt,
        "C8 MTP sparse+tail attention metadata must be int32.");
  }
  CheckOneDeviceAndContiguous(
      query,
      {&query, &packed_kv, &sparse_and_tail_slots, &block_table,
       &actual_seq_lengths_query, &resident_seq_lengths},
      "C8 MTP sparse+tail attention");

  const auto totals = actual_seq_lengths_query.cpu();
  const auto residents = resident_seq_lengths.cpu();
  const auto totals_a = totals.accessor<int32_t, 1>();
  const auto residents_a = residents.accessor<int32_t, 1>();
  std::vector<int64_t> query_counts(static_cast<size_t>(batch));
  int64_t previous = 0;
  int64_t max_count = 0;
  for (int64_t b = 0; b < batch; ++b) {
    const int64_t total = totals_a[b];
    const int64_t count = total - previous;
    TORCH_CHECK(
        count >= kMinQueriesPerRequest && count <= kMaxQueriesPerRequest,
        "C8 MTP sparse+tail attention actual_seq_lengths_query must be "
        "strictly increasing with per-request diffs in [1,4]; request ",
        b, " has diff ", count, ".");
    TORCH_CHECK(
        residents_a[b] >= count,
        "C8 MTP sparse+tail attention resident_seq_lengths[b] must cover "
        "query_counts[b]; request ", b, " has ", residents_a[b], " < ",
        count, ".");
    const int64_t cache_tokens = residents_a[b] - count;
    TORCH_CHECK(
        cache_tokens == 0 ||
            (cache_tokens >= kSparseCount && cache_tokens % kBlockSize == 0),
        "C8 MTP sparse+tail attention cache_tokens must be 0 or "
        "block-aligned >= 2048; request ", b, " has ", cache_tokens, ".");
    query_counts[static_cast<size_t>(b)] = count;
    max_count = std::max(max_count, count);
    previous = total;
  }
  TORCH_CHECK(
      previous == query.size(0),
      "C8 MTP sparse+tail attention actual_seq_lengths_query last value "
      "must equal the packed query row count T.");
  TORCH_CHECK(
      sparse_and_tail_slots.size(2) >= kSparseCount + max_count,
      "C8 MTP sparse+tail slots row width must cover 2048 + the largest "
      "per-request query count.");
  return query_counts;
}

}  // namespace

at::Tensor SparseTailAttentionMtpC8Npu(
    const at::Tensor& query,
    const at::Tensor& packed_kv,
    const at::Tensor& sparse_and_tail_slots,
    const at::Tensor& block_table,
    const at::Tensor& actual_seq_lengths_query,
    const at::Tensor& resident_seq_lengths,
    double scale_value) {
  CheckSparseTailAttentionMtpC8Inputs(
      query, packed_kv, sparse_and_tail_slots, block_table,
      actual_seq_lengths_query, resident_seq_lengths);

  auto attention_out = at::empty(
      {query.size(0), query.size(1), kCkvDim}, query.options());
  // CANN/ACLNN versions differ on required zero-size output descriptors.
  // The local operator never returns LSE, so stable one-element placeholders
  // are safer and are ignored by the device kernel.
  auto softmax_max = at::empty({1}, query.options().dtype(at::kFloat));
  auto softmax_sum = at::empty({1}, query.options().dtype(at::kFloat));

  const c10::optional<at::Tensor> no_external_scale = c10::nullopt;
  constexpr int64_t key_quant_mode = 2;
  constexpr int64_t value_quant_mode = 2;
  constexpr int64_t sparse_block_size = 1;
  std::string layout_query = "TND";
  std::string layout_kv = "PA_BSND";
  char* layout_query_ptr = layout_query.data();
  char* layout_kv_ptr = layout_kv.data();
  constexpr int64_t sparse_mode = 3;
  constexpr int64_t all_tokens = std::numeric_limits<int64_t>::max();
  constexpr int64_t attention_mode = 2;
  constexpr int64_t quant_scale_repo_mode = 1;
  constexpr int64_t tile_size = 128;
  constexpr int64_t rope_head_dim = kKpeDim;
  constexpr bool return_softmax_lse = false;

  auto keepalive = std::make_tuple(
      query, packed_kv, sparse_and_tail_slots, block_table,
      actual_seq_lengths_query, resident_seq_lengths,
      attention_out, softmax_max, softmax_sum);
  EXEC_NPU_CMD_ORDERED(
      aclnnA5SparseTailAttentionMtpC8,
      keepalive,
      query,
      packed_kv,
      packed_kv,
      sparse_and_tail_slots,
      no_external_scale,
      no_external_scale,
      block_table,
      actual_seq_lengths_query,
      resident_seq_lengths,
      scale_value,
      key_quant_mode,
      value_quant_mode,
      sparse_block_size,
      layout_query_ptr,
      layout_kv_ptr,
      sparse_mode,
      all_tokens,
      all_tokens,
      attention_mode,
      quant_scale_repo_mode,
      tile_size,
      rope_head_dim,
      return_softmax_lse,
      attention_out,
      softmax_max,
      softmax_sum);
  return attention_out;
}

at::Tensor SparseTailAttentionMtpC8Meta(
    const at::Tensor& query,
    const at::Tensor&,
    const at::Tensor&,
    const at::Tensor&,
    const at::Tensor&,
    const at::Tensor&,
    double) {
  TORCH_CHECK(
      query.dim() == 3 && query.size(0) > 0 &&
          query.size(1) >= 1 && query.size(1) <= 64 &&
          query.size(2) == kC8QueryDim,
      "C8 MTP sparse+tail attention query must be [T,Q_HEAD,576].");
  return at::empty(
      {query.size(0), query.size(1), kCkvDim}, query.options());
}

}  // namespace nanovllm_dsa_a5_impl

TORCH_LIBRARY_IMPL(nanovllm_dsa, PrivateUse1, m) {
  m.impl(
      "sparse_tail_attention_mtp_c8",
      &nanovllm_dsa_a5_impl::SparseTailAttentionMtpC8Npu);
}

TORCH_LIBRARY_IMPL(nanovllm_dsa, Meta, m) {
  m.impl(
      "sparse_tail_attention_mtp_c8",
      &nanovllm_dsa_a5_impl::SparseTailAttentionMtpC8Meta);
}
