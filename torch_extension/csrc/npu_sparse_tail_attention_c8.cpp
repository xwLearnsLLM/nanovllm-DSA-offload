#include <tuple>

#include "ops_common.h"

namespace nanovllm_dsa_a5_impl {
namespace {

constexpr int64_t kC8QueryDim = kCkvDim + kKpeDim;

void CheckSparseTailAttentionC8Inputs(
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
      "C8 sparse+tail attention query must be TND [T,Q_HEAD,576] "
      "with 1 <= Q_HEAD <= 64.");
  TORCH_CHECK(
      query.scalar_type() == at::kBFloat16 ||
          query.scalar_type() == at::kHalf,
      "C8 sparse+tail attention query must be BF16 or FP16.");
  TORCH_CHECK(
      packed_kv.dim() == 4 && packed_kv.size(0) > 0 &&
          packed_kv.size(1) == kBlockSize && packed_kv.size(2) == 1 &&
          packed_kv.size(3) == kPackedKvDim && packed_kv.element_size() == 1,
      "C8 sparse+tail attention packed KV must be a one-byte tensor "
      "[blocks,128,1,656].");
  TORCH_CHECK(
      packed_kv.scalar_type() == at::ScalarType::Float8_e4m3fn ||
          packed_kv.scalar_type() == at::kChar,
      "C8 sparse+tail attention packed KV must be FP8 E4M3FN or INT8.");
  TORCH_CHECK(
      sparse_and_tail_slots.dim() == 3 &&
          sparse_and_tail_slots.size(0) == query.size(0) &&
          sparse_and_tail_slots.size(1) == 1 &&
          sparse_and_tail_slots.size(2) >= kSparseCount &&
          sparse_and_tail_slots.scalar_type() == at::kInt,
      "C8 sparse+tail slots must be int32 [T,1,2048+max_tail_tokens].");
  const int64_t batch = actual_seq_lengths_query.numel();
  TORCH_CHECK(
      batch > 0 && actual_seq_lengths_query.dim() == 1 &&
          resident_seq_lengths.dim() == 1 &&
          resident_seq_lengths.numel() == batch &&
          block_table.dim() == 2 && block_table.size(0) == batch &&
          block_table.size(1) > 0,
      "C8 sparse+tail attention batch metadata shapes are inconsistent.");
  for (const at::Tensor* tensor : {
           &block_table, &actual_seq_lengths_query,
           &resident_seq_lengths}) {
    TORCH_CHECK(
        tensor->scalar_type() == at::kInt,
        "C8 sparse+tail attention metadata must be int32.");
  }
  CheckOneDeviceAndContiguous(
      query,
      {&query, &packed_kv, &sparse_and_tail_slots, &block_table,
       &actual_seq_lengths_query, &resident_seq_lengths},
      "C8 sparse+tail attention");
}

}  // namespace

at::Tensor SparseTailAttentionC8Npu(
    const at::Tensor& query,
    const at::Tensor& packed_kv,
    const at::Tensor& sparse_and_tail_slots,
    const at::Tensor& block_table,
    const at::Tensor& actual_seq_lengths_query,
    const at::Tensor& resident_seq_lengths,
    double scale_value) {
  CheckSparseTailAttentionC8Inputs(
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
      aclnnA5SparseTailAttentionC8,
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

at::Tensor SparseTailAttentionC8Meta(
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
      "C8 sparse+tail attention query must be [T,Q_HEAD,576].");
  return at::empty(
      {query.size(0), query.size(1), kCkvDim}, query.options());
}

void CheckC8StagedBuffers(
    const at::Tensor& query,
    const at::Tensor& sparse_and_tail_slots,
    const at::Tensor& miss_counts,
    const at::Tensor& partial,
    const at::Tensor& max,
    const at::Tensor& sum) {
  const int64_t batch = query.size(0);
  TORCH_CHECK(
      sparse_and_tail_slots.dim() == 3 &&
          sparse_and_tail_slots.size(0) == batch &&
          sparse_and_tail_slots.size(1) == 1 &&
          sparse_and_tail_slots.size(2) >= kSparseCount &&
          sparse_and_tail_slots.scalar_type() == at::kInt,
      "C8 staged slots must be int32 [B,1,2048+max_tail_tokens].");
  TORCH_CHECK(
      miss_counts.dim() == 1 && miss_counts.size(0) == batch &&
          miss_counts.scalar_type() == at::kInt,
      "C8 staged miss_counts must be int32 [B].");
  TORCH_CHECK(
      partial.sizes() == at::IntArrayRef({batch, query.size(1), kCkvDim}) &&
          partial.scalar_type() == at::kFloat,
      "C8 staged P must be fp32 [B,Q_HEAD,512].");
  TORCH_CHECK(
      max.sizes() == at::IntArrayRef({1, batch, query.size(1)}) &&
          sum.sizes() == max.sizes() && max.scalar_type() == at::kFloat &&
          sum.scalar_type() == at::kFloat,
      "C8 staged M/L must be fp32 [1,B,Q_HEAD].");
}

struct C8StagedAttrs {
  explicit C8StagedAttrs(double scale) : scale_value(scale) {}
  double scale_value;
  int64_t key_quant_mode = 2;
  int64_t value_quant_mode = 2;
  int64_t sparse_block_size = 1;
  std::string query_layout = "TND";
  std::string kv_layout = "PA_BSND";
  int64_t sparse_mode = 3;
  int64_t all_tokens = std::numeric_limits<int64_t>::max();
  int64_t attention_mode = 2;
  int64_t quant_scale_repo_mode = 1;
  int64_t tile_size = 128;
  int64_t rope_head_dim = kKpeDim;
  bool return_softmax_lse = false;
};

void SparseTailAttentionC8Stage1Npu(
    const at::Tensor& query, const at::Tensor& packed_kv,
    const at::Tensor& sparse_and_tail_slots,
    const at::Tensor& block_table,
    const at::Tensor& actual_seq_lengths_query,
    const at::Tensor& resident_seq_lengths,
    const at::Tensor& miss_counts, double scale_value,
    const at::Tensor& partial_out, const at::Tensor& softmax_max,
    const at::Tensor& softmax_sum) {
  CheckSparseTailAttentionC8Inputs(
      query, packed_kv, sparse_and_tail_slots, block_table,
      actual_seq_lengths_query, resident_seq_lengths);
  CheckC8StagedBuffers(
      query, sparse_and_tail_slots, miss_counts, partial_out,
      softmax_max, softmax_sum);
  CheckOneDeviceAndContiguous(
      query, {&miss_counts, &partial_out, &softmax_max, &softmax_sum},
      "C8 staged attention");
  const c10::optional<at::Tensor> no_scale = c10::nullopt;
  C8StagedAttrs attrs(scale_value);
  auto keepalive = std::make_tuple(
      query, packed_kv, sparse_and_tail_slots, block_table,
      actual_seq_lengths_query, resident_seq_lengths, miss_counts,
      partial_out, softmax_max, softmax_sum);
  EXEC_NPU_CMD_ORDERED(
      aclnnA5SparseTailAttentionC8Stage1, keepalive,
      query, packed_kv, packed_kv, sparse_and_tail_slots,
      no_scale, no_scale, block_table, actual_seq_lengths_query,
      resident_seq_lengths, miss_counts, attrs.scale_value,
      attrs.key_quant_mode, attrs.value_quant_mode,
      attrs.sparse_block_size, attrs.query_layout.data(),
      attrs.kv_layout.data(), attrs.sparse_mode, attrs.all_tokens,
      attrs.all_tokens, attrs.attention_mode,
      attrs.quant_scale_repo_mode, attrs.tile_size,
      attrs.rope_head_dim, attrs.return_softmax_lse,
      partial_out, softmax_max, softmax_sum);
}

void SparseTailAttentionC8Stage2Npu(
    const at::Tensor& query, const at::Tensor& packed_kv,
    const at::Tensor& sparse_and_tail_slots,
    const at::Tensor& block_table,
    const at::Tensor& actual_seq_lengths_query,
    const at::Tensor& resident_seq_lengths,
    const at::Tensor& miss_counts, double scale_value,
    const at::Tensor& previous_p, const at::Tensor& previous_m,
    const at::Tensor& previous_l, const at::Tensor& attention_out) {
  CheckSparseTailAttentionC8Inputs(
      query, packed_kv, sparse_and_tail_slots, block_table,
      actual_seq_lengths_query, resident_seq_lengths);
  CheckC8StagedBuffers(
      query, sparse_and_tail_slots, miss_counts, previous_p,
      previous_m, previous_l);
  TORCH_CHECK(
      attention_out.sizes() == previous_p.sizes() &&
          attention_out.scalar_type() == query.scalar_type(),
      "C8 staged output must be [B,Q_HEAD,512] with query dtype.");
  CheckOneDeviceAndContiguous(
      query, {&miss_counts, &previous_p, &previous_m, &previous_l,
              &attention_out}, "C8 staged attention");
  const c10::optional<at::Tensor> no_scale = c10::nullopt;
  C8StagedAttrs attrs(scale_value);
  auto keepalive = std::make_tuple(
      query, packed_kv, sparse_and_tail_slots, block_table,
      actual_seq_lengths_query, resident_seq_lengths, miss_counts,
      previous_p, previous_m, previous_l, attention_out);
  EXEC_NPU_CMD_ORDERED(
      aclnnA5SparseTailAttentionC8Stage2, keepalive,
      query, packed_kv, packed_kv, sparse_and_tail_slots,
      no_scale, no_scale, block_table, actual_seq_lengths_query,
      resident_seq_lengths, miss_counts, previous_p, previous_m,
      previous_l, attrs.scale_value, attrs.key_quant_mode,
      attrs.value_quant_mode, attrs.sparse_block_size,
      attrs.query_layout.data(), attrs.kv_layout.data(),
      attrs.sparse_mode, attrs.all_tokens, attrs.all_tokens,
      attrs.attention_mode, attrs.quant_scale_repo_mode,
      attrs.tile_size, attrs.rope_head_dim, attrs.return_softmax_lse,
      attention_out);
}

void SparseTailAttentionC8Stage1Meta(
    const at::Tensor& query, const at::Tensor&,
    const at::Tensor& sparse_and_tail_slots, const at::Tensor&,
    const at::Tensor&, const at::Tensor&, const at::Tensor& miss_counts,
    double, const at::Tensor& partial_out, const at::Tensor& softmax_max,
    const at::Tensor& softmax_sum) {
  CheckC8StagedBuffers(
      query, sparse_and_tail_slots, miss_counts, partial_out,
      softmax_max, softmax_sum);
}

void SparseTailAttentionC8Stage2Meta(
    const at::Tensor& query, const at::Tensor&,
    const at::Tensor& sparse_and_tail_slots, const at::Tensor&,
    const at::Tensor&, const at::Tensor&, const at::Tensor& miss_counts,
    double, const at::Tensor& previous_p, const at::Tensor& previous_m,
    const at::Tensor& previous_l, const at::Tensor& attention_out) {
  CheckC8StagedBuffers(
      query, sparse_and_tail_slots, miss_counts, previous_p,
      previous_m, previous_l);
  TORCH_CHECK(
      attention_out.sizes() == previous_p.sizes() &&
          attention_out.scalar_type() == query.scalar_type(),
      "C8 staged output must be [B,Q_HEAD,512] with query dtype.");
}

}  // namespace nanovllm_dsa_a5_impl

TORCH_LIBRARY_IMPL(nanovllm_dsa, PrivateUse1, m) {
  m.impl(
      "sparse_tail_attention_c8",
      &nanovllm_dsa_a5_impl::SparseTailAttentionC8Npu);
  m.impl("sparse_tail_attention_c8_stage1",
         &nanovllm_dsa_a5_impl::SparseTailAttentionC8Stage1Npu);
  m.impl("sparse_tail_attention_c8_stage2",
         &nanovllm_dsa_a5_impl::SparseTailAttentionC8Stage2Npu);
}

TORCH_LIBRARY_IMPL(nanovllm_dsa, Meta, m) {
  m.impl(
      "sparse_tail_attention_c8",
      &nanovllm_dsa_a5_impl::SparseTailAttentionC8Meta);
  m.impl("sparse_tail_attention_c8_stage1",
         &nanovllm_dsa_a5_impl::SparseTailAttentionC8Stage1Meta);
  m.impl("sparse_tail_attention_c8_stage2",
         &nanovllm_dsa_a5_impl::SparseTailAttentionC8Stage2Meta);
}
