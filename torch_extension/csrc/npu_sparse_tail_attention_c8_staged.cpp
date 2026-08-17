#include <limits>
#include <string>
#include <tuple>

#include "ops_common.h"

namespace nanovllm_dsa_a5_impl {

constexpr int64_t kHifloat8Dtype = 34;

std::tuple<at::Tensor, at::Tensor, at::Tensor> C8StateOutNpu(
    const at::Tensor& query,
    const at::Tensor& packed_kv,
    const at::Tensor& topk_slots,
    const at::Tensor& block_table,
    const at::Tensor& actual_q,
    const at::Tensor& actual_kv,
    const at::Tensor& miss_counts,
    const at::Tensor& cache_tokens,
    double scale_value,
    at::Tensor partial_out,
    at::Tensor softmax_max,
    at::Tensor softmax_sum,
    c10::optional<int64_t> kv_dtype) {
  TORCH_CHECK(
      query.dim() == 3 && query.size(2) == kCkvDim + kKpeDim,
      "C8 state query must be [T,N,576].");
  TORCH_CHECK(
      packed_kv.dim() == 4 && packed_kv.size(1) == kBlockSize &&
          packed_kv.size(2) == 1 && packed_kv.size(3) == kPackedKvDim &&
          packed_kv.element_size() == 1,
      "C8 state packed KV must be one-byte [blocks,128,1,656].");
  TORCH_CHECK(
      topk_slots.dim() == 3 && topk_slots.size(0) == query.size(0) &&
          topk_slots.size(1) == 1 && topk_slots.size(2) == kSparseCount,
      "C8 state topk_slots must be [T,1,2048].");
  TORCH_CHECK(
      miss_counts.dim() == 1 && miss_counts.size(0) == query.size(0) &&
          actual_q.dim() == 1 && actual_kv.sizes() == actual_q.sizes() &&
          cache_tokens.sizes() == actual_q.sizes() && block_table.dim() == 2 &&
          block_table.size(0) == actual_q.size(0),
      "C8 state metadata shapes are inconsistent.");
  TORCH_CHECK(
      partial_out.dim() == 3 && partial_out.size(0) == query.size(0) &&
          partial_out.size(1) == query.size(1) &&
          partial_out.size(2) == kCkvDim &&
          softmax_max.dim() == 3 && softmax_max.size(0) == 1 &&
          softmax_max.size(1) == query.size(0) &&
          softmax_max.size(2) == query.size(1) &&
          softmax_sum.sizes() == softmax_max.sizes(),
      "C8 state outputs must be P[T,N,512], M/L[1,T,N].");
  for (const at::Tensor* tensor :
       {&topk_slots, &block_table, &actual_q, &actual_kv, &miss_counts,
        &cache_tokens}) {
    TORCH_CHECK(tensor->scalar_type() == at::kInt, "C8 state metadata must be int32.");
  }
  for (const at::Tensor* tensor : {&partial_out, &softmax_max, &softmax_sum}) {
    TORCH_CHECK(tensor->scalar_type() == at::kFloat, "C8 state outputs must be FP32.");
  }
  CheckOneDeviceAndContiguous(
      query,
      {&query, &packed_kv, &topk_slots, &block_table, &actual_q,
       &actual_kv, &miss_counts, &cache_tokens, &partial_out, &softmax_max,
       &softmax_sum},
      "C8 state");

  c10::optional<at::Tensor> no_scale;
  c10::optional<at::Tensor> optional_block_table = block_table;
  c10::optional<at::Tensor> optional_actual_q = actual_q;
  c10::optional<at::Tensor> optional_actual_kv = actual_kv;
  std::string query_layout = "TND";
  std::string kv_layout = "PA_BSND";
  char* query_layout_ptr = query_layout.data();
  char* kv_layout_ptr = kv_layout.data();
  const int64_t key_quant_mode = 2;
  const int64_t value_quant_mode = 2;
  const int64_t sparse_block_size = 1;
  const int64_t sparse_mode = 3;
  const int64_t all_tokens = std::numeric_limits<int64_t>::max();
  const int64_t attention_mode = 2;
  const int64_t quant_scale_repo_mode = 1;
  const int64_t tile_size = 128;
  const int64_t rope_head_dim = 64;
  int64_t stage_mode = 1;
  auto keepalive = std::make_tuple(
      query, packed_kv, topk_slots, block_table, actual_q, actual_kv,
      miss_counts, cache_tokens, partial_out, softmax_max, softmax_sum);
  auto launch = [&](const auto& packed_kv_arg) {
    EXEC_NPU_CMD_ORDERED(
        aclnnA5SparseTailAttentionC8State,
        keepalive,
        query,
        packed_kv_arg,
        packed_kv_arg,
        topk_slots,
        no_scale,
        no_scale,
        optional_block_table,
        optional_actual_q,
        optional_actual_kv,
        miss_counts,
        cache_tokens,
        scale_value,
        key_quant_mode,
        value_quant_mode,
        sparse_block_size,
        query_layout_ptr,
        kv_layout_ptr,
        sparse_mode,
        all_tokens,
        all_tokens,
        attention_mode,
        quant_scale_repo_mode,
        tile_size,
        rope_head_dim,
        stage_mode,
        partial_out,
        softmax_max,
        softmax_sum);
  };
  if (kv_dtype.has_value()) {
    TORCH_CHECK(
        kv_dtype.value() == kHifloat8Dtype,
        "C8 state kv_dtype override only supports torch_npu.hifloat8.");
    TORCH_CHECK(
        packed_kv.scalar_type() == at::kByte,
        "C8 state HIFLOAT8 storage must use a torch.uint8 tensor.");
    const TensorWrapper packed_kv_wrapper{
        packed_kv, aclDataType::ACL_HIFLOAT8};
    launch(packed_kv_wrapper);
  } else {
    TORCH_CHECK(
        packed_kv.scalar_type() == at::ScalarType::Float8_e4m3fn ||
            packed_kv.scalar_type() == at::kChar,
        "C8 state packed KV must be native float8_e4m3fn/int8, or "
        "torch.uint8 with kv_dtype=torch_npu.hifloat8.");
    launch(packed_kv);
  }
  return std::make_tuple(partial_out, softmax_max, softmax_sum);
}

at::Tensor C8Stage2OutNpu(
    const at::Tensor& query,
    const at::Tensor& packed_kv,
    const at::Tensor& topk_slots,
    const at::Tensor& block_table,
    const at::Tensor& actual_q,
    const at::Tensor& actual_kv,
    const at::Tensor& miss_counts,
    const at::Tensor& cache_tokens,
    double scale_value,
    const at::Tensor& previous_p,
    const at::Tensor& previous_m,
    const at::Tensor& previous_l,
    at::Tensor attention_out,
    c10::optional<int64_t> kv_dtype) {
  TORCH_CHECK(
      query.dim() == 3 && query.size(2) == kCkvDim + kKpeDim &&
          packed_kv.dim() == 4 && packed_kv.size(1) == kBlockSize &&
          packed_kv.size(2) == 1 && packed_kv.size(3) == kPackedKvDim &&
          packed_kv.element_size() == 1 &&
          topk_slots.dim() == 3 && topk_slots.size(0) == query.size(0) &&
          topk_slots.size(1) == 1 && topk_slots.size(2) == kSparseCount,
      "C8 Stage2 query/KV/topk shapes are invalid.");
  TORCH_CHECK(
      actual_q.dim() == 1 && actual_kv.sizes() == actual_q.sizes() &&
          cache_tokens.sizes() == actual_q.sizes() &&
          block_table.dim() == 2 && block_table.size(0) == actual_q.size(0) &&
          miss_counts.dim() == 1 && miss_counts.size(0) == query.size(0),
      "C8 Stage2 metadata shapes are inconsistent.");
  TORCH_CHECK(
      previous_p.dim() == 3 && previous_p.size(0) == query.size(0) &&
          previous_p.size(1) == query.size(1) &&
          previous_p.size(2) == kCkvDim &&
          previous_m.dim() == 3 && previous_m.size(0) == 1 &&
          previous_m.size(1) == previous_p.size(0) &&
          previous_m.size(2) == previous_p.size(1) &&
          previous_l.sizes() == previous_m.sizes(),
      "C8 Stage2 requires P[T,N,512] and M/L[1,T,N] Stage1 state.");
  for (const at::Tensor* tensor : {&topk_slots, &block_table, &actual_q,
                                   &actual_kv, &miss_counts, &cache_tokens}) {
    TORCH_CHECK(tensor->scalar_type() == at::kInt,
                "C8 Stage2 metadata must be int32.");
  }
  for (const at::Tensor* tensor : {&previous_p, &previous_m, &previous_l}) {
    TORCH_CHECK(tensor->scalar_type() == at::kFloat,
                "C8 Stage2 state must be FP32.");
  }
  TORCH_CHECK(
      attention_out.sizes() == previous_p.sizes() &&
          attention_out.scalar_type() == query.scalar_type(),
      "C8 Stage2 output must match query dtype and be [T,N,512].");
  CheckOneDeviceAndContiguous(
      query,
      {&query, &packed_kv, &topk_slots, &block_table, &actual_q, &actual_kv,
       &miss_counts, &cache_tokens, &previous_p, &previous_m, &previous_l,
       &attention_out},
      "C8 Stage2");

  c10::optional<at::Tensor> no_scale;
  c10::optional<at::Tensor> optional_block_table = block_table;
  c10::optional<at::Tensor> optional_actual_q = actual_q;
  c10::optional<at::Tensor> optional_actual_kv = actual_kv;
  std::string query_layout = "TND";
  std::string kv_layout = "PA_BSND";
  char* query_layout_ptr = query_layout.data();
  char* kv_layout_ptr = kv_layout.data();
  const int64_t key_quant_mode = 2;
  const int64_t value_quant_mode = 2;
  const int64_t sparse_block_size = 1;
  const int64_t sparse_mode = 3;
  const int64_t all_tokens = std::numeric_limits<int64_t>::max();
  const int64_t attention_mode = 2;
  const int64_t quant_scale_repo_mode = 1;
  const int64_t tile_size = 128;
  const int64_t rope_head_dim = 64;
  int64_t stage_mode = 2;
  auto keepalive = std::make_tuple(
      query, packed_kv, topk_slots, block_table, actual_q, actual_kv,
      miss_counts, cache_tokens, previous_p, previous_m, previous_l,
      attention_out);
  auto launch = [&](const auto& packed_kv_arg) {
    EXEC_NPU_CMD_ORDERED(
        aclnnA5SparseTailAttentionC8Stage2,
        keepalive,
        query,
        packed_kv_arg,
        packed_kv_arg,
        topk_slots,
        no_scale,
        no_scale,
        optional_block_table,
        optional_actual_q,
        optional_actual_kv,
        miss_counts,
        cache_tokens,
        previous_p,
        previous_m,
        previous_l,
        scale_value,
        key_quant_mode,
        value_quant_mode,
        sparse_block_size,
        query_layout_ptr,
        kv_layout_ptr,
        sparse_mode,
        all_tokens,
        all_tokens,
        attention_mode,
        quant_scale_repo_mode,
        tile_size,
        rope_head_dim,
        stage_mode,
        attention_out);
  };
  if (kv_dtype.has_value()) {
    TORCH_CHECK(kv_dtype.value() == kHifloat8Dtype,
                "C8 Stage2 kv_dtype only supports torch_npu.hifloat8.");
    TORCH_CHECK(packed_kv.scalar_type() == at::kByte,
                "C8 Stage2 HIFLOAT8 storage must use torch.uint8.");
    const TensorWrapper packed_kv_wrapper{
        packed_kv, aclDataType::ACL_HIFLOAT8};
    launch(packed_kv_wrapper);
  } else {
    TORCH_CHECK(
        packed_kv.scalar_type() == at::ScalarType::Float8_e4m3fn ||
            packed_kv.scalar_type() == at::kChar,
        "C8 Stage2 packed KV must be float8_e4m3fn/int8, or uint8 with "
        "kv_dtype=torch_npu.hifloat8.");
    launch(packed_kv);
  }
  return attention_out;
}

void CheckC8ProbeInputs(
    const at::Tensor& query,
    const at::Tensor& packed_kv,
    const at::Tensor& sparse_indices,
    const at::Tensor& block_table,
    const at::Tensor& actual_q,
    const at::Tensor& actual_kv,
    const at::Tensor& miss_counts,
    const at::Tensor& cache_tokens,
    const at::Tensor& attention_out,
    const char* name) {
  TORCH_CHECK(
      query.dim() == 3 && query.size(2) == kCkvDim + kKpeDim &&
          packed_kv.dim() == 4 && packed_kv.size(1) == kBlockSize &&
          packed_kv.size(2) == 1 && packed_kv.size(3) == kPackedKvDim &&
          packed_kv.element_size() == 1,
      name, " query/KV shapes are invalid.");
  TORCH_CHECK(
      sparse_indices.dim() == 3 &&
          sparse_indices.size(0) == query.size(0) &&
          sparse_indices.size(1) == 1 && sparse_indices.size(2) > 0,
      name, " sparse_indices must be int32 [T,1,K].");
  TORCH_CHECK(
      actual_q.dim() == 1 && actual_kv.sizes() == actual_q.sizes() &&
          cache_tokens.sizes() == actual_q.sizes() &&
          block_table.dim() == 2 && block_table.size(0) == actual_q.size(0) &&
          miss_counts.dim() == 1 && miss_counts.size(0) == query.size(0),
      name, " metadata shapes are inconsistent.");
  TORCH_CHECK(
      attention_out.dim() == 3 &&
          attention_out.size(0) == query.size(0) &&
          attention_out.size(1) == query.size(1) &&
          attention_out.size(2) == kCkvDim &&
          attention_out.scalar_type() == query.scalar_type(),
      name, " attention_out must be caller-owned [T,N,512].");
  for (const at::Tensor* tensor : {&sparse_indices, &block_table, &actual_q,
                                   &actual_kv, &miss_counts, &cache_tokens}) {
    TORCH_CHECK(tensor->scalar_type() == at::kInt,
                name, " metadata must be int32.");
  }
  CheckOneDeviceAndContiguous(
      query,
      {&query, &packed_kv, &sparse_indices, &block_table, &actual_q,
       &actual_kv, &miss_counts, &cache_tokens, &attention_out},
      name);
}

void C8PmlProbeOutNpu(
    const at::Tensor& query,
    const at::Tensor& packed_kv,
    const at::Tensor& sparse_indices,
    const at::Tensor& block_table,
    const at::Tensor& actual_q,
    const at::Tensor& actual_kv,
    const at::Tensor& miss_counts,
    const at::Tensor& cache_tokens,
    double scale_value,
    bool probe_enabled,
    at::Tensor attention_out,
    at::Tensor partial_out,
    at::Tensor softmax_max,
    at::Tensor softmax_sum,
    c10::optional<int64_t> kv_dtype) {
  CheckC8ProbeInputs(
      query, packed_kv, sparse_indices, block_table, actual_q, actual_kv,
      miss_counts, cache_tokens, attention_out, "C8 PML probe");
  TORCH_CHECK(
      partial_out.sizes() == attention_out.sizes() &&
          partial_out.scalar_type() == at::kFloat &&
          softmax_max.dim() == 3 && softmax_max.size(0) == 1 &&
          softmax_max.size(1) == query.size(0) &&
          softmax_max.size(2) == query.size(1) &&
          softmax_max.scalar_type() == at::kFloat &&
          softmax_sum.sizes() == softmax_max.sizes() &&
          softmax_sum.scalar_type() == at::kFloat,
      "C8 PML probe outputs must be P[T,N,512], M/L[1,T,N] FP32.");
  CheckOneDeviceAndContiguous(
      query, {&partial_out, &softmax_max, &softmax_sum}, "C8 PML probe");

  c10::optional<at::Tensor> no_scale;
  c10::optional<at::Tensor> optional_block_table = block_table;
  c10::optional<at::Tensor> optional_actual_q = actual_q;
  c10::optional<at::Tensor> optional_actual_kv = actual_kv;
  std::string query_layout = "TND";
  std::string kv_layout = "PA_BSND";
  char* query_layout_ptr = query_layout.data();
  char* kv_layout_ptr = kv_layout.data();
  const int64_t key_quant_mode = 2;
  const int64_t value_quant_mode = 2;
  const int64_t sparse_block_size = 1;
  const int64_t sparse_mode = 3;
  const int64_t all_tokens = std::numeric_limits<int64_t>::max();
  const int64_t attention_mode = 2;
  const int64_t quant_scale_repo_mode = 1;
  const int64_t tile_size = 128;
  const int64_t rope_head_dim = 64;
  int64_t stage_mode = 3;
  bool probe_enabled_attr = probe_enabled;
  auto keepalive = std::make_tuple(
      query, packed_kv, sparse_indices, block_table, actual_q, actual_kv,
      miss_counts, cache_tokens, attention_out, partial_out, softmax_max,
      softmax_sum);
  auto launch = [&](const auto& packed_kv_arg) {
    EXEC_NPU_CMD_ORDERED(
        aclnnA5SparseTailAttentionC8PmlProbe, keepalive, query,
        packed_kv_arg, packed_kv_arg, sparse_indices, no_scale, no_scale,
        optional_block_table, optional_actual_q, optional_actual_kv,
        miss_counts, cache_tokens, scale_value, key_quant_mode,
        value_quant_mode, sparse_block_size, query_layout_ptr,
        kv_layout_ptr, sparse_mode, all_tokens, all_tokens, attention_mode,
        quant_scale_repo_mode, tile_size, rope_head_dim, stage_mode,
        probe_enabled_attr, attention_out, partial_out, softmax_max,
        softmax_sum);
  };
  if (kv_dtype.has_value()) {
    TORCH_CHECK(kv_dtype.value() == kHifloat8Dtype &&
                    packed_kv.scalar_type() == at::kByte,
                "C8 PML probe HIFLOAT8 requires uint8 storage.");
    const TensorWrapper packed_kv_wrapper{
        packed_kv, aclDataType::ACL_HIFLOAT8};
    launch(packed_kv_wrapper);
  } else {
    TORCH_CHECK(
        packed_kv.scalar_type() == at::ScalarType::Float8_e4m3fn ||
            packed_kv.scalar_type() == at::kChar,
        "C8 PML probe packed KV dtype is unsupported.");
    launch(packed_kv);
  }
}

void C8TndProbeOutNpu(
    const at::Tensor& query,
    const at::Tensor& packed_kv,
    const at::Tensor& sparse_indices,
    const at::Tensor& block_table,
    const at::Tensor& actual_q,
    const at::Tensor& actual_kv,
    const at::Tensor& miss_counts,
    const at::Tensor& cache_tokens,
    double scale_value,
    bool probe_enabled,
    at::Tensor attention_out,
    c10::optional<int64_t> kv_dtype) {
  CheckC8ProbeInputs(
      query, packed_kv, sparse_indices, block_table, actual_q, actual_kv,
      miss_counts, cache_tokens, attention_out, "C8 TND probe");
  TORCH_CHECK(!probe_enabled || sparse_indices.size(2) == kSparseCount,
              "Enabled C8 TND probe requires compact [T,1,2048] slots.");

  c10::optional<at::Tensor> no_scale;
  c10::optional<at::Tensor> optional_block_table = block_table;
  c10::optional<at::Tensor> optional_actual_q = actual_q;
  c10::optional<at::Tensor> optional_actual_kv = actual_kv;
  std::string query_layout = "TND";
  std::string kv_layout = "PA_BSND";
  char* query_layout_ptr = query_layout.data();
  char* kv_layout_ptr = kv_layout.data();
  const int64_t key_quant_mode = 2;
  const int64_t value_quant_mode = 2;
  const int64_t sparse_block_size = 1;
  const int64_t sparse_mode = 3;
  const int64_t all_tokens = std::numeric_limits<int64_t>::max();
  const int64_t attention_mode = 2;
  const int64_t quant_scale_repo_mode = 1;
  const int64_t tile_size = 128;
  const int64_t rope_head_dim = 64;
  int64_t stage_mode = 4;
  bool probe_enabled_attr = probe_enabled;
  auto keepalive = std::make_tuple(
      query, packed_kv, sparse_indices, block_table, actual_q, actual_kv,
      miss_counts, cache_tokens, attention_out);
  auto launch = [&](const auto& packed_kv_arg) {
    EXEC_NPU_CMD_ORDERED(
        aclnnA5SparseTailAttentionC8TndProbe, keepalive, query,
        packed_kv_arg, packed_kv_arg, sparse_indices, no_scale, no_scale,
        optional_block_table, optional_actual_q, optional_actual_kv,
        miss_counts, cache_tokens, scale_value, key_quant_mode,
        value_quant_mode, sparse_block_size, query_layout_ptr,
        kv_layout_ptr, sparse_mode, all_tokens, all_tokens, attention_mode,
        quant_scale_repo_mode, tile_size, rope_head_dim, stage_mode,
        probe_enabled_attr, attention_out);
  };
  if (kv_dtype.has_value()) {
    TORCH_CHECK(kv_dtype.value() == kHifloat8Dtype &&
                    packed_kv.scalar_type() == at::kByte,
                "C8 TND probe HIFLOAT8 requires uint8 storage.");
    const TensorWrapper packed_kv_wrapper{
        packed_kv, aclDataType::ACL_HIFLOAT8};
    launch(packed_kv_wrapper);
  } else {
    TORCH_CHECK(
        packed_kv.scalar_type() == at::ScalarType::Float8_e4m3fn ||
            packed_kv.scalar_type() == at::kChar,
        "C8 TND probe packed KV dtype is unsupported.");
    launch(packed_kv);
  }
}
} // namespace nanovllm_dsa_a5_impl

TORCH_LIBRARY_IMPL(nanovllm_dsa, PrivateUse1, m) {
  m.impl("_sparse_tail_attention_c8_state_out", &nanovllm_dsa_a5_impl::C8StateOutNpu);
  m.impl("_sparse_tail_attention_c8_stage2_out", &nanovllm_dsa_a5_impl::C8Stage2OutNpu);
  m.impl("_sparse_tail_attention_c8_pml_probe_out", &nanovllm_dsa_a5_impl::C8PmlProbeOutNpu);
  m.impl("_sparse_tail_attention_c8_tnd_probe_out", &nanovllm_dsa_a5_impl::C8TndProbeOutNpu);
}
