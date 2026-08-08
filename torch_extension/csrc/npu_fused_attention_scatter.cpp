#include <tuple>

#include "ops_common.h"

namespace nanovllm_dsa_a5_impl {
namespace {

void CheckFusedInputs(
    const at::Tensor& query,
    const at::Tensor& hbm_ckv,
    const at::Tensor& sparse_slots,
    const at::Tensor& cache_tokens,
    const at::Tensor& hbm_block_table,
    const at::Tensor& actual_q,
    const at::Tensor& actual_kv,
    const at::Tensor& query_rope,
    const at::Tensor& hbm_kpe,
    const at::Tensor& dram_kpe,
    const at::Tensor& dram_ckv,
    const at::Tensor& dram_block_table,
    const at::Tensor& source_token_ids,
    const at::Tensor& copy_counts) {
  CheckAttentionInputs(
      query, hbm_ckv, hbm_ckv, sparse_slots, cache_tokens,
      hbm_block_table, actual_q, actual_kv, query_rope, hbm_kpe);
  TORCH_CHECK(
      query.scalar_type() == at::kBFloat16,
      "Fused SCATTER/SFA operators support BF16 only.");
  const int64_t batch = query.size(0);
  TORCH_CHECK(
      dram_kpe.dim() == 3 && dram_kpe.size(1) == kBlockSize &&
          dram_kpe.size(2) == kKpeDim &&
          dram_ckv.dim() == 3 && dram_ckv.size(0) == dram_kpe.size(0) &&
          dram_ckv.size(1) == kBlockSize && dram_ckv.size(2) == kCkvDim,
      "Fused DRAM KPE/CKV must be [blocks,128,64/512].");
  TORCH_CHECK(
      dram_block_table.dim() == 2 &&
          dram_block_table.size(0) == batch &&
          dram_block_table.size(1) > 0 &&
          dram_block_table.size(1) * kBlockSize <= kMaxSourceCapacity &&
          source_token_ids.dim() == 2 &&
          source_token_ids.size(0) == batch &&
          source_token_ids.size(1) == kSparseCount &&
          copy_counts.dim() == 1 && copy_counts.size(0) == batch,
      "Fused DRAM metadata must be block_table[B,*], "
      "source_ids[B,2048], copy_counts[B].");
  TORCH_CHECK(
      dram_kpe.scalar_type() == at::kBFloat16 &&
          dram_ckv.scalar_type() == at::kBFloat16,
      "Fused DRAM and HBM KV tensors must be BF16.");
  for (const at::Tensor* tensor :
       {&dram_block_table, &source_token_ids, &copy_counts}) {
    TORCH_CHECK(
        tensor->scalar_type() == at::kInt,
        "Fused DRAM metadata must be int32.");
  }
  CheckOneDeviceAndContiguous(
      query,
      {&dram_kpe, &dram_ckv, &dram_block_table,
       &source_token_ids, &copy_counts},
      "Fused SCATTER/SFA");
}

using FusedResult = std::tuple<at::Tensor, at::Tensor, at::Tensor>;

}  // namespace

FusedResult FusedAttentionScatterNpu(
    const at::Tensor& query,
    at::Tensor hbm_ckv,
    const at::Tensor& sparse_slots,
    const at::Tensor& cache_tokens,
    const at::Tensor& hbm_block_table,
    const at::Tensor& actual_q,
    const at::Tensor& actual_kv,
    const at::Tensor& query_rope,
    at::Tensor hbm_kpe,
    const at::Tensor& dram_kpe,
    const at::Tensor& dram_ckv,
    const at::Tensor& dram_block_table,
    const at::Tensor& source_token_ids,
    const at::Tensor& copy_counts,
    double scale_value) {
  CheckFusedInputs(
      query, hbm_ckv, sparse_slots, cache_tokens, hbm_block_table,
      actual_q, actual_kv, query_rope, hbm_kpe, dram_kpe, dram_ckv,
      dram_block_table, source_token_ids, copy_counts);
  auto output = at::empty_like(query);
  auto softmax_max = at::empty({1}, query.options().dtype(at::kFloat));
  auto softmax_sum = at::empty({1}, query.options().dtype(at::kFloat));
  SparseAttentionAclnnAttrs attrs(scale_value);
  char* query_layout = attrs.query_layout.data();
  char* kv_layout = attrs.kv_layout.data();
  auto keepalive = std::make_tuple(
      query, hbm_ckv, sparse_slots, hbm_block_table, actual_q, actual_kv,
      query_rope, hbm_kpe, cache_tokens, dram_kpe, dram_ckv,
      dram_block_table, source_token_ids, copy_counts, output,
      softmax_max, softmax_sum);
  EXEC_NPU_CMD_ORDERED(
      aclnnA5SparseAndTailAttentionAndScatterCopy,
      keepalive,
      query,
      hbm_ckv,
      hbm_ckv,
      sparse_slots,
      hbm_block_table,
      actual_q,
      actual_kv,
      query_rope,
      hbm_kpe,
      cache_tokens,
      dram_kpe,
      dram_ckv,
      dram_block_table,
      source_token_ids,
      copy_counts,
      attrs.scale_value,
      attrs.sparse_block_size,
      query_layout,
      kv_layout,
      attrs.sparse_mode,
      attrs.all_tokens,
      attrs.all_tokens,
      attrs.attention_mode,
      attrs.return_softmax_lse,
      output,
      softmax_max,
      softmax_sum,
      hbm_kpe,
      hbm_ckv);
  return std::make_tuple(output, hbm_kpe, hbm_ckv);
}

FusedResult FusedAttentionScatterMtePipelineNpu(
    const at::Tensor& query,
    at::Tensor hbm_ckv,
    const at::Tensor& sparse_slots,
    const at::Tensor& cache_tokens,
    const at::Tensor& hbm_block_table,
    const at::Tensor& actual_q,
    const at::Tensor& actual_kv,
    const at::Tensor& query_rope,
    at::Tensor hbm_kpe,
    const at::Tensor& dram_kpe,
    const at::Tensor& dram_ckv,
    const at::Tensor& dram_block_table,
    const at::Tensor& source_token_ids,
    const at::Tensor& copy_counts,
    double scale_value,
    int64_t prefetch_rows_per_step) {
  CheckFusedInputs(
      query, hbm_ckv, sparse_slots, cache_tokens, hbm_block_table,
      actual_q, actual_kv, query_rope, hbm_kpe, dram_kpe, dram_ckv,
      dram_block_table, source_token_ids, copy_counts);
  TORCH_CHECK(
      prefetch_rows_per_step >= 0 && prefetch_rows_per_step <= 16,
      "prefetch_rows_per_step must be in [0,16].");
  auto output = at::empty_like(query);
  auto softmax_max = at::empty({1}, query.options().dtype(at::kFloat));
  auto softmax_sum = at::empty({1}, query.options().dtype(at::kFloat));
  SparseAttentionAclnnAttrs attrs(scale_value);
  char* query_layout = attrs.query_layout.data();
  char* kv_layout = attrs.kv_layout.data();
  auto keepalive = std::make_tuple(
      query, hbm_ckv, sparse_slots, hbm_block_table, actual_q, actual_kv,
      query_rope, hbm_kpe, cache_tokens, dram_kpe, dram_ckv,
      dram_block_table, source_token_ids, copy_counts, output,
      softmax_max, softmax_sum);
  EXEC_NPU_CMD_ORDERED(
      aclnnA5SparseAndTailAttentionAndScatterCopyMtePipeline,
      keepalive,
      query,
      hbm_ckv,
      hbm_ckv,
      sparse_slots,
      hbm_block_table,
      actual_q,
      actual_kv,
      query_rope,
      hbm_kpe,
      cache_tokens,
      dram_kpe,
      dram_ckv,
      dram_block_table,
      source_token_ids,
      copy_counts,
      attrs.scale_value,
      attrs.sparse_block_size,
      query_layout,
      kv_layout,
      attrs.sparse_mode,
      attrs.all_tokens,
      attrs.all_tokens,
      attrs.attention_mode,
      attrs.return_softmax_lse,
      prefetch_rows_per_step,
      output,
      softmax_max,
      softmax_sum,
      hbm_kpe,
      hbm_ckv);
  return std::make_tuple(output, hbm_kpe, hbm_ckv);
}

FusedResult FusedAttentionScatterMeta(
    const at::Tensor& query,
    at::Tensor hbm_ckv,
    const at::Tensor&,
    const at::Tensor&,
    const at::Tensor&,
    const at::Tensor&,
    const at::Tensor&,
    const at::Tensor&,
    at::Tensor hbm_kpe,
    const at::Tensor&,
    const at::Tensor&,
    const at::Tensor&,
    const at::Tensor&,
    const at::Tensor&,
    double) {
  TORCH_CHECK(
      query.scalar_type() == at::kBFloat16,
      "Fused SCATTER/SFA operators support BF16 only.");
  return std::make_tuple(at::empty_like(query), hbm_kpe, hbm_ckv);
}

FusedResult FusedAttentionScatterMtePipelineMeta(
    const at::Tensor& query,
    at::Tensor hbm_ckv,
    const at::Tensor&,
    const at::Tensor&,
    const at::Tensor&,
    const at::Tensor&,
    const at::Tensor&,
    const at::Tensor&,
    at::Tensor hbm_kpe,
    const at::Tensor&,
    const at::Tensor&,
    const at::Tensor&,
    const at::Tensor&,
    const at::Tensor&,
    double,
    int64_t prefetch_rows_per_step) {
  TORCH_CHECK(
      query.scalar_type() == at::kBFloat16,
      "Fused SCATTER/SFA MTE pipeline supports BF16 only.");
  TORCH_CHECK(
      prefetch_rows_per_step >= 0 && prefetch_rows_per_step <= 16,
      "prefetch_rows_per_step must be in [0,16].");
  return std::make_tuple(at::empty_like(query), hbm_kpe, hbm_ckv);
}

}  // namespace nanovllm_dsa_a5_impl

TORCH_LIBRARY_IMPL(nanovllm_dsa, PrivateUse1, m) {
  m.impl(
      "sparse_and_tail_attention_and_scatter_copy",
      &nanovllm_dsa_a5_impl::FusedAttentionScatterNpu);
  m.impl(
      "sparse_and_tail_attention_and_scatter_copy_mte_pipeline",
      &nanovllm_dsa_a5_impl::FusedAttentionScatterMtePipelineNpu);
}

TORCH_LIBRARY_IMPL(nanovllm_dsa, Meta, m) {
  m.impl(
      "sparse_and_tail_attention_and_scatter_copy",
      &nanovllm_dsa_a5_impl::FusedAttentionScatterMeta);
  m.impl(
      "sparse_and_tail_attention_and_scatter_copy_mte_pipeline",
      &nanovllm_dsa_a5_impl::FusedAttentionScatterMtePipelineMeta);
}
