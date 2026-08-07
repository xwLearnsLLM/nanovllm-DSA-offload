#include <tuple>

#include "ops_common.h"

namespace nanovllm_dsa_a5_impl {

int64_t CopyMetadataCapacity(const at::Tensor& tensor) {
  if (tensor.dim() == 2) {
    return tensor.size(1);
  }
  TORCH_CHECK(
      tensor.dim() == 3 && tensor.size(1) == 1,
      "packed SCATTER source/destination metadata must be [B,K] or [B,1,K].");
  return tensor.size(2);
}

int64_t AttentionCapacity(int64_t max_tail_tokens) {
  TORCH_CHECK(
      max_tail_tokens >= 0 &&
          max_tail_tokens <= kMaxSourceCapacity - kSparseCount,
      "packed SCATTER max_tail_tokens must be in [0,260096].");
  return kSparseCount + max_tail_tokens;
}

void CheckPackedScatterInputs(
    const at::Tensor& hbm_kv_bytes,
    const at::Tensor& dram_kv_bytes,
    const at::Tensor& hbm_block_table,
    const at::Tensor& dram_block_table,
    const at::Tensor& source_token_ids,
    const at::Tensor& destination_slots,
    const at::Tensor& copy_counts,
    const at::Tensor& cache_tokens,
    const at::Tensor& candidate_lens,
    const at::Tensor& actual_seq_lengths_kv) {
  TORCH_CHECK(
      hbm_kv_bytes.dim() == 4 && hbm_kv_bytes.size(0) > 0 &&
          hbm_kv_bytes.size(1) == kBlockSize &&
          hbm_kv_bytes.size(2) == 1 &&
          hbm_kv_bytes.size(3) == kPackedKvDim &&
          dram_kv_bytes.dim() == 4 && dram_kv_bytes.size(0) > 0 &&
          dram_kv_bytes.size(1) == kBlockSize &&
          dram_kv_bytes.size(2) == 1 &&
          dram_kv_bytes.size(3) == kPackedKvDim,
      "packed SCATTER KV byte views must be [blocks,128,1,656].");
  TORCH_CHECK(
      hbm_kv_bytes.scalar_type() == at::kChar &&
          dram_kv_bytes.scalar_type() == at::kChar,
      "packed SCATTER expects int8 byte views of C8 KV caches.");
  TORCH_CHECK(copy_counts.dim() == 1, "packed SCATTER copy_counts must be int32[B].");
  const int64_t batch = copy_counts.size(0);
  TORCH_CHECK(
      batch > 0 && cache_tokens.dim() == 1 &&
          cache_tokens.size(0) == batch && candidate_lens.dim() == 1 &&
          candidate_lens.size(0) == batch &&
          actual_seq_lengths_kv.dim() == 1 &&
          actual_seq_lengths_kv.size(0) == batch &&
          hbm_block_table.dim() == 2 &&
          hbm_block_table.size(0) == batch &&
          hbm_block_table.size(1) > 0 &&
          dram_block_table.dim() == 2 &&
          dram_block_table.size(0) == batch &&
          dram_block_table.size(1) > 0 &&
          source_token_ids.size(0) == batch &&
          destination_slots.size(0) == batch &&
          source_token_ids.sizes() == destination_slots.sizes() &&
          CopyMetadataCapacity(source_token_ids) == kSparseCount &&
          dram_block_table.size(1) * kBlockSize <= kMaxSourceCapacity,
      "packed SCATTER metadata shapes are inconsistent.");
  for (const at::Tensor* tensor :
       {&hbm_block_table, &dram_block_table, &source_token_ids,
        &destination_slots, &copy_counts, &cache_tokens,
        &candidate_lens, &actual_seq_lengths_kv}) {
    TORCH_CHECK(
        tensor->scalar_type() == at::kInt,
        "packed SCATTER metadata must be int32.");
  }
  CheckOneDeviceAndContiguous(
      hbm_kv_bytes,
      {&hbm_kv_bytes, &dram_kv_bytes, &hbm_block_table,
       &dram_block_table, &source_token_ids, &destination_slots,
       &copy_counts, &cache_tokens, &candidate_lens,
       &actual_seq_lengths_kv},
      "packed SCATTER");
}

void CheckPackedScatterOutputs(
    const at::Tensor& hbm_kv_bytes,
    const at::Tensor& attention_slots,
    const at::Tensor& resident_seq_lengths,
    int64_t expected_batch,
    int64_t max_tail_tokens) {
  const int64_t expected_capacity = AttentionCapacity(max_tail_tokens);
  TORCH_CHECK(
      attention_slots.dim() == 3 &&
          attention_slots.size(0) == expected_batch &&
          attention_slots.size(1) == 1 &&
          attention_slots.size(2) == expected_capacity &&
          resident_seq_lengths.dim() == 1 &&
          resident_seq_lengths.size(0) == expected_batch,
      "packed SCATTER outputs must be attention_slots [B,1,2048+max_tail_tokens] "
      "and resident_seq_lengths [B].");
  for (const at::Tensor* tensor : {&attention_slots, &resident_seq_lengths}) {
    TORCH_CHECK(
        tensor->scalar_type() == at::kInt &&
            tensor->device() == hbm_kv_bytes.device() &&
            tensor->is_contiguous(),
        "packed SCATTER metadata outputs must be contiguous int32 tensors.");
  }
}

std::tuple<at::Tensor, at::Tensor, at::Tensor>
ScatterCopyC8OutNpu(
    at::Tensor hbm_kv_bytes,
    const at::Tensor& dram_kv_bytes,
    const at::Tensor& hbm_block_table,
    const at::Tensor& dram_block_table,
    const at::Tensor& source_token_ids,
    const at::Tensor& destination_slots,
    const at::Tensor& copy_counts,
    const at::Tensor& cache_tokens,
    const at::Tensor& candidate_lens,
    const at::Tensor& actual_seq_lengths_kv,
    int64_t max_tail_tokens,
    at::Tensor attention_slots,
    at::Tensor resident_seq_lengths) {
  CheckPackedScatterInputs(
      hbm_kv_bytes, dram_kv_bytes, hbm_block_table, dram_block_table,
      source_token_ids, destination_slots, copy_counts, cache_tokens,
      candidate_lens, actual_seq_lengths_kv);
  CheckPackedScatterOutputs(
      hbm_kv_bytes, attention_slots, resident_seq_lengths,
      copy_counts.size(0), max_tail_tokens);
  auto keepalive = std::make_tuple(
      hbm_kv_bytes, dram_kv_bytes, hbm_block_table, dram_block_table,
      source_token_ids, destination_slots, copy_counts, cache_tokens,
      candidate_lens, actual_seq_lengths_kv, attention_slots,
      resident_seq_lengths);
  EXEC_NPU_CMD_ORDERED(
      aclnnA5PackedKvcacheScatterCopy,
      keepalive,
      hbm_kv_bytes,
      dram_kv_bytes,
      hbm_block_table,
      dram_block_table,
      source_token_ids,
      destination_slots,
      copy_counts,
      cache_tokens,
      candidate_lens,
      actual_seq_lengths_kv,
      attention_slots,
      resident_seq_lengths,
      hbm_kv_bytes,
      attention_slots,
      resident_seq_lengths);
  return std::make_tuple(
      hbm_kv_bytes, attention_slots, resident_seq_lengths);
}

std::tuple<at::Tensor, at::Tensor, at::Tensor>
ScatterCopyC8Npu(
    at::Tensor hbm_kv_bytes,
    const at::Tensor& dram_kv_bytes,
    const at::Tensor& hbm_block_table,
    const at::Tensor& dram_block_table,
    const at::Tensor& source_token_ids,
    const at::Tensor& destination_slots,
    const at::Tensor& copy_counts,
    const at::Tensor& cache_tokens,
    const at::Tensor& candidate_lens,
    const at::Tensor& actual_seq_lengths_kv,
    int64_t max_tail_tokens) {
  auto options = copy_counts.options().dtype(at::kInt);
  auto attention_slots = at::empty(
      {copy_counts.size(0), 1, AttentionCapacity(max_tail_tokens)}, options);
  auto resident_seq_lengths = at::empty({copy_counts.size(0)}, options);
  return ScatterCopyC8OutNpu(
      hbm_kv_bytes, dram_kv_bytes, hbm_block_table, dram_block_table,
      source_token_ids, destination_slots, copy_counts, cache_tokens,
      candidate_lens, actual_seq_lengths_kv, max_tail_tokens,
      attention_slots, resident_seq_lengths);
}

std::tuple<at::Tensor, at::Tensor, at::Tensor>
ScatterCopyC8Meta(
    at::Tensor hbm_kv_bytes,
    const at::Tensor&,
    const at::Tensor&,
    const at::Tensor&,
    const at::Tensor&,
    const at::Tensor&,
    const at::Tensor& copy_counts,
    const at::Tensor&,
    const at::Tensor&,
    const at::Tensor&,
    int64_t max_tail_tokens) {
  auto options = copy_counts.options().dtype(at::kInt);
  return std::make_tuple(
      hbm_kv_bytes,
      at::empty(
          {copy_counts.size(0), 1, AttentionCapacity(max_tail_tokens)},
          options),
      at::empty({copy_counts.size(0)}, options));
}

std::tuple<at::Tensor, at::Tensor, at::Tensor>
ScatterCopyC8OutMeta(
    at::Tensor hbm_kv_bytes,
    const at::Tensor&,
    const at::Tensor&,
    const at::Tensor&,
    const at::Tensor&,
    const at::Tensor&,
    const at::Tensor& copy_counts,
    const at::Tensor&,
    const at::Tensor&,
    const at::Tensor&,
    int64_t max_tail_tokens,
    at::Tensor attention_slots,
    at::Tensor resident_seq_lengths) {
  CheckPackedScatterOutputs(
      hbm_kv_bytes, attention_slots, resident_seq_lengths,
      copy_counts.size(0), max_tail_tokens);
  return std::make_tuple(
      hbm_kv_bytes, attention_slots, resident_seq_lengths);
}

}  // namespace nanovllm_dsa_a5_impl

TORCH_LIBRARY_IMPL(nanovllm_dsa, PrivateUse1, m) {
  m.impl("scatter_copy_c8", &nanovllm_dsa_a5_impl::ScatterCopyC8Npu);
  m.impl("scatter_copy_c8_out", &nanovllm_dsa_a5_impl::ScatterCopyC8OutNpu);
}

TORCH_LIBRARY_IMPL(nanovllm_dsa, Meta, m) {
  m.impl("scatter_copy_c8", &nanovllm_dsa_a5_impl::ScatterCopyC8Meta);
  m.impl("scatter_copy_c8_out", &nanovllm_dsa_a5_impl::ScatterCopyC8OutMeta);
}
