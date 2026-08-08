#include <tuple>

#include "ops_common.h"

namespace nanovllm_dsa_a5_impl {

void CheckScatterInputs(
    const at::Tensor& hbm_kpe,
    const at::Tensor& hbm_ckv,
    const at::Tensor& dram_kpe,
    const at::Tensor& dram_ckv,
    const at::Tensor& hbm_block_table,
    const at::Tensor& dram_block_table,
    const at::Tensor& source_token_ids,
    const at::Tensor& destination_slots,
    const at::Tensor& copy_counts) {
  TORCH_CHECK(
      hbm_kpe.dim() == 3 && hbm_kpe.size(1) == kBlockSize &&
          hbm_kpe.size(2) == kKpeDim,
      "HBM KPE must be [blocks,128,64].");
  TORCH_CHECK(
      hbm_ckv.dim() == 3 && hbm_ckv.size(1) == kBlockSize &&
          hbm_ckv.size(2) == kCkvDim,
      "HBM CKV must be [blocks,128,512].");
  TORCH_CHECK(
      dram_kpe.dim() == 3 && dram_kpe.size(1) == kBlockSize &&
          dram_kpe.size(2) == kKpeDim &&
          dram_ckv.dim() == 3 && dram_ckv.size(1) == kBlockSize &&
          dram_ckv.size(2) == kCkvDim,
      "DRAM KPE/CKV must be [blocks,128,64/512].");
  TORCH_CHECK(
      hbm_kpe.size(0) == hbm_ckv.size(0) &&
          dram_kpe.size(0) == dram_ckv.size(0),
      "CKV/KPE block counts must agree in each memory tier.");
  TORCH_CHECK(
      hbm_block_table.dim() == 2 && dram_block_table.dim() == 2 &&
          source_token_ids.dim() == 2 && destination_slots.dim() == 2 &&
          copy_counts.dim() == 1 &&
          source_token_ids.sizes() == destination_slots.sizes() &&
          source_token_ids.size(0) == copy_counts.size(0) &&
          hbm_block_table.size(0) == copy_counts.size(0) &&
          dram_block_table.size(0) == copy_counts.size(0) &&
          hbm_block_table.size(1) > 0 && dram_block_table.size(1) > 0 &&
          source_token_ids.size(1) > 0 && source_token_ids.size(1) <= 65536 &&
          dram_block_table.size(1) * kBlockSize <= kMaxSourceCapacity,
      "SCATTER metadata batch dimensions are inconsistent.");
  const auto dtype = hbm_kpe.scalar_type();
  TORCH_CHECK(
      dtype == at::kBFloat16 || dtype == at::kHalf,
      "SCATTER supports bf16/fp16.");
  for (const at::Tensor* tensor : {&hbm_ckv, &dram_kpe, &dram_ckv}) {
    TORCH_CHECK(tensor->scalar_type() == dtype, "All SCATTER KV dtypes must match.");
  }
  for (const at::Tensor* tensor :
       {&hbm_block_table, &dram_block_table, &source_token_ids,
        &destination_slots, &copy_counts}) {
    TORCH_CHECK(tensor->scalar_type() == at::kInt, "SCATTER metadata must be int32.");
  }
  CheckOneDeviceAndContiguous(
      hbm_kpe,
      {&hbm_kpe, &hbm_ckv, &dram_kpe, &dram_ckv, &hbm_block_table,
       &dram_block_table, &source_token_ids, &destination_slots, &copy_counts},
      "SCATTER");
}

std::tuple<at::Tensor, at::Tensor> KvcacheScatterCopyNpu(
    at::Tensor hbm_kpe,
    at::Tensor hbm_ckv,
    const at::Tensor& dram_kpe,
    const at::Tensor& dram_ckv,
    const at::Tensor& hbm_block_table,
    const at::Tensor& dram_block_table,
    const at::Tensor& source_token_ids,
    const at::Tensor& destination_slots,
    const at::Tensor& copy_counts) {
  CheckScatterInputs(
      hbm_kpe, hbm_ckv, dram_kpe, dram_ckv, hbm_block_table,
      dram_block_table, source_token_ids, destination_slots, copy_counts);
  auto keepalive = std::make_tuple(
      hbm_kpe, hbm_ckv, dram_kpe, dram_ckv, hbm_block_table,
      dram_block_table, source_token_ids, destination_slots, copy_counts);
  EXEC_NPU_CMD_ORDERED(
      aclnnA5KvcacheScatterCopy,
      keepalive,
      hbm_kpe,
      hbm_ckv,
      dram_kpe,
      dram_ckv,
      hbm_block_table,
      dram_block_table,
      source_token_ids,
      destination_slots,
      copy_counts,
      hbm_kpe,
      hbm_ckv);
  return std::make_tuple(hbm_kpe, hbm_ckv);
}

std::tuple<at::Tensor, at::Tensor> KvcacheScatterCopyMeta(
    at::Tensor hbm_kpe,
    at::Tensor hbm_ckv,
    const at::Tensor&,
    const at::Tensor&,
    const at::Tensor&,
    const at::Tensor&,
    const at::Tensor&,
    const at::Tensor&,
    const at::Tensor&) {
  return std::make_tuple(hbm_kpe, hbm_ckv);
}

}  // namespace nanovllm_dsa_a5_impl

TORCH_LIBRARY_IMPL(nanovllm_dsa, PrivateUse1, m) {
  m.impl("kvcache_scatter_copy", &nanovllm_dsa_a5_impl::KvcacheScatterCopyNpu);
}

TORCH_LIBRARY_IMPL(nanovllm_dsa, Meta, m) {
  m.impl("kvcache_scatter_copy", &nanovllm_dsa_a5_impl::KvcacheScatterCopyMeta);
}
