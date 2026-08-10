#ifndef NANOVLLM_KVCACHE_OFFLOAD_COPY_TORCH_ADPT_H
#define NANOVLLM_KVCACHE_OFFLOAD_COPY_TORCH_ADPT_H

namespace vllm_ascend {

inline void npu_kvcache_offload_copy(
    const at::Tensor& hbm_kv_cache,
    at::Tensor dram_kv_cache,
    const at::Tensor& hbm_block_table,
    const at::Tensor& dram_block_table,
    const at::Tensor& copy_counts) {
  TORCH_CHECK(hbm_kv_cache.dim() >= 2 &&
                  dram_kv_cache.dim() == hbm_kv_cache.dim(),
              "KVCache offload cache tensors must have matching rank >= 2.");
  TORCH_CHECK(hbm_kv_cache.size(0) > 0 && dram_kv_cache.size(0) > 0,
              "KVCache offload physical block counts must be positive.");
  TORCH_CHECK(hbm_kv_cache.numel() > 0 && dram_kv_cache.numel() > 0,
              "KVCache offload blocks must contain at least one byte.");
  for (int64_t dim = 1; dim < hbm_kv_cache.dim(); ++dim) {
    TORCH_CHECK(hbm_kv_cache.size(dim) == dram_kv_cache.size(dim),
                "KVCache offload cache tensors must have matching trailing dimensions.");
  }
  TORCH_CHECK(hbm_block_table.dim() == 2 &&
                  dram_block_table.dim() == 2 && copy_counts.dim() == 1,
              "KVCache offload block tables must be [B, C] and counts [B].");
  TORCH_CHECK(hbm_block_table.sizes() == dram_block_table.sizes(),
              "KVCache offload HBM and DRAM block-table shapes must match.");
  TORCH_CHECK(copy_counts.size(0) == hbm_block_table.size(0),
              "KVCache offload batch dimensions must match.");
  TORCH_CHECK(hbm_block_table.size(1) > 0 &&
                  hbm_block_table.size(1) <= 65536,
              "KVCache offload block-table capacity must be in [1, 65536].");
  TORCH_CHECK(hbm_kv_cache.scalar_type() == at::kChar &&
                  dram_kv_cache.scalar_type() == at::kChar,
              "KVCache offload cache tensors must be int8.");
  TORCH_CHECK(hbm_block_table.scalar_type() == at::kInt &&
                  dram_block_table.scalar_type() == at::kInt &&
                  copy_counts.scalar_type() == at::kInt,
              "KVCache offload block tables and counts must be int32.");
  TORCH_CHECK(hbm_kv_cache.is_contiguous() && dram_kv_cache.is_contiguous() &&
                  hbm_block_table.is_contiguous() &&
                  dram_block_table.is_contiguous() &&
                  copy_counts.is_contiguous(),
              "All KVCache offload inputs must be contiguous.");
  TORCH_CHECK(hbm_kv_cache.device().is_privateuseone(),
              "KVCache offload inputs must be on NPU.");
  const auto device = hbm_kv_cache.device();
  TORCH_CHECK(dram_kv_cache.device() == device &&
                  hbm_block_table.device() == device &&
                  dram_block_table.device() == device &&
                  copy_counts.device() == device,
              "All KVCache offload inputs must be on the same NPU device.");

  auto keepalive = std::make_tuple(
      hbm_kv_cache, dram_kv_cache, hbm_block_table,
      dram_block_table, copy_counts);
  EXEC_NPU_CMD_ORDERED(
      aclnnNanovllmKvcacheOffloadCopy,
      keepalive,
      hbm_kv_cache,
      dram_kv_cache,
      hbm_block_table,
      dram_block_table,
      copy_counts);
}

}  // namespace vllm_ascend

#endif
