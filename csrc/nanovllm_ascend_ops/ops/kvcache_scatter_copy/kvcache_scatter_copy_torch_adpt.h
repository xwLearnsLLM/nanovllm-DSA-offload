#ifndef NANOVLLM_KVCACHE_SCATTER_COPY_TORCH_ADPT_H
#define NANOVLLM_KVCACHE_SCATTER_COPY_TORCH_ADPT_H

namespace vllm_ascend {

inline void npu_scatter_copy(
    const at::Tensor& src_ids,
    const at::Tensor& dst_slots,
    const at::Tensor& copy_counts,
    const at::Tensor& hbm_block_table,
    const at::Tensor& dram_block_table,
    at::Tensor hbm_k_rope,
    at::Tensor hbm_kv_cache,
    const at::Tensor& dram_k_rope,
    const at::Tensor& dram_kv_cache) {
  TORCH_CHECK(hbm_k_rope.dim() == 3 && hbm_k_rope.size(1) == 128 &&
                  hbm_k_rope.size(2) == 64,
              "SCATTER HBM KPE must be [blocks, 128, 64].");
  TORCH_CHECK(hbm_k_rope.device().is_privateuseone(),
              "SCATTER inputs must be on NPU.");
  TORCH_CHECK(hbm_kv_cache.dim() == 3 && hbm_kv_cache.size(1) == 128 &&
                  hbm_kv_cache.size(2) == 512,
              "SCATTER HBM CKV must be [blocks, 128, 512].");
  TORCH_CHECK(dram_k_rope.dim() == 3 && dram_k_rope.size(1) == 128 &&
                  dram_k_rope.size(2) == 64,
              "SCATTER DRAM KPE must be [blocks, 128, 64].");
  TORCH_CHECK(dram_kv_cache.dim() == 3 && dram_kv_cache.size(1) == 128 &&
                  dram_kv_cache.size(2) == 512,
              "SCATTER DRAM CKV must be [blocks, 128, 512].");
  TORCH_CHECK(hbm_block_table.dim() == 2 && dram_block_table.dim() == 2 &&
                  src_ids.dim() == 2 && dst_slots.dim() == 2 &&
                  copy_counts.dim() == 1,
              "SCATTER tables/indices must be [B, C] and counts [B].");
  TORCH_CHECK(src_ids.sizes() == dst_slots.sizes(),
              "SCATTER source and destination shapes must match.");
  TORCH_CHECK(src_ids.size(1) > 0,
              "SCATTER input capacity must be positive.");
  TORCH_CHECK(src_ids.size(1) <= 65536,
              "SCATTER capacity must be <= 65536.");
  TORCH_CHECK(copy_counts.size(0) == src_ids.size(0) &&
                  hbm_block_table.size(0) == src_ids.size(0) &&
                  dram_block_table.size(0) == src_ids.size(0),
              "SCATTER batch dimensions must match.");
  const auto kv_dtype = hbm_k_rope.scalar_type();
  TORCH_CHECK(kv_dtype == at::kHalf || kv_dtype == at::kBFloat16,
              "SCATTER CKV/KPE must be fp16 or bf16.");
  TORCH_CHECK(hbm_kv_cache.scalar_type() == kv_dtype &&
                  dram_k_rope.scalar_type() == kv_dtype &&
                  dram_kv_cache.scalar_type() == kv_dtype,
              "All SCATTER CKV/KPE tensors must have the same dtype.");
  TORCH_CHECK(hbm_block_table.scalar_type() == at::kInt &&
                  dram_block_table.scalar_type() == at::kInt &&
                  src_ids.scalar_type() == at::kInt &&
                  dst_slots.scalar_type() == at::kInt &&
                  copy_counts.scalar_type() == at::kInt,
              "SCATTER tables, indices, slots and counts must be int32.");
  TORCH_CHECK(hbm_k_rope.is_contiguous() && hbm_kv_cache.is_contiguous() &&
                  dram_k_rope.is_contiguous() && dram_kv_cache.is_contiguous() &&
                  hbm_block_table.is_contiguous() && dram_block_table.is_contiguous() &&
                  src_ids.is_contiguous() && dst_slots.is_contiguous() &&
                  copy_counts.is_contiguous(),
              "All SCATTER inputs must be contiguous.");
  const auto device = hbm_k_rope.device();
  TORCH_CHECK(hbm_kv_cache.device() == device && dram_k_rope.device() == device &&
                  dram_kv_cache.device() == device && hbm_block_table.device() == device &&
                  dram_block_table.device() == device && src_ids.device() == device &&
                  dst_slots.device() == device && copy_counts.device() == device,
              "All SCATTER inputs must be on the same NPU device.");

  auto keepalive = std::make_tuple(
      hbm_k_rope, hbm_kv_cache, dram_k_rope, dram_kv_cache,
      hbm_block_table, dram_block_table, src_ids, dst_slots, copy_counts);
  EXEC_NPU_CMD_ORDERED(
      aclnnNanovllmKvcacheScatterCopy,
      keepalive,
      hbm_k_rope,
      hbm_kv_cache,
      dram_k_rope,
      dram_kv_cache,
      hbm_block_table,
      dram_block_table,
      src_ids,
      dst_slots,
      copy_counts);
}

}  // namespace vllm_ascend

#endif
