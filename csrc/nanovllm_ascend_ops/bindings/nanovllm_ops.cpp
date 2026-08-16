#include <acl/acl.h>
#include <torch/extension.h>
#include <torch/library.h>

#include "common/torch_adapter/op_api_common.h"

thread_local char g_hashBuf[kHashBufSize];
thread_local int g_hashOffset = 0;

#include "ops/fused_li_manage_mtp/fused_li_manage_mtp_torch_adpt.h"

namespace {

void fused_li_manage_mtp_torch_op(
    const at::Tensor& query,
    const at::Tensor& index_weights,
    const at::Tensor& index_key_cache,
    const at::Tensor& index_block_table,
    const at::Tensor& num_candidate_tokens,
    const at::Tensor& num_cache_tokens,
    const at::Tensor& req_pool_entries,
    at::Tensor cache_slots_pool,
    at::Tensor topk_src_ids,
    at::Tensor topk_dst_slots,
    at::Tensor miss_src_ids,
    at::Tensor miss_dst_slots,
    at::Tensor miss_counts) {
  vllm_ascend::npu_fused_li_manage_mtp(
      query, index_weights, index_key_cache, index_block_table,
      num_candidate_tokens, num_cache_tokens, req_pool_entries,
      cache_slots_pool, topk_src_ids, topk_dst_slots, miss_src_ids,
      miss_dst_slots, miss_counts);
}

void fused_li_manage_mtp_meta(
    const at::Tensor& query,
    const at::Tensor& index_weights,
    const at::Tensor& index_key_cache,
    const at::Tensor& index_block_table,
    const at::Tensor& num_candidate_tokens,
    const at::Tensor& num_cache_tokens,
    const at::Tensor& req_pool_entries,
    at::Tensor cache_slots_pool,
    at::Tensor topk_src_ids,
    at::Tensor topk_dst_slots,
    at::Tensor miss_src_ids,
    at::Tensor miss_dst_slots,
    at::Tensor miss_counts) {
  (void)query;
  (void)index_weights;
  (void)index_key_cache;
  (void)index_block_table;
  (void)num_candidate_tokens;
  (void)num_cache_tokens;
  (void)req_pool_entries;
  (void)cache_slots_pool;
  (void)topk_src_ids;
  (void)topk_dst_slots;
  (void)miss_src_ids;
  (void)miss_dst_slots;
  (void)miss_counts;
}

}  // namespace

TORCH_LIBRARY(nanovllm_dsa, ops) {
  ops.def(
      "fused_li_manage_mtp(Tensor query, Tensor index_weights,"
      " Tensor index_key_cache, Tensor index_block_table,"
      " Tensor num_candidate_tokens, Tensor num_cache_tokens,"
      " Tensor req_pool_entries, Tensor(a!) cache_slots_pool,"
      " Tensor(b!) topk_src_ids, Tensor(c!) topk_dst_slots,"
      " Tensor(d!) miss_src_ids, Tensor(e!) miss_dst_slots,"
      " Tensor(f!) miss_counts) -> ()");
}

TORCH_LIBRARY_IMPL(nanovllm_dsa, PrivateUse1, ops) {
  ops.impl("fused_li_manage_mtp", &fused_li_manage_mtp_torch_op);
}

TORCH_LIBRARY_IMPL(nanovllm_dsa, Meta, ops) {
  ops.impl("fused_li_manage_mtp", &fused_li_manage_mtp_meta);
}

PYBIND11_MODULE(_C, m) {
  m.doc() = "standalone GLM MTP3 fused_li_manage_mtp registration";
}
