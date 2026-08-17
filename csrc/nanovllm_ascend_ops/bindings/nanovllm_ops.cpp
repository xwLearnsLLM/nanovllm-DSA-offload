#include <array>
#include <string>
#include <tuple>

#include <acl/acl.h>
#include <torch/extension.h>
#include <torch/library.h>

#include "common/torch_adapter/op_api_common.h"

thread_local char g_hashBuf[kHashBufSize];
thread_local int g_hashOffset = 0;

#include "ops/fused_copy_sfa_mtp/fused_copy_sfa_mtp_torch_adpt.h"
#include "ops/fused_li_manage_mtp/fused_li_manage_mtp_torch_adpt.h"
#include "ops/kvcache_scatter_copy/kvcache_scatter_copy_torch_adpt.h"
#include "ops/sparse_tail_attention_mtp/sparse_tail_attention_mtp_torch_adpt.h"

namespace {

void fused_li_manage_mtp_torch_op(
    const at::Tensor& index_weights,
    const at::Tensor& query_dequant_scale,
    const at::Tensor& query,
    const at::Tensor& index_key_dequant_scale,
    const at::Tensor& index_key_cache,
    const at::Tensor& index_block_table,
    const at::Tensor& actual_seq_lengths_query,
    const at::Tensor& actual_seq_lengths_key,
    const at::Tensor& offload_seq_lengths_key,
    const at::Tensor& req_valid,
    const at::Tensor& req_pool_entries,
    at::Tensor cache_state,
    at::Tensor cache_slots_pool,
    at::Tensor topk_src_ids,
    at::Tensor topk_dst_slots,
    at::Tensor topk_miss_counts,
    at::Tensor miss_src_ids,
    at::Tensor miss_dst_slots,
    at::Tensor miss_counts) {
  vllm_ascend::npu_fused_li_manage_mtp(
      index_weights, query_dequant_scale, query, index_key_dequant_scale,
      index_key_cache, index_block_table, actual_seq_lengths_query,
      actual_seq_lengths_key, offload_seq_lengths_key, req_valid, req_pool_entries,
      cache_state,
      cache_slots_pool, topk_src_ids, topk_dst_slots, topk_miss_counts, miss_src_ids,
      miss_dst_slots, miss_counts);
}

void fused_li_manage_mtp_meta(
    const at::Tensor& index_weights,
    const at::Tensor& query_dequant_scale,
    const at::Tensor& query,
    const at::Tensor& index_key_dequant_scale,
    const at::Tensor& index_key_cache,
    const at::Tensor& index_block_table,
    const at::Tensor& actual_seq_lengths_query,
    const at::Tensor& actual_seq_lengths_key,
    const at::Tensor& offload_seq_lengths_key,
    const at::Tensor& req_valid,
    const at::Tensor& req_pool_entries,
    at::Tensor cache_state,
    at::Tensor cache_slots_pool,
    at::Tensor topk_src_ids,
    at::Tensor topk_dst_slots,
    at::Tensor topk_miss_counts,
    at::Tensor miss_src_ids,
    at::Tensor miss_dst_slots,
    at::Tensor miss_counts) {
  (void)query;
  (void)index_weights;
  (void)index_key_cache;
  (void)index_block_table;
  (void)query_dequant_scale;
  (void)index_key_dequant_scale;
  (void)actual_seq_lengths_query;
  (void)actual_seq_lengths_key;
  (void)offload_seq_lengths_key;
  (void)req_valid;
  (void)req_pool_entries;
  (void)cache_state;
  (void)cache_slots_pool;
  (void)topk_src_ids;
  (void)topk_dst_slots;
  (void)topk_miss_counts;
  (void)miss_src_ids;
  (void)miss_dst_slots;
  (void)miss_counts;
}

void scatter_copy_torch_op(
    const at::Tensor& src_ids,
    const at::Tensor& dst_slots,
    const at::Tensor& copy_counts,
    const at::Tensor& hbm_block_table,
    const at::Tensor& dram_block_table,
    at::Tensor hbm_k_rope,
    at::Tensor hbm_kv_cache,
    const at::Tensor& dram_k_rope,
    const at::Tensor& dram_kv_cache) {
  vllm_ascend::npu_scatter_copy(
      src_ids, dst_slots, copy_counts, hbm_block_table, dram_block_table,
      hbm_k_rope, hbm_kv_cache, dram_k_rope, dram_kv_cache);
}

void scatter_copy_meta(
    const at::Tensor& src_ids,
    const at::Tensor& dst_slots,
    const at::Tensor& copy_counts,
    const at::Tensor& hbm_block_table,
    const at::Tensor& dram_block_table,
    at::Tensor hbm_k_rope,
    at::Tensor hbm_kv_cache,
    const at::Tensor& dram_k_rope,
    const at::Tensor& dram_kv_cache) {
  (void)src_ids;
  (void)dst_slots;
  (void)copy_counts;
  (void)hbm_block_table;
  (void)dram_block_table;
  (void)hbm_k_rope;
  (void)hbm_kv_cache;
  (void)dram_k_rope;
  (void)dram_kv_cache;
}

void sparse_tail_attention_mtp_torch_op(
    const at::Tensor& query_rope,
    const at::Tensor& query,
    const at::Tensor& actual_seq_lengths_query,
    const at::Tensor& actual_seq_lengths_kv,
    const at::Tensor& num_cache_tokens,
    const at::Tensor& topk_dst_slots,
    const at::Tensor& hbm_block_table,
    const at::Tensor& hbm_k_rope,
    const at::Tensor& hbm_kv_cache,
    double scale_value,
    at::Tensor attention_out) {
  vllm_ascend::npu_sparse_tail_attention_mtp(
      query_rope, query, actual_seq_lengths_query, actual_seq_lengths_kv,
      num_cache_tokens, topk_dst_slots, hbm_block_table, hbm_k_rope,
      hbm_kv_cache, scale_value, attention_out);
}

void sparse_tail_attention_mtp_meta(
    const at::Tensor& query_rope,
    const at::Tensor& query,
    const at::Tensor& actual_seq_lengths_query,
    const at::Tensor& actual_seq_lengths_kv,
    const at::Tensor& num_cache_tokens,
    const at::Tensor& topk_dst_slots,
    const at::Tensor& hbm_block_table,
    const at::Tensor& hbm_k_rope,
    const at::Tensor& hbm_kv_cache,
    double scale_value,
    at::Tensor attention_out) {
  (void)query_rope;
  (void)query;
  (void)actual_seq_lengths_query;
  (void)actual_seq_lengths_kv;
  (void)num_cache_tokens;
  (void)topk_dst_slots;
  (void)hbm_block_table;
  (void)hbm_k_rope;
  (void)hbm_kv_cache;
  (void)scale_value;
  (void)attention_out;
}

void fused_copy_sfa_mtp_torch_op(
    const at::Tensor& query_rope,
    const at::Tensor& query,
    const at::Tensor& actual_seq_lengths_query,
    const at::Tensor& actual_seq_lengths_kv,
    const at::Tensor& num_cache_tokens,
    const at::Tensor& topk_dst_slots,
    const at::Tensor& topk_src_ids,
    const at::Tensor& miss_src_ids,
    const at::Tensor& miss_dst_slots,
    const at::Tensor& miss_counts,
    const at::Tensor& hbm_block_table,
    const at::Tensor& dram_block_table,
    at::Tensor hbm_k_rope,
    at::Tensor hbm_kv_cache,
    const at::Tensor& dram_k_rope,
    const at::Tensor& dram_kv_cache,
    double scale_value,
    at::Tensor attention_out) {
  vllm_ascend::npu_fused_copy_sfa_mtp(
      query_rope, query, actual_seq_lengths_query, actual_seq_lengths_kv,
      num_cache_tokens, topk_dst_slots, topk_src_ids, miss_src_ids,
      miss_dst_slots, miss_counts, hbm_block_table, dram_block_table,
      hbm_k_rope, hbm_kv_cache, dram_k_rope, dram_kv_cache, scale_value,
      attention_out);
}

void fused_copy_sfa_mtp_meta(
    const at::Tensor& query_rope,
    const at::Tensor& query,
    const at::Tensor& actual_seq_lengths_query,
    const at::Tensor& actual_seq_lengths_kv,
    const at::Tensor& num_cache_tokens,
    const at::Tensor& topk_dst_slots,
    const at::Tensor& topk_src_ids,
    const at::Tensor& miss_src_ids,
    const at::Tensor& miss_dst_slots,
    const at::Tensor& miss_counts,
    const at::Tensor& hbm_block_table,
    const at::Tensor& dram_block_table,
    at::Tensor hbm_k_rope,
    at::Tensor hbm_kv_cache,
    const at::Tensor& dram_k_rope,
    const at::Tensor& dram_kv_cache,
    double scale_value,
    at::Tensor attention_out) {
  (void)query_rope;
  (void)query;
  (void)actual_seq_lengths_query;
  (void)actual_seq_lengths_kv;
  (void)num_cache_tokens;
  (void)topk_dst_slots;
  (void)topk_src_ids;
  (void)miss_src_ids;
  (void)miss_dst_slots;
  (void)miss_counts;
  (void)hbm_block_table;
  (void)dram_block_table;
  (void)hbm_k_rope;
  (void)hbm_kv_cache;
  (void)dram_k_rope;
  (void)dram_kv_cache;
  (void)scale_value;
  (void)attention_out;
}

}  // namespace

TORCH_LIBRARY(nanovllm_dsa, ops) {
  ops.def(
      "fused_li_manage_mtp(Tensor index_weights, Tensor query_dequant_scale,"
      " Tensor query, Tensor index_key_dequant_scale, Tensor index_key_cache,"
      " Tensor index_block_table, Tensor actual_seq_lengths_query,"
      " Tensor actual_seq_lengths_key, Tensor offload_seq_lengths_key,"
      " Tensor req_valid, Tensor req_pool_entries, Tensor(a!) cache_state,"
      " Tensor(b!) cache_slots_pool, Tensor(c!) topk_src_ids,"
      " Tensor(d!) topk_dst_slots, Tensor(e!) topk_miss_counts,"
      " Tensor(f!) miss_src_ids, Tensor(g!) miss_dst_slots,"
      " Tensor(h!) miss_counts) -> ()");
  ops.def(
      "scatter_copy(Tensor src_ids, Tensor dst_slots, Tensor copy_counts,"
      " Tensor hbm_block_table, Tensor dram_block_table,"
      " Tensor(a!) hbm_k_rope, Tensor(b!) hbm_kv_cache,"
      " Tensor dram_k_rope, Tensor dram_kv_cache) -> ()");
  ops.def(
      "sparse_tail_attention_mtp(Tensor query_rope, Tensor query,"
      " Tensor actual_seq_lengths_query, Tensor actual_seq_lengths_kv,"
      " Tensor num_cache_tokens, Tensor topk_dst_slots,"
      " Tensor hbm_block_table, Tensor hbm_k_rope, Tensor hbm_kv_cache,"
      " float scale_value, Tensor(a!) attention_out) -> ()");
  ops.def(
      "fused_copy_sfa_mtp(Tensor query_rope, Tensor query,"
      " Tensor actual_seq_lengths_query, Tensor actual_seq_lengths_kv,"
      " Tensor num_cache_tokens, Tensor topk_dst_slots,"
      " Tensor topk_src_ids, Tensor miss_src_ids, Tensor miss_dst_slots,"
      " Tensor miss_counts, Tensor hbm_block_table,"
      " Tensor dram_block_table, Tensor(a!) hbm_k_rope,"
      " Tensor(b!) hbm_kv_cache, Tensor dram_k_rope,"
      " Tensor dram_kv_cache, float scale_value,"
      " Tensor(c!) attention_out) -> ()");
}

TORCH_LIBRARY_IMPL(nanovllm_dsa, PrivateUse1, ops) {
  ops.impl("fused_li_manage_mtp", &fused_li_manage_mtp_torch_op);
  ops.impl("scatter_copy", &scatter_copy_torch_op);
  ops.impl("sparse_tail_attention_mtp", &sparse_tail_attention_mtp_torch_op);
  ops.impl("fused_copy_sfa_mtp", &fused_copy_sfa_mtp_torch_op);
}

TORCH_LIBRARY_IMPL(nanovllm_dsa, Meta, ops) {
  ops.impl("fused_li_manage_mtp", &fused_li_manage_mtp_meta);
  ops.impl("scatter_copy", &scatter_copy_meta);
  ops.impl("sparse_tail_attention_mtp", &sparse_tail_attention_mtp_meta);
  ops.impl("fused_copy_sfa_mtp", &fused_copy_sfa_mtp_meta);
}

PYBIND11_MODULE(_C, m) {
  m.doc() = "standalone GLM MTP3 offloading operator registration";
}
