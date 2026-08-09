#include <array>
#include <string>
#include <tuple>

#include <acl/acl.h>
#include <torch/extension.h>
#include <torch/library.h>

#include "common/torch_adapter/op_api_common.h"

thread_local char g_hashBuf[kHashBufSize];
thread_local int g_hashOffset = 0;

namespace vllm_ascend {
extern void batch_matmul_transpose_impl(
    void* stream,
    void* gm_a,
    void* gm_b,
    void* gm_c,
    void* gm_tiling_data,
    const uint32_t block_dim);

extern void mla_preprocess_impl(
    void* stream,
    void* hidden_state,
    void* quant_scale1,
    void* quant_offset1,
    void* wdqkv,
    void* bias1,
    void* gamma2,
    void* beta2,
    void* quant_scale2,
    void* quant_offset2,
    void* gamma3,
    void* sin1,
    void* cos1,
    void* sin2,
    void* cos2,
    void* keycache,
    void* slot_mapping,
    void* wuq,
    void* bias2,
    void* wuk,
    void* descale1,
    void* descale2,
    void* ctkv_scale,
    void* qnope_scale,
    void* q,
    void* keycache_out,
    void* q2,
    void* keycache_out2,
    void* inner_out,
    void* workspace,
    void* tiling,
    const uint32_t block_dim);
}  // namespace vllm_ascend

#include "ops/batch_matmul_transpose/batch_matmul_transpose_torch_adpt.h"
#include "ops/dsa_indexer_project/dsa_indexer_project_torch_adpt.h"
#include "ops/fused_copy_sfa/fused_copy_sfa_torch_adpt.h"
#include "ops/fused_copy_sfa_mtp/fused_copy_sfa_mtp_torch_adpt.h"
#include "ops/fused_li_manage/fused_li_manage_torch_adpt.h"
#include "ops/fused_li_manage_mtp/fused_li_manage_mtp_torch_adpt.h"
#include "ops/kvcache_scatter_copy/kvcache_scatter_copy_torch_adpt.h"
#include "ops/matmul_allreduce_add_rmsnorm/matmul_allreduce_add_rmsnorm_torch_adpt.h"
#include "ops/mla_preprocess/mla_preprocess_torch_adpt.h"
#include "ops/moe_gating_top_k/moe_gating_top_k_torch_adpt.h"
#include "ops/sparse_tail_attention/sparse_tail_attention_torch_adpt.h"
#include "ops/sparse_tail_attention_mtp/sparse_tail_attention_mtp_torch_adpt.h"

namespace {

void fused_li_manage_torch_op(
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
    at::Tensor miss_counts) {
  vllm_ascend::npu_fused_li_manage(
      query, index_weights, index_key_cache, index_block_table,
      num_candidate_tokens, num_cache_tokens, req_pool_entries,
      cache_slots_pool, topk_src_ids, topk_dst_slots, miss_counts);
}

void fused_li_manage_meta(
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
  (void)miss_counts;
}

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

void sparse_tail_attention_torch_op(
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
  vllm_ascend::npu_sparse_tail_attention(
      query_rope, query, actual_seq_lengths_query, actual_seq_lengths_kv,
      num_cache_tokens, topk_dst_slots, hbm_block_table, hbm_k_rope,
      hbm_kv_cache, scale_value, attention_out);
}

void sparse_tail_attention_meta(
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

void fused_copy_sfa_torch_op(
    const at::Tensor& query_rope,
    const at::Tensor& query,
    const at::Tensor& actual_seq_lengths_query,
    const at::Tensor& actual_seq_lengths_kv,
    const at::Tensor& num_cache_tokens,
    const at::Tensor& topk_dst_slots,
    const at::Tensor& topk_src_ids,
    const at::Tensor& miss_counts,
    const at::Tensor& hbm_block_table,
    const at::Tensor& dram_block_table,
    at::Tensor hbm_k_rope,
    at::Tensor hbm_kv_cache,
    const at::Tensor& dram_k_rope,
    const at::Tensor& dram_kv_cache,
    double scale_value,
    at::Tensor attention_out) {
  vllm_ascend::npu_fused_copy_sfa(
      query_rope, query, actual_seq_lengths_query, actual_seq_lengths_kv,
      num_cache_tokens, topk_dst_slots, topk_src_ids, miss_counts,
      hbm_block_table, dram_block_table, hbm_k_rope, hbm_kv_cache,
      dram_k_rope, dram_kv_cache, scale_value, attention_out);
}

void fused_copy_sfa_meta(
    const at::Tensor& query_rope,
    const at::Tensor& query,
    const at::Tensor& actual_seq_lengths_query,
    const at::Tensor& actual_seq_lengths_kv,
    const at::Tensor& num_cache_tokens,
    const at::Tensor& topk_dst_slots,
    const at::Tensor& topk_src_ids,
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

std::tuple<at::Tensor, at::Tensor, at::Tensor> moe_gating_top_k_torch_op(
    const at::Tensor& x,
    int64_t k,
    int64_t k_group,
    int64_t group_count,
    int64_t group_select_mode,
    int64_t renorm,
    int64_t norm_type,
    bool out_flag,
    double routed_scaling_factor,
    double eps,
    const c10::optional<at::Tensor>& bias_opt) {
  return vllm_ascend::moe_gating_top_k(
      x, k, k_group, group_count, group_select_mode, renorm, norm_type,
      out_flag, routed_scaling_factor, eps, bias_opt);
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> moe_gating_top_k_meta(
    const at::Tensor& x,
    int64_t k,
    int64_t k_group,
    int64_t group_count,
    int64_t group_select_mode,
    int64_t renorm,
    int64_t norm_type,
    bool out_flag,
    double routed_scaling_factor,
    double eps,
    const c10::optional<at::Tensor>& bias_opt) {
  (void)k_group;
  (void)group_count;
  (void)group_select_mode;
  (void)renorm;
  (void)norm_type;
  (void)out_flag;
  (void)routed_scaling_factor;
  (void)eps;
  (void)bias_opt;
  auto y = at::empty({x.size(0), k}, x.options());
  auto expert_idx = at::empty({x.size(0), k}, x.options().dtype(at::kInt));
  auto out = at::empty(
      {x.size(0), x.size(1)}, x.options().dtype(at::kFloat));
  return std::make_tuple(y, expert_idx, out);
}

std::tuple<at::Tensor, at::Tensor>
matmul_allreduce_add_rmsnorm_torch_op(
    const at::Tensor& x1,
    const at::Tensor& x2,
    const at::Tensor& residual,
    const at::Tensor& gamma,
    c10::string_view group_tp,
    int64_t tp_rank_size,
    int64_t tp_rank_id,
    double epsilon,
    bool is_trans_b,
    bool is_gather_add_out) {
  return vllm_ascend::matmul_allreduce_add_rmsnorm(
      x1, x2, residual, gamma, group_tp, tp_rank_size, tp_rank_id, epsilon,
      is_trans_b, is_gather_add_out);
}

std::tuple<at::Tensor, at::Tensor> matmul_allreduce_add_rmsnorm_meta(
    const at::Tensor& x1,
    const at::Tensor& x2,
    const at::Tensor& residual,
    const at::Tensor& gamma,
    c10::string_view group_tp,
    int64_t tp_rank_size,
    int64_t tp_rank_id,
    double epsilon,
    bool is_trans_b,
    bool is_gather_add_out) {
  (void)x1;
  (void)x2;
  (void)gamma;
  (void)group_tp;
  (void)tp_rank_size;
  (void)tp_rank_id;
  (void)epsilon;
  (void)is_trans_b;
  (void)is_gather_add_out;
  return std::make_tuple(at::empty_like(residual), at::empty_like(residual));
}

void batch_matmul_transpose_torch_op(
    const at::Tensor& tensor_a,
    const at::Tensor& tensor_b,
    at::Tensor tensor_c,
    c10::optional<c10::string_view> format_mode,
    c10::optional<c10::string_view> quant_mode) {
  vllm_ascend::batch_matmul_transpose(
      tensor_a, tensor_b, tensor_c, format_mode, quant_mode);
}

void batch_matmul_transpose_meta(
    const at::Tensor& tensor_a,
    const at::Tensor& tensor_b,
    at::Tensor tensor_c,
    c10::optional<c10::string_view> format_mode,
    c10::optional<c10::string_view> quant_mode) {
  (void)tensor_a;
  (void)tensor_b;
  (void)tensor_c;
  (void)format_mode;
  (void)quant_mode;
}

void dsa_indexer_query_rope_inplace_torch_op(
    at::Tensor q_inout,
    const at::Tensor& cos,
    const at::Tensor& sin,
    int64_t rope_dim) {
  vllm_ascend::dsa_indexer_query_rope_inplace(q_inout, cos, sin, rope_dim);
}

void dsa_indexer_query_rope_inplace_meta(
    at::Tensor q_inout,
    const at::Tensor& cos,
    const at::Tensor& sin,
    int64_t rope_dim) {
  (void)q_inout;
  (void)cos;
  (void)sin;
  (void)rope_dim;
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
mla_preprocess_torch_op(
    const at::Tensor& hidden_state,
    const at::Tensor& wdqkv,
    const c10::optional<at::Tensor>& descale0,
    const at::Tensor& gamma1,
    const c10::optional<at::Tensor>& beta1,
    const at::Tensor& wuq,
    const c10::optional<at::Tensor>& descale1,
    const at::Tensor& gamma2,
    const at::Tensor& cos,
    const at::Tensor& sin,
    const at::Tensor& wuk,
    const at::Tensor& kv_cache,
    const at::Tensor& kv_cache_rope,
    const at::Tensor& slotmapping,
    at::Tensor q_out0,
    at::Tensor kv_cache_out0,
    at::Tensor q_out1,
    at::Tensor kv_cache_out1,
    at::Tensor inner_out,
    const c10::optional<at::Tensor>& quant_scale0,
    const c10::optional<at::Tensor>& quant_offset0,
    const c10::optional<at::Tensor>& bias0,
    const c10::optional<at::Tensor>& quant_scale1,
    const c10::optional<at::Tensor>& quant_offset1,
    const c10::optional<at::Tensor>& bias1,
    const c10::optional<at::Tensor>& ctkv_scale,
    const c10::optional<at::Tensor>& q_nope_scale,
    c10::optional<c10::string_view> cache_mode,
    c10::optional<c10::string_view> quant_mode,
    bool enable_inner_out) {
  vllm_ascend::mla_preprocess(
      hidden_state, wdqkv, descale0, gamma1, beta1, wuq, descale1, gamma2,
      cos, sin, wuk, kv_cache, kv_cache_rope, slotmapping, quant_scale0,
      quant_offset0, bias0, quant_scale1, quant_offset1, bias1, ctkv_scale,
      q_nope_scale, cache_mode, quant_mode, enable_inner_out, q_out0,
      kv_cache_out0, q_out1, kv_cache_out1, inner_out);
  return std::make_tuple(
      q_out0, kv_cache_out0, q_out1, kv_cache_out1, inner_out);
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
mla_preprocess_meta(
    const at::Tensor& hidden_state,
    const at::Tensor& wdqkv,
    const c10::optional<at::Tensor>& descale0,
    const at::Tensor& gamma1,
    const c10::optional<at::Tensor>& beta1,
    const at::Tensor& wuq,
    const c10::optional<at::Tensor>& descale1,
    const at::Tensor& gamma2,
    const at::Tensor& cos,
    const at::Tensor& sin,
    const at::Tensor& wuk,
    const at::Tensor& kv_cache,
    const at::Tensor& kv_cache_rope,
    const at::Tensor& slotmapping,
    at::Tensor q_out0,
    at::Tensor kv_cache_out0,
    at::Tensor q_out1,
    at::Tensor kv_cache_out1,
    at::Tensor inner_out,
    const c10::optional<at::Tensor>& quant_scale0,
    const c10::optional<at::Tensor>& quant_offset0,
    const c10::optional<at::Tensor>& bias0,
    const c10::optional<at::Tensor>& quant_scale1,
    const c10::optional<at::Tensor>& quant_offset1,
    const c10::optional<at::Tensor>& bias1,
    const c10::optional<at::Tensor>& ctkv_scale,
    const c10::optional<at::Tensor>& q_nope_scale,
    c10::optional<c10::string_view> cache_mode,
    c10::optional<c10::string_view> quant_mode,
    bool enable_inner_out) {
  (void)hidden_state;
  (void)wdqkv;
  (void)descale0;
  (void)gamma1;
  (void)beta1;
  (void)wuq;
  (void)descale1;
  (void)gamma2;
  (void)cos;
  (void)sin;
  (void)wuk;
  (void)kv_cache;
  (void)kv_cache_rope;
  (void)slotmapping;
  (void)quant_scale0;
  (void)quant_offset0;
  (void)bias0;
  (void)quant_scale1;
  (void)quant_offset1;
  (void)bias1;
  (void)ctkv_scale;
  (void)q_nope_scale;
  (void)cache_mode;
  (void)quant_mode;
  (void)enable_inner_out;
  return std::make_tuple(
      q_out0, kv_cache_out0, q_out1, kv_cache_out1, inner_out);
}

}  // namespace

TORCH_LIBRARY(nanovllm_dsa, ops) {
  ops.def(
      "fused_li_manage(Tensor query, Tensor index_weights,"
      " Tensor index_key_cache, Tensor index_block_table,"
      " Tensor num_candidate_tokens, Tensor num_cache_tokens,"
      " Tensor req_pool_entries, Tensor(a!) cache_slots_pool,"
      " Tensor(b!) topk_src_ids, Tensor(c!) topk_dst_slots,"
      " Tensor(d!) miss_counts) -> ()");
  ops.def(
      "fused_li_manage_mtp(Tensor query, Tensor index_weights,"
      " Tensor index_key_cache, Tensor index_block_table,"
      " Tensor num_candidate_tokens, Tensor num_cache_tokens,"
      " Tensor req_pool_entries, Tensor(a!) cache_slots_pool,"
      " Tensor(b!) topk_src_ids, Tensor(c!) topk_dst_slots,"
      " Tensor(d!) miss_src_ids, Tensor(e!) miss_dst_slots,"
      " Tensor(f!) miss_counts) -> ()");
  ops.def(
      "scatter_copy(Tensor src_ids, Tensor dst_slots, Tensor copy_counts,"
      " Tensor hbm_block_table, Tensor dram_block_table,"
      " Tensor(a!) hbm_k_rope, Tensor(b!) hbm_kv_cache,"
      " Tensor dram_k_rope, Tensor dram_kv_cache) -> ()");
  ops.def(
      "sparse_tail_attention(Tensor query_rope, Tensor query,"
      " Tensor actual_seq_lengths_query, Tensor actual_seq_lengths_kv,"
      " Tensor num_cache_tokens, Tensor topk_dst_slots,"
      " Tensor hbm_block_table, Tensor hbm_k_rope, Tensor hbm_kv_cache,"
      " float scale_value, Tensor(a!) attention_out) -> ()");
  ops.def(
      "sparse_tail_attention_mtp(Tensor query_rope, Tensor query,"
      " Tensor actual_seq_lengths_query, Tensor actual_seq_lengths_kv,"
      " Tensor num_cache_tokens, Tensor topk_dst_slots,"
      " Tensor hbm_block_table, Tensor hbm_k_rope, Tensor hbm_kv_cache,"
      " float scale_value, Tensor(a!) attention_out) -> ()");
  ops.def(
      "fused_copy_sfa(Tensor query_rope, Tensor query,"
      " Tensor actual_seq_lengths_query, Tensor actual_seq_lengths_kv,"
      " Tensor num_cache_tokens, Tensor topk_dst_slots,"
      " Tensor topk_src_ids, Tensor miss_counts,"
      " Tensor hbm_block_table, Tensor dram_block_table,"
      " Tensor(a!) hbm_k_rope, Tensor(b!) hbm_kv_cache,"
      " Tensor dram_k_rope, Tensor dram_kv_cache, float scale_value,"
      " Tensor(c!) attention_out) -> ()");
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

  ops.def(
      "moe_gating_top_k(Tensor x, int k, int k_group, int group_count,"
      " int group_select_mode, int renorm, int norm_type, bool out_flag,"
      " float routed_scaling_factor, float eps, Tensor? bias_opt=None)"
      " -> (Tensor, Tensor, Tensor)");
  ops.def(
      "matmul_allreduce_add_rmsnorm(Tensor x1, Tensor x2, Tensor residual,"
      " Tensor gamma, str group_tp, int tp_rank_size, int tp_rank_id,"
      " float epsilon, bool is_trans_b, bool is_gather_add_out)"
      " -> (Tensor, Tensor)");
  ops.def(
      "batch_matmul_transpose(Tensor tensor_a, Tensor tensor_b,"
      " Tensor(a!) tensor_c, str? format_mode=None, str? quant_mode=None)"
      " -> ()");
  ops.def(
      "dsa_indexer_query_rope_inplace(Tensor(a!) q_inout, Tensor cos,"
      " Tensor sin, int rope_dim) -> ()");
  ops.def(
      "mla_preprocess(Tensor hidden_state, Tensor wdqkv, Tensor? descale0,"
      " Tensor gamma1, Tensor? beta1, Tensor wuq, Tensor? descale1,"
      " Tensor gamma2, Tensor cos, Tensor sin, Tensor wuk, Tensor kv_cache,"
      " Tensor kv_cache_rope, Tensor slotmapping, Tensor(a!) q_out0,"
      " Tensor(b!) kv_cache_out0, Tensor(c!) q_out1,"
      " Tensor(d!) kv_cache_out1, Tensor(e!) inner_out,"
      " Tensor? quant_scale0=None, Tensor? quant_offset0=None,"
      " Tensor? bias0=None, Tensor? quant_scale1=None,"
      " Tensor? quant_offset1=None, Tensor? bias1=None,"
      " Tensor? ctkv_scale=None, Tensor? q_nope_scale=None,"
      " str? cache_mode=None, str? quant_mode=None,"
      " bool enable_inner_out=False)"
      " -> (Tensor(a!), Tensor(b!), Tensor(c!), Tensor(d!), Tensor(e!))");
}

TORCH_LIBRARY_IMPL(nanovllm_dsa, PrivateUse1, ops) {
  ops.impl("fused_li_manage", &fused_li_manage_torch_op);
  ops.impl("fused_li_manage_mtp", &fused_li_manage_mtp_torch_op);
  ops.impl("scatter_copy", &scatter_copy_torch_op);
  ops.impl("sparse_tail_attention", &sparse_tail_attention_torch_op);
  ops.impl("sparse_tail_attention_mtp", &sparse_tail_attention_mtp_torch_op);
  ops.impl("fused_copy_sfa", &fused_copy_sfa_torch_op);
  ops.impl("fused_copy_sfa_mtp", &fused_copy_sfa_mtp_torch_op);
  ops.impl("moe_gating_top_k", &moe_gating_top_k_torch_op);
  ops.impl(
      "matmul_allreduce_add_rmsnorm",
      &matmul_allreduce_add_rmsnorm_torch_op);
  ops.impl("batch_matmul_transpose", &batch_matmul_transpose_torch_op);
  ops.impl(
      "dsa_indexer_query_rope_inplace",
      &dsa_indexer_query_rope_inplace_torch_op);
  ops.impl("mla_preprocess", &mla_preprocess_torch_op);
}

TORCH_LIBRARY_IMPL(nanovllm_dsa, Meta, ops) {
  ops.impl("fused_li_manage", &fused_li_manage_meta);
  ops.impl("fused_li_manage_mtp", &fused_li_manage_mtp_meta);
  ops.impl("scatter_copy", &scatter_copy_meta);
  ops.impl("sparse_tail_attention", &sparse_tail_attention_meta);
  ops.impl("sparse_tail_attention_mtp", &sparse_tail_attention_mtp_meta);
  ops.impl("fused_copy_sfa", &fused_copy_sfa_meta);
  ops.impl("fused_copy_sfa_mtp", &fused_copy_sfa_mtp_meta);
  ops.impl("moe_gating_top_k", &moe_gating_top_k_meta);
  ops.impl(
      "matmul_allreduce_add_rmsnorm",
      &matmul_allreduce_add_rmsnorm_meta);
  ops.impl("batch_matmul_transpose", &batch_matmul_transpose_meta);
  ops.impl(
      "dsa_indexer_query_rope_inplace",
      &dsa_indexer_query_rope_inplace_meta);
  ops.impl("mla_preprocess", &mla_preprocess_meta);
}

PYBIND11_MODULE(_C, m) {
  m.doc() = "nano-vLLM Ascend torch.library operator registration";
}
