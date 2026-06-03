#include <optional>
#include <string>
#include <vector>

#include <torch/extension.h>
#include <pybind11/stl.h>
#include <acl/acl.h>

#include "aclnn_torch_adapter/op_api_common.h"

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

#include "batch_matmul_transpose/batch_matmul_transpose_torch_adpt.h"
#include "cann_ops/lightning_indexer_vllm/lightning_indexer_vllm_torch_adpt.h"
#include "cann_ops/moe_gating_top_k/moe_gating_top_k_torch_adpt.h"
#include "cann_ops/paged_scatter_copy_h2d/paged_scatter_copy_h2d_torch_adpt.h"
#include "cann_ops/qk_score/qk_score_torch_adpt.h"
#include "cann_ops/dsa_update_index/dsa_update_index_torch_adpt.h"
#include "cann_ops/sparse_flash_attention/sparse_flash_attention_torch_adpt.h"
#include "dsa_indexer_project/dsa_indexer_project_torch_adpt.h"
#include "mla_preprocess/mla_preprocess_torch_adpt.h"

namespace py = pybind11;

namespace {

constexpr const char* kDsaIndexerProjectBindingVersion =
    "dsa_indexer_project_post_csrc_v1";

c10::optional<at::Tensor> optional_tensor(const py::object& obj) {
  if (obj.is_none()) {
    return c10::nullopt;
  }
  return obj.cast<at::Tensor>();
}

c10::optional<c10::string_view> optional_string_view(
    const py::object& obj,
    std::optional<std::string>& storage) {
  if (obj.is_none()) {
    return c10::nullopt;
  }
  storage = obj.cast<std::string>();
  return c10::string_view(storage->data(), storage->size());
}

at::Tensor npu_lightning_indexer_py(
    const at::Tensor& query,
    const at::Tensor& key,
    const at::Tensor& weights,
    py::object actual_seq_lengths_query,
    py::object actual_seq_lengths_key,
    py::object block_table,
    std::string layout_query,
    std::string layout_key,
    int64_t sparse_count,
    int64_t sparse_mode) {
  return vllm_ascend::npu_lightning_indexer(
      query,
      key,
      weights,
      optional_tensor(actual_seq_lengths_query),
      optional_tensor(actual_seq_lengths_key),
      optional_tensor(block_table),
      c10::string_view(layout_query.data(), layout_query.size()),
      c10::string_view(layout_key.data(), layout_key.size()),
      sparse_count,
      sparse_mode);
}

at::Tensor npu_qk_score_py(
    const at::Tensor& query,
    const at::Tensor& key,
    const at::Tensor& weights,
    py::object actual_seq_lengths_query,
    py::object actual_seq_lengths_key,
    py::object block_table,
    std::string layout_query,
    std::string layout_key) {
  return vllm_ascend::npu_qk_score(
      query,
      key,
      weights,
      optional_tensor(actual_seq_lengths_query),
      optional_tensor(actual_seq_lengths_key),
      optional_tensor(block_table),
      c10::string_view(layout_query.data(), layout_query.size()),
      c10::string_view(layout_key.data(), layout_key.size()));
}

void npu_qk_score_bf16_out_py(
    const at::Tensor& query,
    const at::Tensor& key,
    const at::Tensor& weights,
    py::object actual_seq_lengths_query,
    py::object actual_seq_lengths_key,
    py::object block_table,
    int64_t block_count,
    at::Tensor& score_out,
    std::string layout_query,
    std::string layout_key) {
  vllm_ascend::npu_qk_score_bf16_out(
      query,
      key,
      weights,
      optional_tensor(actual_seq_lengths_query),
      optional_tensor(actual_seq_lengths_key),
      optional_tensor(block_table),
      block_count,
      score_out,
      c10::string_view(layout_query.data(), layout_query.size()),
      c10::string_view(layout_key.data(), layout_key.size()));
}

void paged_scatter_copy_h2d_py(
    at::Tensor npu_krope_cache,
    at::Tensor npu_knope_cache,
    const at::Tensor& cpu_krope_cache,
    const at::Tensor& cpu_knope_cache,
    const at::Tensor& npu_block_table,
    const at::Tensor& cpu_block_table,
    const at::Tensor& npu_dst_token_index,
    const at::Tensor& cpu_src_token_index,
    const at::Tensor& copy_counts,
    int64_t block_size) {
  vllm_ascend::paged_scatter_copy_h2d(
      npu_krope_cache,
      npu_knope_cache,
      cpu_krope_cache,
      cpu_knope_cache,
      npu_block_table,
      cpu_block_table,
      npu_dst_token_index,
      cpu_src_token_index,
      copy_counts,
      block_size);
}

at::Tensor paged_scatter_copy_h2d_alloc_host_mapped_empty_py(
    const at::Tensor& dtype_template,
    std::vector<int64_t> sizes) {
  return vllm_ascend::paged_scatter_copy_h2d_alloc_host_mapped_empty(
      dtype_template,
      at::IntArrayRef(sizes));
}

at::Tensor npu_sparse_flash_attention_py(
    const at::Tensor& query,
    const at::Tensor& key,
    const at::Tensor& value,
    const at::Tensor& sparse_indices,
    double scale_value,
    int64_t sparse_block_size,
    py::object block_table,
    py::object actual_seq_lengths_query,
    py::object actual_seq_lengths_kv,
    py::object query_rope,
    py::object key_rope,
    std::string layout_query,
    std::string layout_kv,
    int64_t sparse_mode) {
  return vllm_ascend::npu_sparse_flash_attention(
      query,
      key,
      value,
      sparse_indices,
      scale_value,
      sparse_block_size,
      optional_tensor(block_table),
      optional_tensor(actual_seq_lengths_query),
      optional_tensor(actual_seq_lengths_kv),
      optional_tensor(query_rope),
      optional_tensor(key_rope),
      c10::string_view(layout_query.data(), layout_query.size()),
      c10::string_view(layout_kv.data(), layout_kv.size()),
      sparse_mode);
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> moe_gating_top_k_py(
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
    py::object bias_opt) {
  return vllm_ascend::moe_gating_top_k(
      x,
      k,
      k_group,
      group_count,
      group_select_mode,
      renorm,
      norm_type,
      out_flag,
      routed_scaling_factor,
      eps,
      optional_tensor(bias_opt));
}

void dsa_update_index_py(
    at::Tensor score,
    at::Tensor selected_idx,
    const at::Tensor& seq_len,
    const at::Tensor& selected_len,
    int64_t k,
    at::Tensor promote_idx,
    at::Tensor demote_idx) {
  vllm_ascend::dsa_update_index(
      score,
      selected_idx,
      seq_len,
      selected_len,
      k,
      promote_idx,
      demote_idx);
}

void batch_matmul_transpose_py(
    const at::Tensor& tensor_a,
    const at::Tensor& tensor_b,
    at::Tensor tensor_c,
    py::object format_mode,
    py::object quant_mode) {
  std::optional<std::string> format_storage;
  std::optional<std::string> quant_storage;
  auto format_view = optional_string_view(format_mode, format_storage);
  auto quant_view = optional_string_view(quant_mode, quant_storage);
  vllm_ascend::batch_matmul_transpose(
      tensor_a,
      tensor_b,
      tensor_c,
      format_view,
      quant_view);
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
mla_preprocess_py(
    const at::Tensor& hidden_state,
    const at::Tensor& wdqkv,
    py::object descale0,
    const at::Tensor& gamma1,
    py::object beta1,
    const at::Tensor& wuq,
    py::object descale1,
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
    py::object quant_scale0,
    py::object quant_offset0,
    py::object bias0,
    py::object quant_scale1,
    py::object quant_offset1,
    py::object bias1,
    py::object ctkv_scale,
    py::object q_nope_scale,
    py::object cache_mode,
    py::object quant_mode,
    bool enable_inner_out) {
  std::optional<std::string> cache_mode_storage;
  std::optional<std::string> quant_mode_storage;
  auto cache_mode_view = optional_string_view(cache_mode, cache_mode_storage);
  auto quant_mode_view = optional_string_view(quant_mode, quant_mode_storage);
  vllm_ascend::mla_preprocess(
      hidden_state,
      wdqkv,
      optional_tensor(descale0),
      gamma1,
      optional_tensor(beta1),
      wuq,
      optional_tensor(descale1),
      gamma2,
      cos,
      sin,
      wuk,
      kv_cache,
      kv_cache_rope,
      slotmapping,
      optional_tensor(quant_scale0),
      optional_tensor(quant_offset0),
      optional_tensor(bias0),
      optional_tensor(quant_scale1),
      optional_tensor(quant_offset1),
      optional_tensor(bias1),
      optional_tensor(ctkv_scale),
      optional_tensor(q_nope_scale),
      cache_mode_view,
      quant_mode_view,
      enable_inner_out,
      q_out0,
      kv_cache_out0,
      q_out1,
      kv_cache_out1,
      inner_out);
  return std::make_tuple(
      q_out0,
      kv_cache_out0,
      q_out1,
      kv_cache_out1,
      inner_out);
}

}  // namespace

PYBIND11_MODULE(_C, m) {
  m.def(
      "npu_lightning_indexer",
      &npu_lightning_indexer_py,
      py::arg("query"),
      py::arg("key"),
      py::arg("weights"),
      py::arg("actual_seq_lengths_query") = py::none(),
      py::arg("actual_seq_lengths_key") = py::none(),
      py::arg("block_table") = py::none(),
      py::arg("layout_query") = "BSND",
      py::arg("layout_key") = "BSND",
      py::arg("sparse_count") = 2048,
      py::arg("sparse_mode") = 3);
  m.def(
      "npu_qk_score",
      &npu_qk_score_py,
      py::arg("query"),
      py::arg("key"),
      py::arg("weights"),
      py::arg("actual_seq_lengths_query") = py::none(),
      py::arg("actual_seq_lengths_key") = py::none(),
      py::arg("block_table") = py::none(),
      py::arg("layout_query") = "BSND",
      py::arg("layout_key") = "PA_BSND");
  m.def(
      "npu_qk_score_bf16_out",
      &npu_qk_score_bf16_out_py,
      py::arg("query"),
      py::arg("key"),
      py::arg("weights"),
      py::arg("actual_seq_lengths_query"),
      py::arg("actual_seq_lengths_key"),
      py::arg("block_table"),
      py::arg("block_count"),
      py::arg("score_out"),
      py::arg("layout_query") = "TND",
      py::arg("layout_key") = "PA_BSND");
  m.def(
      "paged_scatter_copy_h2d",
      &paged_scatter_copy_h2d_py,
      py::arg("npu_krope_cache"),
      py::arg("npu_knope_cache"),
      py::arg("cpu_krope_cache"),
      py::arg("cpu_knope_cache"),
      py::arg("npu_block_table"),
      py::arg("cpu_block_table"),
      py::arg("npu_dst_token_index"),
      py::arg("cpu_src_token_index"),
      py::arg("copy_counts"),
      py::arg("block_size") = 128);
  m.def(
      "paged_scatter_copy_h2d_alloc_host_mapped_empty",
      &paged_scatter_copy_h2d_alloc_host_mapped_empty_py,
      py::arg("dtype_template"),
      py::arg("sizes"));
  m.def(
      "npu_sparse_flash_attention",
      &npu_sparse_flash_attention_py,
      py::arg("query"),
      py::arg("key"),
      py::arg("value"),
      py::arg("sparse_indices"),
      py::arg("scale_value") = 1.0,
      py::arg("sparse_block_size") = 1,
      py::arg("block_table") = py::none(),
      py::arg("actual_seq_lengths_query") = py::none(),
      py::arg("actual_seq_lengths_kv") = py::none(),
      py::arg("query_rope") = py::none(),
      py::arg("key_rope") = py::none(),
      py::arg("layout_query") = "BSND",
      py::arg("layout_kv") = "BSND",
      py::arg("sparse_mode") = 3);
  m.def(
      "moe_gating_top_k",
      &moe_gating_top_k_py,
      py::arg("x"),
      py::arg("k"),
      py::arg("k_group"),
      py::arg("group_count"),
      py::arg("group_select_mode"),
      py::arg("renorm"),
      py::arg("norm_type"),
      py::arg("out_flag"),
      py::arg("routed_scaling_factor"),
      py::arg("eps"),
      py::arg("bias_opt") = py::none());
  m.def(
      "dsa_update_index",
      &dsa_update_index_py,
      py::arg("score"),
      py::arg("selected_idx"),
      py::arg("seq_len"),
      py::arg("selected_len"),
      py::arg("k"),
      py::arg("promote_idx"),
      py::arg("demote_idx"));
  m.def(
      "batch_matmul_transpose",
      &batch_matmul_transpose_py,
      py::arg("tensor_a"),
      py::arg("tensor_b"),
      py::arg("tensor_c"),
      py::arg("format_mode") = py::none(),
      py::arg("quant_mode") = py::none());
  m.def(
      "mla_preprocess",
      &mla_preprocess_py,
      py::arg("hidden_state"),
      py::arg("wdqkv"),
      py::arg("descale0"),
      py::arg("gamma1"),
      py::arg("beta1"),
      py::arg("wuq"),
      py::arg("descale1"),
      py::arg("gamma2"),
      py::arg("cos"),
      py::arg("sin"),
      py::arg("wuk"),
      py::arg("kv_cache"),
      py::arg("kv_cache_rope"),
      py::arg("slotmapping"),
      py::arg("q_out0"),
      py::arg("kv_cache_out0"),
      py::arg("q_out1"),
      py::arg("kv_cache_out1"),
      py::arg("inner_out"),
      py::arg("quant_scale0") = py::none(),
      py::arg("quant_offset0") = py::none(),
      py::arg("bias0") = py::none(),
      py::arg("quant_scale1") = py::none(),
      py::arg("quant_offset1") = py::none(),
      py::arg("bias1") = py::none(),
      py::arg("ctkv_scale") = py::none(),
      py::arg("q_nope_scale") = py::none(),
      py::arg("cache_mode") = py::none(),
      py::arg("quant_mode") = py::none(),
      py::arg("enable_inner_out") = false);
  m.def(
      "dsa_indexer_project_binding_version",
      []() { return kDsaIndexerProjectBindingVersion; });
  m.def(
      "dsa_indexer_project_post",
      &vllm_ascend::dsa_indexer_project_post,
      py::arg("q_in"),
      py::arg("k_in"),
      py::arg("weights_in"),
      py::arg("cos"),
      py::arg("sin"),
      py::arg("score_scale"),
      py::arg("rope_dim"));
  m.def(
      "dsa_indexer_project_post_out",
      &vllm_ascend::dsa_indexer_project_post_out,
      py::arg("q_in"),
      py::arg("k_in"),
      py::arg("weights_in"),
      py::arg("cos"),
      py::arg("sin"),
      py::arg("q_out"),
      py::arg("k_out"),
      py::arg("weights_out"),
      py::arg("score_scale"),
      py::arg("rope_dim"));
}
