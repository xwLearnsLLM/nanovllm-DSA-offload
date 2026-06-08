#include <optional>
#include <string>
#include <vector>

#include <torch/extension.h>
#include <torch/library.h>
#include <pybind11/stl.h>
#include <acl/acl.h>

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
#include "ops/lightning_indexer/lightning_indexer_vllm_torch_adpt.h"
#include "ops/gather_selection_kv_cache/gather_selection_kv_cache_torch_adpt.h"
#include "ops/moe_gating_top_k/moe_gating_top_k_torch_adpt.h"
#include "ops/sparse_flash_attention/sparse_flash_attention_torch_adpt.h"
#include "ops/dsa_indexer_project/dsa_indexer_project_torch_adpt.h"
#include "ops/mla_preprocess/mla_preprocess_torch_adpt.h"

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

void npu_gather_selection_kv_cache_py(
    const at::Tensor& selection_k_rope,
    const at::Tensor& selection_kv_cache,
    const at::Tensor& selection_kv_block_table,
    const at::Tensor& selection_kv_block_status,
    const at::Tensor& req_pool_entries,
    const at::Tensor& selection_topk_indices,
    const at::Tensor& full_k_rope,
    const at::Tensor& full_kv_cache,
    const at::Tensor& full_kv_block_table,
    const at::Tensor& full_kv_actual_seq) {
  vllm_ascend::npu_gather_selection_kv_cache(
      selection_k_rope,
      selection_kv_cache,
      selection_kv_block_table,
      selection_kv_block_status,
      req_pool_entries,
      selection_topk_indices,
      full_k_rope,
      full_kv_cache,
      full_kv_block_table,
      full_kv_actual_seq);
}

std::tuple<at::Tensor, at::Tensor> lightning_indexer_torch_op(
    const at::Tensor& query,
    const at::Tensor& key,
    const at::Tensor& weights,
    const at::Tensor& actual_seq_lengths_query,
    const at::Tensor& actual_seq_lengths_key,
    const at::Tensor& block_table,
    c10::string_view layout_query,
    c10::string_view layout_key,
    int64_t sparse_count,
    int64_t sparse_mode,
    int64_t pre_tokens,
    int64_t next_tokens,
    bool return_value) {
  (void)pre_tokens;
  (void)next_tokens;
  (void)return_value;
  auto output = vllm_ascend::npu_lightning_indexer(
      query,
      key,
      weights,
      c10::optional<at::Tensor>(actual_seq_lengths_query),
      c10::optional<at::Tensor>(actual_seq_lengths_key),
      c10::optional<at::Tensor>(block_table),
      layout_query,
      layout_key,
      sparse_count,
      sparse_mode);
  return std::make_tuple(output, output);
}

std::tuple<at::Tensor, at::Tensor> lightning_indexer_meta(
    const at::Tensor& query,
    const at::Tensor& key,
    const at::Tensor& weights,
    const at::Tensor& actual_seq_lengths_query,
    const at::Tensor& actual_seq_lengths_key,
    const at::Tensor& block_table,
    c10::string_view layout_query,
    c10::string_view layout_key,
    int64_t sparse_count,
    int64_t sparse_mode,
    int64_t pre_tokens,
    int64_t next_tokens,
    bool return_value) {
  (void)weights;
  (void)actual_seq_lengths_query;
  (void)actual_seq_lengths_key;
  (void)block_table;
  (void)sparse_mode;
  (void)pre_tokens;
  (void)next_tokens;
  (void)return_value;
  TORCH_CHECK(query.dim() >= 2, "lightning_indexer query must have at least 2 dims.");
  TORCH_CHECK(key.dim() >= 3, "lightning_indexer key must have at least 3 dims.");
  TORCH_CHECK(sparse_count > 0, "sparse_count must be > 0.");
  at::Tensor output;
  if (std::string(layout_query) == "BSND") {
    output = at::empty({query.size(0), query.size(1), key.size(2), sparse_count}, query.options().dtype(at::kInt));
  } else {
    const int64_t n_dim_index = (std::string(layout_key) == "TND") ? 1 : 2;
    output = at::empty({query.size(0), key.size(n_dim_index), sparse_count}, query.options().dtype(at::kInt));
  }
  // TorchAir's generic custom-op path expects tuple/list meta_outputs.
  // The second tensor is intentionally ignored by Python callers.
  return std::make_tuple(output, output);
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> gather_selection_kv_cache_torch_op(
    at::Tensor selection_k_rope,
    at::Tensor selection_kv_cache,
    const at::Tensor& selection_kv_block_table,
    at::Tensor selection_kv_block_status,
    const at::Tensor& req_pool_entries,
    const at::Tensor& selection_topk_indices,
    const at::Tensor& full_k_rope,
    const at::Tensor& full_kv_cache,
    const at::Tensor& full_kv_block_table,
    const at::Tensor& full_kv_actual_seq) {
  vllm_ascend::npu_gather_selection_kv_cache(
      selection_k_rope,
      selection_kv_cache,
      selection_kv_block_table,
      selection_kv_block_status,
      req_pool_entries,
      selection_topk_indices,
      full_k_rope,
      full_kv_cache,
      full_kv_block_table,
      full_kv_actual_seq);
  return std::make_tuple(selection_k_rope, selection_kv_cache, selection_kv_block_status);
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> gather_selection_kv_cache_meta(
    at::Tensor selection_k_rope,
    at::Tensor selection_kv_cache,
    const at::Tensor& selection_kv_block_table,
    at::Tensor selection_kv_block_status,
    const at::Tensor& req_pool_entries,
    const at::Tensor& selection_topk_indices,
    const at::Tensor& full_k_rope,
    const at::Tensor& full_kv_cache,
    const at::Tensor& full_kv_block_table,
    const at::Tensor& full_kv_actual_seq) {
  (void)selection_k_rope;
  (void)selection_kv_cache;
  (void)selection_kv_block_table;
  (void)selection_kv_block_status;
  (void)req_pool_entries;
  (void)selection_topk_indices;
  (void)full_k_rope;
  (void)full_kv_cache;
  (void)full_kv_block_table;
  (void)full_kv_actual_seq;
  return std::make_tuple(selection_k_rope, selection_kv_cache, selection_kv_block_status);
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

TORCH_LIBRARY(nanovllm_dsa, ops) {
  ops.def(
      "lightning_indexer(Tensor query, Tensor key, Tensor weights,"
      " Tensor actual_seq_lengths_query, Tensor actual_seq_lengths_key,"
      " Tensor block_table, str layout_query, str layout_key,"
      " int sparse_count, int sparse_mode, int pre_tokens, int next_tokens,"
      " bool return_value) -> (Tensor, Tensor)");
  ops.def(
      "gather_selection_kv_cache(Tensor(a!) selection_k_rope,"
      " Tensor(b!) selection_kv_cache, Tensor selection_kv_block_table,"
      " Tensor(c!) selection_kv_block_status, Tensor req_pool_entries,"
      " Tensor selection_topk_indices, Tensor full_k_rope, Tensor full_kv_cache,"
      " Tensor full_kv_block_table, Tensor full_kv_actual_seq)"
      " -> (Tensor, Tensor, Tensor)");
}

TORCH_LIBRARY_IMPL(nanovllm_dsa, PrivateUse1, ops) {
  ops.impl("lightning_indexer", &lightning_indexer_torch_op);
  ops.impl("gather_selection_kv_cache", &gather_selection_kv_cache_torch_op);
}

TORCH_LIBRARY_IMPL(nanovllm_dsa, Meta, ops) {
  ops.impl("lightning_indexer", &lightning_indexer_meta);
  ops.impl("gather_selection_kv_cache", &gather_selection_kv_cache_meta);
}

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
      "npu_gather_selection_kv_cache",
      &npu_gather_selection_kv_cache_py,
      py::arg("selection_k_rope"),
      py::arg("selection_kv_cache"),
      py::arg("selection_kv_block_table"),
      py::arg("selection_kv_block_status"),
      py::arg("req_pool_entries"),
      py::arg("selection_topk_indices"),
      py::arg("full_k_rope"),
      py::arg("full_kv_cache"),
      py::arg("full_kv_block_table"),
      py::arg("full_kv_actual_seq"));
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
