#include <optional>
#include <string>

#include <torch/extension.h>
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
}  // namespace vllm_ascend

#include "batch_matmul_transpose/batch_matmul_transpose_torch_adpt.h"
#include "cann_ops/lightning_indexer_vllm/lightning_indexer_vllm_torch_adpt.h"
#include "cann_ops/moe_gating_top_k/moe_gating_top_k_torch_adpt.h"
#include "cann_ops/sparse_flash_attention/sparse_flash_attention_torch_adpt.h"

namespace py = pybind11;

namespace {

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
}
