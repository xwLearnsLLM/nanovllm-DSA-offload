#include <initializer_list>
#include <limits>
#include <string>
#include <tuple>

#include <torch/extension.h>
#include <torch/library.h>

#include "torch_npu/csrc/framework/OpCommand.h"

namespace {
constexpr int64_t kBlockSize = 128;
constexpr int64_t kKpeDim = 64;
constexpr int64_t kCkvDim = 512;
constexpr int64_t kIndexerDim = 128;
constexpr int64_t kSparseCount = 2048;
constexpr int64_t kMaxSourceCapacity = 1 << 18;

void CheckOneDeviceAndContiguous(
    const at::Tensor& reference,
    std::initializer_list<const at::Tensor*> tensors,
    const char* op_name) {
  TORCH_CHECK(reference.device().is_privateuseone(), op_name, " inputs must be on NPU.");
  for (const at::Tensor* tensor : tensors) {
    TORCH_CHECK(
        tensor->device() == reference.device() && tensor->is_contiguous(),
        op_name, " inputs must be contiguous tensors on one NPU.");
  }
}

void CheckLiduCommon(
    const at::Tensor& query,
    const at::Tensor& key,
    const at::Tensor& weights,
    const at::Tensor& req_pool_entries,
    const at::Tensor& cache_slots_pool,
    const at::Tensor& cache_tokens,
    const at::Tensor& candidate_lens,
    const at::Tensor& block_table) {
  TORCH_CHECK(
      query.dim() == 3 && query.size(0) > 0 &&
          (query.size(1) == 32 || query.size(1) == 64) &&
          query.size(2) == kIndexerDim,
      "LIDU query must be [B,32|64,128].");
  const int64_t batch = query.size(0);
  TORCH_CHECK(
      key.dim() == 4 && key.size(0) > 0 && key.size(1) == kBlockSize &&
          key.size(2) == 1 && key.size(3) == kIndexerDim,
      "LIDU key must be [blocks,128,1,128].");
  TORCH_CHECK(
      weights.dim() == 2 && weights.size(0) == batch &&
          weights.size(1) == query.size(1),
      "LIDU weights must match query [B,N].");
  TORCH_CHECK(
      req_pool_entries.dim() == 1 && req_pool_entries.size(0) == batch &&
          cache_tokens.dim() == 1 && cache_tokens.size(0) == batch &&
          candidate_lens.dim() == 1 && candidate_lens.size(0) == batch,
      "LIDU request metadata must be int32[B].");
  TORCH_CHECK(
      cache_slots_pool.dim() == 2 && cache_slots_pool.size(0) > 0 &&
          cache_slots_pool.size(1) > 0 &&
          cache_slots_pool.size(1) <= kMaxSourceCapacity,
      "LIDU cache_slots_pool must be [pool_size,capacity], capacity <= 2^18.");
  TORCH_CHECK(
      block_table.dim() == 2 && block_table.size(0) == batch &&
          block_table.size(1) > 0 &&
          block_table.size(1) * kBlockSize == cache_slots_pool.size(1),
      "LIDU block_table capacity must equal cache_slots_pool.shape[1].");
  TORCH_CHECK(
      query.scalar_type() == at::kBFloat16 || query.scalar_type() == at::kHalf,
      "LIDU floating tensors must be bf16 or fp16.");
  TORCH_CHECK(
      key.scalar_type() == query.scalar_type() &&
          weights.scalar_type() == query.scalar_type(),
      "LIDU query/key/weights dtypes must match.");
  for (const at::Tensor* tensor :
       {&req_pool_entries, &cache_slots_pool, &cache_tokens,
        &candidate_lens, &block_table}) {
    TORCH_CHECK(tensor->scalar_type() == at::kInt, "LIDU metadata must be int32.");
  }
  CheckOneDeviceAndContiguous(
      query,
      {&query, &key, &weights, &req_pool_entries, &cache_slots_pool,
       &cache_tokens, &candidate_lens, &block_table},
      "LIDU");
}

void CheckLiduOutputs(
    const at::Tensor& query,
    const at::Tensor& source_ids,
    const at::Tensor& destination_slots,
    const at::Tensor& miss_counts) {
  const int64_t batch = query.size(0);
  TORCH_CHECK(
      source_ids.dim() == 3 && source_ids.size(0) == batch &&
          source_ids.size(1) == 1 && source_ids.size(2) == kSparseCount &&
          destination_slots.sizes() == source_ids.sizes() &&
          miss_counts.dim() == 1 && miss_counts.size(0) == batch,
      "LIDU out buffers must be source/destination [B,1,2048], miss_counts [B].");
  for (const at::Tensor* tensor : {&source_ids, &destination_slots, &miss_counts}) {
    TORCH_CHECK(
        tensor->scalar_type() == at::kInt && tensor->device() == query.device() &&
            tensor->is_contiguous(),
        "LIDU out buffers must be contiguous int32 tensors on the query NPU.");
  }
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor>
LiduDecodeUpdateOutNpu(
    const at::Tensor& query,
    const at::Tensor& key,
    const at::Tensor& weights,
    const at::Tensor& req_pool_entries,
    at::Tensor cache_slots_pool,
    const at::Tensor& cache_tokens,
    const at::Tensor& candidate_lens,
    const at::Tensor& block_table,
    at::Tensor source_ids,
    at::Tensor destination_slots,
    at::Tensor miss_counts) {
  CheckLiduCommon(
      query, key, weights, req_pool_entries, cache_slots_pool,
      cache_tokens, candidate_lens, block_table);
  CheckLiduOutputs(query, source_ids, destination_slots, miss_counts);

  at_npu::native::OpCommand cmd;
  cmd.Name("LightningIndexerDecodeUpdateA5")
      .Input(query)
      .Input(key)
      .Input(weights)
      .Input(req_pool_entries)
      .Input(cache_slots_pool)
      .Input(cache_tokens)
      .Input(candidate_lens)
      .Input(block_table)
      .Output(source_ids)
      .Output(destination_slots)
      .Output(miss_counts)
      .Output(cache_slots_pool)
      .Run();
  return std::make_tuple(
      source_ids, destination_slots, miss_counts, cache_slots_pool);
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor>
LiduDecodeUpdateNpu(
    const at::Tensor& query,
    const at::Tensor& key,
    const at::Tensor& weights,
    const at::Tensor& req_pool_entries,
    at::Tensor cache_slots_pool,
    const at::Tensor& cache_tokens,
    const at::Tensor& candidate_lens,
    const at::Tensor& block_table) {
  auto options = query.options().dtype(at::kInt);
  auto source_ids = at::empty({query.size(0), 1, kSparseCount}, options);
  auto destination_slots = at::empty_like(source_ids);
  auto miss_counts = at::empty({query.size(0)}, options);
  return LiduDecodeUpdateOutNpu(
      query, key, weights, req_pool_entries, cache_slots_pool,
      cache_tokens, candidate_lens, block_table,
      source_ids, destination_slots, miss_counts);
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor>
LiduDecodeUpdateMeta(
    const at::Tensor& query,
    const at::Tensor&,
    const at::Tensor&,
    const at::Tensor&,
    at::Tensor cache_slots_pool,
    const at::Tensor&,
    const at::Tensor&,
    const at::Tensor&) {
  auto options = query.options().dtype(at::kInt);
  return std::make_tuple(
      at::empty({query.size(0), 1, kSparseCount}, options),
      at::empty({query.size(0), 1, kSparseCount}, options),
      at::empty({query.size(0)}, options),
      cache_slots_pool);
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor>
LiduDecodeUpdateOutMeta(
    const at::Tensor&,
    const at::Tensor&,
    const at::Tensor&,
    const at::Tensor&,
    at::Tensor cache_slots_pool,
    const at::Tensor&,
    const at::Tensor&,
    const at::Tensor&,
    at::Tensor source_ids,
    at::Tensor destination_slots,
    at::Tensor miss_counts) {
  return std::make_tuple(
      source_ids, destination_slots, miss_counts, cache_slots_pool);
}

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

std::tuple<at::Tensor, at::Tensor> ScatterCopyNpu(
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
  at_npu::native::OpCommand cmd;
  cmd.Name("A5KvcacheScatterCopy")
      .Input(hbm_kpe)
      .Input(hbm_ckv)
      .Input(dram_kpe)
      .Input(dram_ckv)
      .Input(hbm_block_table)
      .Input(dram_block_table)
      .Input(source_token_ids)
      .Input(destination_slots)
      .Input(copy_counts)
      .Output(hbm_kpe)
      .Output(hbm_ckv)
      .Run();
  return std::make_tuple(hbm_kpe, hbm_ckv);
}

std::tuple<at::Tensor, at::Tensor> ScatterCopyMeta(
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

void CheckAttentionInputs(
    const at::Tensor& query,
    const at::Tensor& key,
    const at::Tensor& value,
    const at::Tensor& sparse_slots,
    const at::Tensor& cache_tokens,
    const at::Tensor& block_table,
    const at::Tensor& actual_q,
    const at::Tensor& actual_kv,
    const at::Tensor& query_rope,
    const at::Tensor& key_rope) {
  TORCH_CHECK(
      query.dim() == 3 && query.size(0) > 0 && query.size(2) == kCkvDim &&
          ((query.size(1) >= 1 && query.size(1) <= 64) || query.size(1) == 128),
      "SFA query must be [B,N,512], 1 <= N <= 64 or N == 128.");
  const int64_t batch = query.size(0);
  const int64_t heads = query.size(1);
  TORCH_CHECK(
      key.dim() == 4 && key.size(1) == kBlockSize && key.size(2) == 1 &&
          key.size(3) == kCkvDim && value.sizes() == key.sizes(),
      "SFA key/value must be [blocks,128,1,512].");
  TORCH_CHECK(
      value.is_same(key),
      "SFA is the MLA path and requires value to alias key.");
  TORCH_CHECK(
      key_rope.dim() == 4 && key_rope.size(0) == key.size(0) &&
          key_rope.size(1) == kBlockSize && key_rope.size(2) == 1 &&
          key_rope.size(3) == kKpeDim &&
          query_rope.dim() == 3 && query_rope.size(0) == batch &&
          query_rope.size(1) == heads && query_rope.size(2) == kKpeDim,
      "SFA RoPE tensors must be [blocks,128,1,64] and [B,N,64].");
  TORCH_CHECK(
      sparse_slots.dim() == 3 && sparse_slots.size(0) == batch &&
          sparse_slots.size(1) == 1 && sparse_slots.size(2) == kSparseCount &&
          cache_tokens.dim() == 1 && cache_tokens.size(0) == batch &&
          actual_q.dim() == 1 && actual_q.size(0) == batch &&
          actual_kv.dim() == 1 && actual_kv.size(0) == batch &&
          block_table.dim() == 2 && block_table.size(0) == batch &&
          block_table.size(1) > 0 &&
          block_table.size(1) * kBlockSize <= kMaxSourceCapacity,
      "SFA metadata shapes are invalid.");
  const auto dtype = query.scalar_type();
  TORCH_CHECK(dtype == at::kBFloat16 || dtype == at::kHalf, "SFA supports bf16/fp16.");
  for (const at::Tensor* tensor : {&key, &value, &query_rope, &key_rope}) {
    TORCH_CHECK(tensor->scalar_type() == dtype, "All SFA floating dtypes must match.");
  }
  for (const at::Tensor* tensor :
       {&sparse_slots, &cache_tokens, &block_table, &actual_q, &actual_kv}) {
    TORCH_CHECK(tensor->scalar_type() == at::kInt, "SFA metadata must be int32.");
  }
  CheckOneDeviceAndContiguous(
      query,
      {&query, &key, &value, &sparse_slots, &cache_tokens, &block_table,
       &actual_q, &actual_kv, &query_rope, &key_rope},
      "SFA");
}

void AddAttentionAttrs(at_npu::native::OpCommand& cmd, double scale_value) {
  cmd.Attr("scale_value", static_cast<float>(scale_value))
      .Attr("sparse_block_size", static_cast<int64_t>(1))
      .Attr("layout_query", std::string("TND"))
      .Attr("layout_kv", std::string("PA_BSND"))
      .Attr("sparse_mode", static_cast<int64_t>(3))
      .Attr("pre_tokens", std::numeric_limits<int64_t>::max())
      .Attr("next_tokens", std::numeric_limits<int64_t>::max())
      .Attr("attention_mode", static_cast<int64_t>(2))
      .Attr("return_softmax_lse", false);
}

at::Tensor SparseAndTailAttentionNpu(
    const at::Tensor& query,
    const at::Tensor& key,
    const at::Tensor& value,
    const at::Tensor& sparse_slots,
    const at::Tensor& cache_tokens,
    const at::Tensor& block_table,
    const at::Tensor& actual_q,
    const at::Tensor& actual_kv,
    const at::Tensor& query_rope,
    const at::Tensor& key_rope,
    double scale_value) {
  CheckAttentionInputs(
      query, key, value, sparse_slots, cache_tokens, block_table,
      actual_q, actual_kv, query_rope, key_rope);
  auto output = at::empty_like(query);
  auto softmax_max = at::empty({1}, query.options().dtype(at::kFloat));
  auto softmax_sum = at::empty({1}, query.options().dtype(at::kFloat));
  at_npu::native::OpCommand cmd;
  cmd.Name("A5SparseAndTailAttention")
      .Input(query)
      .Input(key)
      .Input(value)
      .Input(sparse_slots)
      .Input(block_table)
      .Input(actual_q)
      .Input(actual_kv)
      .Input(query_rope)
      .Input(key_rope)
      .Input(cache_tokens)
      .Output(output)
      .Output(softmax_max)
      .Output(softmax_sum);
  AddAttentionAttrs(cmd, scale_value);
  cmd.Run();
  return output;
}

at::Tensor SparseAndTailAttentionMeta(
    const at::Tensor& query,
    const at::Tensor&,
    const at::Tensor&,
    const at::Tensor&,
    const at::Tensor&,
    const at::Tensor&,
    const at::Tensor&,
    const at::Tensor&,
    const at::Tensor&,
    const at::Tensor&,
    double) {
  return at::empty_like(query);
}
}  // namespace

TORCH_LIBRARY(nanovllm_dsa, m) {
  m.def(
      "lidu_decode_update(Tensor query, Tensor key, Tensor weights, "
      "Tensor req_pool_entries, Tensor(a!) cache_slots_pool, "
      "Tensor cache_tokens, Tensor candidate_lens, Tensor block_table) "
      "-> (Tensor, Tensor, Tensor, Tensor(a!))");
  m.def(
      "lidu_decode_update_out(Tensor query, Tensor key, Tensor weights, "
      "Tensor req_pool_entries, Tensor(a!) cache_slots_pool, "
      "Tensor cache_tokens, Tensor candidate_lens, Tensor block_table, "
      "Tensor(b!) source_ids, Tensor(c!) destination_slots, "
      "Tensor(d!) miss_counts) "
      "-> (Tensor(b!), Tensor(c!), Tensor(d!), Tensor(a!))");
  m.def(
      "scatter_copy(Tensor(a!) hbm_kpe, Tensor(b!) hbm_ckv, "
      "Tensor dram_kpe, Tensor dram_ckv, Tensor hbm_block_table, "
      "Tensor dram_block_table, Tensor source_token_ids, "
      "Tensor destination_slots, Tensor copy_counts) "
      "-> (Tensor(a!), Tensor(b!))");
  m.def(
      "sparse_and_tail_attention(Tensor query, Tensor key, Tensor value, "
      "Tensor sparse_slots, Tensor cache_tokens, Tensor block_table, "
      "Tensor actual_seq_lengths_query, Tensor actual_seq_lengths_kv, "
      "Tensor query_rope, Tensor key_rope, float scale_value) -> Tensor");
}

TORCH_LIBRARY_IMPL(nanovllm_dsa, PrivateUse1, m) {
  m.impl("lidu_decode_update", &LiduDecodeUpdateNpu);
  m.impl("lidu_decode_update_out", &LiduDecodeUpdateOutNpu);
  m.impl("scatter_copy", &ScatterCopyNpu);
  m.impl("sparse_and_tail_attention", &SparseAndTailAttentionNpu);
}

TORCH_LIBRARY_IMPL(nanovllm_dsa, Meta, m) {
  m.impl("lidu_decode_update", &LiduDecodeUpdateMeta);
  m.impl("lidu_decode_update_out", &LiduDecodeUpdateOutMeta);
  m.impl("scatter_copy", &ScatterCopyMeta);
  m.impl("sparse_and_tail_attention", &SparseAndTailAttentionMeta);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {}
