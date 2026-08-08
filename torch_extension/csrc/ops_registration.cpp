#include <torch/extension.h>
#include <torch/library.h>

TORCH_LIBRARY(nanovllm_dsa, m) {
  m.def(
      "fused_li_manage(Tensor query, Tensor key, Tensor weights, "
      "Tensor req_pool_entries, Tensor(a!) cache_slots_pool, "
      "Tensor cache_tokens, Tensor candidate_lens, Tensor block_table) "
      "-> (Tensor, Tensor, Tensor, Tensor(a!))");
  m.def(
      "fused_li_manage_out(Tensor query, Tensor key, Tensor weights, "
      "Tensor req_pool_entries, Tensor(a!) cache_slots_pool, "
      "Tensor cache_tokens, Tensor candidate_lens, Tensor block_table, "
      "Tensor(b!) source_ids, Tensor(c!) destination_slots, "
      "Tensor(d!) miss_counts) "
      "-> (Tensor(b!), Tensor(c!), Tensor(d!), Tensor(a!))");
  m.def(
      "_fused_li_manage_c8_cache_update(Tensor topk_indices, Tensor req_pool_entries, "
      "Tensor(a!) cache_slots_pool, Tensor cache_tokens, "
      "Tensor candidate_lens) "
      "-> (Tensor, Tensor, Tensor, Tensor(a!))");
  m.def(
      "_fused_li_manage_c8_cache_update_out(Tensor topk_indices, "
      "Tensor req_pool_entries, Tensor(a!) cache_slots_pool, "
      "Tensor cache_tokens, Tensor candidate_lens, "
      "Tensor(b!) source_ids, Tensor(c!) destination_slots, "
      "Tensor(d!) miss_counts) "
      "-> (Tensor(b!), Tensor(c!), Tensor(d!), Tensor(a!))");
  m.def(
      "fused_li_manage_c8(Tensor query, Tensor key, Tensor weights, "
      "Tensor query_dequant_scale, Tensor key_dequant_scale, "
      "Tensor actual_seq_lengths_query, Tensor req_pool_entries, "
      "Tensor(a!) cache_slots_pool, Tensor cache_tokens, "
      "Tensor candidate_lens, Tensor block_table) "
      "-> (Tensor, Tensor, Tensor, Tensor(a!))");
  m.def(
      "fused_li_manage_c8_out(Tensor query, Tensor key, "
      "Tensor weights, Tensor query_dequant_scale, "
      "Tensor key_dequant_scale, Tensor actual_seq_lengths_query, "
      "Tensor req_pool_entries, Tensor(a!) cache_slots_pool, "
      "Tensor cache_tokens, Tensor candidate_lens, "
      "Tensor block_table, Tensor(b!) source_ids, "
      "Tensor(c!) destination_slots, Tensor(d!) miss_counts) "
      "-> (Tensor(b!), Tensor(c!), Tensor(d!), Tensor(a!))");
  m.def(
      "kvcache_scatter_copy(Tensor(a!) hbm_kpe, Tensor(b!) hbm_ckv, "
      "Tensor dram_kpe, Tensor dram_ckv, Tensor hbm_block_table, "
      "Tensor dram_block_table, Tensor source_token_ids, "
      "Tensor destination_slots, Tensor copy_counts) "
      "-> (Tensor(a!), Tensor(b!))");
  m.def(
      "kvcache_scatter_copy_c8(Tensor(a!) hbm_kv_bytes, Tensor dram_kv_bytes, "
      "Tensor hbm_block_table, Tensor dram_block_table, "
      "Tensor source_token_ids, Tensor destination_slots, "
      "Tensor copy_counts, Tensor cache_tokens, Tensor candidate_lens, "
      "Tensor actual_seq_lengths_kv, int max_tail_tokens) "
      "-> (Tensor(a!), Tensor, Tensor)");
  m.def(
      "kvcache_scatter_copy_c8_out(Tensor(a!) hbm_kv_bytes, Tensor dram_kv_bytes, "
      "Tensor hbm_block_table, Tensor dram_block_table, "
      "Tensor source_token_ids, Tensor destination_slots, "
      "Tensor copy_counts, Tensor cache_tokens, Tensor candidate_lens, "
      "Tensor actual_seq_lengths_kv, int max_tail_tokens, "
      "Tensor(b!) attention_slots, Tensor(c!) resident_seq_lengths) "
      "-> (Tensor(a!), Tensor(b!), Tensor(c!))");
  m.def(
      "sparse_tail_attention(Tensor query, Tensor key, Tensor value, "
      "Tensor sparse_slots, Tensor cache_tokens, Tensor block_table, "
      "Tensor actual_seq_lengths_query, Tensor actual_seq_lengths_kv, "
      "Tensor query_rope, Tensor key_rope, float scale_value) -> Tensor");
  m.def(
      "fused_copy_sparse_tail_attention("
      "Tensor query, Tensor(a!) hbm_ckv, Tensor sparse_slots, "
      "Tensor cache_tokens, Tensor hbm_block_table, "
      "Tensor actual_seq_lengths_query, Tensor actual_seq_lengths_kv, "
      "Tensor query_rope, Tensor(b!) hbm_kpe, Tensor dram_kpe, "
      "Tensor dram_ckv, Tensor dram_block_table, "
      "Tensor source_token_ids, Tensor copy_counts, float scale_value, "
      "int prefetch_rows_per_step=5) "
      "-> (Tensor, Tensor(b!), Tensor(a!))");
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {}
