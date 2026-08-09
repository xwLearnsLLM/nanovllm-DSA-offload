# GLM-5.1 DSA 卸载算子

本仓库内置以下七个算子，不依赖外部算子仓库：

| 算子 | 用途 | MTP3 |
| --- | --- | :---: |
| `fused_li_manage` | query_len=1 的 LightningIndexer、命中/淘汰与 request-pool 更新 | 否 |
| `fused_li_manage_mtp` | 四路 query 的 top-2048 并集、命中/淘汰与 request-pool 更新 | 是 |
| `scatter_copy` | DRAM→HBM 的 CKV/KPE 搬移 | 共用 |
| `sparse_tail_attention` | 单 query 的 top-2048 + dense tail Attention | 否 |
| `sparse_tail_attention_mtp` | 四个验证位置各自的 top-2048 + causal dense tail Attention | 是 |
| `fused_copy_sfa` | 融合 `scatter_copy + sparse_tail_attention`，支持 bs>24 | 否 |
| `fused_copy_sfa_mtp` | MTP3 union miss 搬移与四行 causal sparse Attention | 是 |

七个算子都只有一个公开入口：调用方预先创建 mutable/output buffer，算子
原地写入并返回 `None`。不存在 allocating 入口、`_out` 后缀或 alias 输出。
全部 `torch.ops.nanovllm_dsa.*` 注册都有 Meta/Fake 可见实现，可用于 eager
和 `FULL_DECODE_ONLY` capture/replay。

## 关键接口与边界

```python
fused_li_manage(
    query, index_weights, index_key_cache, index_block_table,
    num_candidate_tokens, num_cache_tokens, req_pool_entries,
    cache_slots_pool, topk_src_ids, topk_dst_slots, miss_counts,
) -> None

fused_li_manage_mtp(
    query, index_weights, index_key_cache, index_block_table,
    num_candidate_tokens, num_cache_tokens, req_pool_entries,
    cache_slots_pool, topk_src_ids, topk_dst_slots,
    miss_src_ids, miss_dst_slots, miss_counts,
) -> None

scatter_copy(
    src_ids, dst_slots, copy_counts, hbm_block_table, dram_block_table,
    hbm_k_rope, hbm_kv_cache, dram_k_rope, dram_kv_cache,
) -> None

sparse_tail_attention(
    query_rope, query, actual_seq_lengths_query, actual_seq_lengths_kv,
    num_cache_tokens, topk_dst_slots, hbm_block_table,
    hbm_k_rope, hbm_kv_cache, scale_value, attention_out,
) -> None

sparse_tail_attention_mtp(
    query_rope, query, actual_seq_lengths_query, actual_seq_lengths_kv,
    num_cache_tokens, topk_dst_slots, hbm_block_table,
    hbm_k_rope, hbm_kv_cache, scale_value, attention_out,
) -> None

fused_copy_sfa(
    query_rope, query, actual_seq_lengths_query, actual_seq_lengths_kv,
    num_cache_tokens, topk_dst_slots, topk_src_ids, miss_counts,
    hbm_block_table, dram_block_table, hbm_k_rope, hbm_kv_cache,
    dram_k_rope, dram_kv_cache, scale_value, attention_out,
) -> None

fused_copy_sfa_mtp(
    query_rope, query, actual_seq_lengths_query, actual_seq_lengths_kv,
    num_cache_tokens, topk_dst_slots, topk_src_ids,
    miss_src_ids, miss_dst_slots, miss_counts,
    hbm_block_table, dram_block_table, hbm_k_rope, hbm_kv_cache,
    dram_k_rope, dram_kv_cache, scale_value, attention_out,
) -> None
```

关键边界：

- 单 query LIM 的 `topk_src_ids/topk_dst_slots` 为 `[B,1,2048]`；MTP3
  LIM 的对应输出为 `[B*4,1,2048]`，union miss 输出为 `[B,8192]`。
- 单 query 的 miss 位于 top-k row 前缀；MTP3 的 `topk_src_ids` 在 HBM
  hit 位置写 `-1`，唯一搬移集合由 `miss_src_ids/miss_dst_slots` 给出。
- 单 query LIM 保留 21-bit source index 能力；MTP3 LIM 暂只支持 18-bit。
- `scatter_copy` 同时支持非 MTP 的 2048 capacity 和 MTP3 的 8192
  capacity，不感知 query_len。
- `sparse_tail_attention_mtp` 保证四个验证位置分别使用各自 top-2048
  和 causal dense tail。
- `fused_copy_sfa` 支持 bs>24；`fused_copy_sfa_mtp` 当前的 Attention
  数值偏差留待性能优化阶段修复。

其余五个本仓算子 `moe_gating_top_k`、
`matmul_allreduce_add_rmsnorm`、`batch_matmul_transpose`、
`dsa_indexer_query_rope_inplace`、`mla_preprocess` 也统一从
`torch.ops.nanovllm_dsa` 暴露；它们的接口、行为和内核没有改动。

## 编译环境

```bash
unset NANOVLLM_CUST_OPAPI_LIB
unset ASCEND_CUSTOM_OPP_PATH

export ASCEND_HOME_PATH=/usr/local/Ascend/cann-8.5.1
export CANN_INSTALL_PATH=/usr/local/Ascend/cann-8.5.1
export PYTHONPATH=$PWD:$PYTHONPATH
export PYTHONUNBUFFERED=1
export NANOVLLM_CANN_BUILD_JOBS=64
export NANOVLLM_EXT_BUILD_JOBS=1
export SOC_VERSION=ascend910_9391

bash scripts/build_nanovllm_ops.sh
```

## 单卡 UT 公共环境

```bash
unset NANOVLLM_OFFLOAD_MODE
unset NANOVLLM_PROFILE_DECODE_OUTPUT
unset NANOVLLM_CUST_OPAPI_LIB
unset ASCEND_CUSTOM_OPP_PATH
unset NANOVLLM_NUM_SPECULATIVE_TOKENS

export ASCEND_HOME_PATH=/usr/local/Ascend/cann-8.5.1
export CANN_INSTALL_PATH=/usr/local/Ascend/cann-8.5.1
export PYTHONUNBUFFERED=1
export PYTHONPATH=$PWD:$PYTHONPATH
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export ASCEND_LAUNCH_BLOCKING=0
export ASCEND_RT_VISIBLE_DEVICES=4
```

## 非 MTP UT

```bash
# 21-bit 边界、request pool、完整 top-2048 与 32/64 heads
python3 ut_ops/test_fused_li_manage.py \
  --device npu:0 --heads 32,64 \
  --seq-lens 262144,262272,1048576,2097151 \
  --batch-size 1 --cache-tokens 6144 --miss-count 300 \
  --warmup 1 --iters 3 --seed 7

# fused_li_manage → scatter_copy 语义和 graph replay
python3 ut_ops/test_fused_li_manage_scatter.py \
  --device npu:0 --heads 32,64 --seed 7 \
  --warmup 2 --iters 10 --graph-replays 3

# fused_li_manage 性能
python3 ut_ops/test_fused_li_manage_perf.py \
  --device npu:0 --heads 32 --batch-sizes 24 \
  --seq-lens 20992 --cache-tokens 6144 \
  --miss-ranges 0:0,0:200,0:300,0:2048 \
  --warmup 10 --iters 100 --seed 7

# SCATTER 的 2048/8192 capacity、graph 与带宽
python3 ut_ops/test_scatter_copy.py \
  --device npu:0 --batch-size 24 --source-len 20992 \
  --cache-tokens 8192 --copy-cap 8192 \
  --miss-counts 0,256,1024,2048,4096,6144,8192 \
  --warmup 10 --iters 100 --seed 7

# 独立 sparse Attention
python3 ut_ops/test_sparse_tail_attention.py \
  --device npu:0 --heads 4 --batch-size 24 \
  --cache-tokens 6144 --tail-tokens 64 \
  --warmup 10 --iters 100 --seed 7

# 融合算子语义与性能
python3 ut_ops/test_fused_copy_sfa.py \
  --device npu:0 --mode all --batch-size 24 --heads 4 \
  --source-len 20992 --cache-tokens 6144 --tail-tokens 64 \
  --miss-min 0 --miss-max 2048 --warmup 10 --iters 100 --seed 7

# 24-core 边界与 bs>24 均衡分核
for bs in 24 25 48 64; do
  python3 ut_ops/test_fused_copy_sfa.py \
    --device npu:0 --mode check --batch-size "$bs" --heads 4 \
    --source-len 20992 --cache-tokens 6144 --tail-tokens 64 \
    --miss-min 0 --miss-max 300 --graph-replays 3 --seed 7
done
```

## MTP3 UT

```bash
# MTP3 LIM 语义、乱序 request pool、动态图 metadata 和性能
python3 ut_ops/test_fused_li_manage_mtp.py \
  --device npu:0 --batch-size 24 --source-len 20992 \
  --cache-tokens 8192 --graph-replays 3 \
  --warmup 10 --iters 100 --seed 7

# 四行因果 sparse Attention、caller-owned output 和 graph replay
python3 ut_ops/test_sparse_tail_attention_mtp.py \
  --device npu:0 --heads 2 --batch-size 24 \
  --cache-tokens 8192 --tail-tokens 64 --graph-replays 3 \
  --warmup 10 --iters 100 --seed 7

# 完整 MTP 卸载链的数据依赖
python3 ut_ops/test_mtp_offload_chain.py \
  --device npu:0 --batch-size 4 --heads 2 \
  --source-len 20992 --cache-tokens 8192 --tail-tokens 64 \
  --graph-replays 3 --seed 7 --allow-fused-attention-diff

# fused_copy_sfa_mtp：split 对照、caller-owned output、graph replay 与时延打印
python3 ut_ops/test_fused_copy_sfa_mtp.py \
  --device npu:0 --batch-size 4 --heads 2 \
  --source-len 20992 --cache-tokens 8192 --tail-tokens 64 \
  --perf-miss-count 300 --graph-replays 3 --warmup 10 --iters 100 \
  --seed 7 --allow-fused-attention-diff

# Target qlen=4 MLA 与逐 token 参考结果及 graph replay
python3 ut_ops/test_glm_mtp_target_verify.py \
  --device npu:0 --batch-size 4 --heads 2 --prefix-len 4096 \
  --query-lens 2,4 --graph-replays 3 --seed 7 \
  --atol 0.04 --rtol 0.02 --min-cosine 0.999
```
