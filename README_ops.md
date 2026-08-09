# DSA decode 卸载算子

本仓库内置以下七个算子，不依赖外部算子仓库：

| 算子 | 用途 | 被谁使用 |
| --- | --- | :---: |
| `fused_li_manage` | query_len=1 的 LightningIndexer、命中/淘汰与 request-pool 更新 | MTP0 |
| `fused_li_manage_mtp` | 四路 query 的 top-2048 并集、命中/淘汰与 request-pool 更新 | MTP3 |
| `scatter_copy` | DRAM→HBM 的 CKV/KPE 搬移 | MTP0 & MTP3 |
| `sparse_tail_attention` | 单 query 的 top-2048 + dense tail Attention | MTP0 |
| `sparse_tail_attention_mtp` | 四个验证位置各自的 top-2048 + causal dense tail Attention | MTP3 |
| `fused_copy_sfa` | 融合 `scatter_copy + sparse_tail_attention`，支持 bs>24 | MTP0 |
| `fused_copy_sfa_mtp` | MTP3 union miss 搬移与四行 causal sparse Attention | MTP3 |

七个算子都只有一个公开入口：调用方预先创建 mutable/output buffer，算子原地写入并返回 `None`。不存在 allocating 入口、`_out` 后缀或 alias 输出。
全部 `torch.ops.nanovllm_dsa.*` 注册都有 Meta/Fake 可见实现，可用于 eager 和 `FULL_DECODE_ONLY` capture/replay。

## 关键接口与边界
```css
fused_li_manage (
    query,                    # bf16/fp16[B,N,128]                         , 只读, index query，N=32或64
    index_weights,            # bf16/fp16[B,N]                             , 只读, 各index head的聚合权重
    index_key_cache,          # bf16/fp16[INDEX_BLOCKS,128,1,128]          , 只读, index key cache
    index_block_table,        # int32[B,INDEX_MAX_BLOCKS]                  , 只读, index cache block table
    num_candidate_tokens,     # int32[B]                                   , 只读, 每个请求参与稀疏选择的prefill满块token数
    num_cache_tokens,         # int32[B]                                   , 只读, 每个请求的HBM缓存token预算C
    req_pool_entries,         # int32[B]                                   , 只读, 每个请求对应的request-pool行号
    cache_slots_pool,         # int32[POOL_SIZE,SOURCE_CAPACITY]           , 读写, source token到HBM逻辑slot的持久映射
    topk_src_ids,             # int32[B,1,2048]                            , 只写, top2048 source token ID，前miss_counts个为需要搬移的token
    topk_dst_slots,           # int32[B,1,2048]                            , 只写, top2048 token对应的HBM逻辑slot，前miss_counts个也是搬移dest
    miss_counts               # int32[B]                                   , 只写, 每个请求需要从DRAM搬入HBM的token数
) -> None
```

```css
fused_li_manage_mtp (
    query,                    # bf16/fp16[B*4,32,128]                      , 只读, MTP3的4路index query
    index_weights,            # bf16/fp16[B*4,32]                          , 只读, 4路query各index head的聚合权重
    index_key_cache,          # bf16/fp16[INDEX_BLOCKS,128,1,128]          , 只读, index key cache
    index_block_table,        # int32[B,INDEX_MAX_BLOCKS]                  , 只读, index cache block table
    num_candidate_tokens,     # int32[B]                                   , 只读, 每个请求参与稀疏选择的prefill满块token数
    num_cache_tokens,         # int32[B]                                   , 只读, 每个请求的HBM缓存token预算C
    req_pool_entries,         # int32[B]                                   , 只读, 每个请求对应的request-pool行号
    cache_slots_pool,         # int32[POOL_SIZE,SOURCE_CAPACITY]           , 读写, source token到HBM逻辑slot的持久映射
    topk_src_ids,             # int32[B*4,1,2048]                          , 只写, 4路top2048的source token ID，HBM hit位置为-1
    topk_dst_slots,           # int32[B*4,1,2048]                          , 只写, 4路top2048 token对应的HBM逻辑slot
    miss_src_ids,             # int32[B,8192]                              , 只写, 4路top2048并集中的unique miss source ID，仅前miss_counts个有效
    miss_dst_slots,           # int32[B,8192]                              , 只写, unique miss对应的HBM逻辑slot，仅前miss_counts个有效
    miss_counts               # int32[B]                                   , 只写, 每个请求unique union miss的token数
) -> None
```

```css
scatter_copy (
    src_ids,                  # int32[B,COPY_CAPACITY]                     , 只读, 需要搬移的token在source DRAM中的逻辑位置
    dst_slots,                # int32[B,COPY_CAPACITY]                     , 只读, 搬移目标在HBM中的逻辑slot
    copy_counts,              # int32[B]                                   , 只读, 每行src_ids/dst_slots中有效元素的数量
    hbm_block_table,          # int32[B,HBM_MAX_BLOCKS]                    , 只读, HBM block table
    dram_block_table,         # int32[B,DRAM_MAX_BLOCKS]                   , 只读, DRAM block table
    hbm_k_rope,               # bf16/fp16[HBM_BLOCKS,128,64]               , 读写, HBM KV cache (rope)，搬移dest
    hbm_kv_cache,             # bf16/fp16[HBM_BLOCKS,128,512]              , 读写, HBM KV cache (nope)，搬移dest
    dram_k_rope,              # bf16/fp16[DRAM_BLOCKS,128,64]              , 只读, DRAM KV cache (rope)，搬移source
    dram_kv_cache             # bf16/fp16[DRAM_BLOCKS,128,512]             , 只读, DRAM KV cache (nope)，搬移source
) -> None
```

```css
sparse_tail_attention (
    query_rope,               # bf16/fp16[B,N,64]                          , 只读, SparseAttn的query (rope)
    query,                    # bf16/fp16[B,N,512]                         , 只读, SparseAttn的query (nope)
    actual_seq_lengths_query, # int32[B]                                   , 只读, TND累计query长度
    actual_seq_lengths_kv,    # int32[B]                                   , 只读, 每个请求参与Attention的C+tail长度
    num_cache_tokens,         # int32[B]                                   , 只读, 每个请求的HBM缓存token预算C
    topk_dst_slots,           # int32[B,1,2048]                            , 只读, top2048 token在HBM中的逻辑slot
    hbm_block_table,          # int32[B,HBM_MAX_BLOCKS]                    , 只读, HBM block table
    hbm_k_rope,               # bf16/fp16[HBM_BLOCKS,128,1,64]             , 只读, HBM KV cache (rope)
    hbm_kv_cache,             # bf16/fp16[HBM_BLOCKS,128,1,512]            , 只读, HBM KV cache (nope)
    scale_value,              # float                                      , 只读, attention scale
    attention_out             # bf16/fp16[B,N,512]                         , 只写, SparseAttn结果
) -> None
```

```css
sparse_tail_attention_mtp (
    query_rope,               # bf16/fp16[B*4,N,64]                        , 只读, MTP3四路SparseAttn query (rope)
    query,                    # bf16/fp16[B*4,N,512]                       , 只读, MTP3四路SparseAttn query (nope)
    actual_seq_lengths_query, # int32[B]                                   , 只读, TND累计query长度，值为[4,8,...,B*4]
    actual_seq_lengths_kv,    # int32[B]                                   , 只读, 每个请求第4路query对应的最终KV长度
    num_cache_tokens,         # int32[B]                                   , 只读, 每个请求的HBM缓存token预算C
    topk_dst_slots,           # int32[B*4,1,2048]                          , 只读, 4路top2048 token在HBM中的逻辑slot
    hbm_block_table,          # int32[B,HBM_MAX_BLOCKS]                    , 只读, HBM block table
    hbm_k_rope,               # bf16/fp16[HBM_BLOCKS,128,1,64]             , 只读, HBM KV cache (rope)
    hbm_kv_cache,             # bf16/fp16[HBM_BLOCKS,128,1,512]            , 只读, HBM KV cache (nope)
    scale_value,              # float                                      , 只读, attention scale
    attention_out             # bf16/fp16[B*4,N,512]                       , 只写, 四路causal SparseAttn结果
) -> None
```

```css
fused_copy_sfa (
    query_rope,               # bf16/fp16[B,N,64]                          , 只读, SparseAttn的query (rope)
    query,                    # bf16/fp16[B,N,512]                         , 只读, SparseAttn的query (nope)
    actual_seq_lengths_query, # int32[B]                                   , 只读, TND累计query长度
    actual_seq_lengths_kv,    # int32[B]                                   , 只读, 每个请求参与Attention的C+tail长度
    num_cache_tokens,         # int32[B]                                   , 只读, 每个请求的HBM缓存token预算C
    topk_dst_slots,           # int32[B,1,2048]                            , 只读, top2048 token在HBM中的逻辑slot，前miss_counts个也是搬移dest
    topk_src_ids,             # int32[B,2048]                              , 只读, top2048 source token ID，仅前miss_counts个需要从DRAM搬移
    miss_counts,              # int32[B]                                   , 只读, 每个请求需要从DRAM搬入HBM的token数
    hbm_block_table,          # int32[B,HBM_MAX_BLOCKS]                    , 只读, HBM block table
    dram_block_table,         # int32[B,DRAM_MAX_BLOCKS]                   , 只读, DRAM block table
    hbm_k_rope,               # bf16/fp16[HBM_BLOCKS,128,1,64]             , 读写, HBM KV cache (rope)，既是Attention输入也是搬移dest
    hbm_kv_cache,             # bf16/fp16[HBM_BLOCKS,128,1,512]            , 读写, HBM KV cache (nope)，既是Attention输入也是搬移dest
    dram_k_rope,              # bf16/fp16[DRAM_BLOCKS,128,64]              , 只读, DRAM KV cache (rope)，搬移source
    dram_kv_cache,            # bf16/fp16[DRAM_BLOCKS,128,512]             , 只读, DRAM KV cache (nope)，搬移source
    scale_value,              # float                                      , 只读, attention scale
    attention_out             # bf16/fp16[B,N,512]                         , 只写, SparseAttn结果
) -> None
```

```css
fused_copy_sfa_mtp (
    query_rope,               # bf16/fp16[B*4,N,64]                        , 只读, MTP3四路SparseAttn query (rope)
    query,                    # bf16/fp16[B*4,N,512]                       , 只读, MTP3四路SparseAttn query (nope)
    actual_seq_lengths_query, # int32[B]                                   , 只读, TND累计query长度，值为[4,8,...,B*4]
    actual_seq_lengths_kv,    # int32[B]                                   , 只读, 每个请求第4路query对应的最终KV长度
    num_cache_tokens,         # int32[B]                                   , 只读, 每个请求的HBM缓存token预算C
    topk_dst_slots,           # int32[B*4,1,2048]                          , 只读, 4路top2048 token在HBM中的逻辑slot
    topk_src_ids,             # int32[B*4,1,2048]                          , 只读, 4路top2048 source token ID，HBM hit位置为-1
    miss_src_ids,             # int32[B,8192]                              , 只读, 4路top2048并集中的unique miss source ID，仅前miss_counts个有效
    miss_dst_slots,           # int32[B,8192]                              , 只读, unique miss对应的HBM逻辑slot，仅前miss_counts个有效
    miss_counts,              # int32[B]                                   , 只读, 每个请求unique union miss的token数
    hbm_block_table,          # int32[B,HBM_MAX_BLOCKS]                    , 只读, HBM block table
    dram_block_table,         # int32[B,DRAM_MAX_BLOCKS]                   , 只读, DRAM block table
    hbm_k_rope,               # bf16/fp16[HBM_BLOCKS,128,1,64]             , 读写, HBM KV cache (rope)，既是Attention输入也是搬移dest
    hbm_kv_cache,             # bf16/fp16[HBM_BLOCKS,128,1,512]            , 读写, HBM KV cache (nope)，既是Attention输入也是搬移dest
    dram_k_rope,              # bf16/fp16[DRAM_BLOCKS,128,64]              , 只读, DRAM KV cache (rope)，搬移source
    dram_kv_cache,            # bf16/fp16[DRAM_BLOCKS,128,512]             , 只读, DRAM KV cache (nope)，搬移source
    scale_value,              # float                                      , 只读, attention scale
    attention_out             # bf16/fp16[B*4,N,512]                       , 只写, 四路causal SparseAttn结果
) -> None
```


关键边界：

- 单 query LIM 的 `topk_src_ids/topk_dst_slots` 为 `[B,1,2048]`；MTP3 LIM 的对应输出为 `[B*4,1,2048]`，union miss 输出为 `[B,8192]`。
- 单 query 的 miss 位于 top-k row 前缀；MTP3 的 `topk_src_ids` 在 HBM hit 位置写 `-1`，唯一搬移集合由 `miss_src_ids/miss_dst_slots` 给出。
- 单 query LIM 保留 21-bit source index 能力；MTP3 LIM 暂只支持 18-bit。
- `scatter_copy` 同时支持非 MTP 的 2048 capacity 和 MTP3 的 8192 capacity，不感知 query_len。
- `sparse_tail_attention_mtp` 保证四个验证位置分别使用各自 top-2048 和 causal dense tail。
- `fused_copy_sfa` 支持 bs>24；`fused_copy_sfa_mtp` 当前的 Attention 数值偏差留待性能优化阶段修复。

其余五个本仓算子 `moe_gating_top_k`、`matmul_allreduce_add_rmsnorm`、`batch_matmul_transpose`、`dsa_indexer_query_rope_inplace`、mla_preprocess` 也统一从
`torch.ops.nanovllm_dsa` 暴露；它们的接口、行为和内核没有改动。

## 编译

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
