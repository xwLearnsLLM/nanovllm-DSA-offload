# Ascend 950 nano-vLLM BF16 offload 算子

本仓库是独立的 Ascend 950 / CANN 9.1 decode KV-cache 卸载算子工程。BF16/FP16 路径提供拆分链路和 MTE-pipeline 融合链路。

所有构建、运行和测试源码均已保存在本仓库；`ops_dsa_offload_a5` 只作为来源记录，不是依赖，可以归档。

## 来源

- LIDU：普通路径基于 `ops_li_update_a5@0362e7e` 的生产 P5 内核，增加 request pool、逐请求 C、动态 pool 行跨度、mutable alias 和 caller-owned out；MTP 路径迁移同仓 `lightning_indexer_decode_update_mtp_a5`，仅重命名并隔离其专用 Arch35 payload，保持原始 packed-query 行为。
- SCATTER：基于 `ops_dsa_offload_a5@01f2065` 已验证的 Ascend 950 swapped-memory DRAM→HBM 路径。
- SFA：基于 `vllm-ascend-v0.23.0-custom@6af99b372` 的官方 Arch35 SFA 叠加 sparse+tail 语义。官方接口说明见 [aclnnSparseFlashAttention](https://github.com/vllm-project/vllm-ascend/blob/main/csrc/attention/sparse_flash_attention/docs/aclnnSparseFlashAttention.md)。
- 融合算子：基于 `ops_dsa_offload_a5@d58629f` 的 MTE-pipeline 版本；公共 SFA 基础代码复用本仓库副本，避免污染拆分链路基线。

## 算子一览表

| 公开入口 | 实际执行 | 作用 | 定位 |
| --- | --- | --- | --- |
| `fused_li_manage` / `fused_li_manage_out` | `A5FusedLiManage` | BF16/FP16 LightningIndexer top-2048 与 request-pool 更新；`_out` 写入 caller-owned buffer | eager / 稳定图主链 |
| `fused_li_manage_mtp` | `A5FusedLiManageMtp` | BF16/FP16 packed MTP query 的逐 query top-2048、请求级并集与一次性 8192-token 缓存更新 | MTP decode |
| `kvcache_scatter_copy` | `A5KvcacheScatterCopy` | 按 miss-prefix 将分离的 BF16/FP16 CKV、KPE 从 swapped-memory DRAM 搬到 HBM | 稳定图主链 |
| `sparse_tail_attention` | `A5SparseTailAttention` | 对 top-2048 sparse slots 与 tail 执行 BF16/FP16 MLA | 稳定图主链 |
| `fused_copy_sparse_tail_attention` | `A5FusedCopySparseTailAttention` | BF16 MTE pipeline，将 DRAM→HBM 搬运与 sparse+tail Attention 融合，可配置每步预取行数 | 稳定图主链 |

## 接口

Torch 算子 namespace 为 `nanovllm_dsa`；Python 包名为 `nanovllm_dsa_a5`，C++ wrapper 的内部 namespace 为 `nanovllm_dsa_a5_impl`。

共同语义：`B` 是 decode batch，`C=cache_tokens[b]` 是请求固定的 HBM token 预算，`candidate_lens[b]` 是参与稀疏选择的 prefill 满块 token 数。`cache_slots_pool[req_pool_entries[b], token_id]` 保存 source token 到 HBM slot 的映射，`-1` 表示未缓存。LIDU 输出中前 `miss_counts[b]` 项为 miss，SCATTER 只搬运这段；全部 2048 个 `destination_slots` 都供 Attention 使用。稳定图使用 caller-owned `_out` 接口。

可选链路：`fused_li_manage_out → kvcache_scatter_copy → sparse_tail_attention` 或 `fused_li_manage_out → fused_copy_sparse_tail_attention`。融合入口只接受 BF16。

```python
torch.ops.nanovllm_dsa.fused_li_manage(
    query,                  # bf16/fp16[B, INDEX_HEADS, 128]，INDEX_HEADS=32|64
    key,                    # bf16/fp16[INDEX_BLOCKS, 128, 1, 128]
    weights,                # bf16/fp16[B, INDEX_HEADS]
    req_pool_entries,       # int32[B]，batch row -> request-pool row
    cache_slots_pool,       # int32[POOL_SIZE, SOURCE_CAPACITY]，in/out
    cache_tokens,           # int32[B]，逐请求 C；C=0 时该请求 no-op
    candidate_lens,         # int32[B]
    block_table,            # int32[B, SOURCE_CAPACITY/128]
) -> (
    source_ids,             # int32[B,1,2048]，miss-prefix + hit-suffix
    destination_slots,      # int32[B,1,2048]，与 source_ids 对齐
    miss_counts,            # int32[B]
    cache_slots_alias,      # cache_slots_pool alias
)

torch.ops.nanovllm_dsa.fused_li_manage_out(
    query, key, weights, req_pool_entries, cache_slots_pool,
    cache_tokens, candidate_lens, block_table,
    source_ids,             # caller-owned int32[B,1,2048]，in/out
    destination_slots,      # caller-owned int32[B,1,2048]，in/out
    miss_counts,            # caller-owned int32[B]，in/out
) -> (source_ids_alias, destination_slots_alias, miss_counts_alias, cache_slots_alias)

torch.ops.nanovllm_dsa.fused_li_manage_mtp(
    query,                  # bf16/fp16[T,INDEX_HEADS,128]，INDEX_HEADS=32|64
    key,                    # bf16/fp16[INDEX_BLOCKS,BLOCK,1,128]
    weights,                # bf16/fp16[T,INDEX_HEADS]
    cache_slots,            # int32[B,262144]，in/out；每个请求固定缓存 8192 tokens
    actual_seq_lengths_query, # cumulative int32[B]；每个请求包含 1..4 个 packed query
    actual_seq_lengths_key, # int32[B]
    block_table,            # int32[B,MAX_BLOCKS]
) -> (
    topk_index,             # int32[T,1,2048]
    topk_slots,             # int32[T,1,2048]
    miss_index,             # int32[B,8192]；请求内所有 query 的 top-k 并集 miss
    miss_slots,             # int32[B,8192]
    miss_count,             # int32[B]
)

torch.ops.nanovllm_dsa.kvcache_scatter_copy(
    hbm_kpe,                # bf16/fp16[HBM_BLOCKS,128,64]，in/out
    hbm_ckv,                # bf16/fp16[HBM_BLOCKS,128,512]，in/out
    dram_kpe,               # swapped-memory bf16/fp16[DRAM_BLOCKS,128,64]
    dram_ckv,               # swapped-memory bf16/fp16[DRAM_BLOCKS,128,512]
    hbm_block_table,        # int32[B,HBM_MAX_BLOCKS]
    dram_block_table,       # int32[B,DRAM_MAX_BLOCKS]
    source_token_ids,       # int32[B,COPY_CAP]，取 LIDU miss-prefix
    destination_slots,      # int32[B,COPY_CAP]
    copy_counts,            # int32[B]，即 miss_counts
) -> (hbm_kpe_alias, hbm_ckv_alias)

torch.ops.nanovllm_dsa.sparse_tail_attention(
    query,                  # bf16/fp16[B,Q_HEAD,512]，1<=Q_HEAD<=64
    key,                    # bf16/fp16[HBM_BLOCKS,128,1,512]
    value,                  # 必须与 key alias
    sparse_slots,           # int32[B,1,2048]，即 LIDU destination_slots
    cache_tokens,           # int32[B]
    block_table,            # int32[B,HBM_MAX_BLOCKS]
    actual_seq_lengths_query, # cumulative int32[B]，decode 为 [1,...,B]
    actual_seq_lengths_kv,  # int32[B]，HBM resident 长度；C>0 时为 C+tail
    query_rope,             # bf16/fp16[B,Q_HEAD,64]
    key_rope,               # bf16/fp16[HBM_BLOCKS,128,1,64]
    scale_value,            # float
) -> attention_out          # bf16/fp16[B,Q_HEAD,512]

torch.ops.nanovllm_dsa.fused_copy_sparse_tail_attention(
    query,                  # bf16[B,Q_HEAD,512]，1<=Q_HEAD<=64
    hbm_ckv,                # bf16[HBM_BLOCKS,128,1,512]，in/out
    sparse_slots,           # int32[B,1,2048]，LIDU destination_slots
    cache_tokens,           # int32[B]
    hbm_block_table,        # int32[B,HBM_MAX_BLOCKS]
    actual_seq_lengths_query, actual_seq_lengths_kv,
    query_rope,             # bf16[B,Q_HEAD,64]
    hbm_kpe,                # bf16[HBM_BLOCKS,128,1,64]，in/out
    dram_kpe, dram_ckv,     # swapped-memory bf16[DRAM_BLOCKS,128,64/512]
    dram_block_table,       # int32[B,DRAM_MAX_BLOCKS]
    source_token_ids,       # int32[B,2048]，LIDU source_ids
    copy_counts,            # int32[B]，LIDU miss_counts
    scale_value,            # float
    prefetch_rows_per_step=5, # int，0..16
) -> (attention_out, hbm_kpe_alias, hbm_ckv_alias)
```

## 编译

从仓库根目录执行：

```bash
unset ASCEND_CUSTOM_OPP_PATH
unset NANOVLLM_A5_INSTALL_OPP_PATH
unset NANOVLLM_CUST_OPAPI_LIB
unset A5_SOC_VERSION
unset SOC_VERSION
unset CANN_INSTALL_PATH
export ASCEND_HOME_PATH=/usr/local/Ascend/ascend-toolkit/latest
export CANN_INSTALL_PATH=$ASCEND_HOME_PATH
source "$ASCEND_HOME_PATH/set_env.sh"
export ASCEND_RT_VISIBLE_DEVICES=0
export ASCEND_LAUNCH_BLOCKING=0
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1
export PYTHONPATH=$PWD/torch_extension:$PYTHONPATH
export SOC_VERSION=ascend950
export NANOVLLM_A5_OPS_PYTHON=python3
export NANOVLLM_A5_OPS_BUILD_JOBS=64
```

```bash
bash build.sh
```

## 算子测试

BF16 测试固定执行正确性检查；`--iters=0` 仅检查正确性，`--iters>0` 随后继续计时。

```bash
python3 tests/test_fused_li_manage.py --device npu:0 --heads 32,64 --batch-sizes 24 --source-lens 20096 --cache-tokens 6144 --miss-ranges 0:300 --iters 0 --seed 7
```

该 LIDU 命令还会强制覆盖不同 candidate length、`C=0/2048/3072/6144/8192/12288/16256`、零 miss、2048 miss、乱序 pool entries、hit slot 保持、重复更新映射、inactive pool guard、caller-owned out 和 `262144` 的 18-bit token-index 边界。

```bash
python3 tests/test_fused_li_manage_mtp.py --device npu:0 --bs 24 --min-seqlen 32768 --max-seqlen 65536 --q-heads 64 --queries-per-request 0 --min-miss-count 0 --max-miss-count 300 --seed 7
```

MTP 门禁逐 query 对照原生 LightningIndexer，并检查因果可见前缀、请求级 top-k 并集去重、并集 miss、一次性缓存更新、hit slot 保持、victim 淘汰和 8192-token 缓存基数。

```bash
for count in 0 1 100 300 2048; do python3 tests/test_kvcache_scatter_copy.py --device npu:0 --batch-size 24 --source-len 65536 --hbm-slots 4096 --copy-cap 2048 --copy-min "$count" --copy-max "$count" --warmup 3 --iters 10 --seed 7; done
```

SCATTER 使用 `empty_with_swapped_memory` 创建真实 DRAM tensor；每次正确性调用前 poison HBM 目标，并验证 CKV、KPE、随机 block tables 和未触碰 guard。

```bash
python3 tests/test_sparse_tail_attention.py --device npu:0 --heads 8 --batch-sizes 24 --source-lens 20096 --cache-tokens 6144 --tail-tokens 64 --iters 0 --seed 7
```

SFA check 固定覆盖 dense `C=0`、2048-token sparse-only、常用 `C=6144/tail=64` 和最大档 `C=12288/tail=257`，再检查命令指定的配置；所有结果均与独立 CPU FP32 golden 比较。

```bash
python3 tests/test_fused_copy_sparse_tail_attention.py --device npu:0 --batch-size 24 --heads 8 --source-len 65536 --cache-tokens 8192 --tail-tokens 64 --miss-min 0 --miss-max 300 --warmup 10 --iters 100 --seed 7
```

该门禁将拆分链路与转正后的融合算子分别从真实 swapped-memory DRAM 搬运；每条链路调用前独立 poison HBM 目标，并校验精确 CKV/KPE 写回、guard、CPU FP32 Attention golden 和时延。融合算子默认使用 `prefetch_rows_per_step=5`。

```bash
python3 tests/test_bf16_graph.py --device npu:0 --case pure-long --attention-path split --replays 4 --seed 7
```

```bash
python3 tests/test_bf16_graph.py --device npu:0 --case mixed --attention-path fused --prefetch-rows-per-step 5 --replays 4 --seed 7
```

同一图门禁覆盖拆分与融合两条 BF16 链路；capture 为零 miss，replay 交替产生非零 miss，并校验输出地址、pool 更新、真实 DRAM→HBM 搬运和 Attention golden。

## 性能矩阵

`q_head=8` 的三算子总时延需与相同输入下的来源版本对照，目标是不超过参考时延的 `1.10x`。

LIDU：

```bash
python3 tests/test_fused_li_manage.py --device npu:0 --heads 32 --batch-sizes 1,4,8,12,16,24,32 --source-lens 12288,20096,65536,131072 --cache-tokens 6144 --miss-ranges 0:0,0:300,300:300 --warmup 10 --iters 100 --seed 7
```

与 `ops_li_update_a5/README.md` 历史表格完全同口径的确认命令如下；单测采用每轮随机顺序的成对 NPU Event 计时，并打印 `delta_vs_reference_us`：

```bash
python3 tests/test_fused_li_manage.py --device npu:0 --heads 32,64 --batch-sizes 1,8,16,24,32,48,64 --source-lens 65536,131072 --cache-tokens 12288 --miss-ranges 100:200 --pool-extra 0 --warmup 10 --iters 1000 --seed 1234
```

SCATTER：

```bash
for bs in 1 4 8 12 16 24 32; do for len in 12288 20096 65536 131072; do python3 tests/test_kvcache_scatter_copy.py --device npu:0 --batch-size "$bs" --source-len "$len" --hbm-slots 8192 --copy-cap 2048 --copy-min 0 --copy-max 0 --warmup 10 --iters 100 --seed 7; python3 tests/test_kvcache_scatter_copy.py --device npu:0 --batch-size "$bs" --source-len "$len" --hbm-slots 8192 --copy-cap 2048 --copy-min 0 --copy-max 300 --warmup 10 --iters 100 --seed 7; python3 tests/test_kvcache_scatter_copy.py --device npu:0 --batch-size "$bs" --source-len "$len" --hbm-slots 8192 --copy-cap 2048 --copy-min 300 --copy-max 300 --warmup 10 --iters 100 --seed 7; done; done
```

SFA：

```bash
python3 tests/test_sparse_tail_attention.py --device npu:0 --heads 8 --batch-sizes 1,4,8,12,16,24,32 --source-lens 12288,20096,65536,131072 --cache-tokens 6144 --tail-tokens 64 --warmup 10 --iters 100 --seed 7
```

融合链路（默认预取深度 5）：

```bash
python3 tests/test_fused_copy_sparse_tail_attention.py --device npu:0 --batch-size 24 --heads 8 --source-len 65536 --cache-tokens 8192 --tail-tokens 64 --miss-min 0 --miss-max 300 --warmup 10 --iters 100 --seed 7
```
