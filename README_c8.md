# Ascend 950 nano-vLLM C8 offload 算子

本仓库是独立的 Ascend 950 / CANN 9.1 decode KV-cache 卸载算子工程。C8 路径使用官方 A5 C8 LightningIndexer、request-pool update、packed-C8 SCATTER 和原生 C8 QSFA。

## 来源

- Indexer：使用 A5 官方 `torch_npu.npu_quant_lightning_indexer`，采用 FP8 E4M3 query/key、BF16 weights、FP32 query/key scale、`TND/PA_BSND`、`sparse_count=2048` 和 `sparse_mode=3`；C8 LIDU 接续完成 request-pool hit/miss、淘汰和更新。
- Attention：packed KV ABI 与 A5 原生 `npu_kv_quant_sparse_flash_attention` 对齐。ModelSlim 的 GLM-5.1 W4A4C8 量化说明见 [GLM-5 量化 README](https://gitcode.com/Ascend/msmodelslim/blob/26.1.0/example/GLM-5/README.md)。

## 算子一览表

| 公开入口 | 实际执行 | 作用 | 定位 |
| --- | --- | --- | --- |
| `fused_li_manage_c8` / `fused_li_manage_c8_out` | 官方 `npu_quant_lightning_indexer` + request-pool update | C8 top-2048 与 request-pool 更新；`_out` 写入 caller-owned buffer | eager / 稳定图主链 |
| `kvcache_scatter_copy_c8` / `kvcache_scatter_copy_c8_out` | `A5KvcacheScatterCopyC8` | 整行搬运 656-byte packed KV，并生成 sparse+tail slots 和 resident length；`_out` 写入 caller-owned metadata | eager / 稳定图主链 |
| `sparse_tail_attention_c8` | Python custom op → 原生 `npu_kv_quant_sparse_flash_attention` | 对 packed C8 KV 的 top-2048 sparse slots 与 tail 执行量化 MLA | 稳定图主链 |

## 接口

Torch 算子 namespace 为 `nanovllm_dsa`；Python 包名为 `nanovllm_dsa_a5`，C++ wrapper 的内部 namespace 为 `nanovllm_dsa_a5_impl`。

共同语义：`B` 是 decode batch，`C=cache_tokens[b]` 是请求固定的 HBM token 预算，`candidate_lens[b]` 是参与稀疏选择的 prefill 满块 token 数。`cache_slots_pool[req_pool_entries[b], token_id]` 保存 source token 到 HBM slot 的映射，`-1` 表示未缓存。LIDU 输出中前 `miss_counts[b]` 项为 miss，SCATTER 只搬运这段；全部 2048 个 `destination_slots` 都供 Attention 使用。稳定图使用 caller-owned `_out` 接口。

链路：`fused_li_manage_c8_out → kvcache_scatter_copy_c8_out → sparse_tail_attention_c8`。每个 packed KV token 固定为 656 bytes：`512 FP8 E4M3 + 64 BF16 RoPE + 4 FP32 scales`。

```python
torch.ops.nanovllm_dsa.fused_li_manage_c8(
    query,                  # float8_e4m3fn[B,INDEX_HEADS,128]，INDEX_HEADS=32|64
    key,                    # float8_e4m3fn[INDEX_BLOCKS,128,1,128]
    weights,                # bf16[B,INDEX_HEADS]
    query_dequant_scale,    # fp32[B,INDEX_HEADS]
    key_dequant_scale,      # fp32[INDEX_BLOCKS,128,1]
    actual_seq_lengths_query, # cumulative int32[B]，decode 为 [1,...,B]
    req_pool_entries,       # int32[B]
    cache_slots_pool,       # int32[POOL_SIZE,SOURCE_CAPACITY]，in/out
    cache_tokens,           # int32[B]
    candidate_lens,         # int32[B]
    block_table,            # int32[B,SOURCE_CAPACITY/128]
) -> (
    source_ids,             # int32[B,1,2048]，miss-prefix + hit-suffix
    destination_slots,      # int32[B,1,2048]
    miss_counts,            # int32[B]
    cache_slots_alias,
)

torch.ops.nanovllm_dsa.fused_li_manage_c8_out(
    query, key, weights, query_dequant_scale, key_dequant_scale,
    actual_seq_lengths_query, req_pool_entries, cache_slots_pool,
    cache_tokens, candidate_lens, block_table,
    source_ids,             # caller-owned int32[B,1,2048]，in/out
    destination_slots,      # caller-owned int32[B,1,2048]，in/out
    miss_counts,            # caller-owned int32[B]，in/out
) -> (source_ids_alias, destination_slots_alias, miss_counts_alias, cache_slots_alias)

torch.ops.nanovllm_dsa.kvcache_scatter_copy_c8_out(
    hbm_packed_kv_bytes,    # int8 view[HBM_BLOCKS,128,1,656]，in/out
    dram_packed_kv_bytes,   # swapped-memory int8 view[DRAM_BLOCKS,128,1,656]
    hbm_block_table,        # int32[B,HBM_MAX_BLOCKS]
    dram_block_table,       # int32[B,DRAM_MAX_BLOCKS]
    source_token_ids,       # int32[B,1,2048] 或 int32[B,2048]
    destination_slots,      # 同 source_token_ids
    copy_counts,            # int32[B]，即 miss_counts
    cache_tokens,           # int32[B]
    candidate_lens,         # int32[B]
    actual_seq_lengths_kv,  # int32[B]，原始逻辑 KV 长度
    max_tail_tokens,        # int，固定图的 tail 容量
    attention_slots,        # caller-owned int32[B,1,2048+max_tail_tokens]
    resident_seq_lengths,   # caller-owned int32[B]
) -> (
    hbm_packed_kv_alias,
    attention_slots_alias,  # C>0: top2048 slots + [C,C+tail)；C=0: [0,actual_len)
    resident_lengths_alias, # C>0: C+tail；C=0: actual_len
)

torch.ops.nanovllm_dsa.sparse_tail_attention_c8(
    query,                  # bf16/fp16[T,Q_HEAD,576]，1<=Q_HEAD<=64
    packed_kv,              # float8_e4m3fn view[HBM_BLOCKS,128,1,656]
    attention_slots,        # int32[T,1,2048+max_tail_tokens]
    hbm_block_table,        # int32[B,HBM_MAX_BLOCKS]
    actual_seq_lengths_query, # cumulative int32[B]
    resident_seq_lengths,   # int32[B]，直接使用 packed SCATTER 输出
    scale_value,            # float
) -> attention_out          # bf16/fp16[T,Q_HEAD,512]
```

非 `_out` 的 `kvcache_scatter_copy_c8` 参数相同，但由算子分配 `attention_slots` 和 `resident_seq_lengths`。C8 LIDU 的组合接口内部完成官方 A5 Quant LightningIndexer 和 request-pool 更新。全部框架接口提供 Fake/Meta；LIDU、SCATTER 的 schema 显式声明 mutable alias。

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

C8 LIDU 直接用官方 A5 C8 LightningIndexer 作为 top-2048 基线，并覆盖 mixed C、乱序 request-pool、零/随机/2048 miss、hit slot 保持、重复更新映射、caller-owned out 和 `262144` 的 18-bit token-index 边界：

```bash
python3 tests/test_fused_li_manage_c8.py --device npu:0 --mode check --heads 32,64 --batch-sizes 24 --source-lens 20096 --cache-tokens 6144 --miss-ranges 0:300 --seed 7
```

packed SCATTER 使用真实 swapped memory；分配式与 caller-owned 两条路径都重新 poison HBM，并校验完整 656 bytes、guard、动态 tail metadata、`resident_seq_lengths` 和输出地址：

```bash
python3 tests/test_kvcache_scatter_copy_c8.py --device npu:0 --batch-size 24 --source-len 20096 --cache-tokens 6144 --tail-tokens 257 --max-tail-tokens 512 --copy-min 0 --copy-max 300 --warmup 10 --iters 100 --seed 7
```

C8 QSFA 使用独立 CPU FP32 golden 校验 FP8 latent、BF16 RoPE 和 FP32 scales 的 packed layout。首轮门禁固定 `q_head=8`；本仓库不支持 `q_head>64`：

```bash
for C in 3072 6144 8192 12288; do for tail in 0 1 64 127 257; do python3 tests/test_sparse_tail_attention_c8.py --device npu:0 --mode check --batch-size 24 --heads 8 --cache-tokens "$C" --tail-tokens "$tail" --max-tail-tokens 512 --seed 7; done; done
```

```bash
python3 tests/test_sparse_tail_attention_c8.py --device npu:0 --mode check --batch-size 1 --heads 8 --cache-tokens 0 --tail-tokens 2048 --max-tail-tokens 2048 --seed 7
```

图门禁捕获框架链路 `fused_li_manage_c8_out → kvcache_scatter_copy_c8_out → sparse_tail_attention_c8`；不依赖 capture 阶段执行，先用一次 replay 验证零 miss 和完整输出写回，再恢复初始 pool 进行多次非零 miss replay：

```bash
python3 tests/test_offload_split_c8_graph.py --device npu:0 --case pure-long --batch-size 2 --heads 8 --index-heads 32 --source-len 4096 --cache-tokens 3072 --tail-tokens 64 --max-tail-tokens 256 --miss-min 256 --miss-max 512 --replays 4 --seed 7
```

```bash
python3 tests/test_offload_split_c8_graph.py --device npu:0 --case mixed --batch-size 2 --heads 8 --index-heads 32 --source-len 4096 --cache-tokens 3072 --tail-tokens 64 --max-tail-tokens 256 --miss-min 256 --miss-max 512 --replays 4 --seed 7
```

## 性能矩阵

LIDU：

```bash
python3 tests/test_fused_li_manage_c8.py --device npu:0 --mode bench --heads 32 --batch-sizes 1,4,8,12,16,24,32 --source-lens 12288,20096,65536,131072 --cache-tokens 6144 --miss-ranges 0:0,0:300,300:300 --warmup 10 --iters 100 --seed 7
```
