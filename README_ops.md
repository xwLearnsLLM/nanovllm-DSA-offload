# GLM-5.1 DSA 卸载算子

本仓库内置以下六个算子，不依赖外部算子仓库：

| 算子 | 用途 | MTP3 |
| --- | --- | :---: |
| `fused_li_manage` | query_len=1 的 LightningIndexer、命中/淘汰与 request-pool 更新 | 否 |
| `fused_li_manage_mtp` | 四路 query 的 top-2048 并集、命中/淘汰与 request-pool 更新 | 是 |
| `scatter_copy` | DRAM→HBM 的 CKV/KPE 搬移 | 共用 |
| `sparse_tail_attention` | 单 query 的 top-2048 + dense tail Attention | 否 |
| `sparse_tail_attention_mtp` | 四个验证位置各自的 top-2048 + causal dense tail Attention | 是 |
| `fused_copy_sfa` | 融合 `scatter_copy + sparse_tail_attention`，支持 bs>24 | 否 |

全部 `torch.ops.nanovllm_dsa.*` 注册都有 Meta/Fake 可见实现；带 `_out` 的接口使用调用方持久 buffer，供 `FULL_DECODE_ONLY` capture/replay。

## 关键接口与边界

### `fused_li_manage`

```python
source_ids, destination_slots, miss_counts, cache_alias = (
    torch.ops.nanovllm_dsa.fused_li_manage(
        query,              # bf16/fp16 [B, 32|64, 128]
        key,                # bf16/fp16 [blocks, 128, 1, 128]
        weights,            # bf16/fp16 [B, 32|64]
        req_pool_entries,   # int32 [B]
        cache_slots_pool,   # int32 [pool_size, source_capacity], mutable
        cache_tokens,       # int32 [B]
        candidate_lens,     # int32 [B]
        block_table,        # int32 [B, max_blocks]
    )
)
```

输出前三项分别为 `[B,1,2048]`、`[B,1,2048]` 和 `[B]`。每行前 `miss_counts[b]` 个 source/slot 用于搬移，完整 destination row 供 Attention 使用。算子本身支持 21-bit source index；当前 nanovllm 端到端配置仍将 source 限制在 18-bit，放开框架限制前不能把算子能力等同于整网能力。

### `fused_li_manage_mtp`

```python
topk_slots, miss_ids, miss_slots, miss_counts, cache_alias = (
    torch.ops.nanovllm_dsa.fused_li_manage_mtp(
        query,              # bf16/fp16 [B*4, 32, 128]
        key,
        weights,            # bf16/fp16 [B*4, 32]
        req_pool_entries,   # int32 [B]
        cache_slots_pool,   # mutable request pool
        cache_tokens,
        candidate_lens,
        block_table,
    )
)
```

固定 `query_len=4`、32 heads、top-k=2048。四路 top-k 先求有序并集，再基于同一份旧缓存完成一次命中、淘汰和状态更新。`topk_slots` 为 `[B*4,1,2048]`；两项 miss 输出为 `[B,8192]`。该算子暂时只支持 18-bit source index。

### `scatter_copy`

`scatter_copy` 同时服务非 MTP 和 MTP。它不理解 query_len，只消费 `[B, copy_capacity]` 的 source IDs、destination slots 与 `[B]` copy counts，因此同一实现可以处理非 MTP 的 2048 capacity 和 MTP 并集的 8192 capacity。

### Attention 与融合路径

- `sparse_tail_attention` 消费 `[B,1,2048]` slots。
- `sparse_tail_attention_mtp` 消费 `[B*4,1,2048]` slots，并保证第 0～3 个验证位置只看到各自允许的 causal tail。
- `fused_copy_sfa` 仅用于非 MTP `offload_fuse`。它使用 quotient/remainder 均衡分核，支持 bs=24、25、48、64 等跨 24-core 边界 batch。
- MTP 暂不支持 `offload_fuse`；后续需要单独实现 `fused_copy_sfa_mtp`。

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
unset NANOVLLM_ENABLE_DSA_OFFLOAD
unset NANOVLLM_OFFLOAD_MODE
unset NANOVLLM_GS_MISS_RATE_ON_LAYERS
unset NANOVLLM_LIDU_MISS_COUNT_ON_LAYERS
unset NANOVLLM_PROFILE_DECODE_OUTPUT
unset NANOVLLM_CUST_OPAPI_LIB
unset ASCEND_CUSTOM_OPP_PATH
unset NANOVLLM_DSA_BOUNDARY_PROBE
unset NANOVLLM_NUM_SPECULATIVE_TOKENS
unset NANOVLLM_GS_PARALLEL_COPY

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
# MTP-LIDU 语义、乱序 request pool、动态图 metadata 和性能
python3 ut_ops/test_fused_li_manage_mtp.py \
  --device npu:0 --batch-size 24 --source-len 20992 \
  --cache-tokens 8192 --graph-replays 3 \
  --warmup 10 --iters 100 --seed 7

# 四行因果 sparse Attention、_out 和 graph replay
python3 ut_ops/test_sparse_tail_attention_mtp.py \
  --device npu:0 --heads 2 --batch-size 24 \
  --cache-tokens 8192 --tail-tokens 64 --graph-replays 3 \
  --warmup 10 --iters 100 --seed 7

# 完整 MTP 卸载链的数据依赖
python3 ut_ops/test_mtp_offload_chain.py \
  --device npu:0 --batch-size 4 --heads 2 \
  --source-len 20992 --cache-tokens 8192 --tail-tokens 64 \
  --graph-replays 3 --seed 7

# Target qlen=4 MLA 与逐 token 参考结果及 graph replay
python3 ut_ops/test_glm_mtp_target_verify.py \
  --device npu:0 --batch-size 4 --heads 2 --prefix-len 4096 \
  --query-lens 2,4 --graph-replays 3 --seed 7 \
  --atol 0.04 --rtol 0.02 --min-cosine 0.999
```
