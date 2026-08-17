# GLM MTP3 offloading operators

> `fused_li_manage_mtp` current ABI (the legacy block below is retained only
> for historical context): `index_weights, query_dequant_scale, query,
> index_key_dequant_scale, index_key_cache, index_block_table,
> actual_seq_lengths_query, actual_seq_lengths_key,
> offload_seq_lengths_key, req_valid, req_pool_entries, cache_state,
> cache_slots_pool, topk_src_ids, topk_dst_slots, miss_src_ids,
> miss_dst_slots, miss_counts`.

这是 GLM MTP3 decode offloading 算子的独立调优工程，不包含 nanovllm 推理框架，也不依赖 `nanovllm-DSA-offload-mtp` 的源码或编译产物。当前公开并测试 `fused_li_manage_mtp`、`scatter_copy`、`sparse_tail_attention_mtp` 和实验性的 `fused_copy_sfa_mtp`。

　

## 算子接口

```python
torch.ops.nanovllm_dsa.fused_li_manage_mtp(  # new ABI: variable 1-4 queries/request
    query,                    # bf16/fp16 [B*4, 32, 128]，只读
    index_weights,            # bf16/fp16 [B*4, 32]，只读
    index_key_cache,          # bf16/fp16 [blocks, 128, 1, 128]，只读
    index_block_table,        # int32 [B, max_source_blocks]，只读
    num_candidate_tokens,     # int32 [B]，只读
    num_cache_tokens,         # int32 [B]，只读
    req_pool_entries,         # int32 [B]，只读
    cache_slots_pool,         # int32 [pool_size, source_capacity]，读写
    topk_src_ids,             # int32 [B*4, 1, 2048]，只写
    topk_dst_slots,           # int32 [B*4, 1, 2048]，只写
    miss_src_ids,             # int32 [B, 8192]，只写
    miss_dst_slots,           # int32 [B, 8192]，只写
    miss_counts,              # int32 [B]，只写
) -> None
```

固定语义：GLM MTP3、每请求 4 个 query、每个 query 独立 top2048，再对四路并集执行一次命中、淘汰和 cache state 更新。`topk_src_ids` 的 HBM hit 位置为 `-1`；`miss_src_ids` 和 `miss_dst_slots` 每行只有前 `miss_counts[b]` 个元素有效。

　

## 全量编译

修改 host tiling、op-api、Torch binding、CMake 或首次部署时运行：

```bash
export ASCEND_HOME_PATH=/usr/local/Ascend/cann-8.5.1
export CANN_INSTALL_PATH=/usr/local/Ascend/cann-8.5.1
export PYTHONPATH=$PWD:$PYTHONPATH
export PYTHONUNBUFFERED=1
export SOC_VERSION=ascend910_9391
export NANOVLLM_CANN_BUILD_JOBS=64
export NANOVLLM_EXT_BUILD_JOBS=1

bash scripts/build_nanovllm_ops.sh
```

产物只写入本仓库：

- `nanovllm/_C*.so`
- `nanovllm/_cann_ops_custom/`

工程不会安装或覆盖系统级自定义算子。

　

## 增量编译

至少成功完成一次全量编译后，如果只修改 AscendC device kernel 或其本地 device header，可以运行：

```bash
export ASCEND_HOME_PATH=/usr/local/Ascend/cann-8.5.1
export SOC_VERSION=ascend910_9391
export NANOVLLM_CANN_BUILD_JOBS=64

bash scripts/rebuild_nanovllm_cann_kernel.sh fused_li_manage_mtp

# 仅重编实验性的 source-aware COPYSFA-MTP 内核
bash scripts/rebuild_nanovllm_cann_kernel.sh fused_copy_sfa_mtp
```

　

## 单算子测试

```bash
unset NANOVLLM_CUST_OPAPI_LIB
unset ASCEND_CUSTOM_OPP_PATH

export ASCEND_HOME_PATH=/usr/local/Ascend/cann-8.5.1
export PYTHONUNBUFFERED=1
export PYTHONPATH=$PWD:$PYTHONPATH
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export ASCEND_LAUNCH_BLOCKING=0
export ASCEND_RT_VISIBLE_DEVICES=10

python3 ut_ops/test_fused_li_manage_mtp.py \
  --device npu:0 \
  --batch-size 24 \
  --source-len 20992 \
  --cache-tokens 8192 \
  --perf-query-miss-count 200 \
  --perf-query-noise 0.25 \
  --graph-replays 3 \
  --warmup 10 \
  --iters 100 \
  --seed 7
```

```bash
python3 ut_ops/test_fused_li_manage_mtp.py \
  --device npu:0 \
  --batch-size 12 \
  --source-len 40064 \
  --cache-tokens 8192 \
  --perf-query-miss-count 500 \
  --perf-query-noise 0.25 \
  --graph-replays 3 \
  --warmup 10 \
  --iters 100 \
  --seed 7
```

UT 覆盖 BF16/FP16 语义、乱序 request-pool、混合 MTP0～MTP3、offload/tail source 范围、空槽扫描与满槽状态切换、请求跳过边界、动态 ACLGraph replay，以及 `B=24` 和 `B=12` 两种典型负载时延。性能日志中的 `index_management_mtp3_us` 定义为：

```text
fused_lim_mtp3_us - official_li_mtp3_us
```

非 MTP 的单 query LIM 基准不属于本单算子工程，跨版本对比时使用主仓库已记录的基线。

　

## 实验性 COPYSFA-MTP

`fused_copy_sfa_mtp` 保持 nanovllm 的固定 ABI：调用方提供输出 buffer，HBM cache 原地更新，算子不返回 alias。当前实现是 source-aware 单内核研究版本。

```python
torch.ops.nanovllm_dsa.fused_copy_sfa_mtp(
    query_rope,
    query,
    actual_seq_lengths_query,
    actual_seq_lengths_kv,
    num_cache_tokens,
    topk_dst_slots,
    topk_src_ids,
    miss_src_ids,
    miss_dst_slots,
    miss_counts,
    hbm_block_table,
    dram_block_table,
    hbm_k_rope,              # mutable
    hbm_kv_cache,            # mutable
    dram_k_rope,
    dram_kv_cache,
    scale_value,
    attention_out,           # output
) -> None
```

诊断测试：

```bash
python3 ut_ops/test_fused_copy_sfa_mtp.py \
  --device npu:0 \
  --batch-size 4 \
  --heads 2 \
  --source-len 20992 \
  --cache-tokens 8192 \
  --tail-tokens 64 \
  --perf-miss-count 300 \
  --graph-replays 2 \
  --warmup 0 \
  --iters 1 \
  --skip-performance \
  --diagnose-attention
```

当前精度问题及后续约束记录在 `TODO.md`；在精度与时延都验收通过前，不合回 nanovllm。
