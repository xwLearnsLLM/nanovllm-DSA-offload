# COPYSFA-MTP 单算子调优工程

这是 GLM MTP3 `fused_copy_sfa_mtp` 的独立调优工程，不包含 nanovllm 推理框架，也不依赖 `nanovllm-DSA-offload-mtp` 的源码或编译产物。`scatter_copy` 和 `sparse_tail_attention_mtp` 仅作为 split 性能与精度基线保留。

　

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
export ASCEND_RT_VISIBLE_DEVICES=4

python3 ut_ops/test_fused_copy_sfa_mtp.py \
  --device npu:0 \
  --batch-size 24 \
  --heads 8 \
  --source-len 65536 \
  --cache-tokens 8192 \
  --tail-tokens 64 \
  --perf-miss-count 300 \
  --perf-miss-overlap-rate 0.3333333333333333 \
  --graph-replays 2 \
  --warmup 10 \
  --iters 100 \
  --seed 7
```

UT 直接构造语义一致的 TopK/miss/slot metadata。语义压力场景覆盖最高 8192 个 union misses；性能场景用 `--perf-miss-count` 指定每请求的 unique union misses，用 `--perf-miss-overlap-rate` 指定这些 miss 在 4 个 query 间的重合率，并与 `scatter_copy + sparse_tail_attention_mtp` 比较。完整 top2048 的重合度不是性能场景的约束。

miss 重合率定义为：

```text
(4 个 query 的 miss occurrence 总数 - unique union misses)
----------------------------------------------------------
                 3 * unique union misses
```

取值 `0` 表示各 query 的 miss 完全不重复，`1` 表示每个 miss 都出现在 4 个 query 中；默认 `1/3` 表示每个 unique miss 平均出现在 2 个 query 中。因 token 数取整，日志会同时打印请求值、实际值和各 query 的实际 miss 数，并分别打印 split 路径的 `kvcache_scatter_copy`、`sparse_tail_attention_mtp` 及二者整体时延。

　

## 算子接口

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
    topk_miss_counts,        # [4B], miss-prefix length for each query
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

后续优化约束记录在 `TODO.md`；在典型负载时延验收完成前，不合回 nanovllm。
