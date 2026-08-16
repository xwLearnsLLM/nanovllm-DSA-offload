# fused_li_manage_mtp

这是 GLM MTP3 `fused_li_manage_mtp` 的独立调优工程，不包含 nanovllm 推理框架，也不依赖其它仓库的源码、算子或编译产物。

　

## 算子接口

```python
torch.ops.nanovllm_dsa.fused_li_manage_mtp(
    query,                    # bf16/fp16 [B*4, H, 128]，只读，H=32/64
    index_weights,            # bf16/fp16 [B*4, H]，只读
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

固定语义：每请求 4 个 query，每个 query 独立执行 top2048，再对四路并集执行一次命中判断、淘汰和 cache state 更新。`topk_src_ids` 的 HBM hit 位置为 `-1`；`miss_src_ids` 和 `miss_dst_slots` 每行只有前 `miss_counts[b]` 个元素有效。

　

## 编译

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

每次都全量重新编译 `fused_li_manage_mtp`。产物只写入本仓库：

- `nanovllm/_C*.so`
- `nanovllm/_cann_ops_custom/`

工程不会安装或覆盖系统级自定义算子。

　

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

UT 只验证两类内容：

1. 算子语义正确，包括 BF16/FP16、乱序 request pool、重复更新和 ACLGraph replay。
2. 与官方 `torch_npu.npu_lightning_indexer` 的 MTP3 路径比较时延。

性能日志中的索引管理额外时延定义为：

```text
index_management_mtp3_us = fused_lim_mtp3_us - official_li_mtp3_us
```

官方 MTP3 时延基线使用 TND query、累计 query 长度 `[4, 8, ..., 4B]` 和 `sparse_mode=3`。语义 golden 使用四路独立 qlen=1 官方 LightningIndexer，因为 LIM-MTP 的四路 query 均搜索同一份固定 prefill source。
