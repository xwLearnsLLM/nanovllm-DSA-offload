# fused_li_manage_mtp

这是 `fused_li_manage_mtp`（LIM-MTP）的独立调优工程。仓库只编译、注册和测试这一个自定义算子，不包含 nanovllm 推理框架，也不依赖 `nanovllm-DSA-offload-mtp` 的源码或编译产物。

　

## 算子接口

```python
torch.ops.nanovllm_dsa.fused_li_manage_mtp(
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

UT 覆盖 BF16/FP16 语义、乱序 request-pool、动态 ACLGraph replay，以及 `B=24` 典型负载时延。性能日志中的 `index_management_mtp3_us` 定义为：

```text
fused_lim_mtp3_us - official_li_mtp3_us
```

非 MTP 的单 query LIM 基准不属于本单算子工程，跨版本对比时使用主仓库已记录的基线。
