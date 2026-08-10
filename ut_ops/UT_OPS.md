# Ascend 算子验收

先运行 `bash scripts/build_nanovllm_ops.sh`，再在单张 NPU 上执行测试。公共环境变量：

```bash
unset NANOVLLM_PROFILE_DECODE_OUTPUT
unset NANOVLLM_CUST_OPAPI_LIB
unset ASCEND_CUSTOM_OPP_PATH
unset NANOVLLM_LIDU_MISS_COUNT_ON_LAYERS
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export ASCEND_LAUNCH_BLOCKING=0
export ASCEND_RT_VISIBLE_DEVICES=4
export PYTHONUNBUFFERED=1
export PYTHONPATH=$PWD:$PYTHONPATH
export ASCEND_HOME_PATH=/usr/local/Ascend/cann-8.5.1
```

`fused_li_manage` 的 18/21-bit 边界、request-pool、重复更新及原生 LightningIndexer 对照：

```bash
python3 ut_ops/test_fused_li_manage.py --device npu:0 --heads 32,64 --seq-lens 262144,262272,1048576,2097151 --batch-size 1 --cache-tokens 6144 --miss-count 300 --warmup 1 --iters 3 --seed 7
```

`fused_li_manage` + `scatter_copy` 语义、request pool、本地 bs=24 调度和 graph replay：

```bash
python3 ut_ops/test_fused_li_manage_scatter.py --device npu:0 --heads 32,64 --seed 7 --warmup 2 --iters 10 --graph-replays 3
```

`fused_li_manage` 索引管理时延，计时前会恢复初始 `cache_slots`：

```bash
python3 ut_ops/test_fused_li_manage_perf.py --device npu:0 --heads 32 --batch-sizes 24 --seq-lens 20992 --cache-tokens 6144 --miss-ranges 0:0,0:200,0:300,0:2048 --warmup 10 --iters 100 --seed 7
```

独立 `scatter_copy`，使用真实 swapped-memory DRAM source 和 poison 校验：

```bash
python3 ut_ops/test_scatter_copy.py --device npu:0 --batch-size 24 --source-len 20992 --hbm-slots 6144 --copy-min 0 --copy-max 300 --copy-cap 2048 --warmup 10 --iters 100 --seed 7
```

独立 INT8 `kvcache_offload_copy`，使用普通 HBM source、真实 swapped-memory DRAM destination、动态 block 数和 poison guard 校验：

```bash
python3 ut_ops/test_kvcache_offload_copy.py --device npu:0 --batch-size 24 --copy-cap 32 --copy-min 0 --copy-max 16 --block-size 128 --cache-dim 512 --warmup 10 --iters 100 --seed 7
```

额外用非 32-byte 对齐且跨 32 KiB tile 的 block 覆盖 `DataCopyPad` 尾块路径：

```bash
python3 ut_ops/test_kvcache_offload_copy.py --device npu:0 --batch-size 2 --copy-cap 3 --copy-min 0 --copy-max 2 --block-size 127 --cache-dim 513 --warmup 1 --iters 3 --seed 17
```

独立 `sparse_tail_attention`：

```bash
python3 ut_ops/test_sparse_tail_attention.py --device npu:0 --heads 4 --batch-size 24 --cache-tokens 6144 --tail-tokens 64 --warmup 10 --iters 100 --min-speedup 1.0 --seed 7
```

`fused_copy_sfa` 与 `scatter_copy + sparse_tail_attention` 对照；测试会独立 poison 两条路径，分别验证 DRAM→HBM 搬移和 Attention：

```bash
python3 ut_ops/test_fused_copy_sfa.py --device npu:0 --mode all --batch-size 24 --heads 4 --source-len 20992 --cache-tokens 6144 --tail-tokens 64 --miss-min 0 --miss-max 2048 --warmup 10 --iters 100 --seed 7
```

GLM Indexer projection 与 interleaved RoPE：

```bash
python3 ut_ops/test_glm_dsa_indexer.py --device npu:0 --batch-size 24 --full-len 4096 --topk 2048 --block-size 128 --seed 7 --bmm-warmup 10 --bmm-iters 100
```

ModelSlim W4A8 routed expert：

```bash
export NANOVLLM_MODEL=/mnt/models/GLM-5.1-w4a8/
python3 ut_ops/test_glm_w4a8_moe.py --model "$NANOVLLM_MODEL" --device npu:0 --layer 3 --expert 0 --tokens 2 --warmup 2 --iters 10
```
