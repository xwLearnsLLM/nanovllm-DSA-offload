# Ascend 算子验收

先运行 `bash scripts/build_nanovllm_ops.sh`，再在单张 NPU 上执行下列测试。公共环境：

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

LIDU + SCATTER 语义、request pool 和 graph replay：

```bash
python3 ut_ops/test_lidu_scatter.py --device npu:0 --heads 32 --seed 7 --warmup 2 --iters 10 --graph-replays 3
```

LIDU 索引管理时延：

```bash
python3 ut_ops/test_lidu_perf.py --device npu:0 --heads 32 --batch-sizes 24 --seq-lens 20992 --cache-tokens 6144 --miss-ranges 0:0,0:200,0:300,0:2048 --warmup 10 --iters 100 --seed 7
```

独立 SCATTER：

```bash
python3 ut_ops/test_scatter_copy.py --device npu:0 --batch-size 24 --source-len 20992 --cache-tokens 8192 --copy-cap 8192 --miss-counts 0,256,1024,2048,4096,6144,8192 --warmup 10 --iters 100 --seed 7
```

MTP3-LIDU，以及其 8192-capacity 输出到 SCATTER 的 eager/graph 链式语义：

```bash
python3 ut_ops/test_lidu_mtp.py --device npu:0 --batch-size 24 --source-len 20992 --cache-tokens 8192 --graph-replays 3 --warmup 10 --iters 100 --min-speedup 1.0 --seed 7
```

Sparse-and-tail Attention：

```bash
python3 ut_ops/test_sparse_and_tail_attention.py --device npu:0 --heads 8 --batch-size 24 --cache-tokens 6144 --tail-tokens 64 --warmup 10 --iters 100 --min-speedup 1.0 --seed 7
```

融合 SCATTER + sparse-and-tail Attention，测试使用真实 swapped-memory DRAM source，并分别验证搬移和 Attention：

```bash
python3 ut_ops/test_fused_attention_scatter.py --device npu:0 --mode check --batch-size 24 --heads 8 --source-len 20992 --cache-tokens 6144 --tail-tokens 64 --miss-min 0 --miss-max 2048 --seed 7
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

MTP3 sparse-and-tail Attention (four causal query rows, alloc/out, graph replay,
and comparison with four serial single-query launches):

```bash
python3 ut_ops/test_sparse_and_tail_attention_mtp.py --device npu:0 --heads 2 --batch-size 24 --cache-tokens 8192 --tail-tokens 64 --graph-replays 3 --warmup 10 --iters 100 --min-speedup 1.0 --seed 7
```
