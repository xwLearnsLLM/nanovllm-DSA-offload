# DSA offload 的 FULL_DECODE_ONLY 模式

该模式只优化稳定 decode 热路径：prefill、prefill 后的首个 decode、短序列、
长短混合 batch，以及未配置 capture size 的 batch 都走 eager。只有以下条件同时
满足时，才重放一个包含 61 层模型 forward、`lightning_indexer`、
`gather_selection_kv_cache`、MLA 和 MoE 的完整 ACLGraph：

- 当前不是 prefill，也不是首个 decode；
- batch 中每个请求都已经进入 DSA offload；
- 当前 batch size 与一个 capture size 完全相等。

DSA graph 不做 bucket padding。`gather_selection_status` 是请求级持久状态，使用
较大的图 bucket 填充额外行可能覆盖其他请求的状态，因此这里优先保证正确性和
稳定性。做 `bs=16` 性能测试时只 capture `16` 即可。

实现采用和 vLLM-Ascend `FULL_DECODE_ONLY` 一致的两级结构：

1. TorchAir `npugraph_ex` 优化 Dynamo 捕获到的 FX region；
2. 外层 `torch.npu.NPUGraph` capture/replay 完整 decode forward。

FIA-v2 的 `actual_seq_kvlen` 是 host list，不能仅靠静态 tensor 更新。因此每层
FIA 都被记录为可更新的 graph task，每次 replay 都刷新当前 sparse KV 长度。
DSA 的动态 block table、candidate length、pool entry 和 slot mapping 则复制到
静态 NPU tensor 后再 replay。

## 重新编译 pybind 扩展

本次修改把 `gather_selection_kv_cache` 的原地 cache/status 更新写入
`torch.library` schema。必须重新编译 `_C*.so`。CANN kernel 没有修改，已经安装
过本仓库 OPP 时可以跳过 OPP 重编译：

```bash
NANOVLLM_SKIP_CANN_OPP_BUILD=1 \
NANOVLLM_EXT_BUILD_JOBS=1 \
SOC_VERSION=ascend910_9391 \
PYTHONPATH=$PWD:$PYTHONPATH \
bash scripts/build_nanovllm_ops.sh
```

若机器上还没有安装这份仓库的 CANN custom OPP，则去掉
`NANOVLLM_SKIP_CANN_OPP_BUILD=1`，执行一次完整编译。

编译后可先确认新 schema 已加载：

```bash
PYTHONPATH=$PWD:$PYTHONPATH python3 - <<'PY'
import torch
import nanovllm.ops  # noqa: F401

schema = str(torch.ops.nanovllm_dsa.gather_selection_kv_cache.default._schema)
print(schema)
assert schema.count("!") >= 3, "stale nanovllm/_C*.so"
PY
```

## TP4、bs=16、约 16K prompt 的测试命令

```bash
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export ASCEND_LAUNCH_BLOCKING=0
export NANOVLLM_MODEL=/mnt/models/Deepseek-V3.2-Pruned-95B-BF/
export ASCEND_RT_VISIBLE_DEVICES=9,10,11,12
export NANOVLLM_TP_SIZE=4
export NANOVLLM_ENABLE_EXPERT_PARALLEL=1

export NANOVLLM_KVCACHE_BLOCK_SIZE=128
export NANOVLLM_HBM_NUM_BLOCKS=450
export NANOVLLM_DRAM_NUM_BLOCKS=2300
export NANOVLLM_MAX_MODEL_LEN=65536
export NANOVLLM_MAX_PREFILL_SEQS_PER_STEP=1
export NANOVLLM_MAX_DECODE_SEQS_PER_STEP=16
export NANOVLLM_ENABLE_DECODE_MLAPO=1

export NANOVLLM_DECODE_GRAPH_MODE=full_decode_only
export NANOVLLM_DECODE_GRAPH_CAPTURE_SIZES=16
export NANOVLLM_DECODE_GRAPH_WARMUP_ITERS=1
export NANOVLLM_ENFORCE_EAGER=0
export NANOVLLM_LOG_DECODE_LAYER_TIMING=0
unset NANOVLLM_DSA_QUERY_ONLY_BACKEND

export NANOVLLM_IGNORE_EOS=1
export NANOVLLM_MAX_GEN_TOKENS=16
export NANOVLLM_PROMPT_LENGTHS=16200,16201,16202,16203,16204,16205,16206,16207,16208,16209,16210,16211,16212,16213,16214,16215

PYTHONPATH=$PWD:$PYTHONPATH python3 example/test.py
```

`NANOVLLM_HBM_NUM_BLOCKS` 和 `NANOVLLM_DRAM_NUM_BLOCKS` 仍需按机器可用内存调整。
对上述 16 个约 16K 请求，DRAM/Index cache 至少需要覆盖全部请求的 prefill 满块。

## 如何确认稳定 decode 确实进入完整图

启动和运行日志必须出现以下三类信息：

```text
DSA FULL_DECODE_ONLY: enabling npugraph_ex FX optimization with one outer ACLGraph ...
DSA FULL_DECODE_ONLY: captured complete decode graph for batch_size=16 with 61 refreshable MLA tasks.
DSA FULL_DECODE_ONLY: first complete graph replay entered for exact batch_size=16.
```

正常退出时还会打印统计：

```text
DSA FULL_DECODE_ONLY final stats: {... 'captures': 1, 'replays': N, ...}
```

验收时要求 `replays > 0`。`eager_first_decode > 0` 是预期行为，不表示降级；TPOT
应只统计日志中首次 replay 之后的稳定 decode step。若 `eager_mixed_batch` 或
`eager_uncaptured_batch` 增长，说明当前调度 batch 不满足本次精确图的条件。

## Profile

先在不开 profiler 的情况下确认 replay 和稳定 TPOT。之后可沿用仓库内置 profiler：

```bash
export NANOVLLM_NPU_PROFILE=1
export NANOVLLM_NPU_PROFILE_DIR=./npu_trace_full_decode_dsa
export NANOVLLM_NPU_PROFILE_SKIP_FIRST=3
export NANOVLLM_NPU_PROFILE_STEPS=6

PYTHONPATH=$PWD:$PYTHONPATH python3 example/test.py
```

分析时应把完整图外的 `prepare_decode`、LM head 和 sampler 与图内模型 forward
分开看。最终目标是让图内相对 baseline 的新增主成本主要收敛到
`LightningIndexerVllm` 和 `GatherSelectionKvCache`；query-only indexer 的小算子
调度应由完整图吸收，若 profile 中仍单独占用较多时间再针对它处理。
