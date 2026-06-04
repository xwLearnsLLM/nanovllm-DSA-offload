# 准备工作

## 推128专家正常模型（16卡910C）的公共配置

```bash
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export ASCEND_LAUNCH_BLOCKING=0
export NANOVLLM_MODEL=/home/models/DeepSeek-V3.2-REAP-345B-A37B-BF16/  # 模型路径
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15 # 8 卡
export NANOVLLM_TP_SIZE=16                                       # TP16
export NANOVLLM_ENABLE_EXPERT_PARALLEL=1
export NANOVLLM_KVCACHE_BLOCK_SIZE=128
export NANOVLLM_HBM_NUM_BLOCKS=200                              # 500个HBM blocks
export NANOVLLM_DRAM_NUM_BLOCKS=800                             # 2000个DRAM blocks 以及 2000个HBM IndexCache Blocks
export NANOVLLM_MAX_MODEL_LEN=65536
export NANOVLLM_MAX_PREFILL_SEQS_PER_STEP=1                     # prefill最大batch-size设为1，避免爆显存
export NANOVLLM_MAX_DECODE_SEQS_PER_STEP=256                    # decode最大batch-size设为256
export NANOVLLM_IGNORE_EOS=1
```

## 推32专家残障模型（8卡910C）的公共配置

```bash
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export ASCEND_LAUNCH_BLOCKING=0
export NANOVLLM_MODEL=/mnt/models/Deepseek-V3.2-Pruned-95B-BF/  # 模型路径
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7                # 8 卡
export NANOVLLM_TP_SIZE=8                                       # TP8 
export NANOVLLM_ENABLE_EXPERT_PARALLEL=1
export NANOVLLM_KVCACHE_BLOCK_SIZE=128
export NANOVLLM_HBM_NUM_BLOCKS=500                              # 500个HBM blocks
export NANOVLLM_DRAM_NUM_BLOCKS=2000                            # 2000个DRAM blocks 以及 2000个HBM IndexCache Blocks
export NANOVLLM_MAX_MODEL_LEN=65536
export NANOVLLM_MAX_PREFILL_SEQS_PER_STEP=1                     # prefill最大batch-size设为1，避免爆显存
export NANOVLLM_MAX_DECODE_SEQS_PER_STEP=256                    # decode最大batch-size设为256
export NANOVLLM_IGNORE_EOS=1
```

# 运行

## 2026-06-03 23:09：优化纯长序列固定 Tx=128 的 `dsa_index_update` Python 热路径

这次只改 Python 封装和调度元数据，没有改 CANN kernel，理论上不需要重新编译算子。目标场景是所有 batch 都是长序列、都会卸载，并且 `NANOVLLM_DSA_OFFLOAD_FIXED_TX=128`。

下一次请在昇腾上先跑这个 sweep，看纯长快路径下 `dsa_index_update` 自身时延：

```bash
PYTHONPATH=$PWD:$PYTHONPATH ASCEND_RT_VISIBLE_DEVICES=0 python3 ut_ops/bench_dsa_index_update_sweep.py --device npu:0 --batch-sizes 1,2,4,8,10,16 --candidate-lens 8192,16384,32768,65536 --selected-lens 2560 --warmup 10 --iters 50
```

然后跑纯长 batch=10 的 decode timing：

```bash
PYTHONPATH=$PWD:$PYTHONPATH NANOVLLM_DSA_QK_SCORE_BF16_OUT=1 NANOVLLM_DSA_QUERY_ONLY_BACKEND=auto NANOVLLM_DSA_QUERY_ONLY_WARMUP_TOKENS=1,2,4,8,16,32,64,128 NANOVLLM_MAX_GEN_TOKENS=8 NANOVLLM_ENABLE_DECODE_MLAPO=1 NANOVLLM_DSA_OFFLOAD_FIXED_TX=128 NANOVLLM_DSA_INDEX_UPDATE_USE_CANN=1 NANOVLLM_DSA_CHECK=0 NANOVLLM_PROMPT_LENGTHS=17000,17001,17002,17003,17004,17005,17006,17007,17008,17009 NANOVLLM_LOG_DECODE_LAYER_TIMING=1 NANOVLLM_DECODE_LAYER_TIMING_SYNC=0 NANOVLLM_PROFILE_LAYER_IDS=mid python3 example/test.py
```

## 2026-06-04 10:05：固定启用 decode query-only BF16 `weights_proj`，并把 q BMM 上限收敛到 64

这次没有新增运行时开关；decode query-only 直接使用 cached BF16 `weights_proj`，prefill/full indexer_project 仍走原 FP32 路径。下一次请先跑 probe，确认 q BMM 路径和 BF16 weights_proj 仍然对齐：

```bash
PYTHONPATH=$PWD:$PYTHONPATH ASCEND_RT_VISIBLE_DEVICES=0 python3 ut_ops/probe_indexer_project.py --device npu:0 --tokens 10 --warmup 10 --iters 100 --use-bmm-transpose --reuse-output-buffers --profile-detail --weights-topk 10
```

然后跑不组图的纯长 batch=10 decode timing，观察 `indexer_project` 和最终输出是否正常：

```bash
PYTHONPATH=$PWD:$PYTHONPATH NANOVLLM_DSA_QK_SCORE_BF16_OUT=1 NANOVLLM_MAX_GEN_TOKENS=8 NANOVLLM_ENABLE_DECODE_MLAPO=1 NANOVLLM_DSA_OFFLOAD_FIXED_TX=128 NANOVLLM_DSA_INDEX_UPDATE_USE_CANN=1 NANOVLLM_DSA_CHECK=0 NANOVLLM_PROMPT_LENGTHS=17000,17001,17002,17003,17004,17005,17006,17007,17008,17009 NANOVLLM_LOG_DECODE_LAYER_TIMING=1 NANOVLLM_DECODE_LAYER_TIMING_SYNC=0 NANOVLLM_PROFILE_LAYER_IDS=mid python3 example/test.py
```

## 2026-06-04 11:35：把 decode query-only q BMM 路径真正接入推理

这次会为每层额外缓存一份 head-major `wq_b_bmm_t`，用于 `tokens<=64` 的 query-only q projection。请先跑同步 timing，重点看是否 OOM、`indexer_project` 是否下降、输出是否和上一轮一致：

```bash
PYTHONPATH=$PWD:$PYTHONPATH NANOVLLM_DSA_QK_SCORE_BF16_OUT=1 NANOVLLM_MAX_GEN_TOKENS=8 NANOVLLM_ENABLE_DECODE_MLAPO=1 NANOVLLM_DSA_OFFLOAD_FIXED_TX=128 NANOVLLM_DSA_INDEX_UPDATE_USE_CANN=1 NANOVLLM_DSA_CHECK=0 NANOVLLM_PROMPT_LENGTHS=17000,17001,17002,17003,17004,17005,17006,17007,17008,17009 NANOVLLM_LOG_DECODE_LAYER_TIMING=1 NANOVLLM_DECODE_LAYER_TIMING_SYNC=1 NANOVLLM_PROFILE_LAYER_IDS=mid python3 example/test.py
```

如果同步 timing 正常，再跑正常性能版本：

```bash
PYTHONPATH=$PWD:$PYTHONPATH NANOVLLM_DSA_QK_SCORE_BF16_OUT=1 NANOVLLM_MAX_GEN_TOKENS=8 NANOVLLM_ENABLE_DECODE_MLAPO=1 NANOVLLM_DSA_OFFLOAD_FIXED_TX=128 NANOVLLM_DSA_INDEX_UPDATE_USE_CANN=1 NANOVLLM_DSA_CHECK=0 NANOVLLM_PROMPT_LENGTHS=17000,17001,17002,17003,17004,17005,17006,17007,17008,17009 NANOVLLM_LOG_DECODE_LAYER_TIMING=1 NANOVLLM_DECODE_LAYER_TIMING_SYNC=0 NANOVLLM_PROFILE_LAYER_IDS=mid python3 example/test.py
```


## 2026-06-04 12:05：诊断 query-only TorchAir 组图精度

这次新增 `ut_ops/probe_query_only_torchair_accuracy.py`，用于把 current 路径、functional eager 路径、TorchAir 组图路径放在同一组输入上对齐。先跑 batch=10 的主场景：

```bash
PYTHONPATH=$PWD:$PYTHONPATH ASCEND_RT_VISIBLE_DEVICES=0 python3 ut_ops/probe_query_only_torchair_accuracy.py --device npu:0 --tokens 10 --use-bmm-transpose --weights-dtype bf16 --repeats 3 --warmup 10 --iters 100 --score-proxy-candidates 512 --score-proxy-topk 128
```

然后跑几个 shape，确认问题是否只在特定 batch 触发：

```bash
PYTHONPATH=$PWD:$PYTHONPATH ASCEND_RT_VISIBLE_DEVICES=0 python3 ut_ops/probe_query_only_torchair_accuracy.py --device npu:0 --tokens 1,2,4,8,10,16,32,64 --use-bmm-transpose --weights-dtype bf16 --repeats 3 --warmup 5 --iters 20 --score-proxy-candidates 512 --score-proxy-topk 128
```

如果上面显示 `current_vs_functional q` 已经有明显差异，再额外跑一把不走 q BMM 的版本，用来判断差异是否来自 q BMM 还是 RoPE：

```bash
PYTHONPATH=$PWD:$PYTHONPATH ASCEND_RT_VISIBLE_DEVICES=0 python3 ut_ops/probe_query_only_torchair_accuracy.py --device npu:0 --tokens 10 --weights-dtype bf16 --repeats 3 --warmup 10 --iters 100 --score-proxy-candidates 512 --score-proxy-topk 128
```

## 2026-06-04 12:06：让 query-only TorchAir 路径使用 runtime NPU RoPE 语义

`runlog/44.txt` 显示 q projection 和 weights projection 都能对齐，差异从 RoPE 开始。因此这次把 TorchAir functional 路径从手写 RoPE 改成和 runtime query-only 一样的 `torch_npu.npu_rotary_mul` 路径；同时把 probe 里的 sentinel 默认值改成 NaN，避免合法输出刚好等于 `-123` 时误报。

下一次请在昇腾上先跑这个：

```bash
PYTHONPATH=$PWD:$PYTHONPATH ASCEND_RT_VISIBLE_DEVICES=0 python3 ut_ops/probe_query_only_torchair_accuracy.py --device npu:0 --tokens 10 --use-bmm-transpose --weights-dtype bf16 --repeats 3 --warmup 10 --iters 100 --score-proxy-candidates 512 --score-proxy-topk 128
```

如果上面的 `torchair_vs_current q` 变成 0，再跑纯长 batch=10 的 TorchAir 推理：

```bash
PYTHONPATH=$PWD:$PYTHONPATH NANOVLLM_DSA_QUERY_ONLY_BACKEND=torchair NANOVLLM_DSA_QUERY_ONLY_WARMUP_TOKENS=10 NANOVLLM_DSA_QK_SCORE_BF16_OUT=1 NANOVLLM_MAX_GEN_TOKENS=8 NANOVLLM_ENABLE_DECODE_MLAPO=1 NANOVLLM_DSA_OFFLOAD_FIXED_TX=128 NANOVLLM_DSA_INDEX_UPDATE_USE_CANN=1 NANOVLLM_DSA_CHECK=0 NANOVLLM_PROMPT_LENGTHS=17000,17001,17002,17003,17004,17005,17006,17007,17008,17009 NANOVLLM_LOG_DECODE_LAYER_TIMING=1 NANOVLLM_DECODE_LAYER_TIMING_SYNC=1 NANOVLLM_PROFILE_LAYER_IDS=mid python3 example/test.py
```

## 2026-06-04 17:36：清理运行时开关并固化默认性能路径

这次删除了 `NANOVLLM_DSA_QK_SCORE_BF16_OUT`、`NANOVLLM_FUSE_QKV_A`、`NANOVLLM_FREE_KV_B_PROJ`、`NANOVLLM_DEBUG_DECODE_MLAPO_COMPARE*` 相关代码；`NANOVLLM_DSA_OFFLOAD_FIXED_TX` 默认改为 128，`example/test.py` 的 `NANOVLLM_MAX_MODEL_LEN` 默认改为 65536。

下一次请在昇腾上先跑这个：

```bash
PYTHONPATH=$PWD:$PYTHONPATH NANOVLLM_DSA_QUERY_ONLY_BACKEND=torchair NANOVLLM_DSA_QUERY_ONLY_WARMUP_TOKENS=10 NANOVLLM_MAX_GEN_TOKENS=8 NANOVLLM_ENABLE_DECODE_MLAPO=1 NANOVLLM_DSA_INDEX_UPDATE_USE_CANN=1 NANOVLLM_DSA_CHECK=0 NANOVLLM_PROMPT_LENGTHS=17000,17001,17002,17003,17004,17005,17006,17007,17008,17009 NANOVLLM_LOG_DECODE_LAYER_TIMING=0 NANOVLLM_DECODE_LAYER_TIMING_SYNC=0 NANOVLLM_PROFILE_LAYER_IDS=mid python3 example/test.py
```
