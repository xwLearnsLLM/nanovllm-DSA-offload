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

## 2026-06-04 09:16：放宽 `dsa_indexer_project` 的 q BMM 路径到 batch=16

这次只改 Python 判断逻辑，不需要重新编译 CANN 算子。下一次请在昇腾上先跑 query-only / full indexer_project 单测，确认 batch=10 已经走 `dsa_indexer_project_bmm_transpose` 路径：

```bash
PYTHONPATH=$PWD:$PYTHONPATH ASCEND_RT_VISIBLE_DEVICES=0 python3 ut_ops/probe_indexer_project.py --device npu:0 --tokens 10 --warmup 10 --iters 100 --use-bmm-transpose --reuse-output-buffers --profile-detail
```

然后跑纯长 batch=10 的 decode timing，对比 `indexer_project` 是否下降：

```bash
PYTHONPATH=$PWD:$PYTHONPATH NANOVLLM_DSA_QK_SCORE_BF16_OUT=1 NANOVLLM_DSA_QUERY_ONLY_BACKEND=auto NANOVLLM_DSA_QUERY_ONLY_WARMUP_TOKENS=1,2,4,8,10,16,32,64,128 NANOVLLM_MAX_GEN_TOKENS=8 NANOVLLM_ENABLE_DECODE_MLAPO=1 NANOVLLM_DSA_OFFLOAD_FIXED_TX=128 NANOVLLM_DSA_INDEX_UPDATE_USE_CANN=1 NANOVLLM_DSA_CHECK=0 NANOVLLM_PROMPT_LENGTHS=17000,17001,17002,17003,17004,17005,17006,17007,17008,17009 NANOVLLM_LOG_DECODE_LAYER_TIMING=1 NANOVLLM_DECODE_LAYER_TIMING_SYNC=0 NANOVLLM_PROFILE_LAYER_IDS=mid python3 example/test.py
```

## 2026-06-04 09:20：继续把 q BMM 路径上限放宽到 128

这次仍然只改 Python 判断逻辑，不需要重新编译 CANN 算子。下一次请在昇腾上跑这个 sweep，重点看 `dsa_indexer_project_q_path` 是否为 `dsa_indexer_project_bmm_transpose`，以及不同 tokens 下 BMM 路径是否比 linear 路径更快：

```bash
for t in 10 16 32 64 128; do PYTHONPATH=$PWD:$PYTHONPATH ASCEND_RT_VISIBLE_DEVICES=0 python3 ut_ops/probe_indexer_project.py --device npu:0 --tokens $t --warmup 10 --iters 100 --use-bmm-transpose --reuse-output-buffers --profile-detail; done
```

## 2026-06-04 09:56：验证 `weights_proj` 从 FP32 改成 BF16 的精度和性能风险

这次只改了 `ut_ops/probe_indexer_project.py`，没有改模型热路径。下一次请在昇腾上跑这个，重点看 `INDEXER_DIFF weights_proj_bf16_*`、`INDEXER_TOPK weights_proj_bf16_*` 和 `INDEXER_BENCH weights_proj_*`：

```bash
PYTHONPATH=$PWD:$PYTHONPATH ASCEND_RT_VISIBLE_DEVICES=0 python3 ut_ops/probe_indexer_project.py --device npu:0 --tokens 10 --warmup 10 --iters 100 --use-bmm-transpose --reuse-output-buffers --profile-detail --weights-topk 10
```

如果想看更大 batch/token 数下 BF16 weights_proj 是否仍然稳定，再跑这个 sweep：

```bash
for t in 10 16 32 64 128; do PYTHONPATH=$PWD:$PYTHONPATH ASCEND_RT_VISIBLE_DEVICES=0 python3 ut_ops/probe_indexer_project.py --device npu:0 --tokens $t --warmup 10 --iters 100 --use-bmm-transpose --reuse-output-buffers --weights-topk $t; done
```
