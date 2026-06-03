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

## 2026-06-03 15:25：调整 dsa_index_update 单测，默认校验 hard invariant 而不是逐项对齐 torch

下一次请在昇腾上先跑这个。默认会检查 promote/demote/pool unique、范围、pool 更新位置、非 demote slot 不变等硬约束，并打印和 torch 原型的 overlap 作为参考：

```bash
PYTHONPATH=$PWD:$PYTHONPATH ASCEND_RT_VISIBLE_DEVICES=0 NANOVLLM_DSA_INDEX_UPDATE_USE_CANN=1 python3 ut_ops/probe_dsa_index_update.py --device npu:0 --batch 4 --candidate 8192 --selected 2560 --k 128 --warmup 10 --iters 100
```


运行推理（不对 query_only_indexer_project 进行组图）：

```
PYTHONPATH=$PWD:$PYTHONPATH NANOVLLM_MAX_GEN_TOKENS=8 NANOVLLM_ENABLE_DECODE_MLAPO=1 NANOVLLM_DSA_OFFLOAD_FIXED_TX=128 NANOVLLM_DSA_INDEX_UPDATE_USE_CANN=1 NANOVLLM_DSA_CHECK=0 NANOVLLM_LOG_DECODE_LAYER_TIMING=1 NANOVLLM_DECODE_LAYER_TIMING_SYNC=0 NANOVLLM_PROFILE_LAYER_IDS=mid NANOVLLM_PROMPT_LENGTHS=8200,9000,10000,11000,12000,13000,14000,15000,16000,17000 python3 example/test.py
```

运行推理（对 query_only_indexer_project 进行组图）：

```
PYTHONPATH=$PWD:$PYTHONPATH NANOVLLM_DSA_QUERY_ONLY_BACKEND=auto NANOVLLM_DSA_QUERY_ONLY_WARMUP_TOKENS=1,2,4,8,16,32,64,128 NANOVLLM_MAX_GEN_TOKENS=8 NANOVLLM_ENABLE_DECODE_MLAPO=1 NANOVLLM_DSA_OFFLOAD_FIXED_TX=128 NANOVLLM_DSA_INDEX_UPDATE_USE_CANN=1 NANOVLLM_DSA_CHECK=0 NANOVLLM_LOG_DECODE_LAYER_TIMING=1 NANOVLLM_DECODE_LAYER_TIMING_SYNC=0 NANOVLLM_PROFILE_LAYER_IDS=mid NANOVLLM_PROMPT_LENGTHS=8200,9000,10000,11000,12000,13000,14000,15000,16000,17000 python3 example/test.py
```
