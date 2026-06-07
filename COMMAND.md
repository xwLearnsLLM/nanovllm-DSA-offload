## 推128专家模型（16卡910C）准备工作

先进行一些公用配置：

```bash
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export ASCEND_LAUNCH_BLOCKING=0
export NANOVLLM_MODEL=/var/models/DeepSeek-V3.2-REAP-345B-A37B-BF16/   # 模型路径
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15 # 16卡
export NANOVLLM_TP_SIZE=16                                      # TP16
export NANOVLLM_ENABLE_EXPERT_PARALLEL=1
export NANOVLLM_KVCACHE_BLOCK_SIZE=128
export NANOVLLM_HBM_NUM_BLOCKS=200                              # 200个HBM blocks
export NANOVLLM_DRAM_NUM_BLOCKS=800                             # 800个DRAM blocks 以及 800个HBM IndexCache Blocks
export NANOVLLM_MAX_MODEL_LEN=65536
export NANOVLLM_MAX_PREFILL_SEQS_PER_STEP=1                     # prefill最大batch-size设为1，避免爆显存
export NANOVLLM_MAX_DECODE_SEQS_PER_STEP=256                    # decode最大batch-size设为256
export NANOVLLM_IGNORE_EOS=1
```

## 推32专家残障模型（8卡910C）准备工作

先进行一些公用配置：

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

## 开启或者关闭时延分解打印

```
export NANOVLLM_LOG_DECODE_LAYER_TIMING=0    # 是否打印时延分解
export NANOVLLM_DECODE_LAYER_TIMING_SYNC=0   # 计时时是否 sync 一下
export NANOVLLM_PROFILE_LAYER_IDS=mid        # 打印的层
```

## 2026-06-07 16:19：验证 gather_selection_kv_cache pool entry 机制

下一次请在昇腾上先跑这个：

```bash
NANOVLLM_CANN_BUILD_JOBS=64 NANOVLLM_EXT_BUILD_JOBS=1 SOC_VERSION=ascend910_9391 PYTHONPATH=$PWD:$PYTHONPATH bash scripts/build_nanovllm_ops.sh

PYTHONPATH=$PWD:$PYTHONPATH ASCEND_RT_VISIBLE_DEVICES=0 python3 ut_ops/gather_selection/probe_pool.py --device npu:0 --batch-size 4 --pool-capacity 8 --full-len 4096 --topk 2048 --block-size 128 --warmup 10 --iters 100

PYTHONPATH=$PWD:$PYTHONPATH ASCEND_RT_VISIBLE_DEVICES=0 python3 ut_ops/gather_selection/probe_pool.py --device npu:0 --batch-size 8 --pool-capacity 16 --full-len 16384 --topk 2048 --block-size 128 --no-mixed-short --warmup 10 --iters 100
```

## 2026-06-07 16:46：修复 gather_block_table 残留变量后重跑推理

下一次请在昇腾上先跑这个：

```bash
PYTHONPATH=$PWD:$PYTHONPATH NANOVLLM_MAX_GEN_TOKENS=16 NANOVLLM_PROMPT_LENGTHS=11000,11100,11200,11300,11000,11100,11200,11300,11000,11100,11200,11300,11000,11100,11200,11300 python3 example/test.py
```
