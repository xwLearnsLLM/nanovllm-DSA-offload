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

## 2026-06-03 10:25：清理 DSA 临时日志，新增 NANOVLLM_DSA_CHECK，并精简 timing 字段

下一次请在昇腾上先跑这个：

```bash
PYTHONPATH=$PWD:$PYTHONPATH python3 -m py_compile nanovllm/models/deepseek_v32.py

PYTHONPATH=$PWD:$PYTHONPATH NANOVLLM_MAX_GEN_TOKENS=8 NANOVLLM_ENABLE_DECODE_MLAPO=0 NANOVLLM_DSA_OFFLOAD_FIXED_TX=128 NANOVLLM_DSA_CHECK=1 NANOVLLM_LOG_DECODE_LAYER_TIMING=1 NANOVLLM_DECODE_LAYER_TIMING_SYNC=1 NANOVLLM_PROFILE_LAYER_IDS=mid NANOVLLM_PROMPT_LENGTHS=8192 python3 example/test.py

PYTHONPATH=$PWD:$PYTHONPATH NANOVLLM_MAX_GEN_TOKENS=16 NANOVLLM_ENABLE_DECODE_MLAPO=0 NANOVLLM_DSA_OFFLOAD_FIXED_TX=128 NANOVLLM_DSA_CHECK=0 NANOVLLM_LOG_DECODE_LAYER_TIMING=1 NANOVLLM_DECODE_LAYER_TIMING_SYNC=1 NANOVLLM_PROFILE_LAYER_IDS=mid NANOVLLM_PROMPT_LENGTHS=8192 python3 example/test.py
```

## 2026-06-03 12:53：合入 dsa_index_update CANN 真算子，默认启用，保留 torch 路径开关

下一次请在昇腾上先跑这个：
```bash
PYTHONPATH=$PWD:$PYTHONPATH python3 -m py_compile nanovllm/models/dsa_offload_ops.py ut_ops/probe_dsa_index_update.py

SOC_VERSION=ascend910_9391 PYTHONPATH=$PWD:$PYTHONPATH bash scripts/build_nanovllm_ops.sh

PYTHONPATH=$PWD:$PYTHONPATH ASCEND_RT_VISIBLE_DEVICES=0 python3 ut_ops/probe_dsa_index_update.py --device npu:0 --batch 4 --candidate 8192 --selected 2560 --k 128 --warmup 5 --iters 20

PYTHONPATH=$PWD:$PYTHONPATH NANOVLLM_MAX_GEN_TOKENS=8 NANOVLLM_ENABLE_DECODE_MLAPO=0 NANOVLLM_DSA_OFFLOAD_FIXED_TX=128 NANOVLLM_DSA_INDEX_UPDATE_USE_CANN=1 NANOVLLM_DSA_CHECK=1 NANOVLLM_LOG_DECODE_LAYER_TIMING=1 NANOVLLM_DECODE_LAYER_TIMING_SYNC=1 NANOVLLM_PROFILE_LAYER_IDS=mid NANOVLLM_PROMPT_LENGTHS=8192 python3 example/test.py

PYTHONPATH=$PWD:$PYTHONPATH NANOVLLM_MAX_GEN_TOKENS=8 NANOVLLM_ENABLE_DECODE_MLAPO=0 NANOVLLM_DSA_OFFLOAD_FIXED_TX=128 NANOVLLM_DSA_INDEX_UPDATE_USE_CANN=0 NANOVLLM_DSA_CHECK=1 NANOVLLM_LOG_DECODE_LAYER_TIMING=1 NANOVLLM_DECODE_LAYER_TIMING_SYNC=1 NANOVLLM_PROFILE_LAYER_IDS=mid NANOVLLM_PROMPT_LENGTHS=8192 python3 example/test.py
```

## 2026-06-03 13:10：修复 build_nanovllm_ops.sh 后半段高并发构建不稳定

下一次请在昇腾上先跑这个：
```bash
NANOVLLM_CANN_BUILD_JOBS=64 NANOVLLM_EXT_BUILD_JOBS=1 SOC_VERSION=ascend910_9391 PYTHONPATH=$PWD:$PYTHONPATH bash scripts/build_nanovllm_ops.sh

PYTHONPATH=$PWD:$PYTHONPATH ASCEND_RT_VISIBLE_DEVICES=0 python3 ut_ops/probe_dsa_index_update.py --device npu:0 --batch 4 --candidate 8192 --selected 2560 --k 128 --warmup 5 --iters 20
```
