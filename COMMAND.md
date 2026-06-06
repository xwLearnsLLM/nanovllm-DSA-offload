# 准备工作

## 推128专家正常模型（16卡910C）的公共配置

```bash
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export ASCEND_LAUNCH_BLOCKING=0
export NANOVLLM_MODEL=/home/models/DeepSeek-V3.2-REAP-345B-A37B-BF16/  # 模型路径
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15 # 16 卡
export NANOVLLM_TP_SIZE=16                                       # TP16
export NANOVLLM_ENABLE_EXPERT_PARALLEL=1
export NANOVLLM_KVCACHE_BLOCK_SIZE=128
export NANOVLLM_HBM_NUM_BLOCKS=200                              # 200个 HBM blocks
export NANOVLLM_DRAM_NUM_BLOCKS=800                             # 800个 DRAM blocks 以及 800个 HBM IndexCache Blocks
export NANOVLLM_MAX_MODEL_LEN=65536
export NANOVLLM_MAX_PREFILL_SEQS_PER_STEP=1                     # prefill最大batch-size设为1，避免爆显存
export NANOVLLM_MAX_DECODE_SEQS_PER_STEP=256                    # decode最大batch-size设为256
export NANOVLLM_IGNORE_EOS=1
```

## 2026-06-06：把 gather_selection_kv_cache 合入 nano-vllm csrc

本次把 `npu_gather_selection_kv_cache` 从外部实验目录合入主仓 `csrc/nanovllm_ascend_ops/ops/gather_selection_kv_cache`。
之后 nano-vllm 直接调用自己的 `nanovllm.ops.npu_gather_selection_kv_cache`，不再依赖外部 `gather_selection_custom_ops` 包。

下一次请在昇腾上先跑这个：

```bash
cd /home/w00916487/nanovllm-DSA/nano-vllm-ascend-DeepseekV32-dev_dsa_offload
rm -rf build/nanovllm_ascend_ops nanovllm/_cann_ops_custom
NANOVLLM_CANN_BUILD_JOBS=64 NANOVLLM_EXT_BUILD_JOBS=1 SOC_VERSION=ascend910_9391 PYTHONPATH=$PWD:$PYTHONPATH bash scripts/build_nanovllm_ops.sh

PYTHONPATH=$PWD:$PYTHONPATH \
NANOVLLM_MAX_GEN_TOKENS=8 \
NANOVLLM_ENABLE_DECODE_MLAPO=1 \
NANOVLLM_LOG_DECODE_LAYER_TIMING=1 \
NANOVLLM_DECODE_LAYER_TIMING_SYNC=1 \
NANOVLLM_PROFILE_LAYER_IDS=mid \
NANOVLLM_PROMPT_LENGTHS=8200,9000,10000,11000,12000,13000,14000,15000,16000,17000 \
python3 example/test.py
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

## 2026-06-05 16:30：csrc / ut_ops 重构后的干净构建和 smoke test

这次同步了 `csrc` 扁平化目录、`ut_ops` 分目录和文档命名。下一次请在昇腾上先跑这个：

```bash
rm -rf build/nanovllm_ascend_ops
NANOVLLM_CANN_BUILD_JOBS=64 SOC_VERSION=ascend910_9391 PYTHONPATH=$PWD:$PYTHONPATH bash scripts/build_nanovllm_ops.sh
PYTHONPATH=$PWD:$PYTHONPATH ASCEND_RT_VISIBLE_DEVICES=0 python3 ut_ops/dsa/probe_indexer_score_bf16_out.py --device npu:0 --batch-size 10 --block-count 64 --warmup 10 --iters 100
PYTHONPATH=$PWD:$PYTHONPATH ASCEND_RT_VISIBLE_DEVICES=0 NANOVLLM_DSA_INDEXER_UPDATE_USE_CANN=1 python3 ut_ops/dsa/probe_indexer_update.py --device npu:0 --batch-size 10 --candidate-len 17000 --selected-len 2560 --k 128 --warmup 10 --iters 100
PYTHONPATH=$PWD:$PYTHONPATH ASCEND_RT_VISIBLE_DEVICES=0 python3 ut_ops/indexer_project/probe_query_only_torchair_accuracy.py --device npu:0 --tokens 10 --warmup 10 --iters 100
```

## 2026-06-06：切换为 lightning_indexer + gather_selection decode offload

本次把 decode 阶段旧的 `dsa_indexer_score + dsa_indexer_update + dsa_scatter_h2d`
替换为 `npu_lightning_indexer + npu_gather_selection_kv_cache`。下一次请在昇腾上先跑这个：

```bash
# 1. 先确认 gather_selection 实验算子已经可用
cd /home/w00916487/nanovllm-DSA/nano-vllm-ascend-DeepseekV32-ops_gather_selection_kv_cache
SOC_VERSION=ascend910_9391 GSKV_BUILD_JOBS=32 bash build_and_install.sh
ASCEND_RT_VISIBLE_DEVICES=0 bash run_probe.sh --device npu:0 --copy-mode cpu_to_hbm --batch-size 1 --seq-len 1 --topk 2048 --full-len 16384 --selection-topk-block-size 1 --warmup 10 --iters 100

# 2. 回到 nano-vllm 主仓，重新编译主仓算子
cd /home/w00916487/nanovllm-DSA/nano-vllm-ascend-DeepseekV32-dev_dsa_offload
export GSKV_PYTHONPATH=/home/w00916487/nanovllm-DSA/nano-vllm-ascend-DeepseekV32-ops_gather_selection_kv_cache/python_extension
export GSKV_OPP=${ASCEND_OPP_PATH:-/usr/local/Ascend/ascend-toolkit/latest/opp}
set +u
source ${GSKV_OPP}/vendors/customize/bin/set_env.bash
set -u
export ASCEND_CUSTOM_OPP_PATH=${GSKV_OPP}/vendors/customize${ASCEND_CUSTOM_OPP_PATH:+:${ASCEND_CUSTOM_OPP_PATH}}
rm -rf build/nanovllm_ascend_ops
NANOVLLM_CANN_BUILD_JOBS=64 NANOVLLM_EXT_BUILD_JOBS=1 SOC_VERSION=ascend910_9391 PYTHONPATH=$PWD:$PYTHONPATH bash scripts/build_nanovllm_ops.sh

# 3. 跑纯长序列 smoke test。注意新路径不再使用 NANOVLLM_DSA_OFFLOAD_FIXED_TX / NANOVLLM_DSA_INDEXER_UPDATE_USE_CANN。
PYTHONPATH=$PWD:$GSKV_PYTHONPATH:$PYTHONPATH \
NANOVLLM_MAX_GEN_TOKENS=8 \
NANOVLLM_ENABLE_DECODE_MLAPO=1 \
NANOVLLM_LOG_DECODE_LAYER_TIMING=1 \
NANOVLLM_DECODE_LAYER_TIMING_SYNC=1 \
NANOVLLM_PROFILE_LAYER_IDS=mid \
NANOVLLM_PROMPT_LENGTHS=8200,9000,10000,11000,12000,13000,14000,15000,16000,17000 \
python3 example/test.py

# 4. 再跑一把 timing sync=0，看端到端 TPOT。
PYTHONPATH=$PWD:$GSKV_PYTHONPATH:$PYTHONPATH \
NANOVLLM_MAX_GEN_TOKENS=8 \
NANOVLLM_ENABLE_DECODE_MLAPO=1 \
NANOVLLM_LOG_DECODE_LAYER_TIMING=1 \
NANOVLLM_DECODE_LAYER_TIMING_SYNC=0 \
NANOVLLM_PROFILE_LAYER_IDS=mid \
NANOVLLM_PROMPT_LENGTHS=8200,9000,10000,11000,12000,13000,14000,15000,16000,17000 \
python3 example/test.py
```
