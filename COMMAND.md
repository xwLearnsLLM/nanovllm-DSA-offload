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

## 2026-06-04 17:36：清理运行时开关并固化默认性能路径

这次删除了旧的 BF16 score 开关、`NANOVLLM_FUSE_QKV_A`、`NANOVLLM_FREE_KV_B_PROJ`、`NANOVLLM_DEBUG_DECODE_MLAPO_COMPARE*` 相关代码；`NANOVLLM_DSA_OFFLOAD_FIXED_TX` 默认改为 128，`example/test.py` 的 `NANOVLLM_MAX_MODEL_LEN` 默认改为 65536。

下一次请在昇腾上先跑这个：

```bash
PYTHONPATH=$PWD:$PYTHONPATH NANOVLLM_DSA_QUERY_ONLY_BACKEND=torchair NANOVLLM_DSA_QUERY_ONLY_WARMUP_TOKENS=10 NANOVLLM_MAX_GEN_TOKENS=8 NANOVLLM_ENABLE_DECODE_MLAPO=1 NANOVLLM_DSA_INDEXER_UPDATE_USE_CANN=1 NANOVLLM_DSA_CHECK=0 NANOVLLM_PROMPT_LENGTHS=17000,17001,17002,17003,17004,17005,17006,17007,17008,17009 NANOVLLM_LOG_DECODE_LAYER_TIMING=0 NANOVLLM_DECODE_LAYER_TIMING_SYNC=0 NANOVLLM_PROFILE_LAYER_IDS=mid python3 example/test.py
```

## 2026-06-05 13:06：整理 csrc 目录并统一 DSA indexer 命名

这次把 `csrc/nanovllm_ascend_ops` 下的绑定、公共代码和算子目录重新整理为 `bindings/`、`common/`、`ops/`，并把框架侧 update 接口统一成 `dsa_indexer_update`，把 score 绑定改成 `npu_dsa_indexer_score*`。底层 CANN 内部 op 名仍保持原注册名，所以需要重新编译。

下一次请在昇腾上先跑这个：

```bash
NANOVLLM_CANN_BUILD_JOBS=64 SOC_VERSION=ascend910_9391 PYTHONPATH=$PWD:$PYTHONPATH bash scripts/build_nanovllm_ops.sh
```

## 2026-06-05 15:19：整理 ut_ops 单测目录

这次删除了历史 SFA/prefill/老 indexer 探针，把仍在使用的单测整理到 `ut_ops/dsa/`、`ut_ops/indexer_project/`、`ut_ops/mla/`、`ut_ops/moe/`，并抽出 `ut_ops/common/` 公共工具。下一次请在昇腾上先跑这个：

```bash
PYTHONPATH=$PWD:$PYTHONPATH ASCEND_RT_VISIBLE_DEVICES=0 python3 ut_ops/dsa/probe_indexer_score_bf16_out.py --device npu:0 --batch-size 10 --block-count 64 --warmup 10 --iters 100
PYTHONPATH=$PWD:$PYTHONPATH ASCEND_RT_VISIBLE_DEVICES=0 NANOVLLM_DSA_INDEXER_UPDATE_USE_CANN=1 python3 ut_ops/dsa/probe_indexer_update.py --device npu:0 --batch-size 10 --candidate-len 17000 --selected-len 2560 --k 128 --warmup 10 --iters 100
PYTHONPATH=$PWD:$PYTHONPATH ASCEND_RT_VISIBLE_DEVICES=0 python3 ut_ops/indexer_project/probe_query_only_torchair_accuracy.py --device npu:0 --tokens 10 --warmup 10 --iters 100
```

然后跑两个 renamed op 的最小 probe：

```bash
PYTHONPATH=$PWD:$PYTHONPATH ASCEND_RT_VISIBLE_DEVICES=0 python3 ut_ops/dsa/probe_indexer_score_bf16_out.py --device npu:0 --batch-size 10 --block-count 64 --warmup 10 --iters 100
PYTHONPATH=$PWD:$PYTHONPATH ASCEND_RT_VISIBLE_DEVICES=0 NANOVLLM_DSA_INDEXER_UPDATE_USE_CANN=1 python3 ut_ops/dsa/probe_indexer_update.py --device npu:0 --batch-size 10 --candidate-len 17000 --selected-len 2560 --k 128 --warmup 10 --iters 100
```

## 2026-06-05 14:22：修复 csrc 扁平化后的 kernel include 路径

这次修复 `common/kernels` 移动后残留的旧 include 路径。下一次请在昇腾上重新跑编译：

```bash
NANOVLLM_CANN_BUILD_JOBS=64 SOC_VERSION=ascend910_9391 PYTHONPATH=$PWD:$PYTHONPATH bash scripts/build_nanovllm_ops.sh
```
