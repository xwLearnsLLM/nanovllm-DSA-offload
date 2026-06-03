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

## 2026-06-02 23:02：对齐 vllm-ascend BF16 indexer weights 输入

下一次请在昇腾上先跑这个：

```bash
PYTHONPATH=$PWD:$PYTHONPATH python3 -m py_compile nanovllm/models/deepseek_v32.py nanovllm/models/dsa_indexer_project.py ut_ops/probe_indexer_project.py

PYTHONPATH=$PWD:$PYTHONPATH ASCEND_RT_VISIBLE_DEVICES=0 python3 ut_ops/probe_indexer_project.py --device npu:0 --tokens 4 --warmup 10 --iters 100

export DSA_DUMP=runlog/dsa_index_update_debug_$(date +%Y%m%d_%H%M%S)
PYTHONPATH=$PWD:$PYTHONPATH NANOVLLM_MAX_GEN_TOKENS=8 NANOVLLM_ENABLE_DECODE_MLAPO=0 NANOVLLM_DSA_OFFLOAD_FIXED_TX=128 NANOVLLM_DSA_DEBUG_INDEX_UPDATE_DUMP_PATH=$DSA_DUMP NANOVLLM_PROFILE_LAYER_IDS=mid NANOVLLM_PROMPT_LENGTHS=8192 python3 example/test.py

ls ${DSA_DUMP}_rank*.txt
```

## 2026-06-03 08:56：新增 lightning_indexer 与 dsa_indexer_score 的 topk 对齐单测

下一次请在昇腾上先跑这个：

```bash
PYTHONPATH=$PWD:$PYTHONPATH python3 -m py_compile ut_ops/compare_lightning_indexer_qk_score.py

PYTHONPATH=$PWD:$PYTHONPATH ASCEND_RT_VISIBLE_DEVICES=0 python3 ut_ops/compare_lightning_indexer_qk_score.py --device npu:0 --batch-size 1 --seq-len 8192 --topk 2048 --score-dtype bf16

PYTHONPATH=$PWD:$PYTHONPATH ASCEND_RT_VISIBLE_DEVICES=0 python3 ut_ops/compare_lightning_indexer_qk_score.py --device npu:0 --batch-size 1 --seq-len 8192 --topk 2048 --score-dtype fp32

PYTHONPATH=$PWD:$PYTHONPATH ASCEND_RT_VISIBLE_DEVICES=0 python3 ut_ops/compare_lightning_indexer_qk_score.py --device npu:0 --batch-size 4 --candidate-lens 8192,10000,12000,16384 --topk 2048 --score-dtype bf16
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

## 2026-06-02 21:18:30：修复 debug 校验索引设备不一致问题

下一次请在昇腾上先跑这个：

```bash
PYTHONPATH=$PWD:$PYTHONPATH python3 -m py_compile nanovllm/models/deepseek_v32.py

export DSA_DUMP=runlog/dsa_index_update_debug_$(date +%Y%m%d_%H%M%S)
PYTHONPATH=$PWD:$PYTHONPATH NANOVLLM_MAX_GEN_TOKENS=4 NANOVLLM_ENABLE_DECODE_MLAPO=0 NANOVLLM_DSA_OFFLOAD_FIXED_TX=128 NANOVLLM_DSA_DEBUG_INDEX_UPDATE_DUMP_PATH=$DSA_DUMP NANOVLLM_PROFILE_LAYER_IDS=mid NANOVLLM_PROMPT_LENGTHS=8192 python3 example/test.py

ls ${DSA_DUMP}_rank*.txt
```

## 2026-06-02 22:05:29：修复 prefill HBM 到 DRAM 阻塞拷贝，并补充 score 输入 NaN 诊断

下一次请在昇腾上先跑这个：

```bash
PYTHONPATH=$PWD:$PYTHONPATH python3 -m py_compile nanovllm/models/deepseek_v32.py

export DSA_DUMP=runlog/dsa_index_update_debug_$(date +%Y%m%d_%H%M%S)
PYTHONPATH=$PWD:$PYTHONPATH NANOVLLM_MAX_GEN_TOKENS=4 NANOVLLM_ENABLE_DECODE_MLAPO=0 NANOVLLM_DSA_OFFLOAD_FIXED_TX=128 NANOVLLM_DSA_DEBUG_INDEX_UPDATE_DUMP_PATH=$DSA_DUMP NANOVLLM_PROFILE_LAYER_IDS=mid NANOVLLM_PROMPT_LENGTHS=8192 python3 example/test.py

ls ${DSA_DUMP}_rank*.txt
```

## 2026-06-02 21:52:45：补充 score 摘要和 prefill DRAM materialize 校验

下一次请在昇腾上先跑这个：

```bash
PYTHONPATH=$PWD:$PYTHONPATH python3 -m py_compile nanovllm/models/deepseek_v32.py

export DSA_DUMP=runlog/dsa_index_update_debug_$(date +%Y%m%d_%H%M%S)
PYTHONPATH=$PWD:$PYTHONPATH NANOVLLM_MAX_GEN_TOKENS=4 NANOVLLM_ENABLE_DECODE_MLAPO=0 NANOVLLM_DSA_OFFLOAD_FIXED_TX=128 NANOVLLM_DSA_DEBUG_INDEX_UPDATE_DUMP_PATH=$DSA_DUMP NANOVLLM_PROFILE_LAYER_IDS=mid NANOVLLM_PROMPT_LENGTHS=8192 python3 example/test.py

ls ${DSA_DUMP}_rank*.txt
```
