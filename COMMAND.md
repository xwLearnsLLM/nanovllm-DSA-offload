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

## 2026-06-07 00:11：尝试把 TorchAir 图边界扩到 DSA 小流水

下一次请在昇腾上先跑这个：

```bash
python3 -m py_compile nanovllm/models/dsa_indexer_project.py nanovllm/models/deepseek_v32.py ut_ops/indexer_project/probe_dsa_pipeline_torchair.py

PYTHONPATH=$PWD:$PYTHONPATH ASCEND_RT_VISIBLE_DEVICES=0 python3 ut_ops/indexer_project/probe_dsa_pipeline_torchair.py --device npu:0 --batch-size 10 --pool-capacity 256 --full-len 16384 --topk 2048 --block-size 128 --warmup 10 --iters 100

PYTHONPATH=$PWD:$PYTHONPATH NANOVLLM_DSA_QUERY_ONLY_BACKEND=torchair NANOVLLM_MAX_GEN_TOKENS=16 NANOVLLM_PROMPT_LENGTHS=11000,11100,11200,11300,11000,11100,11200,11300,11000,11100 python3 example/test.py
```

## 2026-06-08 00:31：给 DSA pybind custom op 添加 allow_in_graph 后重测小流水组图

下一次请在昇腾上先跑这个：

```bash
python3 -m py_compile nanovllm/models/dsa_indexer_project.py nanovllm/models/deepseek_v32.py ut_ops/indexer_project/probe_dsa_pipeline_torchair.py

PYTHONPATH=$PWD:$PYTHONPATH ASCEND_RT_VISIBLE_DEVICES=0 python3 ut_ops/indexer_project/probe_dsa_pipeline_torchair.py --device npu:0 --batch-size 10 --pool-capacity 256 --full-len 16384 --topk 2048 --block-size 128 --warmup 10 --iters 100

PYTHONPATH=$PWD:$PYTHONPATH NANOVLLM_DSA_QUERY_ONLY_BACKEND=torchair NANOVLLM_LOG_DECODE_LAYER_TIMING=1 NANOVLLM_DECODE_LAYER_TIMING_SYNC=1 NANOVLLM_PROFILE_LAYER_IDS=mid NANOVLLM_MAX_GEN_TOKENS=16 NANOVLLM_PROMPT_LENGTHS=11000,11100,11200,11300,11000,11100,11200,11300,11000,11100 python3 example/test.py
```

## 2026-06-08 00:42：用 torch.library custom op fake/meta 重新验证 DSA 小流水组图

下一次请在昇腾上先跑这个：

```bash
python3 -m py_compile nanovllm/models/dsa_indexer_project.py nanovllm/models/deepseek_v32.py ut_ops/indexer_project/probe_dsa_pipeline_torchair.py

PYTHONPATH=$PWD:$PYTHONPATH ASCEND_RT_VISIBLE_DEVICES=0 python3 ut_ops/indexer_project/probe_dsa_pipeline_torchair.py --device npu:0 --batch-size 10 --pool-capacity 256 --full-len 16384 --topk 2048 --block-size 128 --warmup 10 --iters 100
```

## 2026-06-08 00:56：补齐 LightningIndexer TorchAir AscendIR 13 参数后重跑小流水单测

下一次请在昇腾上先跑这个：

```bash
python3 -m py_compile nanovllm/models/dsa_indexer_project.py nanovllm/models/deepseek_v32.py ut_ops/indexer_project/probe_dsa_pipeline_torchair.py

PYTHONPATH=$PWD:$PYTHONPATH ASCEND_RT_VISIBLE_DEVICES=0 python3 ut_ops/indexer_project/probe_dsa_pipeline_torchair.py --device npu:0 --batch-size 10 --pool-capacity 256 --full-len 16384 --topk 2048 --block-size 128 --warmup 10 --iters 100
```

## 2026-06-08 01:22：把 lightning/gather 从 Python custom op 改成 C++ TORCH_LIBRARY 注册

下一次请在昇腾上先跑这个：

```bash
NANOVLLM_CANN_BUILD_JOBS=64 NANOVLLM_EXT_BUILD_JOBS=1 SOC_VERSION=ascend910_9391 PYTHONPATH=$PWD:$PYTHONPATH bash scripts/build_nanovllm_ops.sh

python3 -m py_compile nanovllm/models/dsa_indexer_project.py nanovllm/models/deepseek_v32.py ut_ops/indexer_project/probe_dsa_pipeline_torchair.py

PYTHONPATH=$PWD:$PYTHONPATH ASCEND_RT_VISIBLE_DEVICES=0 python3 ut_ops/indexer_project/probe_dsa_pipeline_torchair.py --device npu:0 --batch-size 10 --pool-capacity 256 --full-len 16384 --topk 2048 --block-size 128 --warmup 10 --iters 100

PYTHONPATH=$PWD:$PYTHONPATH NANOVLLM_DSA_QUERY_ONLY_BACKEND=torchair NANOVLLM_LOG_DECODE_LAYER_TIMING=1 NANOVLLM_DECODE_LAYER_TIMING_SYNC=1 NANOVLLM_PROFILE_LAYER_IDS=mid NANOVLLM_MAX_GEN_TOKENS=16 NANOVLLM_PROMPT_LENGTHS=11000,11100,11200,11300,11000,11100,11200,11300,11000,11100 python3 example/test.py
```

## 2026-06-08 01:35：把 C++ lightning_indexer 注册改成单元素 tuple 返回后重跑

下一次请在昇腾上先跑这个：

```bash
NANOVLLM_CANN_BUILD_JOBS=64 NANOVLLM_EXT_BUILD_JOBS=1 SOC_VERSION=ascend910_9391 PYTHONPATH=$PWD:$PYTHONPATH bash scripts/build_nanovllm_ops.sh

python3 -m py_compile nanovllm/models/dsa_indexer_project.py nanovllm/models/deepseek_v32.py ut_ops/indexer_project/probe_dsa_pipeline_torchair.py

PYTHONPATH=$PWD:$PYTHONPATH ASCEND_RT_VISIBLE_DEVICES=0 python3 ut_ops/indexer_project/probe_dsa_pipeline_torchair.py --device npu:0 --batch-size 10 --pool-capacity 256 --full-len 16384 --topk 2048 --block-size 128 --warmup 10 --iters 100
```

## 2026-06-08 10:20：给 TorchAir DSA 小流水增加 C++ op schema 版本检查后重跑

下一次请在昇腾上先跑这个：

```bash
NANOVLLM_CANN_BUILD_JOBS=64 NANOVLLM_EXT_BUILD_JOBS=1 SOC_VERSION=ascend910_9391 PYTHONPATH=$PWD:$PYTHONPATH bash scripts/build_nanovllm_ops.sh

python3 -m py_compile nanovllm/models/dsa_indexer_project.py nanovllm/models/deepseek_v32.py ut_ops/indexer_project/probe_dsa_pipeline_torchair.py

PYTHONPATH=$PWD:$PYTHONPATH ASCEND_RT_VISIBLE_DEVICES=0 python3 ut_ops/indexer_project/probe_dsa_pipeline_torchair.py --device npu:0 --batch-size 10 --pool-capacity 256 --full-len 16384 --topk 2048 --block-size 128 --warmup 10 --iters 100
```

## 2026-06-08 10:42：把 lightning_indexer Torch op 改成双 Tensor 返回后重跑

下一次请在昇腾上先跑这个：

```bash
NANOVLLM_CANN_BUILD_JOBS=64 NANOVLLM_EXT_BUILD_JOBS=1 SOC_VERSION=ascend910_9391 PYTHONPATH=$PWD:$PYTHONPATH bash scripts/build_nanovllm_ops.sh

python3 -m py_compile nanovllm/models/dsa_indexer_project.py nanovllm/models/deepseek_v32.py ut_ops/indexer_project/probe_dsa_pipeline_torchair.py

PYTHONPATH=$PWD:$PYTHONPATH ASCEND_RT_VISIBLE_DEVICES=0 python3 ut_ops/indexer_project/probe_dsa_pipeline_torchair.py --device npu:0 --batch-size 10 --pool-capacity 256 --full-len 16384 --topk 2048 --block-size 128 --warmup 10 --iters 100
```

## 2026-06-08 11:00：把 gather_selection mutable op 改成返回被修改的三个 Tensor 后重跑

下一次请在昇腾上先跑这个：

```bash
NANOVLLM_CANN_BUILD_JOBS=64 NANOVLLM_EXT_BUILD_JOBS=1 SOC_VERSION=ascend910_9391 PYTHONPATH=$PWD:$PYTHONPATH bash scripts/build_nanovllm_ops.sh

python3 -m py_compile nanovllm/models/dsa_indexer_project.py nanovllm/models/deepseek_v32.py ut_ops/indexer_project/probe_dsa_pipeline_torchair.py

PYTHONPATH=$PWD:$PYTHONPATH ASCEND_RT_VISIBLE_DEVICES=0 python3 ut_ops/indexer_project/probe_dsa_pipeline_torchair.py --device npu:0 --batch-size 10 --pool-capacity 256 --full-len 16384 --topk 2048 --block-size 128 --warmup 10 --iters 100
```

## 2026-06-08 11:13：去掉 gather_selection 返回值 alias annotation 后重跑

下一次请在昇腾上先跑这个：

```bash
NANOVLLM_CANN_BUILD_JOBS=64 NANOVLLM_EXT_BUILD_JOBS=1 SOC_VERSION=ascend910_9391 PYTHONPATH=$PWD:$PYTHONPATH bash scripts/build_nanovllm_ops.sh

python3 -m py_compile nanovllm/models/dsa_indexer_project.py nanovllm/models/deepseek_v32.py ut_ops/indexer_project/probe_dsa_pipeline_torchair.py

PYTHONPATH=$PWD:$PYTHONPATH ASCEND_RT_VISIBLE_DEVICES=0 python3 ut_ops/indexer_project/probe_dsa_pipeline_torchair.py --device npu:0 --batch-size 10 --pool-capacity 256 --full-len 16384 --topk 2048 --block-size 128 --warmup 10 --iters 100
```

## 2026-06-08 11:28：把 gather_selection 图内 schema 改成函数式返回并显式使用返回值后重跑

清理 build

```bash
rm -rf build/nanovllm_ascend_ops
rm -rf nanovllm/_C*.so nanovllm/libnanovllm_ascend_kernels.so
rm -rf nanovllm/_cann_ops_custom
```

下一次请在昇腾上先跑这个：

```bash
NANOVLLM_CANN_BUILD_JOBS=64 NANOVLLM_EXT_BUILD_JOBS=1 SOC_VERSION=ascend910_9391 PYTHONPATH=$PWD:$PYTHONPATH bash scripts/build_nanovllm_ops.sh

python3 -m py_compile nanovllm/models/dsa_indexer_project.py nanovllm/models/deepseek_v32.py ut_ops/indexer_project/probe_dsa_pipeline_torchair.py

PYTHONPATH=$PWD:$PYTHONPATH ASCEND_RT_VISIBLE_DEVICES=0 python3 ut_ops/indexer_project/probe_dsa_pipeline_torchair.py --device npu:0 --batch-size 10 --pool-capacity 256 --full-len 16384 --topk 2048 --block-size 128 --warmup 10 --iters 100
```
## 2026-06-08 11:23：给 gather_selection_kv_cache 注册显式 TorchAir converter 后重跑

下一次请在昇腾上先跑这个：

```bash
python3 -m py_compile nanovllm/models/dsa_indexer_project.py nanovllm/models/deepseek_v32.py ut_ops/indexer_project/probe_dsa_pipeline_torchair.py
```

然后 clean 并重新编译算子：

```bash
rm -rf build/nanovllm_ascend_ops
rm -rf csrc/nanovllm_ascend_ops/build csrc/nanovllm_ascend_ops/dist csrc/nanovllm_ascend_ops/*.egg-info
rm -rf nanovllm/_C*.so nanovllm/libnanovllm_ascend_kernels.so nanovllm/_cann_ops_custom
NANOVLLM_EXT_BUILD_JOBS=1 SOC_VERSION=ascend910_9391 PYTHONPATH=$PWD:$PYTHONPATH bash scripts/build_nanovllm_ops.sh
```

最后重跑 DSA 小流水 TorchAir 单测：

```bash
PYTHONPATH=$PWD:$PYTHONPATH ASCEND_RT_VISIBLE_DEVICES=0 python3 ut_ops/indexer_project/probe_dsa_pipeline_torchair.py --device npu:0 --batch-size 10 --pool-capacity 256 --full-len 16384 --topk 2048 --block-size 128 --warmup 10 --iters 100
```
## 2026-06-08 12:58：恢复 gather_selection 的 block_table 引用输出后重跑

下一次请在昇腾上先跑这个：

```bash
python3 -m py_compile nanovllm/models/dsa_indexer_project.py nanovllm/models/deepseek_v32.py ut_ops/indexer_project/probe_dsa_pipeline_torchair.py
```

然后 clean 并重新编译算子：

```bash
rm -rf build/nanovllm_ascend_ops
rm -rf csrc/nanovllm_ascend_ops/build csrc/nanovllm_ascend_ops/dist csrc/nanovllm_ascend_ops/*.egg-info
rm -rf nanovllm/_C*.so nanovllm/libnanovllm_ascend_kernels.so nanovllm/_cann_ops_custom
NANOVLLM_EXT_BUILD_JOBS=1 SOC_VERSION=ascend910_9391 PYTHONPATH=$PWD:$PYTHONPATH bash scripts/build_nanovllm_ops.sh
```

最后重跑 DSA 小流水 TorchAir 单测：

```bash
PYTHONPATH=$PWD:$PYTHONPATH ASCEND_RT_VISIBLE_DEVICES=0 python3 ut_ops/indexer_project/probe_dsa_pipeline_torchair.py --device npu:0 --batch-size 10 --pool-capacity 256 --full-len 16384 --topk 2048 --block-size 128 --warmup 10 --iters 100
```
## 2026-06-08 13:20：TorchAir 小流水改用 BSND LightningIndexer，去掉 gather 前的 reshape

这次只改 Python，不需要重新编译算子。下一次请在昇腾上先跑这个：

```bash
python3 -m py_compile nanovllm/models/dsa_indexer_project.py nanovllm/models/deepseek_v32.py ut_ops/indexer_project/probe_dsa_pipeline_torchair.py
```

然后重跑 DSA 小流水 TorchAir 单测：

```bash
PYTHONPATH=$PWD:$PYTHONPATH ASCEND_RT_VISIBLE_DEVICES=0 python3 ut_ops/indexer_project/probe_dsa_pipeline_torchair.py --device npu:0 --batch-size 10 --pool-capacity 256 --full-len 16384 --topk 2048 --block-size 128 --warmup 10 --iters 100
```
## 2026-06-08 15:38：给 LightningIndexer 也注册显式 TorchAir converter 后重跑

这次只改 Python，不需要重新编译算子。下一次请在昇腾上先跑这个：

```bash
python3 -m py_compile nanovllm/models/dsa_indexer_project.py nanovllm/models/deepseek_v32.py ut_ops/indexer_project/probe_dsa_pipeline_torchair.py
```

然后重跑 DSA 小流水 TorchAir 单测：

```bash
PYTHONPATH=$PWD:$PYTHONPATH ASCEND_RT_VISIBLE_DEVICES=0 python3 ut_ops/indexer_project/probe_dsa_pipeline_torchair.py --device npu:0 --batch-size 10 --pool-capacity 256 --full-len 16384 --topk 2048 --block-size 128 --warmup 10 --iters 100
```
