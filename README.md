# nano-vLLM Ascend DSA offload

本仓库只面向两类模型：

- DeepSeek V3.2 BF16（包括 32 专家裁剪模型和 128 专家 REAP 模型）。
- GLM-5.1-w4a8。Routed experts 保持 ModelSlim W4A8；Attention、dense/shared MLP 的 W8A8 权重在加载时反量化为 BF16。

运行时只保留两种模式：

| `NANOVLLM_ENFORCE_EAGER` | Prefill | 首个 decode | 后续稳定 decode |
| --- | --- | --- | --- |
| `1` | eager | eager | eager |
| `0` | eager | eager | `FULL_DECODE_ONLY` |

LM head 和 sampler 始终在图外。整图只捕获已进入 DSA offload、batch size 与 capture size 完全一致的稳定 decode；首个 decode、短序列、混合 batch 和未捕获 batch 均走 eager。DeepSeek 使用 `npugraph_ex + outer ACLGraph`，GLM 使用 raw outer ACLGraph，因此 GLM 的最终 proof 显示 `npugraph_ex=False` 是正常的。

## 安装与算子编译

Python 依赖见 `requirements.txt`。DeepSeek 原始 FP8 权重需先转换成 BF16；仓库保留以下正式转换入口：

- `scripts/export_deepseek_v32_to_hf_bf16.py`
- `scripts/export_deepseek_v32_pruned.py`
- `scripts/export_deepseek_v32_keep_routed_experts.py`

在昇腾机器的仓库根目录编译自定义算子：

```bash
export PYTHONUNBUFFERED=1
export PYTHONPATH=$PWD:$PYTHONPATH
export NANOVLLM_CANN_BUILD_JOBS=64
export SOC_VERSION=ascend910_9391

bash scripts/build_nanovllm_ops.sh
ls -lh nanovllm/_C*.so nanovllm/libnanovllm_ascend_kernels.so
```

GatherSelection 会对当前支持的 BF16、top-k 2048、block size 128 配置自动使用 all-core 并行 copy tiling，不需要运行时开关。

## DeepSeek V3.2：稳定 decode 整图

下面是 128 专家 BF16、TP16+EP16、batch 6、约 30K token 的完整命令：

```bash
cd /home/w00916487/nanovllm-dsa_offload

unset NANOVLLM_PROFILE_DECODE_OUTPUT

export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export ASCEND_LAUNCH_BLOCKING=0
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
export PYTHONUNBUFFERED=1
export PYTHONPATH=$PWD:$PYTHONPATH

export NANOVLLM_MODEL=/home/models/DeepSeek-V3.2-REAP-345B-A37B-BF16/
export NANOVLLM_TP_SIZE=16
export NANOVLLM_ENABLE_EXPERT_PARALLEL=1
export NANOVLLM_ENFORCE_EAGER=0
export NANOVLLM_KVCACHE_BLOCK_SIZE=128
export NANOVLLM_HBM_NUM_BLOCKS=350
export NANOVLLM_DRAM_NUM_BLOCKS=1500
export NANOVLLM_PREFILL_CHUNK_SIZE=1024
export NANOVLLM_IGNORE_EOS=1
export NANOVLLM_MAX_GEN_TOKENS=16
export NANOVLLM_PROMPT_LENGTHS=30000,30001,30002,30003,30004,30005

python3 example/test.py
```

结束时应看到类似：

```text
DSA FULL_DECODE_ONLY proof: capture_sizes=[6], npugraph_ex=True, captures=1, replays>0, eager_first_decode=1, eager_no_dsa=0, eager_mixed_batch=0, eager_uncaptured_batch=0
```

只需把 `NANOVLLM_ENFORCE_EAGER=1` 即可做同配置 eager 对照。`example/test.py` 会把 `max_model_len` 设置为最长 prompt 加生成长度，并把 prompt 条数同时作为 decode batch 上限和唯一 capture size。

## GLM-5.1-w4a8：短序列 eager smoke

```bash
cd /home/w00916487/nanovllm-dsa_offload

unset NANOVLLM_PROFILE_DECODE_OUTPUT

export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export ASCEND_LAUNCH_BLOCKING=0
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
export PYTHONUNBUFFERED=1
export PYTHONPATH=$PWD:$PYTHONPATH

export NANOVLLM_MODEL=/mnt/models/GLM-5.1-w4a8/
export NANOVLLM_TP_SIZE=16
export NANOVLLM_ENABLE_EXPERT_PARALLEL=1
export NANOVLLM_ENFORCE_EAGER=1
export NANOVLLM_KVCACHE_BLOCK_SIZE=128
export NANOVLLM_HBM_NUM_BLOCKS=64
export NANOVLLM_DRAM_NUM_BLOCKS=64
export NANOVLLM_PREFILL_CHUNK_SIZE=0
export NANOVLLM_MAX_MODEL_LEN=512
export NANOVLLM_MAX_GEN_TOKENS=8
export NANOVLLM_IGNORE_EOS=0

python3 example/glm_short_prompts.py
```

前两个回答应分别为“北京”和“14”。短序列不触发 DSA offload，也不用于整图验收。

## GLM-5.1-w4a8：长序列稳定 decode 整图

```bash
cd /home/w00916487/nanovllm-dsa_offload

unset NANOVLLM_PROFILE_DECODE_OUTPUT

export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export ASCEND_LAUNCH_BLOCKING=0
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
export PYTHONUNBUFFERED=1
export PYTHONPATH=$PWD:$PYTHONPATH

export NANOVLLM_MODEL=/mnt/models/GLM-5.1-w4a8/
export NANOVLLM_TP_SIZE=16
export NANOVLLM_ENABLE_EXPERT_PARALLEL=1
export NANOVLLM_ENFORCE_EAGER=0
export NANOVLLM_KVCACHE_BLOCK_SIZE=128
export NANOVLLM_HBM_NUM_BLOCKS=96
export NANOVLLM_DRAM_NUM_BLOCKS=128
export NANOVLLM_PREFILL_CHUNK_SIZE=1024
export NANOVLLM_IGNORE_EOS=1
export NANOVLLM_MAX_GEN_TOKENS=8
export NANOVLLM_PROMPT_LENGTHS=8200

python3 example/test.py
```

结束时应满足：

```text
DSA FULL_DECODE_ONLY proof: capture_sizes=[1], npugraph_ex=False, captures=1, replays>0, eager_first_decode=1, eager_no_dsa=0, eager_mixed_batch=0, eager_uncaptured_batch=0
```

同一配置分别以 `NANOVLLM_ENFORCE_EAGER=1` 和 `0` 运行时，生成 token IDs 应一致。

## 只采集 TP rank 0 的 decode profile

Profiler 在 rank 0 的第一次 decode forward 前启动，在 `generate()` 结束后停止；不会采集 prefill，也不会采集其他 TP rank。以下是完整的 DeepSeek 示例：

```bash
cd /home/w00916487/nanovllm-dsa_offload

unset PROFILING_MODE
unset PROFILING_OPTIONS
unset PROF_CONFIG_PATH

export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export ASCEND_LAUNCH_BLOCKING=0
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
export PYTHONUNBUFFERED=1
export PYTHONPATH=$PWD:$PYTHONPATH

export NANOVLLM_MODEL=/home/models/DeepSeek-V3.2-REAP-345B-A37B-BF16/
export NANOVLLM_TP_SIZE=16
export NANOVLLM_ENABLE_EXPERT_PARALLEL=1
export NANOVLLM_ENFORCE_EAGER=0
export NANOVLLM_KVCACHE_BLOCK_SIZE=128
export NANOVLLM_HBM_NUM_BLOCKS=350
export NANOVLLM_DRAM_NUM_BLOCKS=1500
export NANOVLLM_PREFILL_CHUNK_SIZE=1024
export NANOVLLM_IGNORE_EOS=1
export NANOVLLM_MAX_GEN_TOKENS=6
export NANOVLLM_PROMPT_LENGTHS=30000,30001,30002,30003,30004,30005
export NANOVLLM_PROFILE_DECODE_OUTPUT=$PWD/profile_rank0_decode

python3 example/test.py
```

关闭采集执行 `unset NANOVLLM_PROFILE_DECODE_OUTPUT`。

## 正式环境变量

| 变量 | 说明 |
| --- | --- |
| `NANOVLLM_MODEL` | DeepSeek V3.2 BF16 或 GLM-5.1-w4a8 模型目录 |
| `NANOVLLM_TP_SIZE` | Tensor parallel 大小 |
| `NANOVLLM_ENABLE_EXPERT_PARALLEL` | 当前两类大模型均推荐设为 `1` |
| `NANOVLLM_ENFORCE_EAGER` | `1` 为全 eager；`0` 为稳定 decode 的 `FULL_DECODE_ONLY` |
| `NANOVLLM_KVCACHE_BLOCK_SIZE` | KV block size，当前推荐并验证值为 `128` |
| `NANOVLLM_HBM_NUM_BLOCKS` | HBM KV block 数量，必须大于 2 |
| `NANOVLLM_DRAM_NUM_BLOCKS` | DRAM KV block 数量，同时决定 HBM IndexCache 容量，必须大于 2 |
| `NANOVLLM_PREFILL_CHUNK_SIZE` | 只允许 `0` 或 `1024`；`1024` 强制单请求纯 prefill chunk |
| `NANOVLLM_PROMPT_LENGTHS` | `example/test.py` 的精确 token 长度列表，条目数即 batch size |
| `NANOVLLM_MAX_GEN_TOKENS` | 每个请求最多生成的 token 数 |
| `NANOVLLM_IGNORE_EOS` | 是否忽略 EOS，性能测试通常设为 `1` |
| `NANOVLLM_PROFILE_DECODE_OUTPUT` | 非空时只采集 TP rank 0 的 decode profile，并写入指定目录 |

Chunk prefill 只降低 prefill 激活峰值；完整请求所需的 HBM KV、DRAM KV 和 IndexCache 容量仍会预先分配。当前没有 prefill/decode 混合 forward。

算子语义与性能 UT 见 `ut_ops/UT_OPS.md`，vLLM-Ascend 的 GLM 对照脚本见 `compare_to_vllm/README.md`。
