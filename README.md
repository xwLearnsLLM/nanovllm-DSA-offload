# nano-vLLM Ascend：DeepSeek V3.2 / GLM-5.1

本仓库面向以下两类模型：

- DeepSeek V3.2 BF16。
- GLM-5.1-w4a8。Routed experts 保持 ModelSlim W4A8；Attention、dense/shared MLP 的 W8A8 权重在加载时反量化为 BF16。

引擎提供三种互斥的 decode KV 模式：

| `offload_mode` / `NANOVLLM_OFFLOAD_MODE` | 行为 |
| --- | --- |
| `none` | 不卸载，完整 MLA KV 保留在 HBM；默认值 |
| `gs` | LightningIndexer + GatherSelectionKVCache |
| `lidu` | LightningIndexerDecodeUpdate + KvcacheScatterCopy |

旧参数 `enable_dsa_offload` 和环境变量 `NANOVLLM_ENABLE_DSA_OFFLOAD` 已删除。

执行方式只保留 eager 和 `FULL_DECODE_ONLY`：

| `NANOVLLM_ENFORCE_EAGER` | Prefill | 第一个 decode | 后续稳定 decode |
| --- | --- | --- | --- |
| `1` | eager | eager | eager |
| `0` | eager | eager | `FULL_DECODE_ONLY` |

LM head 和 sampler 始终在图外。首次 LIDU 缓存初始化也始终 eager；初始化完成后的下一次稳定 decode 才能 replay。DeepSeek 使用 `npugraph_ex + outer ACLGraph`，GLM 使用 raw outer ACLGraph，因此 GLM proof 中 `npugraph_ex=False` 是正常现象。

## LIDU + SCATTER 语义

缓存预算 C 由原始 prompt 长度固定，生成过程中不再改变：

| 原始 prompt 长度 | C |
| --- | ---: |
| `<= 2048` | 0 |
| `2049–8192` | 2048 |
| `8193–16384` | 3072 |
| `16385–32768` | 5120 |
| `32769–65536` | 8192 |
| `>= 65537` | 12288 |

稀疏 source 只包含原始 prompt 的完整 128-token blocks。prompt 末尾非满块和所有 decode token 始终留在 dense tail，不参与 LIDU 选择或 SCATTER 搬移。Attention 对 C 个缓存 token、prompt tail 和所有 decode token 做 dense MLA，因此真正的 top-2048 一定参与计算且不会重复。

每层维护持久化 request pool。`req_pool_entries[b]` 将当前 batch 行映射到该请求的状态行；batch 重排不搬移状态。首次 decode 分块计算 top-C 并初始化 HBM，稳定 decode 由 LIDU 融合 top-2048、hit/miss、eviction 和索引更新，再由 SCATTER 只搬运 miss token。

## 本仓库自包含算子

LIDU 与 SCATTER 的 host、tiling、AscendC kernel、ACLNN adapter、`torch.library` schema、Meta/Fake 路径和 TorchAir converter 均已放入本仓库：

- `csrc/nanovllm_ascend_ops/ops/lightning_indexer_decode_update/`
- `csrc/nanovllm_ascend_ops/ops/kvcache_scatter_copy/`

编译和运行不会从参考工程导入、链接或动态加载这两个算子；它们只依赖本仓库代码和标准 CANN/PyTorch-NPU SDK。参考目录仅用于设计对照。

## 编译

新增了 Ascend 自定义算子，更新代码后必须重新编译：

```bash
cd /home/w00916487/nanovllm-dsa_offload

export PYTHONUNBUFFERED=1
export PYTHONPATH=$PWD:$PYTHONPATH
export NANOVLLM_CANN_BUILD_JOBS=64
export SOC_VERSION=ascend910_9391

bash scripts/build_nanovllm_ops.sh
ls -lh nanovllm/_C*.so nanovllm/libnanovllm_ascend_kernels.so
```

## 先运行 LIDU + SCATTER 单卡 UT

必须先通过算子语义测试，再运行 nano-vLLM 推理。该 UT 覆盖 32/64 heads、所有 C 档位、C=0、乱序 request-pool entries、随机 block tables、初始化 C-copy、稳定 miss-copy、重复零 miss，以及同输入 GS/LIDU 链路时延。

```bash
cd /home/w00916487/nanovllm-dsa_offload

unset NANOVLLM_GS_MISS_RATE_ON_LAYERS
unset NANOVLLM_PROFILE_DECODE_OUTPUT

export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export ASCEND_LAUNCH_BLOCKING=0
export ASCEND_RT_VISIBLE_DEVICES=4
export PYTHONUNBUFFERED=1
export PYTHONPATH=$PWD:$PYTHONPATH

python3 ut_ops/test_lidu_scatter.py \
  --device npu:0 \
  --heads 32,64 \
  --seed 7 \
  --warmup 2 \
  --iters 10
```

最终成功标志是 `LIDU_SCATTER_UT_OK`。`LIDU_GS_COMPARE` 只报告时延，不把 LIDU 必须快于 GS 设为正确性条件。

CPU 状态机测试：

```bash
cd /home/w00916487/nanovllm-dsa_offload

export PYTHONUNBUFFERED=1
export PYTHONPATH=$PWD:$PYTHONPATH

python3 -m pytest -q \
  tests/test_lidu_offload.py \
  tests/test_chunked_prefill.py \
  tests/test_full_decode_graph.py \
  tests/test_glm_dsa_offload.py
```

## DeepSeek V3.2：LIDU 整图

以下是 TP16+EP16、batch 6、约 30K token 的完整命令：

```bash
cd /home/w00916487/nanovllm-dsa_offload

unset NANOVLLM_GS_MISS_RATE_ON_LAYERS
unset NANOVLLM_PROFILE_DECODE_OUTPUT

export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export ASCEND_LAUNCH_BLOCKING=0
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
export PYTHONUNBUFFERED=1
export PYTHONPATH=$PWD:$PYTHONPATH

export NANOVLLM_MODEL=/home/models/DeepSeek-V3.2-REAP-345B-A37B-BF16/
export NANOVLLM_TP_SIZE=16
export NANOVLLM_ENABLE_EXPERT_PARALLEL=1
export NANOVLLM_OFFLOAD_MODE=lidu
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

结束时应满足 `offload_mode=lidu`、`captures=1`、`replays>0`、`eager_first_decode=1`，且初始化后不再出现 `eager_lidu_uninitialized` 增长。

## DeepSeek V3.2：GS 整图对照

```bash
cd /home/w00916487/nanovllm-dsa_offload

unset NANOVLLM_GS_MISS_RATE_ON_LAYERS
unset NANOVLLM_PROFILE_DECODE_OUTPUT

export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export ASCEND_LAUNCH_BLOCKING=0
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
export PYTHONUNBUFFERED=1
export PYTHONPATH=$PWD:$PYTHONPATH

export NANOVLLM_MODEL=/home/models/DeepSeek-V3.2-REAP-345B-A37B-BF16/
export NANOVLLM_TP_SIZE=16
export NANOVLLM_ENABLE_EXPERT_PARALLEL=1
export NANOVLLM_OFFLOAD_MODE=gs
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

## DeepSeek V3.2：不卸载整图对照

不卸载模式必须给完整 prompt KV 留出足够 HBM blocks：

```bash
cd /home/w00916487/nanovllm-dsa_offload

unset NANOVLLM_GS_MISS_RATE_ON_LAYERS
unset NANOVLLM_PROFILE_DECODE_OUTPUT
unset NANOVLLM_DRAM_NUM_BLOCKS

export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export ASCEND_LAUNCH_BLOCKING=0
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
export PYTHONUNBUFFERED=1
export PYTHONPATH=$PWD:$PYTHONPATH

export NANOVLLM_MODEL=/home/models/DeepSeek-V3.2-REAP-345B-A37B-BF16/
export NANOVLLM_TP_SIZE=16
export NANOVLLM_ENABLE_EXPERT_PARALLEL=1
export NANOVLLM_OFFLOAD_MODE=none
export NANOVLLM_ENFORCE_EAGER=0
export NANOVLLM_KVCACHE_BLOCK_SIZE=128
export NANOVLLM_HBM_NUM_BLOCKS=96
export NANOVLLM_PREFILL_CHUNK_SIZE=1024
export NANOVLLM_IGNORE_EOS=1
export NANOVLLM_MAX_GEN_TOKENS=8
export NANOVLLM_PROMPT_LENGTHS=8192

python3 example/test.py
```

## GLM-5.1-w4a8：LIDU 长序列整图

```bash
cd /home/w00916487/nanovllm-dsa_offload

unset NANOVLLM_GS_MISS_RATE_ON_LAYERS
unset NANOVLLM_PROFILE_DECODE_OUTPUT

export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export ASCEND_LAUNCH_BLOCKING=0
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
export PYTHONUNBUFFERED=1
export PYTHONPATH=$PWD:$PYTHONPATH

export NANOVLLM_MODEL=/mnt/models/GLM-5.1-w4a8/
export NANOVLLM_TP_SIZE=16
export NANOVLLM_ENABLE_EXPERT_PARALLEL=1
export NANOVLLM_OFFLOAD_MODE=lidu
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

GLM 结束 proof 应满足 `offload_mode=lidu`、`npugraph_ex=False`、`captures=1` 和 `replays>0`。同一配置把 `NANOVLLM_ENFORCE_EAGER` 分别设为 `1/0` 时，`temperature=0` 的 token IDs 应一致。

## 短序列 smoke

`example/short_prompts.py` 同时支持 DeepSeek 和 GLM。以下 GLM 命令使用默认的非卸载 eager 路径：

```bash
cd /home/w00916487/nanovllm-dsa_offload

unset NANOVLLM_GS_MISS_RATE_ON_LAYERS
unset NANOVLLM_PROFILE_DECODE_OUTPUT
unset NANOVLLM_DRAM_NUM_BLOCKS

export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export ASCEND_LAUNCH_BLOCKING=0
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
export PYTHONUNBUFFERED=1
export PYTHONPATH=$PWD:$PYTHONPATH

export NANOVLLM_MODEL=/mnt/models/GLM-5.1-w4a8/
export NANOVLLM_TP_SIZE=16
export NANOVLLM_ENABLE_EXPERT_PARALLEL=1
export NANOVLLM_OFFLOAD_MODE=none
export NANOVLLM_ENFORCE_EAGER=1
export NANOVLLM_KVCACHE_BLOCK_SIZE=128
export NANOVLLM_HBM_NUM_BLOCKS=64
export NANOVLLM_PREFILL_CHUNK_SIZE=0
export NANOVLLM_MAX_MODEL_LEN=512
export NANOVLLM_MAX_GEN_TOKENS=8
export NANOVLLM_IGNORE_EOS=0

python3 example/short_prompts.py
```

## 关键环境变量

| 变量 | 说明 |
| --- | --- |
| `NANOVLLM_OFFLOAD_MODE` | `none|gs|lidu`，默认 `none` |
| `NANOVLLM_ENFORCE_EAGER` | `1` 全 eager；`0` 后续稳定 decode 使用整图 |
| `NANOVLLM_KVCACHE_BLOCK_SIZE` | LIDU/SCATTER 当前只支持 `128` |
| `NANOVLLM_HBM_NUM_BLOCKS` | HBM KV blocks，必须大于 2 |
| `NANOVLLM_DRAM_NUM_BLOCKS` | `gs/lidu` 必须大于 2；决定 DRAM KV 与 HBM IndexCache 容量 |
| `NANOVLLM_PREFILL_CHUNK_SIZE` | 只允许 `0` 或 `1024`；不引入 prefill/decode 混合 forward |
| `NANOVLLM_PROMPT_LENGTHS` | `example/test.py` 的精确 token 长度列表；条目数即 batch size |
| `NANOVLLM_PROFILE_DECODE_OUTPUT` | 非空时只采集 TP rank 0、从首次 decode 到程序结束的 profile |

Chunk prefill 只降低 prefill 激活峰值；不会减少完整请求所需的 KV/IndexCache 容量。其他正式算子 UT 见 `ut_ops/UT_OPS.md`。
