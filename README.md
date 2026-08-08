# nano-vLLM Ascend：GLM-5.1-w4a8

本仓库只支持 `GLM-5.1-w4a8`。Routed experts 使用 ModelSlim W4A8；Attention、dense/shared MLP 的 W8A8 权重在加载时反量化为 BF16。必须启用 EP。

　

## LIDU 缓存

| 原始 prompt 长度 | HBM 缓存预算 C |
| --- | ---: |
| `<= 2048` | 0 |
| `2049–8192` | 2048 |
| `8193–16384` | 3072 |
| `16385–32768` | 6144 |
| `32769–65536` | 8192 |
| `>= 65537` | 12288 |

修改 `nanovllm/engine/dsa_offload.py` 可以修改以上规则

　

## 编译

修改 C++/AscendC 代码或首次部署后必须重新编译：

```bash
export ASCEND_HOME_PATH=/usr/local/Ascend/cann-8.5.1
PYTHONPATH=$PWD:$PYTHONPATH PYTHONUNBUFFERED=1 NANOVLLM_CANN_BUILD_JOBS=64 SOC_VERSION=ascend910_9391 bash scripts/build_nanovllm_ops.sh
```



## MTP-LIDU 算子验收

仓库内置 `NanovllmLiduDecodeUpdateMtp`，固定处理 GLM MTP3 的每请求 4 个 query。四路 top-2048 先求有序并集，再做一次 request-pool 命中、淘汰和状态更新。活跃请求要求 `C >= min(candidate_len, 8192)`。配套 SCATTER 支持最多 8192 个 union miss；`NanovllmSparseAndTailAttentionMtp` 分别消费四路 top-2048，并按四个验证位置计算各自的因果 tail。整网 eager 路径已经接入三者；target 78 层使用 LIDU 卸载，MTP 单层使用独立 dense HBM KV。

修改或首次拉取该算子后先执行上面的完整编译，再运行：

```bash
unset NANOVLLM_ENABLE_DSA_OFFLOAD
unset NANOVLLM_OFFLOAD_MODE
unset NANOVLLM_GS_MISS_RATE_ON_LAYERS
unset NANOVLLM_LIDU_MISS_COUNT_ON_LAYERS
unset NANOVLLM_PROFILE_DECODE_OUTPUT
unset NANOVLLM_CUST_OPAPI_LIB
unset ASCEND_CUSTOM_OPP_PATH
unset NANOVLLM_DSA_BOUNDARY_PROBE

export ASCEND_HOME_PATH=/usr/local/Ascend/cann-8.5.1
export PYTHONUNBUFFERED=1
export PYTHONPATH=$PWD:$PYTHONPATH
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export ASCEND_LAUNCH_BLOCKING=0
export ASCEND_RT_VISIBLE_DEVICES=4

python3 ut_ops/test_lidu_mtp.py \
  --device npu:0 \
  --batch-size 24 \
  --source-len 20992 \
  --cache-tokens 8192 \
  --graph-replays 3 \
  --warmup 10 \
  --iters 100 \
  --min-speedup 1.0 \
  --seed 7
```

随后验证 MTP3 sparse-and-tail Attention 的 CPU golden、`_out`、动态图回放和时延：

```bash
python3 ut_ops/test_sparse_and_tail_attention_mtp.py \
  --device npu:0 \
  --heads 2 \
  --batch-size 24 \
  --cache-tokens 8192 \
  --tail-tokens 64 \
  --graph-replays 3 \
  --warmup 10 \
  --iters 100 \
  --min-speedup 1.0 \
  --seed 7
```

最后验证整条算子链：

```bash
python3 ut_ops/test_mtp_offload_chain.py \
  --device npu:0 \
  --batch-size 4 \
  --heads 2 \
  --source-len 20992 \
  --cache-tokens 8192 \
  --tail-tokens 64 \
  --graph-replays 3 \
  --seed 7
```

三个成功标志分别为 `MTP_LIDU_UT_OK`、`MTP_SPARSE_TAIL_ATTENTION_UT_OK` 和 `MTP_OFFLOAD_CHAIN_UT_OK`。


## MTP3 + LIDU eager 整网验收

该组合固定使用 `NANOVLLM_OFFLOAD_MODE=offload_split` 和 `NANOVLLM_ENFORCE_EAGER=1`。MTP union 最多为 8192，因此 2049 token 以上的请求会自动把 C 提高到 `min(prefill_full_tokens, 8192)`；原本更大的可调预算保持不变。`NANOVLLM_DRAM_NUM_BLOCKS` 同时决定完整 target source 和 MTP 单层 dense KV 池容量。

```bash
unset NANOVLLM_ENABLE_DSA_OFFLOAD
unset NANOVLLM_GS_MISS_RATE_ON_LAYERS
unset NANOVLLM_LIDU_MISS_COUNT_ON_LAYERS
unset NANOVLLM_PROFILE_DECODE_OUTPUT
unset NANOVLLM_CUST_OPAPI_LIB
unset ASCEND_CUSTOM_OPP_PATH
unset NANOVLLM_DSA_BOUNDARY_PROBE
unset NANOVLLM_MAX_GEN_TOKENS

export ASCEND_HOME_PATH=/usr/local/Ascend/cann-8.5.1
export PYTHONUNBUFFERED=1
export PYTHONPATH=$PWD:$PYTHONPATH
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export ASCEND_LAUNCH_BLOCKING=0
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15

export NANOVLLM_MODEL=/mnt/models/GLM-5.1-w4a8/
export NANOVLLM_TP_SIZE=16
export NANOVLLM_ENABLE_EXPERT_PARALLEL=1
export NANOVLLM_OFFLOAD_MODE=offload_split
export NANOVLLM_ENFORCE_EAGER=1
export NANOVLLM_KVCACHE_BLOCK_SIZE=128
export NANOVLLM_HBM_NUM_BLOCKS=700
export NANOVLLM_DRAM_NUM_BLOCKS=2000
export NANOVLLM_PREFILL_CHUNK_SIZE=1024
export NANOVLLM_NUM_SPECULATIVE_TOKENS=3
export NANOVLLM_IGNORE_EOS=1
export NANOVLLM_MAX_STEPS=8

python3 example/test_dureader.py --prompt_count 8
```



## 推理

示例统一使用 `NANOVLLM_MAX_STEPS` 限制 decode request-step 数；prefill
产生的首 token 不计入。MTP 每轮无论提交 1～4 个 token 都只计一步，最后
一轮会完整提交后再结束。

下面是 TP16、LIDU、融合算子、24 个约 20K prompt、`FULL_DECODE_ONLY` 的完整配置：

```bash
unset NANOVLLM_MAX_GEN_TOKENS

export ASCEND_HOME_PATH=/usr/local/Ascend/cann-8.5.1
export PYTHONUNBUFFERED=1
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export ASCEND_LAUNCH_BLOCKING=0
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
export NANOVLLM_MODEL=/mnt/models/GLM-5.1-w4a8/
export NANOVLLM_TP_SIZE=16
export NANOVLLM_ENABLE_EXPERT_PARALLEL=1
export NANOVLLM_OFFLOAD_MODE=offload_fuse
export NANOVLLM_ENFORCE_EAGER=0
export NANOVLLM_KVCACHE_BLOCK_SIZE=128
export NANOVLLM_HBM_NUM_BLOCKS=1190
export NANOVLLM_DRAM_NUM_BLOCKS=3800
export NANOVLLM_PREFILL_CHUNK_SIZE=1024
export NANOVLLM_IGNORE_EOS=1
export NANOVLLM_MAX_STEPS=20

PYTHONPATH=$PWD:$PYTHONPATH NANOVLLM_PROMPT_LENGTHS=20000,20001,20002,20003,20004,20005,20006,20007,20008,20009,20010,20011,20012,20013,20014,20015,20016,20017,20018,20019,20020,20021,20022,20023 python3 example/test.py
```

如果要开启 profile :

```
NANOVLLM_PROFILE_DECODE_OUTPUT=$PWD/profile python3 example/test.py
```

　

## 非卸载 MTP 验收

非卸载 MTP 支持 `offload_mode=none` 和 greedy。`NANOVLLM_NUM_SPECULATIVE_TOKENS` 仅接受 `0`（关闭）或 `3`（MTP3）；非卸载 MTP3 同时支持 eager 和 `FULL_DECODE_ONLY`。图模式只捕获 exact batch：该 batch 的第一次 MTP decode 保持 eager，随后懒 capture target verification 图和三步 draft 图；batch 缩小时自动回到 eager。

MTP3 加载第 78 层 BF16 MTP 权重和模型根目录的 `rot.safetensors`。本功能只修改 Python，无需重新编译 Ascend 自定义算子。先验证 qlen=4 的 target attention 能 capture、动态刷新 KV 长度并 replay：

```bash
unset NANOVLLM_LIDU_MISS_COUNT_ON_LAYERS
unset NANOVLLM_PROFILE_DECODE_OUTPUT
unset NANOVLLM_CUST_OPAPI_LIB
unset ASCEND_CUSTOM_OPP_PATH
unset NANOVLLM_MAX_GEN_TOKENS

export ASCEND_HOME_PATH=/usr/local/Ascend/cann-8.5.1
export PYTHONUNBUFFERED=1
export PYTHONPATH=$PWD:$PYTHONPATH
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export ASCEND_LAUNCH_BLOCKING=0
export ASCEND_RT_VISIBLE_DEVICES=4

python3 ut_ops/test_glm_mtp_target_verify.py \
  --device npu:0 \
  --batch-size 4 \
  --heads 2 \
  --prefix-len 4096 \
  --query-lens 2,4 \
  --graph-replays 3 \
  --seed 7 \
  --atol 0.04 \
  --rtol 0.02 \
  --min-cosine 0.999
```

下面使用 8 条 DuReader 请求验收 K=3 的完整图模式。`NANOVLLM_MAX_STEPS=32` 表示每个请求执行 32 轮 decode；在 `ignore_eos=1` 下，8 个请求会在同一轮结束，即使命中率不同、最终输出 token 数不同。应看到一次 paired capture，随后 `mtp_target_replays` 和 `mtp_draft_replays` 均大于 0，且末尾不再因 token 上限不同出现 batch=7/6/... 的 eager 尾巴。

```bash
unset NANOVLLM_LIDU_MISS_COUNT_ON_LAYERS
unset NANOVLLM_PROFILE_DECODE_OUTPUT
unset NANOVLLM_DRAM_NUM_BLOCKS
unset NANOVLLM_CUST_OPAPI_LIB
unset ASCEND_CUSTOM_OPP_PATH
unset NANOVLLM_MAX_GEN_TOKENS

export ASCEND_HOME_PATH=/usr/local/Ascend/cann-8.5.1
export PYTHONUNBUFFERED=1
export PYTHONPATH=$PWD:$PYTHONPATH
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export ASCEND_LAUNCH_BLOCKING=0
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15

export NANOVLLM_MODEL=/mnt/models/GLM-5.1-w4a8/
export NANOVLLM_TP_SIZE=16
export NANOVLLM_ENABLE_EXPERT_PARALLEL=1
export NANOVLLM_OFFLOAD_MODE=none
export NANOVLLM_ENFORCE_EAGER=0
export NANOVLLM_KVCACHE_BLOCK_SIZE=128
export NANOVLLM_HBM_NUM_BLOCKS=1200
export NANOVLLM_MAX_STEPS=32
export NANOVLLM_PREFILL_CHUNK_SIZE=1024
export NANOVLLM_NUM_SPECULATIVE_TOKENS=3
export NANOVLLM_IGNORE_EOS=1

python3 example/test_dureader.py --prompt_count 8
```

　

## 推理 longbench/dureader

```bash
unset NANOVLLM_MAX_GEN_TOKENS

export ASCEND_HOME_PATH=/usr/local/Ascend/cann-8.5.1
export PYTHONUNBUFFERED=1
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export ASCEND_LAUNCH_BLOCKING=0
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
export NANOVLLM_MODEL=/mnt/models/GLM-5.1-w4a8/
export NANOVLLM_TP_SIZE=16
export NANOVLLM_ENABLE_EXPERT_PARALLEL=1
export NANOVLLM_OFFLOAD_MODE=offload_fuse
export NANOVLLM_ENFORCE_EAGER=0
export NANOVLLM_KVCACHE_BLOCK_SIZE=128
export NANOVLLM_HBM_NUM_BLOCKS=500
export NANOVLLM_DRAM_NUM_BLOCKS=1000
export NANOVLLM_PREFILL_CHUNK_SIZE=1024
export NANOVLLM_IGNORE_EOS=1
export NANOVLLM_MAX_STEPS=50

PYTHONPATH=$PWD:$PYTHONPATH python3 example/test_dureader.py --prompt_count 2
```

　
