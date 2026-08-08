# nano-vLLM Ascend：GLM-5.1 W4A8

本仓库只维护 `GLM-5.1-w4a8`。运行时要求 BF16、Expert Parallel、128-token KV block，以及 ModelSlim 1.0.0 per-channel W4A8 checkpoint。Routed experts 保持原生 W4A8；Attention、dense/shared MLP 的 W8A8 权重在加载时反量化为 BF16。

## 支持范围

`NANOVLLM_NUM_SPECULATIVE_TOKENS` 只接受 `0` 或 `3`。

| MTP | `NANOVLLM_OFFLOAD_MODE` | eager | `FULL_DECODE_ONLY` | 稳定 decode 路径 |
| ---: | --- | :---: | :---: | --- |
| 0 | `none` | 支持 | 支持 | Dense MLA |
| 0 | `offload_split` | 支持 | 支持 | `fused_li_manage → scatter_copy → sparse_tail_attention` |
| 0 | `offload_fuse` | 支持 | 支持 | `fused_li_manage → fused_copy_sfa`，支持 bs>24 |
| 3 | `none` | 支持 | 支持 | MTP3 + Dense MLA |
| 3 | `offload_split` | 支持 | 支持 | `fused_li_manage_mtp → scatter_copy → sparse_tail_attention_mtp` |
| 3 | `offload_fuse` | 不支持 | 不支持 | 尚未实现 `fused_copy_sfa_mtp` |

图模式只针对后续稳定 decode。Prefill、首次 decode、卸载缓存初始化和首次 lazy capture 允许走 eager；稳定且 batch size 与 capture size 完全一致后才 replay。

## LIDU 缓存预算

非 MTP 请求按原始 prompt 长度选择 HBM token 预算 C：

| prompt 长度 | C |
| --- | ---: |
| `<= 2048` | 0 |
| `2049–8192` | 2048 |
| `8193–16384` | 3072 |
| `16385–32768` | 6144 |
| `32769–65536` | 8192 |
| `>= 65537` | 12288 |

预算集中定义在 `nanovllm/engine/dsa_offload.py` 的 `LIDU_CACHE_TOKEN_BUDGETS`。MTP3 的四路 top-2048 并集最多为 8192，因此启用 MTP 卸载时会保证 `C >= min(prefill_full_tokens, 8192)`。

## 编译

首次部署或修改 C++、host tiling、AscendC kernel 后必须重新编译：

```bash
unset NANOVLLM_CUST_OPAPI_LIB
unset ASCEND_CUSTOM_OPP_PATH

export ASCEND_HOME_PATH=/usr/local/Ascend/cann-8.5.1
export CANN_INSTALL_PATH=/usr/local/Ascend/cann-8.5.1
export PYTHONPATH=$PWD:$PYTHONPATH
export PYTHONUNBUFFERED=1
export NANOVLLM_CANN_BUILD_JOBS=64
export NANOVLLM_EXT_BUILD_JOBS=1
export SOC_VERSION=ascend910_9391

bash scripts/build_nanovllm_ops.sh
```

算子接口、边界和昇腾 UT 命令见 [`README_ops.md`](README_ops.md)。

## 非 MTP：融合卸载图模式

下面运行 TP16、24 个约 20K prompt、`offload_fuse + FULL_DECODE_ONLY`：

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
export NANOVLLM_OFFLOAD_MODE=offload_fuse
export NANOVLLM_ENFORCE_EAGER=0
export NANOVLLM_KVCACHE_BLOCK_SIZE=128
export NANOVLLM_HBM_NUM_BLOCKS=1190
export NANOVLLM_DRAM_NUM_BLOCKS=3800
export NANOVLLM_PREFILL_CHUNK_SIZE=1024
export NANOVLLM_NUM_SPECULATIVE_TOKENS=0
export NANOVLLM_IGNORE_EOS=1
export NANOVLLM_MAX_STEPS=20
export NANOVLLM_PROMPT_LENGTHS=20000,20001,20002,20003,20004,20005,20006,20007,20008,20009,20010,20011,20012,20013,20014,20015,20016,20017,20018,20019,20020,20021,20022,20023

python3 example/test.py
```

## MTP3：分离卸载图模式

下面用 8 条 DuReader 请求验收 `MTP3 + offload_split + FULL_DECODE_ONLY`。Target 的 78 层使用 MTP 卸载链；单个 MTP 层保留独立 dense HBM KV cache。

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
export NANOVLLM_ENFORCE_EAGER=0
export NANOVLLM_KVCACHE_BLOCK_SIZE=128
export NANOVLLM_HBM_NUM_BLOCKS=900
export NANOVLLM_DRAM_NUM_BLOCKS=2500
export NANOVLLM_PREFILL_CHUNK_SIZE=1024
export NANOVLLM_NUM_SPECULATIVE_TOKENS=3
export NANOVLLM_IGNORE_EOS=1
export NANOVLLM_MAX_STEPS=32

python3 example/test_dureader.py --prompt_count 8
```

`NANOVLLM_MAX_STEPS` 表示每个请求最多执行多少个 decode request-step；它取代了旧的 `NANOVLLM_MAX_GEN_TOKENS`。MTP3 的一个 request-step 可以提交 1～4 个 token。

## Decode profile

只采集 TP rank 0、从首次 decode 到程序结束：

```bash
export NANOVLLM_PROFILE_DECODE_OUTPUT=$PWD/profile
python3 example/test.py
```

关闭 profile：

```bash
unset NANOVLLM_PROFILE_DECODE_OUTPUT
```
