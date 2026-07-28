# nano-vLLM Ascend：GLM-5.1-w4a8

本仓库只支持 `GLM-5.1-w4a8`。Routed experts 使用 ModelSlim W4A8；Attention、dense/shared MLP 的 W8A8 权重在加载时反量化为 BF16。必须启用 EP。

| `offload_mode` / `NANOVLLM_OFFLOAD_MODE` | Decode KV 路径 |
| --- | --- |
| `none` | 不卸载，完整 MLA KV 保留在 HBM；默认值 |
| `lidu` | LIDU 索引更新 + DRAM→HBM SCATTER + sparse-and-tail Attention |

执行模式只保留 eager 和 `FULL_DECODE_ONLY`。Prefill 始终 eager，也不做 prefill/decode 混合 forward。`NANOVLLM_ENFORCE_EAGER=1` 时所有 decode 都 eager；设为 `0` 时，首次 decode、LIDU 初始化和首次 capture 允许 eager，后续稳定 decode 使用 raw outer ACLGraph replay。LM head 和 sampler 始终在图外。

Chunk prefill 只支持 `NANOVLLM_PREFILL_CHUNK_SIZE=0` 或 `1024`；设为 `1024` 时，每次只处理一个请求的一段 1024-token prefill。

## LIDU 缓存

| 原始 prompt 长度 | HBM 缓存预算 C |
| --- | ---: |
| `<= 2048` | 0 |
| `2049–8192` | 2048 |
| `8193–16384` | 3072 |
| `16385–32768` | 6144 |
| `32769–65536` | 8192 |
| `>= 65537` | 12288 |

后四档由 [dsa_offload.py](nanovllm/engine/dsa_offload.py) 中的 `LIDU_CACHE_TOKEN_BUDGETS` 集中控制。修改这些 Python 常量不需要重新编译算子，但预算必须是 KV block size 的倍数，并需要相应增加 HBM block 数量。

稀疏 source 只包含原始 prompt 的完整 128-token blocks；prompt 末尾非满块和 decode token 始终留在 dense tail。稳定 decode 的 Attention 覆盖缓存中的 top-2048 和完整 tail。`NANOVLLM_ENABLE_LIDU_FUSED_ATTENTION_SCATTER=1` 使用融合搬移/Attention 算子；设为 `0` 使用 SCATTER 后接 sparse-and-tail Attention。

## 编译

修改 C++/AscendC 代码或首次部署后必须重新编译：

```bash
unset NANOVLLM_CUST_OPAPI_LIB
unset ASCEND_CUSTOM_OPP_PATH
export PYTHONUNBUFFERED=1
export PYTHONPATH=$PWD:$PYTHONPATH
export NANOVLLM_CANN_BUILD_JOBS=64
export SOC_VERSION=ascend910_9391
export ASCEND_HOME_PATH=/usr/local/Ascend/cann-8.5.1
bash scripts/build_nanovllm_ops.sh
```

## 推理

下面是 TP16、LIDU、融合算子、24 个约 20K prompt、`FULL_DECODE_ONLY` 的完整配置：

```bash
unset NANOVLLM_PROFILE_DECODE_OUTPUT
unset NANOVLLM_CUST_OPAPI_LIB
unset ASCEND_CUSTOM_OPP_PATH
unset NANOVLLM_LIDU_MISS_COUNT_ON_LAYERS
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export ASCEND_LAUNCH_BLOCKING=0
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
export PYTHONUNBUFFERED=1
export PYTHONPATH=$PWD:$PYTHONPATH
export ASCEND_HOME_PATH=/usr/local/Ascend/cann-8.5.1
export NANOVLLM_MODEL=/mnt/models/GLM-5.1-w4a8/
export NANOVLLM_TP_SIZE=16
export NANOVLLM_ENABLE_EXPERT_PARALLEL=1
export NANOVLLM_OFFLOAD_MODE=lidu
export NANOVLLM_ENABLE_LIDU_FUSED_ATTENTION_SCATTER=1
export NANOVLLM_ENFORCE_EAGER=0
export NANOVLLM_KVCACHE_BLOCK_SIZE=128
export NANOVLLM_HBM_NUM_BLOCKS=1190
export NANOVLLM_DRAM_NUM_BLOCKS=3800
export NANOVLLM_PREFILL_CHUNK_SIZE=1024
export NANOVLLM_IGNORE_EOS=1
export NANOVLLM_MAX_GEN_TOKENS=20
export NANOVLLM_PROMPT_LENGTHS=20000,20001,20002,20003,20004,20005,20006,20007,20008,20009,20010,20011,20012,20013,20014,20015,20016,20017,20018,20019,20020,20021,20022,20023
python3 example/test.py
```

测试不卸载路径时只需把 `NANOVLLM_OFFLOAD_MODE` 改为 `none`，并把 `NANOVLLM_ENABLE_LIDU_FUSED_ATTENTION_SCATTER` 改为 `0`。测试全 eager 时把 `NANOVLLM_ENFORCE_EAGER` 改为 `1`。

## Profile

内置 profiler 只在 TP rank 0 启动。Eager 模式从第一次 decode 开始采集；图模式等到稳定 graph replay 可用后才开始，程序结束时自动停止：

```bash
mkdir -p profile
export NANOVLLM_PROFILE_DECODE_OUTPUT=$PWD/profile
python3 example/test.py
```

关闭 profile：

```bash
unset NANOVLLM_PROFILE_DECODE_OUTPUT
```

Eager 模式下可用 `NANOVLLM_LIDU_MISS_COUNT_ON_LAYERS=0,30,60` 打印指定层当前 decode batch 的 miss count。

算子验收命令见 [ut_ops/UT_OPS.md](ut_ops/UT_OPS.md)，支持矩阵见 [support.md](support.md)。
