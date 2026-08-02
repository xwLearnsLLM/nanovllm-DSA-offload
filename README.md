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

　

## 推理

下面是 TP16、LIDU、融合算子、24 个约 20K prompt、`FULL_DECODE_ONLY` 的完整配置：

```bash
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
export NANOVLLM_MAX_GEN_TOKENS=20

PYTHONPATH=$PWD:$PYTHONPATH NANOVLLM_PROMPT_LENGTHS=20000,20001,20002,20003,20004,20005,20006,20007,20008,20009,20010,20011,20012,20013,20014,20015,20016,20017,20018,20019,20020,20021,20022,20023 python3 example/test.py
```

如果要开启 profile :

```
NANOVLLM_PROFILE_DECODE_OUTPUT=$PWD/profile python3 example/test.py
```

　

## 推理 longbench/dureader

```bash
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
export NANOVLLM_MAX_GEN_TOKENS=50

PYTHONPATH=$PWD:$PYTHONPATH python3 example/test_dureader.py --prompt_count 2
```

　
