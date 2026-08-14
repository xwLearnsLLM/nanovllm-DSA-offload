# nano-vLLM Ascend：GLM-5.1 / GLM-5.2 W4A8

运行时要求 BF16、Expert Parallel、128-token KV block，以及 ModelSlim 1.0.0 per-channel W4A8 checkpoint。Routed experts 保持原生 W4A8；Attention、dense/shared MLP 的 W8A8 权重在加载时反量化为 BF16。

`main-glm52` 支持 `GLM-5.2-w4a8` 在 20K～64K 范围内使用 `MTP0/MTP3 × none/offload_split/offload_fuse × eager/FULL_DECODE_ONLY`。MTP3 target verification 一次处理每请求四路因果 query；offload 路径使用对应的 `*_mtp` LIM 和 Attention 算子。

　

## 支持范围

### GLM-5.1 / GLM-5.2 MTP

`NANOVLLM_NUM_SPECULATIVE_TOKENS` 只接受 `0` 或 `3`。图模式只针对后续稳定 decode。Prefill、首次 decode、卸载缓存初始化和首次 lazy capture 允许走 eager；稳定且 batch size 与 capture size 完全一致后才 replay。

| MTP | `NANOVLLM_OFFLOAD_MODE` | eager | `FULL_DECODE_ONLY` | 稳定 decode 路径 |
| ---: | --- | :---: | :---: | --- |
| 0 | `none` | 支持 | 支持 | Dense MLA |
| 0 | `offload_split` | 支持 | 支持 | `fused_li_manage → scatter_copy → sparse_tail_attention` |
| 0 | `offload_fuse` | 支持 | 支持 | `fused_li_manage → fused_copy_sfa`，支持 bs>24 |
| 3 | `none` | 支持 | 支持 | MTP3 + Dense MLA |
| 3 | `offload_split` | 支持 | 支持 | `fused_li_manage_mtp → scatter_copy → sparse_tail_attention_mtp` |
| 3 | `offload_fuse` | 支持 | 支持 | `fused_li_manage_mtp → fused_copy_sfa_mtp`；后者内部按序执行 union SCATTER 与 MTP-SFA |

　

### DSA decode 卸载

设置 `NANOVLLM_OFFLOAD_MODE` 可以将 target layer 的历史 KV 从 HBM 卸载到 DRAM；`none` 保持 dense MLA，不使用 DRAM KV 池。卸载只覆盖稳定 decode，prefill 仍会写入完整 KV，首次需要卸载的 decode 会初始化 LIDU 缓存，因此不应将这些阶段纳入 TPOT 对比。

- `offload_split`：LIM 先根据 LightningIndexer 选出 top-2048 token，随后分别执行 DRAM→HBM `scatter_copy` 和 top-2048 + dense tail Attention。
- `offload_fuse`：在稳定 decode 将搬移与 Attention 合并为 `fused_copy_sfa` / `fused_copy_sfa_mtp`；首次 LIDU 初始化仍安全地走 split 路径。
- GLM-5.2 target layer 采用官方 IndexShare schedule：21 个 full layer 运行 LIM，57 个 shared layer 复用 owner 的 token/slot/miss 元数据；78 层仍分别使用自己的 KV payload 执行 Attention，绝不跨层复用 KV 或 Attention 输出。
- MTP3 的 target verification 每请求以四路因果 query 执行；MTP LIM 管理四路 top-2048 的并集，MTP Attention 仍为每个 query 使用其对应的因果可见 KV 长度。MTP draft layer 使用独立的 dense KV source，并在 draft iteration 间复用首次选择结果。

`offload_split` 与 `offload_fuse` 是等价的卸载编排，应该在相同请求和采样设置下输出一致。它们与 `none` 分别使用稀疏 top-2048 + dense-tail 和 dense MLA，长序列生成可能因稀疏近似出现分叉；验收应以任务质量、稳定性以及 split/fuse 一致性为准。

当前卸载配置要求 `NANOVLLM_KVCACHE_BLOCK_SIZE=128`、`kv_lora_rank=512`、`qk_rope_head_dim=64`、`index_topk=2048` 与 `index_n_heads/index_head_dim=32/128`。支持范围为常用 20K～64K 序列，不面向 1M 上下文。七个内置算子的接口与边界见 [`README_ops.md`](README_ops.md)。

　

### HBM缓存预算

修改 `nanovllm/engine/dsa_offload.py` 的 `LIDU_CACHE_TOKEN_BUDGETS` 可以修改缓存预算。默认如下：

| prompt 长度 | 不开MTP的缓存预算 (tokens) | 开MTP的缓存预算 (tokens) |
| :-: | :--: | :--: |
| `<= 2048` | 0 | 0 |
| `2049–8192` | 2048 | `floor(L / 128) × 128`，缓存全部 prefill 满块 |
| `8193–16384` | 6144 | 8192 |
| `16385–32768` | 6144 | 8192 |
| `32769–65536` | 8192 | 8192 |
| `>= 65537` | 12288 | 12288 |

　

## 编译

首次部署或修改 C++、host tiling、AscendC kernel 后必须重新编译：

```bash
export ASCEND_HOME_PATH=/usr/local/Ascend/cann-8.5.1
export CANN_INSTALL_PATH=/usr/local/Ascend/cann-8.5.1
PYTHONPATH=$PWD:$PYTHONPATH PYTHONUNBUFFERED=1 SOC_VERSION=ascend910_9391 NANOVLLM_CANN_BUILD_JOBS=64 NANOVLLM_EXT_BUILD_JOBS=1 bash scripts/build_nanovllm_ops.sh
```

完成过一次全量编译后，如果只修改了 `fused_li_manage_mtp` 的 AscendC kernel 或其依赖的 device header，可以复用原 build 目录，只重新编译并安装对应 kernel：

```bash
bash scripts/rebuild_nanovllm_cann_kernel.sh fused_li_manage_mtp
```

修改 host tiling、算子接口、op-api、PyTorch binding、CMake，或者 build 目录已被删除时，仍须使用上面的全量编译命令。

算子接口、边界和昇腾 UT 命令见 [`README_ops.md`](README_ops.md)。

　

## 运行

先设置一些公共环境变量

```bash
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export ASCEND_LAUNCH_BLOCKING=0
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
export NANOVLLM_MODEL=/mnt/models/GLM-5.1-w4a8/
export NANOVLLM_TP_SIZE=16
export NANOVLLM_ENABLE_EXPERT_PARALLEL=1
export NANOVLLM_ENFORCE_EAGER=0
export NANOVLLM_KVCACHE_BLOCK_SIZE=128
export NANOVLLM_PREFILL_CHUNK_SIZE=1024
export NANOVLLM_IGNORE_EOS=1
export NANOVLLM_MAX_STEPS=20
```

不开MTP，bs=12，seqlen=40k

```bash
PYTHONPATH=$PWD:$PYTHONPATH PYTHONUNBUFFERED=1 NANOVLLM_OFFLOAD_MODE=offload_fuse NANOVLLM_NUM_SPECULATIVE_TOKENS=0 NANOVLLM_HBM_NUM_BLOCKS=800 NANOVLLM_DRAM_NUM_BLOCKS=3900 NANOVLLM_PROMPT_LENGTHS=40000,40001,40002,40003,40004,40005,40006,40007,40008,40009,40010,40011 python3 example/test.py
```

开MTP，bs=12，seqlen=40k

```bash
PYTHONPATH=$PWD:$PYTHONPATH PYTHONUNBUFFERED=1 NANOVLLM_OFFLOAD_MODE=offload_split NANOVLLM_NUM_SPECULATIVE_TOKENS=3 NANOVLLM_HBM_NUM_BLOCKS=800 NANOVLLM_DRAM_NUM_BLOCKS=3900 NANOVLLM_PROMPT_LENGTHS=40000,40001,40002,40003,40004,40005,40006,40007,40008,40009,40010,40011 python3 example/test.py
```

不开MTP，bs=12，longbench/dureader最长的12条（序列长度17k左右）

```bash
NANOVLLM_PREFILL_CHUNK_SIZE=0 PYTHONPATH=$PWD:$PYTHONPATH PYTHONUNBUFFERED=1 NANOVLLM_OFFLOAD_MODE=offload_fuse NANOVLLM_NUM_SPECULATIVE_TOKENS=0 NANOVLLM_HBM_NUM_BLOCKS=800 NANOVLLM_DRAM_NUM_BLOCKS=1500 python3 example/test_dureader.py --prompt_count 12
```

开MTP，bs=12，longbench/dureader最长的12条（序列长度17k左右）

```bash
NANOVLLM_PREFILL_CHUNK_SIZE=0 PYTHONPATH=$PWD:$PYTHONPATH PYTHONUNBUFFERED=1 NANOVLLM_OFFLOAD_MODE=offload_split NANOVLLM_NUM_SPECULATIVE_TOKENS=3 NANOVLLM_HBM_NUM_BLOCKS=800 NANOVLLM_DRAM_NUM_BLOCKS=1500 python3 example/test_dureader.py --prompt_count 12
```

　

## 开profile运行 (导出后可用mindstudio查看)

只采集 TP rank 0、从首次 decode 到程序结束，加上 `NANOVLLM_PROFILE_DECODE_OUTPUT` 环境变量就行

```bash
NANOVLLM_PROFILE_DECODE_OUTPUT=./profile <你要运行的命令>
```

　
