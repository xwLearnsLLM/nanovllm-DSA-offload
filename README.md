# nano-vLLM Ascend：DeepSeek V3.2 / GLM-5.1

本仓库面向以下两类模型：

- DeepSeek V3.2 BF16。
- GLM-5.1-w4a8。Routed experts 保持 ModelSlim W4A8；Attention、dense/shared MLP 的 W8A8 权重在加载时反量化为 BF16。

引擎提供三种互斥的 decode KV 模式：

| `offload_mode` / `NANOVLLM_OFFLOAD_MODE` | 行为 |
| --- | --- |
| `none` | 不卸载，完整 MLA KV 保留在 HBM；默认值 |
| `gs` | LightningIndexer + GatherSelectionKVCache |
| `lidu` | NanovllmLiduDecodeUpdate；GLM 可选融合 SCATTER + sparse-and-tail Attention |

旧参数 `enable_dsa_offload` 和环境变量 `NANOVLLM_ENABLE_DSA_OFFLOAD` 已删除。

执行方式只保留 eager 和 `FULL_DECODE_ONLY`：

| `NANOVLLM_ENFORCE_EAGER` | Prefill | 第一个 decode | 后续稳定 decode |
| --- | --- | --- | --- |
| `1` | eager | eager | eager |
| `0` | eager | eager | `FULL_DECODE_ONLY` |

LM head 和 sampler 始终在图外。首次 LIDU 缓存初始化始终 eager；下一次 initialized decode 用真实 C、block tables 和 request state 延迟 capture，并仍以 eager 产出当前 token；再下一步开始 replay。DeepSeek 使用 `npugraph_ex + outer ACLGraph`，GLM 使用 raw outer ACLGraph，因此 GLM proof 中 `npugraph_ex=False` 是正常现象。

## LIDU + SCATTER 语义

缓存预算 C 由原始 prompt 长度固定，生成过程中不再改变：

| 原始 prompt 长度 | C |
| --- | ---: |
| `<= 2048` | 0 |
| `2049–8192` | 2048 |
| `8193–16384` | 3072 |
| `16385–32768` | 6144 |
| `32769–65536` | 8192 |
| `>= 65537` | 12288 |

后四档统一由 `nanovllm/engine/dsa_offload.py` 中的
`LIDU_CACHE_TOKEN_BUDGETS = (3072, 6144, 8192, 12288)` 控制，按表中顺序对应。
预算必须是 128 的倍数、不得小于 2048、保持非递减，四档上限依次为
`8192/16384/32768/65536`。编译本版本算子一次后，实验时只需修改该元组并重启进程，
不需要再次编译算子；同时要相应增大 `NANOVLLM_HBM_NUM_BLOCKS`。

稀疏 source 只包含原始 prompt 的完整 128-token blocks。prompt 末尾非满块和所有 decode token 始终留在 tail，不参与 LIDU 选择或 SCATTER 搬移。GLM 的 Attention 计算缓存中的 top-2048 加完整 tail；DeepSeek 仍以 dense MLA 计算 C 个缓存 token 加完整 tail。

对于真正发生卸载的 LIDU 请求，final prefill 把完整 source KV 持久化到 DRAM 后会释放全部完整 prompt HBM blocks，只保留 dense tail。C-token HBM arena 此时只做逻辑容量预留，后续请求的 prefill 可以临时借用这些空闲 blocks；该请求第一次进入 decode 前，调度器再原子申请 C blocks（以及必要的新 tail block）并放到 block table 前部。准入检查会保证所有活跃请求最终的 decode footprint 不超过可用 HBM，避免多个 prefill 成功后在首次 decode 才 OOM。GS 的缓存布局不受这一策略影响。

每层维护持久化 request pool。`req_pool_entries[b]` 将当前 batch 行映射到该请求的状态行；batch 重排不搬移状态。首次 decode 分块计算 top-C 并初始化 HBM，稳定 decode 由 LIDU 融合 top-2048、hit/miss、eviction 和索引更新；batch size 不能整除 24 时按 512-token chunk 在 24 个 AI Core 间均衡调度，尤其避免 batch size 小于 24 时闲置算力。GLM 可通过 `NANOVLLM_ENABLE_LIDU_FUSED_ATTENTION_SCATTER=1` 将 miss 搬移与 sparse-and-tail Attention 合并为一个算子；默认值 `0` 保留原来的 SCATTER 后接 Attention 路径。融合路径目前只用于 GLM、LIDU、稳定 decode 且 batch size 不超过 24；首次 decode、初始化 step 和更大 batch 自动回退旧路径。

## 本仓库自包含算子

LIDU、SCATTER、sparse-and-tail Attention 与融合算子的 host、tiling、AscendC kernel、ACLNN adapter、`torch.library` schema 和 Meta/Fake 路径均已放入本仓库：

- `csrc/nanovllm_ascend_ops/ops/lightning_indexer_decode_update/`
- `csrc/nanovllm_ascend_ops/ops/kvcache_scatter_copy/`
- `csrc/nanovllm_ascend_ops/ops/sparse_and_tail_attention/`
- `csrc/nanovllm_ascend_ops/ops/sparse_and_tail_attention_and_scatter_copy/`

编译和运行不会从参考工程导入、链接或动态加载这些算子；它们只依赖本仓库代码和标准 CANN/PyTorch-NPU SDK，并按绝对路径加载本仓库生成的 `libcust_opapi.so`。

## 编译

新增了 Ascend 自定义算子，更新代码后必须重新编译：

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

## GLM-5.1-w4a8 推理

```bash
unset NANOVLLM_ENABLE_DSA_OFFLOAD
unset NANOVLLM_GS_MISS_RATE_ON_LAYERS
unset NANOVLLM_PROFILE_DECODE_OUTPUT
unset NANOVLLM_CUST_OPAPI_LIB
unset ASCEND_CUSTOM_OPP_PATH
unset NANOVLLM_DSA_BOUNDARY_PROBE

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

如果要输出 PROFILE :

```
NANOVLLM_PROFILE_DECODE_OUTPUT=$PWD/profile python3 example/test.py
```

