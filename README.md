# nano-vllm-ascend DeepSeek V3.2 DSA 卸载说明

当前仓库在 decode 阶段使用 DSA sparse budget，把完整 KV cache 放在 DRAM 侧，把参与本次 MLA 的 sparse KV budget 放在 HBM 侧。

## 编译算子

在昇腾机器的仓库根目录执行：

```bash
SOC_VERSION=ascend910_9391 PYTHONPATH=$PWD:$PYTHONPATH bash scripts/build_nanovllm_ops.sh
ls -lh nanovllm/_C*.so nanovllm/libnanovllm_ascend_kernels.so
```

说明：
- `SOC_VERSION=ascend910_9391` 按机器实际 SoC 设置。
- 如果只改了 pybind extension，可以设置 `NANOVLLM_SKIP_CANN_OPP_BUILD=1` 跳过较慢的 OPP 重建。

## 常用运行命令

混合长短序列，并打印 decode timing：

```bash
PYTHONPATH=$PWD:$PYTHONPATH PYTORCH_NPU_ALLOC_CONF=expandable_segments:True ASCEND_LAUNCH_BLOCKING=0 ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 NANOVLLM_MODEL=/mnt/models/Deepseek-V3.2-Pruned-95B-BF/ NANOVLLM_TP_SIZE=8 NANOVLLM_HBM_NUM_BLOCKS=500 NANOVLLM_DRAM_NUM_BLOCKS=2000 NANOVLLM_MAX_MODEL_LEN=65536 NANOVLLM_MAX_PREFILL_SEQS_PER_STEP=1 NANOVLLM_MAX_DECODE_SEQS_PER_STEP=256 NANOVLLM_PROMPT_LENGTHS=256,12288,14000,18000 NANOVLLM_MAX_GEN_TOKENS=5 NANOVLLM_IGNORE_EOS=1 NANOVLLM_LOG_DECODE_LAYER_TIMING=1 NANOVLLM_DECODE_LAYER_TIMING_SYNC=0 NANOVLLM_PROFILE_LAYER_IDS=mid python3 example/test.py
```

混合长短序列，不打印 timing：

```bash
PYTHONPATH=$PWD:$PYTHONPATH PYTORCH_NPU_ALLOC_CONF=expandable_segments:True ASCEND_LAUNCH_BLOCKING=0 ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 NANOVLLM_MODEL=/mnt/models/Deepseek-V3.2-Pruned-95B-BF/ NANOVLLM_TP_SIZE=8 NANOVLLM_HBM_NUM_BLOCKS=500 NANOVLLM_DRAM_NUM_BLOCKS=2000 NANOVLLM_MAX_MODEL_LEN=65536 NANOVLLM_MAX_PREFILL_SEQS_PER_STEP=1 NANOVLLM_MAX_DECODE_SEQS_PER_STEP=256 NANOVLLM_PROMPT_LENGTHS=256,12288,14000,18000 NANOVLLM_MAX_GEN_TOKENS=5 NANOVLLM_IGNORE_EOS=1 python3 example/test.py
```

短序列：

```bash
PYTHONPATH=$PWD:$PYTHONPATH PYTORCH_NPU_ALLOC_CONF=expandable_segments:True ASCEND_LAUNCH_BLOCKING=0 ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 NANOVLLM_MODEL=/mnt/models/Deepseek-V3.2-Pruned-95B-BF/ NANOVLLM_TP_SIZE=8 NANOVLLM_HBM_NUM_BLOCKS=500 NANOVLLM_DRAM_NUM_BLOCKS=2000 NANOVLLM_MAX_MODEL_LEN=65536 NANOVLLM_MAX_PREFILL_SEQS_PER_STEP=1 NANOVLLM_MAX_DECODE_SEQS_PER_STEP=256 NANOVLLM_MAX_GEN_TOKENS=16 NANOVLLM_IGNORE_EOS=1 python3 example/short_prompts.py
```

长序列：

```bash
PYTHONPATH=$PWD:$PYTHONPATH PYTORCH_NPU_ALLOC_CONF=expandable_segments:True ASCEND_LAUNCH_BLOCKING=0 ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 NANOVLLM_MODEL=/mnt/models/Deepseek-V3.2-Pruned-95B-BF/ NANOVLLM_TP_SIZE=8 NANOVLLM_HBM_NUM_BLOCKS=500 NANOVLLM_DRAM_NUM_BLOCKS=2000 NANOVLLM_MAX_MODEL_LEN=65536 NANOVLLM_MAX_PREFILL_SEQS_PER_STEP=1 NANOVLLM_MAX_DECODE_SEQS_PER_STEP=256 NANOVLLM_MAX_GEN_TOKENS=16 NANOVLLM_IGNORE_EOS=1 python3 example/long_prompts.py
```

## 主要环境变量

| 变量 | 默认值 | 含义 |
|---|---:|---|
| `NANOVLLM_MODEL` | `/home/models/Deepseek-V3.2-Pruned-95B-BF/` | 模型目录。 |
| `NANOVLLM_TP_SIZE` | `4` | Tensor parallel world size。 |
| `NANOVLLM_ENABLE_EXPERT_PARALLEL` | `true` | 是否启用 MoE expert parallel。 |
| `NANOVLLM_KVCACHE_BLOCK_SIZE` | `128` | Paged KV cache block size。 |
| `NANOVLLM_HBM_NUM_BLOCKS` | 必填 | HBM KV cache block 数量。 |
| `NANOVLLM_DRAM_NUM_BLOCKS` | 必填 | DRAM KV cache 和 IndexCache block 数量。 |
| `NANOVLLM_MAX_MODEL_LEN` | example 自推导 | engine 最大序列长度，也会影响 sparse token pool 的最大长度。 |
| `NANOVLLM_MAX_PREFILL_SEQS_PER_STEP` | `1` | 单次 prefill step 最多调度多少个新请求。 |
| `NANOVLLM_MAX_DECODE_SEQS_PER_STEP` | example 自推导 | running 队列容量上限和 decode batch size 上限。 |
| `NANOVLLM_PROMPT_LENGTHS` | 未设置 | 逗号分隔的精确 prompt token 长度。 |
| `NANOVLLM_MAX_GEN_TOKENS` | 脚本自定义 | 每个请求最大 decode token 数。 |
| `NANOVLLM_IGNORE_EOS` | `false` | 是否忽略 EOS，持续 decode 到 `max_tokens`。 |
| `NANOVLLM_LOG_DECODE_LAYER_TIMING` | `false` | 是否打印 decode layer timing。 |
| `NANOVLLM_DECODE_LAYER_TIMING_SYNC` | `true` | timing 前后是否同步。 |
| `NANOVLLM_PROFILE_LAYER_IDS` | `0,mid,last` | 打印 timing 的层。 |

## Decode 时延分解字段

| 字段 | 含义 |
|---|---|
| `attention_total` | 单层 attention block 总耗时。 |
| `indexer_project` | 生成 `q_index`、`index_k` 和 DSA score 权重。 |
| `index_cache` | 把当前 token 的 `index_k` 写入 HBM IndexCache。 |
| `dsa_total` | `dsa_indexer_score + dsa_index_update + dsa_scatter_h2d` 总和。 |
| `dsa_indexer_score` | 基于 query 和 IndexCache 计算候选 token 分数。 |
| `dsa_index_update` | 更新 sparse HBM token budget，并输出 promote/demote 信息。 |
| `dsa_scatter_h2d` | 根据 promote 结果把 KV 从 DRAM 拷回 HBM。 |
| `decode_attention_op` | 在 sparse HBM KV budget 上执行 decode MLA。 |
| `moe_total` | attention 后 MLP/MoE block 耗时。 |

