# nano-vllm-ascend DeepSeek V3.2 DSA 卸载说明

昇腾上做 DSA 模型的 decode 阶段 KVcache offload ，节省显存，提升 batch-size

　

##  编译算子

在昇腾机器的仓库根目录执行：

```bash
SOC_VERSION=ascend910_9391 PYTHONPATH=$PWD:$PYTHONPATH bash scripts/build_nanovllm_ops.sh
ls -lh nanovllm/_C*.so nanovllm/libnanovllm_ascend_kernels.so
```

说明：

- `SOC_VERSION=ascend910_9391` 按机器实际 SoC 设置。
- 如果只改了 pybind extension，可以设置 `NANOVLLM_SKIP_CANN_OPP_BUILD=1` 跳过较慢的 OPP 重建。

　

## 准备模型

当前这一版 nano-vllm-ascend 只支持 BF16 的 deepseek_v32 系列的模型。因为BF16非常占显存，所以不建议跑满血 256 专家的原版 DeepSeek-V3.2 ，而是跑 ：

- **32专家残障版 deepseek_v32** ：https://www.modelscope.cn/models/xwLearnsLLM/Deepseek-V3.2-Pruned-95B 。注意，需要先把模型下载下来，然后按照它的 README 的指示，把模型权重文件从 FP8 转成 BF16 。该模型在nanovllm上需要使用 4~8 张昇腾 910C 就能拉起（每张卡 64GB显存）。
- **cerebras公司裁剪128专家版的 deepseek_v32** ： https://www.modelscope.cn/models/cerebras/DeepSeek-V3.2-REAP-345B-A37B 。注意，需要先把模型下载下来，然后借用 [这里](https://www.modelscope.cn/models/xwLearnsLLM/Deepseek-V3.2-Pruned-95B) 的python脚本来把模型权重文件从 FP8 转成 BF16。该模型在nanovllm上需要使用 16 张昇腾 910C 就能拉起（每张卡 64GB显存）。

　

## 推128专家模型（16卡910C）

先进行一些公用配置：

```bash
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export ASCEND_LAUNCH_BLOCKING=0
export NANOVLLM_MODEL=/var/models/DeepSeek-V3.2-REAP-345B-A37B-BF16/   # 模型路径
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15 # 16卡
export NANOVLLM_TP_SIZE=16                                      # TP16
export NANOVLLM_HBM_NUM_BLOCKS=200                              # 200个HBM blocks
export NANOVLLM_DRAM_NUM_BLOCKS=800                             # 800个DRAM blocks 以及 800个HBM IndexCache Blocks
export NANOVLLM_MAX_MODEL_LEN=65536
export NANOVLLM_MAX_PREFILL_SEQS_PER_STEP=1                     # prefill最大batch-size设为1，避免爆显存
export NANOVLLM_MAX_DECODE_SEQS_PER_STEP=256                    # decode最大batch-size设为256
export NANOVLLM_IGNORE_EOS=1
export NANOVLLM_DSA_OFFLOAD_FIXED_TX=128   # 每请求每个decode step 每层换入的token数量
```

然后进入目录，不需要 `pip install -e .` ，直接推：

1. 混合长短序列（随机无意义tokens），并打印 decode 时延分解：

```bash
PYTHONPATH=$PWD:$PYTHONPATH NANOVLLM_MAX_GEN_TOKENS=5 NANOVLLM_PROMPT_LENGTHS=9000,9001,9002,9003,9004,9005,9006 NANOVLLM_LOG_DECODE_LAYER_TIMING=1 NANOVLLM_DECODE_LAYER_TIMING_SYNC=0 NANOVLLM_PROFILE_LAYER_IDS=mid python3 example/test.py
```

2. 混合长短序列（随机无意义tokens），不打印时延分解：

```bash
PYTHONPATH=$PWD:$PYTHONPATH NANOVLLM_MAX_GEN_TOKENS=5 NANOVLLM_PROMPT_LENGTHS=9000,9001,9002,9003,9004,9005,9006 python3 example/test.py
```

3. 真实短序列：

```bash
PYTHONPATH=$PWD:$PYTHONPATH NANOVLLM_MAX_GEN_TOKENS=16 python3 example/short_prompts.py
```

4. 真实长序列：

```bash
PYTHONPATH=$PWD:$PYTHONPATH NANOVLLM_MAX_GEN_TOKENS=16 python3 example/long_prompts.py
```

　

## 推32专家残障模型（8卡910C）

先进行一些公用配置：

```bash
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export ASCEND_LAUNCH_BLOCKING=0
export NANOVLLM_MODEL=/mnt/models/Deepseek-V3.2-Pruned-95B-BF/  # 模型路径
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7                # 8 卡
export NANOVLLM_TP_SIZE=8                                       # TP8
export NANOVLLM_HBM_NUM_BLOCKS=500                              # 500个HBM blocks
export NANOVLLM_DRAM_NUM_BLOCKS=2000                            # 2000个DRAM blocks 以及 2000个HBM IndexCache Blocks
export NANOVLLM_MAX_MODEL_LEN=65536
export NANOVLLM_MAX_PREFILL_SEQS_PER_STEP=1                     # prefill最大batch-size设为1，避免爆显存
export NANOVLLM_MAX_DECODE_SEQS_PER_STEP=256                    # decode最大batch-size设为256
export NANOVLLM_IGNORE_EOS=1
export NANOVLLM_DSA_OFFLOAD_FIXED_TX=128   # 每请求每个decode step 每层换入的token数量
```

然后进入目录，不需要 `pip install -e .` ，直接推：

1. 混合长短序列（随机无意义tokens），并打印 decode 时延分解：

```bash
PYTHONPATH=$PWD:$PYTHONPATH NANOVLLM_MAX_GEN_TOKENS=5 NANOVLLM_PROMPT_LENGTHS=256,12288,14000,18000 NANOVLLM_LOG_DECODE_LAYER_TIMING=1 NANOVLLM_DECODE_LAYER_TIMING_SYNC=0 NANOVLLM_PROFILE_LAYER_IDS=mid python3 example/test.py
```

2. 混合长短序列（随机无意义tokens），不打印时延分解：

```bash
PYTHONPATH=$PWD:$PYTHONPATH NANOVLLM_MAX_GEN_TOKENS=5 NANOVLLM_PROMPT_LENGTHS=256,12288,14000,18000 python3 example/test.py
```

3. 真实短序列：

```bash
PYTHONPATH=$PWD:$PYTHONPATH NANOVLLM_MAX_GEN_TOKENS=16 python3 example/short_prompts.py
```

4. 真实长序列：

```bash
PYTHONPATH=$PWD:$PYTHONPATH NANOVLLM_MAX_GEN_TOKENS=16 python3 example/long_prompts.py
```

　

## 主要环境变量含义

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

　

## Decode 时延分解字段含义

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

