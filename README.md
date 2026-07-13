# nano-vLLM Ascend DeepSeek V3.2 DSA offload

本分支在 nano-vLLM Ascend 的 DeepSeek V3.2 BF16 推理上增加 DSA decode KV cache 卸载。后续以 128 专家模型为主。运行时只保留两种模式：

| `NANOVLLM_ENFORCE_EAGER` | Prefill | 第一个 decode step | 稳定 decode |
| --- | --- | --- | --- |
| `0`（默认） | eager | eager | 满足 DSA 条件时使用 `FULL_DECODE_ONLY + npugraph_ex + ACLGraph` |
| `1` | eager | eager | eager |

LM head 和 sampler 始终在整图之外。

DSA 整图采用精确 batch size。只有 batch 内所有请求均已进入 DSA offload 的稳定 decode，才会 replay 整图；短请求、首个 decode、混合 batch 和未 capture 的 batch size 会明确走 eager，并在最终统计中分别计数。

　

## 准备模型

当前这一版 nano-vllm-ascend 只支持 BF16 的 deepseek_v32 系列的模型。因为BF16非常占显存，所以不建议跑满血 256 专家的原版 DeepSeek-V3.2 ，而是跑 ：

- **32专家残障版 deepseek_v32** ：https://www.modelscope.cn/models/xwLearnsLLM/Deepseek-V3.2-Pruned-95B 。注意，需要先把模型下载下来，然后按照它的 README 的指示，把模型权重文件从 FP8 转成 BF16 。该模型在nanovllm上需要使用 4~8 张昇腾 910C 就能拉起（每张卡 64GB显存）。
- **cerebras公司裁剪128专家版的 deepseek_v32** ： https://www.modelscope.cn/models/cerebras/DeepSeek-V3.2-REAP-345B-A37B 。注意，需要先把模型下载下来，然后借用 [这里](https://www.modelscope.cn/models/xwLearnsLLM/Deepseek-V3.2-Pruned-95B) 的python脚本来把模型权重文件从 FP8 转成 BF16。该模型在nanovllm上需要使用 16 张昇腾 910C 就能拉起（每张卡 64GB显存）。

　

## 编译算子

```bash
NANOVLLM_CANN_BUILD_JOBS=64 SOC_VERSION=ascend910_9391 PYTHONPATH=$PWD:$PYTHONPATH bash scripts/build_nanovllm_ops.sh
```

　

## 推荐验证命令（128 专家模型、TP16）

在仓库根目录执行：

```bash
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export ASCEND_LAUNCH_BLOCKING=0
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
export NANOVLLM_TP_SIZE=16
export NANOVLLM_MODEL=/home/models/DeepSeek-V3.2-REAP-345B-A37B-BF16/
export NANOVLLM_ENABLE_EXPERT_PARALLEL=1
export NANOVLLM_ENFORCE_EAGER=0
export NANOVLLM_KVCACHE_BLOCK_SIZE=128
export NANOVLLM_HBM_NUM_BLOCKS=350
export NANOVLLM_DRAM_NUM_BLOCKS=1500
export NANOVLLM_PREFILL_CHUNK_SIZE=1024   # chunk-prefill模式，可避免激活爆显存

du -sh "$NANOVLLM_MODEL"    # 检查模型存在

PYTHONPATH=$PWD:$PYTHONPATH NANOVLLM_IGNORE_EOS=1 NANOVLLM_MAX_GEN_TOKENS=16 NANOVLLM_PROMPT_LENGTHS=30000,30001 python3 example/test.py 

PYTHONPATH=$PWD:$PYTHONPATH NANOVLLM_IGNORE_EOS=1 NANOVLLM_MAX_GEN_TOKENS=16 NANOVLLM_PROMPT_LENGTHS=30000,30001,30002,30003,30004,30005 python3 example/test.py
```

`example/test.py` 会自动设置：

- `max_model_len = 最长 prompt + max_gen_tokens`
- prefill batch 上限为 1，避免一次 prefill 多条长序列
- decode batch 上限和 graph capture size 均为 prompt 数量；上面的命令即精确 batch size 2
- `temperature=0`、`ignore_eos=True`

运行结束必须看到类似：

```text
DSA FULL_DECODE_ONLY proof: capture_sizes=[2], captures=1, replays=14, eager_first_decode=1, eager_no_dsa=0, eager_mixed_batch=0, eager_uncaptured_batch=0
```

验收要求：

- `captures=1`
- `replays > 0`
- 推荐命令中后三个 eager 回退计数均为 0
- `eager_first_decode=1` 是当前 MLAPO 正确性约束，不是异常回退

如果 `replays=0`，先看最终统计属于 `eager_no_dsa`、`eager_mixed_batch` 还是 `eager_uncaptured_batch`。尤其要确认 HBM/DRAM block 足够让两条请求同时驻留，否则实际稳定 decode batch 会与精确 capture size 不一致。

　

## 只采集 TP rank 0 的 decode profile

保留前面的模型和并行环境变量，直接运行 Python，不要再套 `msprof`：

```bash
PYTHONPATH=$PWD:$PYTHONPATH \
NANOVLLM_PROFILE_DECODE_OUTPUT=./profile_rank0_decode \
NANOVLLM_IGNORE_EOS=1 \
NANOVLLM_MAX_GEN_TOKENS=6 \
NANOVLLM_PROMPT_LENGTHS=30000,30001 \
python3 example/test.py
```

Profiler 在 rank 0 第一次 decode forward 之前启动，在 `generate()` 完成后停止。
Prefill 和 TP rank 1–15 均不采集。结果写入 `profile_rank0_decode`，可用
MindStudio Insight 打开。`MAX_GEN_TOKENS=6` 会采集 1 个 eager decode 和
4 个 full-decode graph replay，通常已包含至少 3 个稳定 decode step。

　

## eager 对照

保留其他环境变量不变，仅执行：

```bash
PYTHONPATH=$PWD:$PYTHONPATH NANOVLLM_PREFILL_CHUNK_SIZE=1024 NANOVLLM_MAX_GEN_TOKENS=16 NANOVLLM_PROMPT_LENGTHS=8200,8201 NANOVLLM_ENFORCE_EAGER=1 python3 example/test.py
```

　

## GatherSelectionKVCache miss rate 调试

该功能只支持 eager 路径，并且只由 TP rank 0 打印。示例：

```bash
PYTHONPATH=$PWD:$PYTHONPATH \
NANOVLLM_ENFORCE_EAGER=1 \
NANOVLLM_GS_MISS_RATE_ON_LAYERS=0,30,60 \
NANOVLLM_PREFILL_CHUNK_SIZE=1024 \
NANOVLLM_IGNORE_EOS=1 \
NANOVLLM_MAX_GEN_TOKENS=6 \
NANOVLLM_PROMPT_LENGTHS=30000,30001 \
python3 example/test.py
```

每个配置层在每个 decode step 的 gather 之前打印当前 batch 各请求的
`|topk - selection| / 2048`、平均 miss rate 和对应的 miss token 数。例如：

```text
GS_MISS_RATE decode_step=2 layer=30 batch_size=2 request_miss_tokens=[512, 480] request_miss_rate=[0.250000, 0.234375] mean_miss_rate=0.242188
```

该调试会执行 NPU 到 CPU 同步，不能用开启该开关后的 TPOT 作为性能数据。
在 full-decode-only 模式下只可能看到 eager first-decode 的输出；若要观察每个
decode step，必须设置 `NANOVLLM_ENFORCE_EAGER=1`。

　

## 主要 bash 参数

| 参数 | 说明 |
| --- | --- |
| `NANOVLLM_MODEL` | BF16 模型目录 |
| `NANOVLLM_TP_SIZE` | TP 大小 |
| `NANOVLLM_ENABLE_EXPERT_PARALLEL` | 是否启用 EP |
| `NANOVLLM_ENFORCE_EAGER` | `0` 为 DSA full-decode-only，`1` 为 eager |
| `NANOVLLM_KVCACHE_BLOCK_SIZE` | KV block size，必须是 16 的倍数，当前推荐 128 |
| `NANOVLLM_HBM_NUM_BLOCKS` | HBM KV block 数，必须大于 2 |
| `NANOVLLM_DRAM_NUM_BLOCKS` | DRAM KV block 数，同时也是 HBM IndexCache block 数，必须大于 2 |
| `NANOVLLM_PREFILL_CHUNK_SIZE` | 仅允许 `0` 或 `1024`；`0` 为整段 prefill，`1024` 为单请求 chunk prefill |
| `NANOVLLM_PROMPT_LENGTHS` | 精确 prompt token 长度，逗号分隔；条目数就是测试 batch size |
| `NANOVLLM_MAX_GEN_TOKENS` | 每个请求生成 token 数，默认 16 |
| `NANOVLLM_PROFILE_DECODE_OUTPUT` | 非空时仅在 TP rank 0 采集从首次 decode 到生成结束的 profile，并写入该目录 |
| `NANOVLLM_GS_MISS_RATE_ON_LAYERS` | eager-only；逗号分隔的层号，例如 `0,30,60`，在 gather 前打印当前 batch 的 miss rate |

　

## Chunk prefill 行为与当前边界

- DSA sparse budget 固定为 2048 token；短于该条件的请求没有卸载收益，会走 eager decode。
- `prefill_chunk_size=1024` 强制 `max_num_prefill_seqs_per_step=1`。每个 forward 最多处理单条请求的 1024 个 token；当前请求完成前不会进入 decode running 队列，也不做 prefill/decode 混合 forward。
- 中间 chunk 只写当前 token 对应的 HBM KV 和 IndexCache，不运行 LM head/sampler，也不会提前执行 DRAM KV finalize 或释放 HBM 中间块；只有最后一个 chunk 完成这些操作并采样首个输出 token。
- Paged MLA FIA 继续使用 `sparse_mode=3` 的右下角因果 mask；query 长度为当前 chunk 长度，KV 有效长度为“历史前缀 + 当前 chunk”的总长度。算子语义见 [CANN FIA V4 文档](https://gitcode.com/cann/ops-transformer/blob/master/attention/fused_infer_attention_score/docs/aclnnFusedInferAttentionScoreV4.md)。
- Chunk prefill 只降低激活值峰值。调度器仍为完整请求预先分配 HBM KV、DRAM KV 和 IndexCache blocks，因此不能解决 cache 容量不足。
- DSA 整图不能用 padding bucket：`gather_selection_status` 是跨 step 持久化状态，虚假 padding 行可能污染真实请求状态。

实现与判定细节见 [FULL_DECODE_ONLY.md](FULL_DECODE_ONLY.md)。

　

## 当前效果

DeepSeek-V3.2-REAP-345B-A37B-BF16 模型，TP16+EP16，序列长度 30k

| 推理框架                | NPU KV 块数量 | NPU Index块数量  | bs     | TPOT         | TPS        |
| --------                | ------------- | ---------------- | ------ | ------------ | ---------- |
| vLLM 0.19 (不卸载)      | -             | -                | 2      | 54 ms        | 37 TPS     | 
| nanovllm (不卸载)       | 700           | 700              | 2      | 58 ms        | 34 TPS     | 
| nanovllm (卸载, GS算子) | 350           | 1500             | 2      | 79 ms        | 25 TPS     | 
| nanovllm (卸载, GS算子) | 350           | 1500             | 6      | 97 ms        | 61 TPS     | 




