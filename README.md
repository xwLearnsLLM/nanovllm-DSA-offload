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

`catlass` 仍然是必要依赖，因为 decode 的 `matmul_allreduce_add_rmsnorm` 融合算子使用它。

若编译命令卡在 catlass 下载，请提前手动下载 catlass 并把 catlass 放到 `csrc/third_party/catlass`。方法如下：

```
mkdir -p csrc/third_party/
cd csrc/third_party/
git clone --depth 1 --branch master https://gitcode.com/cann/catlass.git 
cd ../..
ls csrc/third_party/catlass/include/catlass/catlass.hpp    # 检查关键头文件存在
```

　

## 推荐验证命令（128 专家模型、TP16）

在仓库根目录执行：

```bash
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export ASCEND_LAUNCH_BLOCKING=0
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
export NANOVLLM_MODEL=/home/models/DeepSeek-V3.2-REAP-345B-A37B-BF16/
export NANOVLLM_TP_SIZE=16
export NANOVLLM_ENABLE_EXPERT_PARALLEL=1
export NANOVLLM_ENFORCE_EAGER=0
export NANOVLLM_KVCACHE_BLOCK_SIZE=128
export NANOVLLM_HBM_NUM_BLOCKS=200
export NANOVLLM_DRAM_NUM_BLOCKS=200

du -sh "$NANOVLLM_MODEL"    # 检查模型存在

PYTHONPATH=$PWD:$PYTHONPATH NANOVLLM_MAX_GEN_TOKENS=16 NANOVLLM_PROMPT_LENGTHS=8200,8201 python3 example/test.py
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

　

## eager 对照

保留其他环境变量不变，仅执行：

```bash
PYTHONPATH=$PWD:$PYTHONPATH NANOVLLM_MAX_GEN_TOKENS=16 NANOVLLM_PROMPT_LENGTHS=8200,8201 NANOVLLM_ENFORCE_EAGER=1 python3 example/test.py
```

　

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
| `NANOVLLM_PROMPT_LENGTHS` | 精确 prompt token 长度，逗号分隔；条目数就是测试 batch size |
| `NANOVLLM_MAX_GEN_TOKENS` | 每个请求生成 token 数，默认 16 |

　

## 当前边界

- DSA sparse budget 固定为 2048 token；短于该条件的请求没有卸载收益，会走 eager decode。
- 尚未实现 chunked prefill。长 prompt 仍是单次 prefill forward，过长时可能因激活值 OOM。
- 将来实现 chunked prefill 时，仍保持 prefill 与 decode 分开调度，不做混合 forward。
- DSA 整图不能用 padding bucket：`gather_selection_status` 是跨 step 持久化状态，虚假 padding 行可能污染真实请求状态。

实现与判定细节见 [FULL_DECODE_ONLY.md](FULL_DECODE_ONLY.md)。
