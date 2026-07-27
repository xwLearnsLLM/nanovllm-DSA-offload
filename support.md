# 模型、卸载模式与 FULL_DECODE_ONLY 支持矩阵

本文梳理 `main` 分支当前支持的模型、decode KV 模式，以及各组合进入
`FULL_DECODE_ONLY` 的条件。

## 总体支持矩阵

当前正式支持两类模型、三种互斥的 decode KV 模式。六种组合均支持
eager，也都实现了 `FULL_DECODE_ONLY` 路径；卸载模式能否真正 replay
还取决于运行时条件。

| 模型 | `none` 不卸载 | `gs` 卸载 | `lidu` 卸载 | 整图后端 |
| --- | --- | --- | --- | --- |
| DeepSeek V3.2 BF16 | eager ✅ / 整图 ✅ | eager ✅ / 整图 ✅* | eager ✅ / 整图 ✅* | `npugraph_ex + outer ACLGraph` |
| GLM-5.1 W4A8 | eager ✅ / 整图 ✅ | eager ✅ / 整图 ✅* | eager ✅ / 整图 ✅* | raw outer ACLGraph，`npugraph_ex=False` |

`npugraph_ex=False` 不代表 GLM 没有组图。GLM 会直接用外层 ACLGraph
捕获完整 decode forward，仍然属于 `FULL_DECODE_ONLY`。

## 支持的模型

### DeepSeek V3.2 BF16

支持：

- shared-only/pruned BF16 导出模型；
- 保留 routed experts 的 BF16 导出模型；
- 例如 README 中的 `DeepSeek-V3.2-REAP-345B-A37B-BF16`；
- 兼容 V3.2 架构的 95B BF16 导出模型。

不支持直接加载原始 Hugging Face FP8 模型目录。代码会明确拒绝，并要求
先转换成 BF16。

### GLM-5.1 W4A8

支持当前的 ModelSlim checkpoint 格式：

- routed experts 保持原生 W4A8；
- Attention、dense/shared MLP 的 W8A8 权重在加载时反量化为 BF16；
- 要求存在 `quant_model_description.json`；
- 要求 ModelSlim `version=1.0.0`、`group_size=0`；
- 强制开启 Expert Parallel；
- 当前没有 MTP，运行日志会显示 `MTP is disabled`。

模型识别与格式校验位于 `nanovllm/config.py`。

## Decode KV 模式

公开参数为：

```python
LLM(..., offload_mode="none")
```

示例脚本通过 `NANOVLLM_OFFLOAD_MODE` 传入。仅支持：

- `none`：不卸载，默认；
- `gs`：LightningIndexer + GatherSelectionKVCache；
- `lidu`：LIDU + SCATTER。

旧参数 `enable_dsa_offload` 和环境变量
`NANOVLLM_ENABLE_DSA_OFFLOAD` 已删除。

### `none`：不卸载

- 完整 KV 保留在 HBM；
- Attention 对完整历史 KV 做 dense MLA；
- 不需要 DRAM KV blocks；
- `FULL_DECODE_ONLY` 在启动时预先 capture；
- runtime batch 可以向上 padding 到已配置的 capture size，不要求精确相等。

Dense MLA 整图要求 capture size 不超过 KV cache block size，以便 padding
行使用互不冲突的 null-block slots。

### `gs`：LightningIndexer + GatherSelection

- 旧的 DSA decode 卸载方案；
- GS 同时负责索引管理和 DRAM→HBM 搬移；
- HBM 稀疏预算固定为 2048 token；
- Attention 覆盖选中的 2048 token、prompt tail 和 decode token；
- 支持 `FULL_DECODE_ONLY`，但使用 exact-size capture。

真正 replay 必须同时满足：

1. 当前 batch 确实发生 DSA 卸载；
2. batch 中所有请求都是 offload row；
3. runtime batch 精确匹配某个 capture size。

短请求与长请求混合时，整个 decode step 会退回 eager。

### `lidu`：LIDU + SCATTER

- LIDU 融合 top-2048、hit/miss、eviction 和索引更新；
- SCATTER 只负责 DRAM→HBM 搬移；
- GLM 可用 `enable_lidu_fused_attention_scatter=True`（示例环境变量为 `NANOVLLM_ENABLE_LIDU_FUSED_ATTENTION_SCATTER=1`）改走融合 SCATTER + sparse-and-tail Attention；
- HBM 缓存预算按原始 prompt 长度选择；
- 支持 `FULL_DECODE_ONLY`，使用 exact-size lazy capture；
- 允许 `C=0` 短请求和 `C>0` 长请求混合入图。

默认缓存预算如下：

| 原始 prompt 长度 | HBM 缓存预算 C |
| --- | ---: |
| `<= 2048` | 0 |
| `2049–8192` | 2048 |
| `8193–16384` | 3072 |
| `16385–32768` | 6144 |
| `32769–65536` | 8192 |
| `>= 65537` | 12288 |

真正 replay 必须同时满足：

1. batch 中至少有一个 `C>0` 请求；
2. 所有需要初始化的请求均已完成 LIDU 缓存初始化；
3. runtime batch 精确匹配某个 capture size。

如果 batch 中所有请求都是 `C=0`，则没有真实 DSA update，运行时会走
`eager_no_dsa`。

GLM 和 DeepSeek 在 LIDU 模式下的 Attention 路径不同：

- GLM + LIDU 使用 `sparse_and_tail_attention`，计算缓存中的 top-2048
  加 dense tail；开启融合开关后，稳定 decode 且 batch size 不超过 24 时改用
  `sparse_and_tail_attention_and_scatter_copy`，首次 decode、初始化 step 和更大
  batch 自动回退旧的 SCATTER + Attention 路径；
- DeepSeek + LIDU 仍使用 dense MLA，计算全部 C 个缓存 token
  加 prompt tail 和 decode token；融合开关目前不支持 DeepSeek。

## FULL_DECODE_ONLY 的边界

通过以下环境变量选择执行方式：

```bash
# 全程 eager
export NANOVLLM_ENFORCE_EAGER=1

# 稳定 decode 尝试 FULL_DECODE_ONLY
export NANOVLLM_ENFORCE_EAGER=0
```

共同规则：

- prefill 始终 eager；
- batch 中任意请求首次进入 decode 时，整个 decode step 都走 eager；
- LM head 和 sampler 始终在图外；
- 只有后续稳定 decode 的 model forward 才可能 graph replay。

不同模式的入图节奏：

- `none`：第一个 decode eager，后续符合 capture size 即可 replay；
- `gs`：第一个 decode eager，后续还必须满足全 offload 和 exact-size；
- `lidu`：
  1. 首次 decode eager，完成请求缓存初始化；
  2. 第一个 initialized stable decode 执行 lazy capture，但仍以 eager
     产生当前 token；
  3. 再下一步开始 graph replay。

如果生成过程中因为 EOS 导致 active batch 缩小，而缩小后的 batch 没有
对应 capture size，GS/LIDU 会退回 eager。稳定性能测试通常设置：

```bash
export NANOVLLM_IGNORE_EOS=1
```

以保持固定 batch。

## 结论

当前 `main` 支持：

- DeepSeek V3.2 BF16；
- GLM-5.1 W4A8；
- `none`、`gs`、`lidu` 三种 decode KV 模式；
- 六种“模型 × KV 模式”组合的 eager 和 `FULL_DECODE_ONLY` 路径。

其中，`none` 的整图准入相对宽松；GS/LIDU 只有在真实触发卸载、初始化
完成且 batch 命中 exact capture size 后，稳定 decode 才会真正 replay。
