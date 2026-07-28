# 支持矩阵

本仓库只支持 `GLM-5.1-w4a8`，并要求 ModelSlim 1.0.0 per-channel W4A8 checkpoint、BF16 runtime、Expert Parallel 和 128-token KV blocks。

| KV 模式 | Eager | 稳定 decode `FULL_DECODE_ONLY` | Attention |
| --- | --- | --- | --- |
| `none` | 支持 | 支持 | Dense MLA |
| `lidu` | 支持 | 支持 | top-2048 + dense tail |

`FULL_DECODE_ONLY` 使用 raw outer ACLGraph。Prefill、首次 decode、LIDU 缓存初始化和首次 lazy capture 仍走 eager；只有后续稳定 decode replay 入图。LIDU 支持独立 SCATTER + Attention，以及 `enable_lidu_fused_attention_scatter=True` 的融合路径。

除此之外的模型、KV 卸载模式和执行模式均不在支持范围内。
