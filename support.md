# 支持矩阵

本仓库只支持 `GLM-5.1-w4a8`，并要求 ModelSlim 1.0.0 per-channel W4A8 checkpoint、BF16 runtime、Expert Parallel 和 128-token KV blocks。

| KV 模式 | Eager | 稳定 decode `FULL_DECODE_ONLY` | Attention |
| --- | --- | --- | --- |
| `none` | 支持 | 支持 | Dense MLA |
| `offload_split` | 支持 | 支持 | LIDU + SCATTER + top-2048/tail Attention |
| `offload_fuse` | 支持 | 支持 | LIDU + 融合 SCATTER/Attention |

`FULL_DECODE_ONLY` 使用 raw outer ACLGraph。Prefill、首次 decode、LIDU 缓存初始化和首次 lazy capture 仍走 eager；只有后续稳定 decode replay 入图。`offload_fuse` 在稳定且 batch size 不超过 24 时使用融合算子，首次 decode、初始化 step 和更大 batch 回退分离路径。

除此之外的模型、KV 卸载模式和执行模式均不在支持范围内。
