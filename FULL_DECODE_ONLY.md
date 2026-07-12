# DSA FULL_DECODE_ONLY

## 执行边界

`Config.enforce_eager` 是唯一模式开关：

- `True`：prefill、首个 decode 和稳定 decode 全部 eager。
- `False`：prefill 与首个 decode eager；满足条件的稳定 decode replay 一张完整图。

整图范围从 decode model forward 的 embedding 开始，到最终 hidden states 结束。LM head 与 sampler 不在图内。

## 整图进入条件

一次 decode 必须同时满足：

1. 不是首个 decode step；
2. batch 中每一行都需要 DSA offload 更新；
3. 实际 batch size 等于一个预 capture 的精确 size；
4. 对应整图已完成 capture。

不满足条件时走 eager，并分别累计 `eager_first_decode`、`eager_no_dsa`、`eager_mixed_batch` 或 `eager_uncaptured_batch`。

这里不能像非卸载 baseline 那样把小 batch padding 到 bucket。`gather_selection_status` 按请求跨 decode step 持久化；padding 行执行 gather 会产生额外状态写入，存在污染真实请求的风险。

## 图结构

每个精确 batch size 使用：

- TorchAir `npugraph_ex` 对完整 decode Python forward 做 FX 优化；
- 一张外层 `torch.npu.NPUGraph` 捕获完整稳定 decode；
- 每层 FIA-v2 attention task 在 replay 前更新实际 KV 长度；
- `lightning_indexer` 与 `gather_selection_kv_cache` 通过 `torch.library` schema、PrivateUse1 实现、Meta/Fake 实现和 TorchAir GE converter 进入 Dynamo/TorchAir 图。

运行路径只保留这一种整图实现。旧的局部 TorchAir pipeline cache、piecewise 路径和 `sparse_flash_attention` 探针已经删除。

## 验收

以 README 的两个 8K prompt、生成 16 token 命令为例，最终应看到：

```text
DSA FULL_DECODE_ONLY proof: capture_sizes=[2], captures=1, replays=14, eager_first_decode=1, eager_no_dsa=0, eager_mixed_batch=0, eager_uncaptured_batch=0
DSA decode hot path: compact_ipc_steps=15, average_ipc_bytes=..., metadata_cache_hits=..., metadata_cache_misses=..., graph_metadata_refreshes=..., graph_metadata_reuses=...
```

稳定 batch 下，`compact_ipc_steps` 应随 decode step 增长，`metadata_cache_hits`
和 `graph_metadata_reuses` 应持续增长；miss/refresh 通常只发生在首次进入 decode、
请求集合变化或每 128 token 新增 KV block 时。

实际性能以预热后的稳定 TPOT 为准；前几个 decode step 的时延不作为优化目标。
