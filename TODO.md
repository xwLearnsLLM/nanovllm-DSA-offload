# TODO

- 保持 `fused_copy_sfa_mtp` ABI 不变，并持续要求 HBM payload 与 Attention 输出均和 split 路径逐元素一致。
- 优先优化 `B=4/24/32`、每请求 TopK union≈3K～4K、unique union misses≈300 的典型负载。
- 重点减少同一个 union miss 被多个 query 重复读取 DRAM 的次数，同时保留当前 query gather 与 HBM writeback 流水。
- 优化完成后再合回 nanovllm 主仓库。
