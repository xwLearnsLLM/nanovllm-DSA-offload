# `fused_li_manage` 行为验收

`torch_npu.npu_lightning_indexer` 生成 Top2048 reference；本仓
`torch.ops.nanovllm_dsa.fused_li_manage` 同时完成稀疏选择、命中判断、
淘汰和 request-pool 状态更新。

## 接口约束

- 调用方预先创建 `topk_src_ids`、`topk_dst_slots` 和 `miss_counts`。
- `cache_slots_pool` 与三个输出 buffer 均由算子原地写入；算子返回
  `None`，没有 allocating 入口、`_out` 入口或 alias 输出。
- `req_pool_entries` 映射活跃请求到持久 request-pool row；未映射 row
  必须保持不变。

## 单请求行为

- `topk_src_ids` 包含与 LightningIndexer reference 相同的 2048 个唯一
  token ID。
- `miss_counts` 等于根据更新前 cache state 重新计算出的缺失数。
- `topk_dst_slots` 全部位于 `[0, C)` 且唯一。
- 原 hit token 保持原 slot；更新后
  `new_cache_slots[topk_src_ids[i]] == topk_dst_slots[i]`。
- 更新后的有效 slot 仍唯一覆盖 `0..C-1`。
- 使用相同 query 连续更新时，第二次 `miss_counts == 0`。
- 不约束 miss 段或 hit 段内部顺序。

## 边界与计时

- 单 query LIM 继续覆盖 18-bit/21-bit source ID 边界；MTP LIM 仍只覆盖
  18-bit source ID。
- 性能数据只打印，不用时延阈值决定 UT 成败。
- 计时前恢复 cache state，且恢复与同步不计入算子时延。
