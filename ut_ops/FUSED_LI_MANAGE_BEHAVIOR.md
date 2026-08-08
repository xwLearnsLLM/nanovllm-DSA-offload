# `fused_li_manage` 索引管理基本行为检查

## 测试对象

1. `torch_npu.npu_lightning_indexer`：生成 Top2048 reference。
2. `fused_li_manage_out`：需要验收的 LI + 索引管理算子。

## 测试数据

1. query heads 分别覆盖 32 和 64，query、key、weights 使用可复现数据。
2. 每个 batch row 的 `actual_seq_len` 位于 `[2048, 2097151]`。
3. `req_pool_entries` 是互不重复且合法的 `int32[B]` request-pool row 映射。
4. 每个请求恰好有 C 个有效 cache slot；C 是 128 的倍数且 `2048 <= C < 16384`。
5. 有效 cache slot 唯一，值集合恰好为 `0..C-1`。
6. miss_count 位于 `[0, 2048]`，并满足 `miss_count <= actual_seq_len-C`。
7. 测试数据应避免 Top2048 截止位置出现相差不超过 7 ULP 的 score 近似并列，确保原版 reference 与 score-low3 路径可做确定性的集合比较。

## 每个 batch row 独立验收

1. `topk_index` 包含 2048 个合法且唯一的 token index，其集合与 LightningIndexer reference 相同。
2. `miss_count` 等于构造值，也等于根据 old cache 和实际 `topk_index` 重新计算出的 miss 数量。
3. `topk_index[:miss_count]` 在 old cache 中全部为 `-1`。
4. `topk_index[miss_count:]` 在 old cache 中全部命中。
5. `topk_slots` 全部位于 `[0, C)` 且互不重复。
6. hit 后缀的输出 slot 与 old cache 中的 slot 逐项相等。
7. `new_cache_slots[topk_index[i]] == topk_slots[i]` 对全部 2048 项逐项成立。
8. new cache 的有效 slot 数量、唯一性和值集合 `0..C-1` 保持不变。
9. 未被 `req_pool_entries` 映射的 request-pool row 逐项保持不变。
10. 不恢复 cache 立即进行第二次更新时，`miss_count==0`，全部 Top2048 token 保持命中且slot映射正确。
11. `fused_li_manage` 不支持 IOSORT；只要求 miss 在前、hit 在后，不约束两段内部的 token 顺序。
12. `_out` 接口返回的 `topk_index`、`topk_slots`、`miss_count` 和 `cache_slots` 必须分别alias调用方传入的持久输出tensor。

## 18-bit/21-bit边界

1. `actual_seq_len <= 262144` 时使用精确的 `18-bit index + 14-bit slot` payload。
2. `actual_seq_len > 262144` 时，payload保存index低18 bit和slot，FP32 score低3 bit直接保存index高3 bit。
3. score-low3编解码必须覆盖index高3 bit的全部取值，重建后的token index必须完全一致，score扰动不超过7 ULP。
4. 至少覆盖 `262144`、`262272`、`1048576` 和 `2097151`；最后一个case必须实际选中超过旧18-bit边界的token。
5. `actual_seq_len` 最大为 `2^21-1=2097151`，对齐后的source capacity最大为 `2^21`。

## Reference与计时

1. 当前测试允许在同一 Python 进程中调用原生 `torch_npu.npu_lightning_indexer` 和本仓库 `fused_li_manage`；两者必须使用完全相同的 query、key、weights、序列长度和 block table。
2. `--iters=0` 只做正确性验收；`--iters>0` 时额外使用 NPU Event 统计 LightningIndexer 和 `fused_li_manage` 时延。
3. 每次计时调用 `fused_li_manage` 前恢复独立的 initial cache 副本。
4. cache恢复与NPU同步必须发生在Event起点之前，不计入算子时延。
5. `fused_li_manage - LightningIndexer` 仅作为索引管理额外开销的代理值，其中还包含 cache_slots 访问、score workspace 及缓存更新成本。
