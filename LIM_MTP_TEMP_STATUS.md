# LIM-MTP 算子临时进展说明

> 临时文档：用于记录 `0eb50b1` 与 `0d44013` 的实现状态、步骤划分和当前性能问题，后续实现推进后需同步更新或删除。

## 步骤 1–4 说明

| 步骤 | 名称 | 主要内容 |
| --- | --- | --- |
| 步骤① | 每路判缺失与重排 | 四路 query 分别计算 TopK 2048；根据 `cache_slots` 判断 hit/miss；将 miss 排在前、hit 排在后，并按 token index 排序。 |
| 步骤② | 求四路 miss union | 合并四路有序 miss，使用 MrgSort 和单指针去重得到 union miss，同时需要建立 union miss 与原四路 miss 的对应关系。 |
| 步骤③ | 查找 eviction slot | 从随机 chunk 开始扫描四路 score；过滤无效 cache slot 和任一路超过 TopK 阈值的 token；通过 Sort32/MrgSort 找到 `union_miss_count` 个可淘汰位置，不足时进入 fallback。 |
| 步骤④ | 更新索引并回填 | 淘汰旧 token，将 union miss 写入对应 HBM slot，更新 `cache_slots`，再根据 union 与四路 miss 的对应关系回填四路 `topk_slots`。 |

## `0eb50b1` 与 `0d44013` 对比

| 步骤 | 功能 | `0eb50b1` | `0d44013` |
| --- | --- | --- | --- |
| 步骤① | 四路 TopK | 已完成 | 已完成 |
| 步骤① | 每路 hit/miss 判定与重排 | 已完成 | 已完成 |
| 步骤① | miss 在前、hit 在后 | 已完成 | 已完成 |
| 步骤① | 每路 miss 按 token index 排序 | 已完成 | 已完成 |
| 步骤② | 四路 miss union | 已完成 | 已完成 |
| 步骤② | MrgSort 合并与去重 | 已完成 | 已完成 |
| 步骤② | union miss 与四路 miss 对应关系 | 未完成 | 未完成 |
| 步骤③ | 随机 chunk 扫描 | 已实现 | 已实现 |
| 步骤③ | 四路 threshold candidate key | 已实现 | 已实现 |
| 步骤③ | 512 Sort32/MrgSort | 已实现 | 已实现 |
| 步骤③ | 512/1024/1536/2048 accumulator | 已实现 | 已实现并经过边界验证 |
| 步骤③ | eviction payload | 只保存 token index | `slot14 + index_low18`，long index high3 放在 key |
| 步骤③ | eviction slot 输出 | 没有 | 已输出到 `miss_dst_slots` |
| 步骤③ | scalar 候选校验 | 没有 | 有 |
| 步骤③ | 每个 chunk scalar 压缩 | 没有 | 有 |
| 步骤③ | accumulator scalar 压缩 | 没有 | 有 |
| 步骤③ | `miss_count <= 2048` | 只形成内部候选 | 已输出；通过 0/1/64/512/513/2048 边界测试 |
| 步骤③ | `miss_count > 2048` | 不支持 | 不支持，输出 `-1` |
| 步骤③ | fallback | 没有 | 没有 |
| 步骤④ | 淘汰旧 token | 没有 | 没有 |
| 步骤④ | union miss 写入 cache slot | 没有 | 没有 |
| 步骤④ | 更新 `cache_slots` | 没有 | 没有，仍保持只读 |
| 步骤④ | union slot 回填四路 miss | 没有 | 没有 |
| 步骤④ | 生成更新后的四路 `topk_slots` | 没有 | 没有 |

## 当前版本与时延说明

`0eb50b1` 已完成四路 TopK、hit/miss 重排、miss union，以及步骤③中基于四路阈值的随机 chunk 扫描、Sort32/MrgSort 和最多 2048 个候选的 accumulator，但候选只保存在内部，尚未输出和验证 eviction slot；`0d44013` 在此基础上增加了 packed eviction payload、`miss_dst_slots` 输出，以及 index/slot 映射、四路阈值、slot 范围和重复性校验。由于测试发现 vector 候选中存在无效或错误候选，为保证正确性，当前对每个 chunk 和 accumulator 都执行 scalar 校验与压缩，并在最终输出时再次校验，产生大量串行计算及对 `cache_slots`、四路 score 的随机 GM 读取，因此时延显著增加。
