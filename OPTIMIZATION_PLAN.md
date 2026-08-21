# FusedCopySfaMtp 性能优化方案

## 1. 目标与结论

目标算子为 `fused_copy_sfa_mtp`：在 Ascend 上完成 MTP token 级 DRAM -> HBM KV 搬运，并执行 Sparse Flash Attention (SFA)。

当前最推荐的实施顺序：

1. 接入上游已有的 `topk_miss_counts[4B]`，实现逐 query 的 miss/hit 精确分流。
2. 在不损失核利用率的 batch 区间内，将同一 MTP request 的相邻 query 聚合到同一 AIC + 2 AIV mixed group。
3. 在该 group 内通过 `ready-slot` 标记复用前序 query 已回填到 HBM 的 miss KV，减少跨 query 重复 DRAM 读取。
4. 基于 profiler 决定是否继续做 DRAM look-ahead 预取、DRAM block-table 预取等高复杂度优化。

仅修改 tiling、将 4 个 MTP query 放到同一 core，并不会自动减少 DRAM 访问；必须同时改变 source-aware gather 的数据来源选择，才能获得实质收益。

## 2. 当前实现与约束

### 2.1 输入和数据布局

- 每个 request 有 4 个 MTP query；每个 query 有 2048 个 TopK token。
- 上游 `fused_li_manage_mtp` 已输出：
  - `topk_source_ids[4B, 1, 2048]`
  - `topk_slots[4B, 1, 2048]`
  - `topk_miss_counts[4B]`
  - `miss_source_ids[B, 8192]` / `miss_destination_slots[B, 8192]` / `miss_counts[B]`。
- 已知的数据契约：每个 query 的 `topk_source_ids` 为 miss 前缀、hit 后缀；`topk_miss_counts[q]` 给出前缀长度。miss 使用有效 DRAM source token id，hit 为无效 source id。

当前 fused 算子只接收 union 级 `miss_counts[B]`。一个 request 只要有任意 union miss，4 个 query 都会进入 source-aware 逻辑，即使某个 query 实际为 0 miss。

### 2.2 当前 SFA 调度

MTP 模式下，当前 runtime 按 `(request, query)` 作为基础任务：

```text
任务数 = 4 * B
```

当 `4B < AIC 数` 时，同一 request 的 4 个 query 往往会落到不同的 mixed group。每个 mixed group 包含 1 AIC + 2 AIV，且拥有私有 workspace。

SFA 按 512-token `s2` tile 顺序处理。由于 miss 在前、hit 在后，执行顺序天然为：

```text
miss / mixed tile -> hit tile
```

当前 source-aware 路径本身已经是流式路径：

```text
DRAM(GM) --MTE2--> UB --MTE3--> SFA merge workspace
                                      |--MTE3--> HBM persistent KV cache
```

因此“预取到 workspace”不是一个可以绕过 UB 的独立 DMA；仍需要 UB、MTE 资源和事件同步。

### 2.3 Workspace 约束

SFA 的 merge workspace 是按 AIC group 分配的四槽 ring buffer。仅 KV merge 区的大小约为：

```text
4 * 512 * (512 + 64) * sizeof(KV dtype)
```

bf16/fp16 下约为 2.25 MiB/group。该区域已经参与 Vector0 -> Cube 的流水，不能直接复用为无同步的预取缓存。

## 3. 优化项与优先级

| 优先级 | 优化项 | 复杂度 | 预期收益 | 关键前提 |
|---|---|---:|---|---|
| P0 | 接入逐 query `topk_miss_counts`，miss/hit 分流 | 低 | 稳定的小到中等收益；消除无效 source-aware 开销 | 上游 miss 前缀布局正确 |
| P1 | request-affinity tiling + 跨 query miss HBM 复用 | 中高 | 重合率高且 batch 足够大时，中到大收益 | 不降低核利用率；实现 ready-slot 同步 |
| P2 | DRAM block-table / metadata UB 缓存 | 中 | 高 miss 场景的小到中等收益 | UB 容量、block table 宽度合适 |
| P3 | look-ahead DRAM 预取到专用 staging workspace | 高 | 仅 MTE2/DRAM latency 明显时可能有效 | 额外 UB/workspace 与流水同步 |
| P4 | source token 连续 run 合并搬运 | 高 | 取决于 source token 连续性，收益不稳定 | profile 证明存在连续 run |

## 4. P0：逐 query miss/hit 精确分流

### 4.1 接口改动

为 `fused_copy_sfa_mtp` 新增输入：

```text
topk_miss_counts: int32 [4B]
```

需要同步更新：

- Torch schema、adapter 和调用点；
- OpDef、tiling 输入索引与 shape/dtype 校验；
- kernel 参数和 `InitSourceAwareGather` 参数；
- UT、ACLGraph replay 测试和性能测试。

union `miss_counts[B]` 保留给现有 ABI/调试/后续 union-copy 路径，但 source-aware attention 的 query 分流改用 `topk_miss_counts`。

### 4.2 内核分流

对 query `q`，读取：

```text
m = topk_miss_counts[q]
```

并按如下路径执行：

```text
m == 0:
  全部 2048 token 走纯 HBM gather

0 < m < 2048:
  [0, m)      走 DRAM source-aware gather，并回填 HBM
  [m, 2048)   走纯 HBM gather
  只有边界 512-token tile 为 mixed tile

m == 2048:
  全部走 DRAM source-aware gather
```

例如 `m = 200` 时，首个 512-token tile 为 200 miss + 312 hit，余下 3 个 tile 均是纯 hit。相比当前 union count 非零时所有 tile 都执行 source-aware 元数据加载与来源判断，可显著缩短 hit 后缀的 Vector0 路径。

### 4.3 预期收益与边界

P0 不减少真实 miss 的 DRAM payload 总量；主要减少：

- hit-only query 的 source-id 元数据读；
- hit-only tile 的 source-id 分支和 DRAM offset 判断；
- 无效的 persistent-copy 相关控制流；
- source-aware 与纯 HBM 路径混用导致的 UB/事件开销。

收益取决于 `topk_miss_counts` 分布。典型低 miss 情形下，P0 的收益通常为小到中等，但实现风险低，是后续优化的必要基础。

### 4.4 正确性要求

- 强制校验 `0 <= m <= 2048`。
- debug/UT 中校验 `[0, m)` source id 合法，`[m, 2048)` source id 为 hit 标记；生产路径可保留安全 fallback。
- `topk_slots` 和 `topk_source_ids` 必须同序重排；不能只重排 source id。
- 覆盖 miss count：`0, 1, 255, 256, 511, 512, 513, 2048`。
- 覆盖一组四 query 不均匀场景，例如 `[0, 200, 700, 2048]`。

## 5. P1：request-affinity tiling 和跨 query miss 复用

### 5.1 为什么可行

同一个 MTP request 的 4 个 query 在跨预测步/相邻 query 中通常具有较高 TopK 重合率。当前 query 分散时，多个 group 可能对同一个 source token 各自执行：

```text
DRAM -> UB -> workspace + HBM
```

若将相邻 query 放到同一个 mixed group 并顺序处理：

```text
q0: 首次 miss 从 DRAM 读取，写入 HBM slot
q1/q2/q3: 同 slot 已 ready 时，直接从 HBM gather
```

即可把同一个 source 的多次 DRAM 读取压缩为一次。

### 5.2 仅改 tiling 不足够

上游生成的 `topk_source_ids` 是执行前的 miss 标记。即使 q0 已经写入 HBM，q1 的 source id 仍然表明它原本是 miss；若不额外判断，q1 仍会从 DRAM 读取。

因此 P1 必须配套一个 request-local 的 `ready-slot` 状态：

```text
if source is miss and ready_slot[topk_slot]:
    从 HBM gather
else if source is miss:
    从 DRAM gather
    完成 HBM persistent copy 后设置 ready_slot[topk_slot]
else:
    从 HBM gather
```

`ready-slot` 推荐按 destination slot 建位图，而不是按 source token 建大 hash：相同 source 在缓存映射后应对应相同 slot，位图更小。必须确保 HBM persistent copy 已完成后再置 ready，避免下一个 query 读到被替换前的数据。

### 5.3 tiling 策略

不能无条件将 4 个 query 聚合。小 batch 时这会把可并行任务数从 `4B` 降到 `B`，造成严重欠占用。

建议根据 `B` 和可用 AIC 数 `A` 选择每个 task 的 query 数：

```text
B < A / 2:
    query_group_size = 1

A / 2 <= B < A:
    query_group_size = 2

B >= A:
    query_group_size = 4
```

其中 2-query task 应优先合并相邻 query，例如 `(q0,q1)`、`(q2,q3)`。该策略的目的不是减少总 Attention 计算量，而是在不减少 active mixed group 数的前提下捕获 DRAM miss 重用。

需要在 fused-MTP tiling suffix 中增加 `query_group_size` 等 MTP 专用字段，并替换当前按 `4B` 单元的通用平均分配。workspace 可继续按 group 私有分配，无需跨 group 共享。

### 5.4 收益模型

定义：

```text
D = sum(topk_miss_counts[q])       # 四个 query 的 miss occurrence 数
U = miss_counts_union              # unique miss 数
R = D / U                          # 跨 query 重合倍率
p = 当前总耗时中可被 DRAM source gather 改善的比例
```

理想情况下 DRAM 读取从 `D` 降到约 `U`，端到端理论上界：

```text
speedup <= 1 / (1 - p * (1 - 1 / R))
```

示例：

| DRAM 可改善占比 p | 重合倍率 R | 理论上界 |
|---:|---:|---:|
| 40% | 2 | 1.25x |
| 50% | 2 | 1.33x |
| 50% | 3 | 1.50x |

这只是上界；实际收益还会受 HBM 写回、ready-slot 同步、Cube 计算和 batch 核利用率影响。

### 5.5 限制

- 该方案只保证单次 fused kernel 内的 4 个 MTP query 复用。
- 不同 decode step 是不同 kernel launch，不能依赖“下一次仍调度到同一 core”；跨 launch 复用应由已经回填的 HBM KV cache 提供。
- 不建议第一版缓存/复用全部 hit MLA payload。一个 token 的 CKV + KPE 约 1152 B，四个 TopK 行的完整数据远超 UB；显式 workspace 复用会增加额外 GM 流量。

## 6. P2：低风险的辅助内存优化

### 6.1 DRAM block table 缓存

`GetDramKeyGmOffset` 对每个 miss 读取 `dram_block_table`。对高 miss request，可在每个 mixed group 首次处理该 request 时将活跃 block table 预读到 UB/L1，后续 source token 的物理 block 查询在本地完成。

仅在 block table 宽度和 UB 余量满足条件时启用；需要与 source gather 的 UB 使用情况共同评估。

### 6.2 元数据加载

P0 后，hit-only tile 不应加载 `topk_source_ids`。对 mixed tile，可保持 slot/source 的 512-entry UB 批量加载，避免标量 GM 读取。

## 7. P3：look-ahead DRAM 预取

### 7.1 可行性判断

由于布局为 miss 在前、hit 在后，不能通过“先算 hit，再预取 miss”隐藏首个 miss tile 的 DRAM 延迟；首个 miss tile 必然先被消费。

只有在 profiler 显示 Vector0 明显等待 MTE2/DRAM，且 MTE/GM 带宽尚有余量时，才值得做 look-ahead。

### 7.2 候选实现

- 为未来仍含 miss 的 tile 分配独立 ping-pong UB 和 staging workspace；
- 使用 MTE2 将 DRAM 数据读入 staging UB，并通过 MTE3 写入专用 staging workspace；
- 当前 tile 的 Cube/Vector 后续阶段运行时，发起下一 miss tile 的预取；
- 消费时把 staging 数据并入最终 merge workspace，并维护 token 顺序和 valid-size；
- 通过事件防止 staging buffer 与现有 `loop % 4` merge ring 覆盖。

该方案至少需要额外 1 个 512-token staging tile；实用上建议双缓冲，bf16/fp16 约增加 1.125 MiB/group workspace，且会竞争 MTE2/MTE3。

### 7.3 风险

- 不减少 DRAM 总 payload；
- 可能因额外 workspace 写入/读取而变慢；
- 侵入现有 Vector0 -> Cube pipeline 和事件依赖；
- 改变 tile 处理顺序时需重新验证 online softmax 数值稳定性。

## 8. P4：连续 source run 合并

可以检测连续 DRAM source token，在同一 physical DRAM block 内合并为较大的 DataCopy。

不建议按完整 128-token block 盲目预取：TopK source 通常稀疏，平均每 block 的命中数可能很低，整 block 搬运会过取。应先由 profile 或离线统计确认连续 run 分布，再决定是否实施。

## 9. 验证与性能评估

### 9.1 正确性

- 与 split `scatter_copy + sparse_tail_attention` 对比 attention 输出和 HBM CKV/KPE；
- 覆盖 bf16/fp16、不同 batch、不同 miss 分布、tail token、重复 replay；
- 重点验证 P1：同 source 在多个 query 中出现时，仅首次走 DRAM，后续从完成回填的 HBM slot 读取；
- ACLGraph capture/replay 下验证依赖和 HBM 写回可见性。

### 9.2 必采集指标

```text
batch B、active AIC group 数、query_group_size
每 query topk_miss_counts
D、U、R = D/U
hit-only tile 数、mixed tile 数、all-miss tile 数
DRAM payload bytes、HBM read/write bytes
MTE2 stall、MTE3 stall、Vector0 时间、Cube 时间
端到端 fused latency
```

### 9.3 决策门槛

- P0 后，若 hit-only tile 占比高且 Vector0 时间下降，保留精确分流。
- P1 仅在 `B` 足以维持核满载、且 `R` 显著大于 1 时启用。
- P3 仅在 MTE2/DRAM latency 明显且没有带宽饱和时进入开发。
- P4 仅在 source token 连续 run 统计显示足够长的 run 时开发。

## 10. 推荐实施里程碑

1. **M1 / P0**：新增 `topk_miss_counts` ABI，完成逐 query miss/hit 分流和完整 UT。
2. **M2 / 测量**：采集 miss 分布、`D/U`、MTE/Cube 分解，并建立按 batch 的 baseline。
3. **M3 / P1**：实现 `query_group_size = 1/2/4` 的 request-affinity tiling；先不启用 ready-slot，只验证调度和核利用率。
4. **M4 / P1 完整版**：增加 ready-slot 位图、query 边界同步和 DRAM -> HBM 复用；仅在满足 batch/重合率阈值时启用。
5. **M5 / P2-P4**：根据 profiler 选择 block-table 缓存、look-ahead 预取或连续 run 合并。

