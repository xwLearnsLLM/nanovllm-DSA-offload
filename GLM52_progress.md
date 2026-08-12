# GLM-5.2 W4A8 适配进度与计划

## Checklist

- [x] 从 `main@7c096e0` 创建独立分支 `main-glm52`。
- [x] 完成 GLM-5.2 模型识别、官方 IndexShare 拓扑校验和第一阶段能力门禁（`7b2effb`）。
- [x] 在昇腾上验收 `MTP0 + none + eager`：短序列、21K、接近 64K。
- [x] 完成 MTP0 的 target-layer IndexShare：先 `offload_split + eager`，再 `offload_fuse + eager`。
- [x] 完成 MTP0 的 stable full-decode-only graph。
- [ ] 适配 GLM-5.2 的量化 MTP layer，打通 `MTP3 + none` 的 eager 和 graph。
- [ ] 完成 MTP3 的 target-layer IndexShare offload：split、fuse、eager、graph。
- [ ] 处理 MTP iteration 内的 IndexShare，完成 20K～64K 性能验收、回归和文档收尾。

## 1. 当前状态

- 工作目录：`D:\vLLM-ascend\nanovllm-DSA-offload-mtp`
- 工作分支：`main-glm52`
- 分支基线：`main@7c096e0`
- 第一阶段功能提交：`7b2effb feat: add GLM-5.2 nonoffload eager bring-up`
- 阶段 0、阶段 1 和阶段 2 均已完成并通过当前范围内的验收；阶段 2 的实现提交为 `15f1a7f`，后续修复为 `f9f4123` 和 `7c77e81`。
- 阶段 2 昇腾整网已覆盖 21K、40K 和 64K 的 `MTP0 + offload_split + eager`。21K 和 64K 与 `none` 的 token IDs 完全一致；40K 在后续 greedy token 上存在稀疏 top-2048 + dense-tail Attention 的预期近似分叉。
- 40K eager profile 确认每个 stable decode step 为 21 次 LIM、78 次 SCATTER、78 次 SFA；同一 IndexShare group 内的 miss metadata 一致，KV payload 仍按层独立。
- 阶段 3（MTP0 + offload_fuse + eager）和阶段 4（MTP0 full-decode-only graph）均已完成并通过验收。阶段 5 已打通 `MTP3 + offload_mode=none + eager`，并完成 21K、40K、64K 的 K=0/K=3 token 对齐验收。
- 阶段 5 仅改 Python 模型加载、调度与 target 验证，不需要重新编译 Ascend 自定义算子；下一阶段为 MTP3 nonoffload full-decode-only graph。

相关文档：[`README.md`](README.md)、[`README_ops.md`](README_ops.md)、[`TODO.md`](TODO.md)。

## 2. 范围与非目标

本分支只面向 [Eco-Tech/GLM-5.2-w4a8](https://www.modelscope.cn/models/Eco-Tech/GLM-5.2-w4a8)，同时不能破坏当前 [Eco-Tech/GLM-5.1-w4a8](https://www.modelscope.cn/models/Eco-Tech/GLM-5.1-w4a8) 的既有行为。

本轮目标：

- 只支持常用的 20K～64K 序列。
- 支持 `MTP0/MTP3`。
- 支持 `none/offload_split/offload_fuse`。
- 支持 eager，以及只覆盖后续稳定 decode step 的 full-decode-only graph。
- 优先保证语义正确、输出稳定，再做 steady decode TPOT 优化。

本轮不做：

- 不支持 GLM-5.2 宣称的 1M 上下文。
- 不引入 prefill/decode 混合 forward。
- 不要求 prefill、首次 decode、缓存初始化或首次 lazy capture 很快；这些阶段可以保持 eager。
- 不修改 offloading 强相关 7 个算子的 ABI，除非用户先明确同意。
- 不从外部算子工程引入运行时依赖。算子来源只能是 `torch/torch_npu` 官方算子或本仓自定义算子。

## 3. GLM-5.2 相比 GLM-5.1 的关键差异

### 3.1 主干结构基本不变

两者的 target backbone 对当前 nanovllm 来说基本一致：78 个 target layer、hidden size 6144、256 experts、top8、前三层 dense、后续 MoE，以及 MLA 的显式维度均可继续复用当前实现。

不要被通用配置字段 `head_dim` 的变化误导。需要以 `qk_nope_head_dim`、`qk_rope_head_dim`、`v_head_dim`、`kv_lora_rank` 等显式 MLA 字段为准；当前 kernel 约束仍按已有 512/64 latent KV 布局处理。

### 3.2 Target layer 使用 IndexShare

GLM-5.1 的 78 个 target layer 都独立运行 Indexer。GLM-5.2 的官方 schedule 是 21 个 `full` layer 和 57 个 `shared` layer：

```text
full layers = 0, 1, 2, 6, 10, 14, ..., 74
```

`shared` layer 复用最近一个 `full` layer 的稀疏选择结果。对应 group 为：

```text
[0], [1], [2,3,4,5], [6,7,8,9], ..., [74,75,76,77]
```

这意味着：

- LIM 只在 21 个 `full` layer 执行。
- 78 层各自拥有不同的 KV 数据，所以每层仍必须执行自己的 COPY+SFA，或者 COPYSFA。
- shared layer 复用的是选中 token 的逻辑 ID、最终 HBM slot 和 miss/update metadata，不是复用 KV payload、attention query 或 attention 输出。

### 3.3 MTP layer 的量化方式不同

当前 GLM-5.1 MTP 实现假定 checkpoint 的第 78 层为 FLOAT/BF16，并使用 `GlmFloatSparseMoeBlock`。GLM-5.2 的第 78 层主要是 ModelSlim W4A8/W8A8 权重，因此不能只放宽配置校验；必须真正适配量化权重加载和执行路径。

建议复用 target layer 已有的 W4A8 routed experts、W8A8 attention/dense 权重反量化逻辑，不要再维护一套相似实现。

### 3.4 两种 IndexShare 不要混淆

- `indexer_types`：78 个 target layer 之间的跨层共享，本计划优先实现。
- `index_share_for_mtp_iteration`：同一个 MTP layer 在多次 draft iteration 之间共享 Indexer 结果，属于后续 MTP3 优化。

MTP3 target verification 仍有每请求 4 个 query。`fused_li_manage_mtp` 仍需对这 4 个 query 的 top2048 并集做管理；不能因为 `index_share_for_mtp_iteration=true` 就把它误改成 query_len=1。

## 4. Target-layer IndexShare 的核心设计

### 4.1 Group 共享索引状态，layer 保留独立 KV

每个 IndexShare group 维护一份持久状态：

- `cache_slots_pool`
- 固定地址的 `topk_src_ids/topk_dst_slots`
- MTP3 所需的 `miss_src_ids/miss_dst_slots/miss_counts`
- request-pool entry 和该 group 所需的图内 metadata

group 内每层仍分别维护：

- HBM CKV/KPE cache
- DRAM CKV/KPE cache
- 本层 attention query

只有 `full` layer 需要 Indexer 权重、index key cache 和 LIM 调用。shared layer 直接消费 group 的上一次 LIM 输出。

### 4.2 初始化必须让 group 内物理 slot 对齐

不能让四层先各自建立不同的 token→slot 映射，再直接复用 full layer 的 slot metadata。正确初始化方式是：

1. group 的 full layer 计算初始 top-C logical token IDs。
2. group 只建立一次 logical token→HBM slot 映射。
3. group 内每一层把自己的 DRAM KV payload 拷贝到同一组 logical destination slots。
4. 初始化成功后，group 内所有层共享同一份映射状态，但 KV payload 仍然按层隔离。

请求 finish、abort、preemption、request-pool entry 复用和 batch 重排后，都必须保持这个约束。

### 4.3 稳定 decode 的调用关系

MTP0：

```text
full layer:
  index projection -> fused_li_manage -> 本层 COPY/SFA

shared layer:
  复用 group metadata -> 本层 COPY/SFA
```

MTP3 target verification：

```text
full layer:
  4-query index projection -> fused_li_manage_mtp -> 本层 COPY/SFA-MTP

shared layer:
  复用 group metadata -> 本层 COPY/SFA-MTP
```

预期每个 stable decode step 的算子次数：

| 路径 | LIM 次数 | COPY/SFA 或 COPYSFA 次数 |
| --- | ---: | ---: |
| MTP0 | 21 | 78 |
| MTP3 | 21 | 78 |

如果 profile 中仍出现 78 次 LIM，说明 IndexShare 没有真正生效；如果 COPY/SFA 只有 21 次，则实现错误地复用了不同层的 KV 或 attention 结果。

### 4.4 Graph 设计

- group metadata 必须使用 caller-owned、固定地址的 buffer。
- full layer 的 LIM 写这些 buffer，后续 shared layer 在同一张 full-decode-only graph 内读取。
- 首次 decode、group cache 初始化、首次 lazy capture 可以 eager。
- 只要求后续 batch shape 稳定且所有请求初始化完成的 decode step replay。
- 不为前几个 decode step 的时延增加复杂状态机。

## 5. 分阶段实施与验收

每个阶段单独提交。用户在昇腾机器上 `git pull` 并验证后，才能继续下一阶段；不要把多个尚未验收的阶段堆进一个 commit。

### 阶段 0：验收当前 nonoffload eager bring-up

代码状态：已完成，commit `7b2effb`；昇腾验收已完成。

验收内容：

- 短 prompts 能加载并生成合理答案。
- `prefill_chunk_size=1024` 下完成约 21K prompt。
- 再覆盖一组接近 64K、但为输出 token 预留余量的 prompt。
- `temperature=0` 时，与 vLLM-Ascend 或可信 GLM-5.2 baseline 对比 token IDs。
- 日志应明确显示 `GLM-5.2`、`MTP K=0`、`offload_mode=none`、eager。

失败时只修复当前 nonoffload eager 路径，不提前写 offload。

### 阶段 1：建立 IndexShare group 元数据层

目标是先建立清晰的数据模型，不改变算子 ABI。

实现项：

- 从 `hf_config.indexer_types` 构造 21 个 group，以及 layer→owner 映射。
- GLM-5.1 继续退化为 78 个单层 group，保持原行为。
- 增加 group-owned cache mapping 和固定输出 buffer 的生命周期管理。
- shared layer 不创建 Indexer 权重和 index key cache。
- 序列初始化、abort、preemption、pool entry 回收按 group 清理。

CPU 验收：

- 精确校验 21/57 schedule 和所有 layer→owner 映射。
- 校验 group state 的唯一性、batch 重排、pool entry 复用、abort 和 preemption。
- GLM-5.1 回归不变。

这一阶段可以只建立结构和测试，不必立刻打通整网。

### 阶段 2：MTP0 + offload_split + eager

实现项：

- prefill 只给 21 个 full layer 写 index cache，但 78 层 KV 都正常写 DRAM/HBM。
- 首次 decode 按 group 初始化映射，并逐层填充本层 HBM KV payload。
- stable decode 中仅 full layer 调用 `fused_li_manage`。
- group 内所有层使用同一份 `topk_src_ids/topk_dst_slots/miss_counts`，分别调用 `scatter_copy + sparse_tail_attention`。

验收：

- 先通过 CPU 状态测试和必要的 Ascend 链式 UT，再跑整网。
- `temperature=0` 下与 `offload_mode=none` 的 token IDs 对齐。
- 20K、约 40K、接近 64K 均能运行。
- eager profile 中应为 21 次 LIM、78 次 SCATTER、78 次 SFA。
- 观察每个 group 内 full/shared layer 的 miss metadata 是否一致，KV payload 必须各层独立。

### 阶段 3：MTP0 + offload_fuse + eager

在阶段 2 语义稳定后，让 21 次 LIM 的输出被 group 内各层的 COPYSFA 消费。

验收：

- token IDs 与阶段 2 完全一致。
- profile 中应为 21 次 LIM、78 次 COPYSFA。
- 不以 UT 时延做 Assert，只打印时延；只对语义错误做 Assert。
- 不在这个阶段顺手优化 COPYSFA kernel。

### 阶段 4：MTP0 full-decode-only graph

顺序为 `offload_split` 先入图，再接 `offload_fuse`。

验收：

- 首次 decode 和初始化允许 eager。
- stable decode 最终必须出现 capture/replay，且初始化完成后不因 shared layer 回 eager。
- eager 与 graph 的 token IDs 一致。
- profile 中的 LIM/COPY/SFA 次数仍符合 21/78，不因组图恢复成 78 次 LIM。
- 对比 GLM-5.1 与 GLM-5.2 的 steady TPOT，记录 IndexShare 带来的实际收益。

### 阶段 5：GLM-5.2 量化 MTP layer，先打通 nonoffload eager

实现项：

- 去掉“第 78 层必须全 FLOAT”的 GLM-5.2 限制，但只有在真正支持其 W4A8/W8A8 权重之后才能放开门禁。
- 复用 target layer 的 ModelSlim W4A8 routed experts、router FP32 语义和 W8A8 权重加载逻辑。
- 正确加载第 78 层、shared head、MTP embedding/norm/`eh_proj` 和根目录 `rot.safetensors`。
- 仍只支持 `NANOVLLM_NUM_SPECULATIVE_TOKENS=0/3`。
- 第一版可以继续让 MTP attention 使用 dense MLA，以最小改动先验证 draft/rejection 语义；这属于 bring-up fallback，不代表完成 GLM-5.2 的 MTP IndexShare 性能适配。

验收：

- K=0 与 K=3 的最终 token IDs 一致。
- 普通 prefill 与 chunk prefill 输出一致。
- 使用 `example/test_dureader.py` 的多样化请求检查 accepted drafts 和输出文本。
- 与 vLLM-Ascend 对照首轮 drafts、接受长度和总体 acceptance。

### 阶段 6：MTP3 + none 的 graph，以及 MTP iteration IndexShare

先让现有 nonoffload MTP3 stable decode 正确入图，再单独处理 `index_share_for_mtp_iteration`：

- MTP layer 第一次 draft iteration 计算选择结果。
- 后续 draft iteration 复用该结果，但仍分别计算自己的 attention。
- 这份状态只属于 MTP layer，不要与 target-layer 的 21 个 group state 混在一起。

验收：

- `captures/replays` 正常，K=3 输出与 eager 一致。
- draft iteration 的 Indexer 次数符合模型语义。
- 单独记录该优化前后的 MTP draft graph 时延。

### 阶段 7：MTP3 target IndexShare offload eager

顺序：`offload_split`，再 `offload_fuse`。

实现项：

- 只有 21 个 full target layer 调用 `fused_li_manage_mtp`。
- 每次 LIM 仍处理每请求 4 个 verification query，并保护四路 top2048 的并集。
- 57 个 shared layer 复用 owner 的 `topk_src_ids/topk_dst_slots/miss_src_ids/miss_dst_slots/miss_counts`。
- 78 层分别调用 MTP SCATTER/SFA 或 COPYSFA-MTP。

验收：

- K=3 `none/split/fuse` 最终 token IDs 一致。
- 每请求典型 unique union miss 应按实际模型数据观测，不用人工假定为四路 miss 的简单和。
- eager 逻辑调用应为 21 次 LIM-MTP、78 次 COPY/SFA-MTP 或 78 次 COPYSFA-MTP。当前保守版 `fused_copy_sfa_mtp` 内部仍由 SCATTER+SFA 组成，因此 Ascend Hardware 视图中看到两个内部算子是正常的。
- 先看语义，再看 step latency；不在这一阶段同时改 LIM-MTP/COPYSFA-MTP kernel。

### 阶段 8：MTP3 full-decode-only graph

- 固定 `B*4` verification shape 和 group output buffer 地址。
- split 路径先通过，再接 fuse。
- 首次 decode、cache init 和 lazy capture 可以慢；只优化后续 stable replay。
- 覆盖请求同时结束、EOS、`NANOVLLM_MAX_STEPS`、batch reorder 和 request-pool reuse。

验收：

- eager/graph 结果一致。
- 初始化后连续 stable decode replay，不出现周期性 eager fallback。
- profile 算子次数为 21/78。
- 用 DuReader 多请求验证真实 MTP acceptance 和 effective TPOT。

### 阶段 9：性能、回归与收尾

建议固定比较矩阵：

- seqlen：约 21K、40K、接近 64K。
- batch size：优先 12、24、32；显存不足时明确记录实际配置。
- 模式：MTP0/MTP3 × none/split/fuse × eager/graph。
- baseline：同配置 vLLM-Ascend，以及本仓 GLM-5.1。

重点指标：

- stable step latency / effective TPOT。
- 21 次 LIM 或 LIM-MTP 的合计时延。
- 78 次 SCATTER+SFA 或 COPYSFA 的合计时延。
- decode step 之间的 free gap。
- MTP accepted drafts、输出 token/request-step。
- HBM/DRAM block 占用和最大可运行 batch size。

旧 GLM-5.1 MTP3 profile 中 LIM-MTP 约占 44 ms/78 层。若单次成本近似不变，降为 21 次后理论上可减少约 32 ms/step；这只是方向性估算，最终以 GLM-5.2 实测为准。参考日志：[`runlog/commit_7c096e/dureader12_mtp3_offload_fuse.txt`](runlog/commit_7c096e/dureader12_mtp3_offload_fuse.txt)。

完成后：

- 删除临时诊断代码和临时环境开关。
- 更新 `README.md` 的支持矩阵和完整运行命令。
- 更新本文件 checklist、对应 commit ID 和昇腾验收结果。
- 再跑 GLM-5.1 回归，确认 `main-glm52` 没有破坏原模型。

## 6. 必须遵守的注意事项

1. 只在 `main-glm52` 工作，不修改或强推 `main`。
2. 每次开始前确认分支和工作区；不要覆盖用户已有改动。
3. 每个可独立验收的阶段单独 commit。提交后停止，给出 commit ID 和完整的昇腾运行环境变量，等待用户回传日志。
4. 用户要求运行命令时，必须列出完整环境变量；不要写“其余变量沿用上次”。命令中不需要 `cd`，也不需要用 `tee` 保存日志。
5. 术语统一使用 LIM（`fused_li_manage`/`fused_li_manage_mtp`）和 COPYSFA（`fused_copy_sfa`/`fused_copy_sfa_mtp`），不要在新文档和日志中继续扩散旧称呼。
6. 不改变 7 个 offloading 算子的 ABI 或既定语义；确需改变时，先停下来和用户讨论。
7. 修改 Python 调度/模型代码通常无需重编算子。修改 C++、host tiling、AscendC kernel、op-api、schema 或 CMake 后必须重编，并先跑算子 UT，再跑整网。
8. 算子 UT 只对语义正确性做 Assert；性能只打印，不因设备波动或时延回退让 UT 失败。
9. 当前 nanovllm 中的 `fused_copy_sfa_mtp` 是行为正确、先完成 union SCATTER 再执行四路 SFA 的保守实现。实验性的 COPYSFA-MTP 精度/性能优化在独立算子工程 `D:\vLLM-ascend\ops_lim_mtp` 演进；没有同时通过精度和性能验证前，不要合回本仓。
10. 20K～64K 是产品范围。不要为了 1M 上下文扩展 source-ID 位宽、workspace、block table 或算子 ABI。
11. Chunk prefill 仍是纯 prefill，不能引入 prefill/decode 混合 forward。
12. 输出一致性测试统一使用 `temperature=0`；性能判断排除 prefill、首次 decode、初始化和 graph warmup。
13. shared layer 只能复用选择与 slot metadata，不能复用另一层的 KV payload、query 或 attention 输出。
14. group 映射一致性必须覆盖初始化、连续 update、preemption、abort、batch reorder 和 pool row 复用；这是最容易产生“能生成文本但语义已错”的地方。
15. 保留 graph 所需的 caller-owned 固定输出 buffer，不要在 stable decode 热路径动态创建输出 tensor。
16. `README.md` 中的全角空格是用户用于 GitHub 排版的，不要清理或替换。
17. `tests/test_glm_dsa_offload.py` 当前有一个历史预算预期与主线实际缓存预算不一致的旧测试；不要把它误判成 GLM-5.2 IndexShare 回归，也不要在无关提交中顺手改预算语义。

## 7. Codex 接手流程

新的 Codex 接手后，先执行以下检查，再开始写代码：

1. 阅读本文件、`README.md`、`README_ops.md` 和 `TODO.md`。
2. 确认当前分支是 `main-glm52`，且工作区没有未知改动。
3. 查看 checklist 和最后一个已验收 commit，不要重复已经完成的阶段。
4. 如果上一阶段只完成代码但用户尚未回传昇腾结果，先等待或帮助分析日志，不要越过验收点。
5. 实现当前阶段的最小闭环：代码、CPU/静态测试、必要文档、一个 commit。
6. 向用户说明是否需要重新编译算子，并提供完整的昇腾验收命令。
7. 用户验收后，把结果、commit ID 和关键性能数据补到本文件，再进入下一阶段。

进度记录建议格式：

```text
日期：
阶段：
commit：
代码状态：
CPU/UT：
昇腾整网：
profile/性能：
结论与下一步：
```

---

## 进度记录

### 阶段 0：nonoffload eager bring-up

```text
日期：2025-08-11
阶段：0（验收 nonoffload eager bring-up）
commit：7b2effb
代码状态：已完成，仅改 Python，无需重编算子
CPU/UT：本地 CPU 覆盖模型识别、IndexShare schedule、能力门禁、原有 MTP 和 chunk prefill 回归
昇腾整网：通过
  - 短序列 smoke（bsz=3，prompt≤21）：输出语义正确（"北京"、"14"、英文句），日志显示 GLM-5.2 / MTP K=0 / none / eager
  - 21K（prompt=21000，prefill_chunk_size=1024，21 个 prefill chunk）：输出 "Hawthorn Bridge" 正确，decode mean TPOT=0.1641s
  - 64K（prompt=64000，prefill_chunk_size=1024，63 个 prefill chunk）：输出 "Hawthorn Bridge" 正确，decode mean TPOT=0.1551s
  - token IDs baseline 对齐：64K 在 main-glm52 与 main（GLM-5.1 代码路径）上 token_ids 完全一致
    [39, 672, 339, 1512, 19836, 154827, 39, 672, 339, 1512, 19836, 154842, 39, 672, 339, 1512, 19836]
profile/性能：
  - 21K：prefill TPS=1747 tok/s，decode mean TPOT=0.1641s，decode TPS=6.10 tok/s
  - 64K：prefill TPS=2351 tok/s，decode mean TPOT=0.1551s，decode TPS=6.45 tok/s
结论与下一步：阶段 0 验收通过，进入阶段 1（建立 IndexShare group 元数据层）
```

### 阶段 1：建立 IndexShare group 元数据层

```text
日期：2025-08-10
阶段：1（建立 IndexShare group 元数据层）
commit：8164dbe
代码状态：已完成，仅改 Python，无需重编算子
CPU/UT：30 项新测试全部通过（tests/test_glm_index_share.py）
  - GLM-5.2 group 拓扑：21 groups、21 full/57 shared、layer->owner 映射、group membership
  - GLM-5.1 退化：78 单层 group、全部 owner、无 shared
  - 错误处理：长度不匹配、未知类型、shared 无前导 full、空 types
  - Config 集成：GLM-5.1/5.2 均构建 IndexShareGroupManager 并存储到 hf_config
  - 权重映射：offload 模式下 shared layer indexer 权重跳过、owner 正常加载
  - Scheduler 生命周期：deallocate/preempt/abort/pool reuse/batch reorder 不受影响
  - GLM-5.1 回归：全部原有测试不变（budget 相关旧测试失败为预存问题，见注意事项 17）
昇腾整网：无需整网验收（阶段 1 只建立数据模型，不改变运行时行为）
profile/性能：不适用
结论与下一步：阶段 1 代码完成，等待用户确认后进入阶段 2（MTP0 + offload_split + eager）
```

实现内容：
- `IndexShareGroup`（frozen dataclass）和 `IndexShareGroupManager`（`dsa_offload.py`）
- `Config._configure_glm_version` 构建 group manager 并存储到 `hf_config.nanovllm_index_share_groups`
- `GlmMLAAttention` 新增 `is_index_share_owner`，shared layer 跳过 indexer/indexer_rotary_emb 创建
- `ModelRunner._allocate_mla_cache` 跳过 shared layer 的 `index_cache` 分配
- `GlmMoeDsaForCausalLM.weight_name_mapping` 跳过 shared layer 的 indexer 权重加载

### 阶段 2：MTP0 + offload_split + eager

```text
日期：2026-08-11
阶段：2（MTP0 + offload_split + eager）
commit：15f1a7f；后续修复 f9f4123、7c77e81
代码状态：已完成，仅改 Python，无需重编算子
CPU/UT：新增 IndexShare/offload_split 测试全部通过；scatter_copy 相关 CPU 用例正确跳过
  - GLM-5.2 offload_split config 构建和门禁（允许 split、拒绝 fuse/graph/mtp3）
  - initialize_lidu_row_shared：owner 映射提取、zero-cache noop、未填充映射检测
  - scheduler offload_split 生命周期（allocate/deallocate/preempt/abort）
  - GLM-5.1 回归不变
昇腾整网：通过
  - 21K：修复首次 decode 后，split 与 none 的 token IDs 完全一致
  - 40K：两者共同前缀稳定，后续 greedy token 出现 top-2048 稀疏 Attention 的预期近似分叉；LIDU miss metadata 在 group 内一致，step 3 后稳定为 0
  - 64K：split 与 none 的 token IDs 完全一致
profile/性能：40K profile 为 21 次 LIM、78 次 SCATTER、78 次 SFA；语义验证完成后不以 bsz=1 时延作为性能结论
结论与下一步：阶段 2 验收通过，进入阶段 3（MTP0 + offload_fuse + eager）
```

实现内容：
- `Config._validate_glm52_phase1_runtime` 放宽：允许 `offload_split + MTP0 + eager`，仍拒绝 `offload_fuse`/MTP3/graph
- `dsa_offload_ops.py` 新增 `initialize_lidu_row_shared`：shared layer 从 owner 填充的 `cache_slots_row` 提取映射，调用 `scatter_copy` 拷贝本层 KV
- `GlmMLAAttention` 新增 `_index_share_owner` 引用和 `_lidu_update_shared` 方法：shared layer 跳过 LIM，读取 owner 的 `topk_src_ids/topk_dst_slots/miss_counts`，用本层 KV 执行 SCATTER
- `GlmMLAAttention.forward` prefill/decode 路径：shared layer 跳过 indexer 调用
- `GlmMLAAttention.finalize_prefill_offload`：仅 owner 操作 `cache_slots_pool` 映射，所有层各自持久化 DRAM KV
- `ModelRunner._allocate_mla_cache`：group-owned `lidu_cache_slots`（同 group 的层共享同一 tensor）
- `ModelRunner._setup_index_share_owners`：为每个 shared layer 设置 owner 引用

### 阶段 3：MTP0 + offload_fuse + eager

```text
日期：2026-08-11
阶段：3（MTP0 + offload_fuse + eager）
commit：390cf22
代码状态：已完成，仅改 Python，无需重编算子
CPU/UT：GLM-5.2 fuse eager 配置允许；split/fuse 的 graph 门禁仍正确拒绝；IndexShare 与 DSA offload 回归通过（历史预算断言单独排除）
昇腾整网：通过
  - 20K：清理 profile/miss-count 诊断环境变量后，offload_fuse 与 offload_split 的 token IDs 完全一致
profile/性能：稳定 decode 为 21 次 LIM、78 次 COPYSFA
结论与下一步：阶段 3 验收通过；后续进入阶段 4（MTP0 full-decode-only graph），先实现 offload_split graph
```

### 阶段 4：MTP0 full-decode-only graph

```text
日期：2026-08-12
阶段：4（MTP0 full-decode-only graph）
commit：ea6fe43（offload_split）、8ef5359（offload_fuse）
代码状态：已完成，仅改 Python 图调度与能力门禁，无需重编算子
CPU/UT：IndexShare 39 passed / 1 skipped；DSA offload 39 passed（历史 8.2K 预算断言单独排除）；full-decode graph 21 passed
昇腾整网：通过
  - offload_split 和 offload_fuse 均覆盖 21K、40K、64K；输出符合预期
  - 首次 decode 与 LIDU 初始化保持 eager；首次 initialized stable decode 完成 lazy capture，后续 stable decode 持续 replay
  - 21K fuse：16 个 decode = eager_first_decode 1 + eager_lidu_capture 1 + replay 14；无 uninitialized/uncaptured eager fallback
  - 21K fuse：token IDs 为 [39, 672, 339, 1512, 19836, 154827, 39, 672, 339, 1512, 19836, 154842, 39, 672, 339, 1512, 19836]
profile/性能：split 为 21 次 LIM、78 次 SCATTER、78 次 SFA；fuse 为 21 次 LIM、78 次 COPYSFA。profile 运行包含 lazy capture 与 profiler 开销，尚未记录 GLM-5.1/GLM-5.2 稳定 TPOT 对比
结论与下一步：阶段 4 验收通过；进入阶段 5（GLM-5.2 量化 MTP layer，先打通 MTP3 + none + eager）
```

### 阶段 5：GLM-5.2 量化 MTP layer，MTP3 + none + eager

```text
日期：2026-08-12
阶段：5（MTP3 + none + eager）
commit：e6dcd07；语义修复 4116711
代码状态：已完成，仅改 Python，无需重编算子
CPU/UT：MTP、IndexShare、DSA offload 联合回归 134 passed / 1 skipped（历史 8.2K 预算断言单独排除）；MTP + full-decode graph 回归 77 passed
实现：GLM-5.2 第 78 层复用 W4A8 routed-expert 加载与执行路径；其余 dense 权重继续走 W8A8 通用加载。GLM-5.2 eager target verification 改为逐 token ordinary decode，保证 partial reject 后与 K=0 的因果 cache 语义一致
昇腾整网：通过
  - MTP3 + none + eager 覆盖 21K、40K、64K
  - K=3 与 K=0 的 token IDs 在相同 completion 范围内逐位一致；40K K=3 的 35 个 token 与 K=0 前 35 个 token 一致
  - K=3 的 max_steps 统计 decode 调度轮次，每轮最多提交 4 个 token；因此不能用相同 max_steps 比较 K=0/K=3 的输出长度
profile/性能：本阶段串行 target verification 优先保证语义，暂不以 eager TPOT 作为性能结论
结论与下一步：阶段 5 eager 验收通过；进入阶段 6（MTP3 + none full-decode-only graph）
```
