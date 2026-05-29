# Decode 阶段 KVcache 卸载设计文档

本文档描述 `nano-vllm-ascend-DeepSeekV32` 中 decode 阶段 KVcache 卸载机制的设计。设计目标是在长序列请求进入 decode 后，将 prefill 阶段的大部分 KVcache 从 NPU HBM 卸载到 CPU DRAM，只在 HBM 中保留一份可动态更新的 sparse budget，从而降低单请求显存占用，提高 decode batch size。

本文档描述稳定设计。阶段性实现取舍集中放在最后的“分阶段落地计划”中。

整体结构如下：

```mermaid
flowchart LR
    Req["Sequence 请求<br/>保存请求级元数据"] --> S["Scheduler<br/>负责调度与资源编排"]
    S --> IBM["IndexBlockManager<br/>管理 IndexCache 的申请释放"]
    S --> HBM["HBMBlockManager<br/>管理 HBM KVcache 的申请释放"]
    S --> DBM["DramBlockManager<br/>管理 DRAM KVcache 的申请释放"]
    S --> PEM["PoolEntryManager<br/>管理 sparse pool entry"]

    IBM --> IC[("HBM IndexCache<br/>常驻 DSA 打分表征")]
    HBM --> HKV[("HBM CKV/KPE<br/>保存 sparse、尾块和 decode KV")]
    DBM --> DKV[("CPU DRAM CKV/KPE<br/>保存 prefill 满块")]
    PEM --> Pool[("hbm_cached_tokens_pool<br/>记录 sparse slot 映射")]

    W["Worker / ModelRunner<br/>执行 forward 与搬运算子"] --> IC
    W --> HKV
    W --> DKV
    W --> Pool
```

## 1. 目标和边界

本节界定这项工作的目标和不覆盖的范围。KVcache 卸载会同时影响调度、块管理、模型 forward 和算子接口，先明确边界有助于后续实现保持可验证。

### 1.1 目标

1. 对长序列请求，在 prefill 结束后将 prefill 满块对应的 CKV/KPE 卸载到 CPU DRAM。
2. 在 HBM 中保留一小段 sparse budget，并保留 prefill 尾部非满块和 decode 阶段新产生的 KVcache。
3. 在每个 decode step、每一层根据 DSA 打分结果，从 DRAM 召回当前更重要的 prefill layer_kv_token，并淘汰 HBM sparse budget 中分数较低的 layer_kv_token。
4. 使同一套框架接口同时兼容有损固定 `Tx` 策略和无损动态 `Tx` 策略。
5. 将 model.forward 需要的调度元数据尽量 tensor 化，避免 forward 内部出现细碎 H2D 拷贝。
6. decode forward 中新增逻辑必须支持组 batch，不能依赖逐请求 Python 循环来完成核心算子逻辑。

### 1.2 设计边界

1. IndexCache 常驻 HBM，不随 CKV/KPE 卸载到 DRAM。
2. decode 阶段新产生的 KVcache 不卸载到 DRAM。
3. HBM sparse budget 只管理 prefill 满块中的 layer_kv_token；prefill 尾部非满块和 decode 新 token 作为常驻上下文参与 MLA。
4. 抢占属于调度增强能力，不影响本设计的数据结构和算子接口。
5. SFA decode 与本设计正交。卸载后的 HBM sparse budget 以 paged dense MLA 作为 attention 计算路径。

## 2. 当前基线与可参考机制

本节说明当前 DeepSeek V3.2 基线中的 cache 形态，并总结 Qwen KVStar 改造中可以复用的工程模式。这里参考的是机制，不直接复用稠密 GQA 模型的算法细节。

### 2.1 当前 DeepSeek V3.2 cache 形态

当前 DeepSeek V3.2 在每个 TP rank 上按层分配 CKV/KPE/IndexCache。典型形状如下：

| 缓存 | 形状 | dtype | device | 说明 |
|---|---:|---|---|---|
| CKV cache | `tensor[(L, C, 1, B, 512), bf16, npu]` | bf16 | NPU | MLA latent KV，也可称为 nope cache |
| KPE cache | `tensor[(L, C, 1, B, 64), bf16, npu]` | bf16 | NPU | RoPE 部分的 K cache |
| IndexCache | `tensor[(L, C, B, 1, 128), bf16, npu]` | bf16 | NPU | DSA indexer 使用的 K 表征 |

其中 `L=61`，典型 `B=128`。当前基线中，一个请求的 `block_table` 同时索引 CKV、KPE 和 IndexCache。卸载后，这三类数据生命周期不同，需要拆成独立的 block table。

### 2.2 Qwen KVStar 改造中的可参考机制

已阅读 `D:\work\1.92_bakup\nano-vllm-kvstar-0321\` 中的 Qwen KVStar 改造。以下机制可迁移到 DSA 卸载设计中：

| Qwen KVStar 机制 | DSA 卸载中的对应设计 |
|---|---|
| `CPUBlockManager` 管理 CPU KV block，并通过链式 hash/refcount 支持前缀复用 | `DramBlockManager` 管理 DRAM CKV/KPE 满块，并支持 prefill 满块前缀复用 |
| `GPUBlockManager` 在 prefill 结束后释放被稀疏化的 GPU block | `HBMBlockManager` 在 sparse budget 初始化后释放不再常驻的 HBM KVcache block |
| `Sequence` 同时维护 `gpu_block_table` 和 `cpu_block_table` | `Sequence` 维护 `index_block_table`、`hbm_block_table`、`dram_block_table` |
| `selected_pool` 记录每个请求当前选中的稀疏集合 | `hbm_cached_tokens_pool` 记录每层每请求当前 HBM sparse slot 对应的原始 token id |
| batched incremental topk 同时更新 selected，并输出源/目的块 | `dsa_index_update` 更新 sparse slot 索引，并输出 `promote_idx/demote_idx/copy_counts` |
| scatter copy 根据源/目的物理块执行 H2D/D2H 搬运 | `dsa_scatter_h2d` 根据 promote/demote 和两套 block table 执行 DRAM 到 HBM 搬运 |

DSA 版本与 Qwen KVStar 版本的关键差异是粒度不同：Qwen 版主要按块和 GQA KV head 处理；DSA 卸载需要按 prefill layer_kv_token 处理，并且 IndexCache 必须常驻 HBM 参与每层打分。

## 3. 核心概念和符号

本节统一术语。本文中涉及 KVcache 的 token 均严格指 `layer_kv_token`，也就是一层中一个序列位置对应的 CKV/KPE 表征。

### 3.1 DSA 与 GT2048

DeepSeek V3.2 的 DSA 会为每个 query token 选择最多 2048 个历史 layer_kv_token 参与 attention。本文将某个 decode step、某一层中 DSA 真实选中的最多 2048 个 layer_kv_token 称为 `GT2048`。

如果卸载机制能保证该层该步的 `GT2048` 全部参与 MLA，则 attention 选择集合是无损的；如果不能保证，则是有损的。

### 3.2 KVcache 与 IndexCache

本文中的 KVcache 是 CKV cache 与 KPE cache 的统称：

| 名称 | 内容 | 单个 layer_kv_token 大小 |
|---|---|---:|
| CKV | latent KV，维度 512 | 512 个 bf16 |
| KPE | RoPE K，维度 64 | 64 个 bf16 |

一个 KVcache 物理块包含 `B` 个 layer_kv_token。典型 `B=128` 时，单层单块 CKV/KPE 大小为：

```text
B * (512 + 64) * sizeof(bf16)
= 128 * 576 * 2 bytes
= 147456 bytes
≈ 144 KiB
```

IndexCache 是 DSA indexer 使用的 K 表征，典型形状为：

```text
tensor[(L, Cidx, B, 1, 128), bf16, npu]
```

IndexCache 不卸载，因为 decode 每步每层都需要用它为候选 prefill layer_kv_token 计算重要性分数。

### 3.3 符号表

基础维度：

| 符号 | 含义 |
|---|---|
| `L` | 模型层数，DeepSeek V3.2 为 61 |
| `B` | KVcache block size，典型值为 128 |
| `Dckv` | 单个 layer_kv_token 的 CKV 维度，DeepSeek V3.2 中为 512 |
| `Dkpe` | 单个 layer_kv_token 的 KPE 维度，DeepSeek V3.2 中为 64 |
| `Dkv` | 单个 layer_kv_token 的 CKV/KPE 合计维度，`Dkv = Dckv + Dkpe = 576` |
| `Didx` | IndexCache 单 token 维度，DeepSeek V3.2 中为 128 |
| `Hidx` | DSA indexer 使用的 index head 数 |
| `Cidx` | HBM 上 IndexCache 物理块数量 |
| `Chbm` | HBM 上 KVcache 物理块数量 |
| `Cdram` | DRAM 上 KVcache 物理块数量 |
| `pool_capacity` | `hbm_cached_tokens_pool` 可同时容纳的请求槽位数 |
| `bs` | 一次 decode step 的 batch size |

单请求长度：

| 符号 | 含义 |
|---|---|
| `Sp` | 某请求 prefill token 数量 |
| `Sd` | 某请求截至当前层 MLA 调用前，已经写入 HBM KVcache、并会参与本次 MLA 的 decode token 数；第一步 decode 时 `Sd=1` |
| `Nprefill` | prefill 总块数，`Nprefill = ceil(Sp / B)` |
| `Np` | prefill 满块数量，`Np = Sp // B` |
| `Nr` | prefill 尾部非满块数量，`Nr = Nprefill - Np`，只可能为 0 或 1 |
| `tail_len` | prefill 尾部非满块 token 数，`tail_len = Sp - Np * B` |
| `Nd` | prefill 尾块加 decode 新 token 所需块数，`Nd = ceil((tail_len + Sd) / B)` |
| `Nc` | prefill 满块中可被前缀复用的块数 |
| `Ns` | prefill 满块卸载后，在 HBM 中保留的 sparse budget 块数 |
| `Ts` | HBM 中用于 prefill 历史的 sparse budget token 数，`Ts = Ns * B` |
| `Tp` | 参与候选的 prefill 满块 layer_kv_token 数，`Tp = Np * B` |
| `sparse_kv_len` | MLA 实际看到的 KV 长度，`sparse_kv_len = Ts + tail_len + Sd` |

运行时索引：

| 符号 | 含义 |
|---|---|
| `l` | 模型层 id，范围为 `[0, L)` |
| `b` | decode batch 内请求 id，范围为 `[0, bs)` |
| `e_b` | 第 `b` 个请求在 `hbm_cached_tokens_pool` 中的 pool entry |
| `t` | 原始 prefill 满块 token id，范围为 `[0, Tp)` |
| `s` | HBM sparse budget 本地 slot id，范围为 `[0, Ts)` |
| `i` | promote/demote 对的序号 |
| `Tcopy_b` | 第 `b` 个请求在某层某步实际搬运的 token 数，等于 `copy_counts[b]` |
| `Tx` | 固定有损策略下每请求每层每步搬运的 layer_kv_token 数量 |
| `GT2048` | DSA 在某 step、某层选择的最多 2048 个 layer_kv_token |

全局容量上限：

| 符号 | 含义 |
|---|---|
| `max_model_len` | 引擎配置的最大序列长度，设计默认用户不设置超过 131072 |
| `Nmax_prefill` | 配置范围内 prefill 总块数上限，`Nmax_prefill = ceil(max_model_len / B)` |
| `Nmax_sparse` | 配置范围内 sparse budget 块数上限，由 `Ns` 分段函数取最大值得到 |
| `Tmax_sparse` | `hbm_cached_tokens_pool` 的 token 维上限，`Tmax_sparse = Nmax_sparse * B` |
| `Tmax_candidate` | `score_out` 的 token 维上限，不超过 `max_model_len` |
| `Tmax_copy` | `promote_idx/demote_idx` 的 token 维上限；无损策略下为 2048 |
| `Nmax_index` | tensor 化 `index_block_tables` 的块数上限 |
| `Nmax_hbm` | tensor 化 `hbm_block_tables` 的块数上限 |
| `Nmax_dram` | tensor 化 `dram_block_tables` 的块数上限 |

后文如果写作 `Tp_b`、`Ts_b`、`Sd_b`，表示 batch 内第 `b` 个请求对应的取值。

### 3.4 索引映射与不变量

后文统一使用以下辅助函数描述块内寻址：

```text
blk(x) = x // B
off(x) = x % B
```

原始 prefill 满块 token id `t` 使用原始 prefill 语义；它通过 `dram_block_table[blk(t)]` 定位 DRAM 中的源块。HBM sparse slot id `s` 使用 sparse 语义；它通过 `hbm_block_table[blk(s)]` 定位 HBM sparse budget 中的目标块。

对 batch 内第 `b` 个请求、第 `l` 层：

```text
hbm_cached_tokens_pool[l, e_b, s] = t
```

表示该层 HBM sparse budget 的本地 slot `s` 当前保存的是原始 prefill 满块 token `t`。因此 `hbm_block_table` 可以保持请求级，不需要按层拆分；逐层差异完全由 `hbm_cached_tokens_pool[l, e_b, :]` 表达。

关键不变量如下：

1. `dram_block_table` 只覆盖 prefill 满块，候选 token 范围始终是 `[0, Tp)`。
2. `index_block_table` 使用原始序列语义，服务于 DSA indexer 打分。
3. `hbm_block_table` 在 decode 阶段使用 sparse 语义，前 `Ns` 个逻辑块为 sparse budget。
4. `hbm_cached_tokens_pool[l, e_b, 0:Ts]` 中的每个值都必须落在 `[0, Tp)`。
5. `promote_idx[b, i]` 是原始 prefill 满块 token id `t`，`demote_idx[b, i]` 是本地 sparse slot id `s`。
6. 仅 `0 <= i < Tcopy_b` 的 promote/demote 对有效，`Tcopy_b = copy_counts[b] <= Tmax_copy`。
7. 启用卸载的请求需要满足 `Ts >= 2048`；默认分段函数下实际最小值为 2560。

## 4. 总体数据布局

本节描述卸载后 CKV/KPE/IndexCache 分别在哪里、如何索引，以及 MLA 看到的序列是什么。

### 4.1 三类物理 cache

卸载后 Worker 维护三类物理 cache：

| cache | 形状 | device | 生命周期 |
|---|---:|---|---|
| `index_cache` | `tensor[(L, Cidx, B, 1, Didx), bf16, npu]` | NPU | prefill 满块、尾块、decode 新块均常驻 HBM |
| `hbm_ckv_cache` | `tensor[(L, Chbm, 1, B, Dckv), bf16, npu]` | NPU | prefill 阶段临时完整保存，decode 阶段仅保存 sparse budget、尾块和 decode 新块 |
| `hbm_kpe_cache` | `tensor[(L, Chbm, 1, B, Dkpe), bf16, npu]` | NPU | 同上 |
| `dram_ckv_cache` | `tensor[(L, Cdram, 1, B, Dckv), bf16, cpu]` | CPU DRAM | 保存 prefill 满块 |
| `dram_kpe_cache` | `tensor[(L, Cdram, 1, B, Dkpe), bf16, cpu]` | CPU DRAM | 保存 prefill 满块 |

DRAM cache 应优先使用 pinned memory。最终 H2D 路径由 `dsa_scatter_h2d` 负责，底层可替换为 CANN custom op。

### 4.2 三类 block table

每个请求维护三套 block table：

| block table | 指向 | 用途 |
|---|---|---|
| `index_block_table` | HBM IndexCache 物理块 | DSA indexer 对候选 layer_kv_token 打分 |
| `hbm_block_table` | HBM CKV/KPE 物理块 | MLA 读取 sparse HBM KV，以及 scatter 的目的地址 |
| `dram_block_table` | DRAM CKV/KPE 物理块 | scatter 的源地址，以及 DRAM 前缀复用 |

`hbm_block_table` 在 decode 阶段使用 sparse 语义。对启用卸载的请求，它的前 `Ns` 个逻辑块是 sparse budget，后续逻辑块是 prefill 尾部非满块和 decode 新 token 所在块。

三种 block table 都是请求级元数据，不需要按层拆分。不同层的 sparse budget 选中集合可能不同，但它们共享同一组 HBM sparse slot 物理位置；每一层的 sparse slot 到原始 prefill token id 的映射由 `hbm_cached_tokens_pool[layer, entry, :]` 区分。

三种 block_table 均统一使用**0**作为**null/padding block id**；真实物理块从**1**开始分配。

三类 block table 与物理 cache、逐层 sparse 映射的关系如下：

```mermaid
flowchart TB
    Seq["Sequence 请求元数据"]
    Seq --> IBT["index_block_table<br/>请求级"]
    Seq --> HBT["hbm_block_table<br/>请求级"]
    Seq --> DBT["dram_block_table<br/>请求级"]

    IBT --> IC["HBM IndexCache<br/>DSA indexer 使用"]
    HBT --> HKV["HBM CKV/KPE<br/>MLA 与 scatter 目的地址"]
    DBT --> DKV["CPU DRAM CKV/KPE<br/>scatter 源地址与前缀复用"]

    Pool["hbm_cached_tokens_pool[layer, entry, slot]"]
    Pool --> Slot["每层 sparse slot 到原始 prefill token id 的映射"]
    HBT --> Slot
```

### 4.3 `hbm_cached_tokens_pool`

`hbm_cached_tokens_pool` 记录 HBM sparse budget 中每个 slot 对应的原始 prefill token id：

```text
hbm_cached_tokens_pool = tensor[(L, pool_capacity, Tmax_sparse), int32, npu]
```

其中：

```text
Tmax_sparse = B * Nmax_sparse
```

`Nmax_sparse` 由 sparse budget 分段函数在配置的 `max_model_len` 范围内取最大值得到。若 `max_model_len=131072` 且 `B=128`，默认分段函数下 `Np` 最大为 1024，`Ns=ceil(0.20*1024)=205`，因此：

```text
Tmax_sparse = 205 * 128 = 26240
```

这个形状比 `(L, max_decode_batch, max_model_len)` 更合适，因为 pool 只需要记录 HBM sparse budget 的 slot 到原始 token id 的映射，不记录全量序列。

## 5. Sparse Budget 与长度语义

本节定义 `Ns/Ts` 如何计算，以及 sparse HBM KV 的长度如何传递。这里是整个设计最容易混用原始序列长度和 sparse 长度的地方。

### 5.1 `Ns` 分段函数

`Ns` 表示 prefill 满块卸载后，仍在 HBM 中保留的 sparse budget 块数。默认分段函数如下：

| 条件 | `Ns` |
|---|---|
| `Np < 64` | `Np`，不启用卸载 |
| `64 <= Np < 128` | `ceil(0.30 * Np)` |
| `128 <= Np < 256` | `ceil(0.25 * Np)` |
| `256 <= Np < 512` | `ceil(0.22 * Np)` |
| `512 <= Np` | `ceil(0.20 * Np)` |

比例 `0.30/0.25/0.22/0.20` 应做成配置常量，便于后续调整质量与显存之间的折中。

对启用卸载的请求，设计要求：

```text
Ts = Ns * B >= 2048
```

默认分段函数中，最小启用卸载条件为 `64 <= Np < 128`，此时 `Ns >= ceil(64*0.30)=20`，因此 `Ts >= 2560`。如果后续调整分段比例，也应保持启用卸载时 `Ts >= 2048`。

### 5.2 Sparse MLA 长度

卸载后，MLA 不能使用原始 `Sp + Sd` 作为 KV 长度。MLA 看到的是 sparse HBM KV 序列：

```text
sparse_kv_len = Ts + tail_len + Sd
```

其中：

```text
tail_len = Sp - Np * B
```

也就是：

1. `Ts` 个来自 prefill 满块的 sparse budget layer_kv_token；
2. prefill 最后一个非满块中的 `tail_len` 个 layer_kv_token；
3. decode 阶段已经产生的 `Sd` 个 layer_kv_token。

其中 `Sd` 包含当前 decode step 已经写入 HBM KVcache 的新 token。因此第一步 decode 运行到 MLA 前，`Sd=1`。

传给 MLA 的 `actual_seq_lengths_kv`、`block_table`、slot mapping 都必须使用 sparse 语义，不能混用原始序列长度。KPE 已经包含原始 RoPE 位置信息，因此 prefill layer_kv_token 被放入 sparse slot 后仍可参与 MLA。

MLA 实际看到的 HBM KV 序列可以表示为：

```mermaid
flowchart LR
    A["sparse budget<br/>Ts = Ns * B<br/>来自 prefill 满块"] --> B["prefill 尾块<br/>tail_len tokens"]
    B --> C["decode 常驻区<br/>Sd tokens"]
    A -. "逐层 slot 映射" .-> P["hbm_cached_tokens_pool[layer, entry, :]"]
    P -. "记录原始 prefill token id" .-> O["DRAM 中的原始满块 token"]
```

## 6. 总体算法

本节从算法角度描述 prefill 和 decode 的完整流程，不涉及具体类字段如何实现。

### 6.1 Prefill 流程

对每个请求：

1. Scheduler 为请求申请 IndexCache HBM 块、KVcache HBM 块、KVcache DRAM 块和 pool entry。
2. 如果前缀复用命中，则 Scheduler 计算 `Nc`，Worker 在 prefill forward 前将命中的 DRAM KV 块完整加载到 HBM，使其参与 prefill attention。
3. Worker 执行 prefill forward，写入未命中的 CKV/KPE/IndexCache。
4. 每层 prefill 完成后，将 prefill 满块对应的 CKV/KPE 保存到 DRAM。
5. 根据分段函数计算 `Ns/Ts`，为每层初始化 HBM sparse budget。
6. 将每层选中的 `Ts` 个 prefill layer_kv_token 写入 `hbm_block_table` 的前 `Ns` 个 HBM KVcache 块中。
7. 写入 `hbm_cached_tokens_pool[layer, entry, :Ts]`，记录每个 sparse slot 对应的原始 prefill token id。
8. Scheduler 在 prefill 完成后释放不再常驻的 HBM KVcache 块；其中包括前缀命中时临时加载到 HBM 的前缀 KV 块，以及 prefill forward 计算出的未命中块。

```mermaid
sequenceDiagram
    participant S as Scheduler
    participant W as Worker
    participant I as HBM IndexCache
    participant H as HBM CKV 与 KPE
    participant D as CPU DRAM CKV 与 KPE
    participant P as hbm_cached_tokens_pool

    S->>S: 申请 Index/HBM/DRAM/Pool 资源
    S->>W: 下发 block tables 与 Nc
    alt 存在前缀命中
        W->>H: 将命中的 DRAM KV 满块完整加载到 HBM
    end
    W->>I: 写入未命中的 IndexCache
    W->>H: 写入未命中的 CKV/KPE
    W->>D: 保存 prefill 满块 CKV/KPE
    W->>H: 初始化每层 sparse budget
    W->>P: 写入 sparse slot 到原始 token id 的映射
    W->>S: prefill 完成
    S->>H: 释放临时前缀块与未命中满块
```

### 6.2 Decode 流程

每个 decode step、每一层执行：

1. 生成当前 token 的 CKV/KPE/IndexCache，并写入 decode 常驻 HBM 块。
2. `dsa_indexer_score` 使用当前 query 的 index 表征和 HBM IndexCache，为 DRAM 中可候选的 prefill layer_kv_token 打分。
3. `dsa_index_update` 根据分数和 `hbm_cached_tokens_pool` 更新 sparse budget 的索引集合，输出：
   - `promote_idx`：需要从 DRAM 召回的原始 prefill token id；
   - `demote_idx`：需要在 HBM sparse budget 中覆盖的本地 slot id；
   - `copy_counts`：每个 batch item 本次实际需要拷贝的 layer_kv_token 数量。
4. `dsa_scatter_h2d` 根据 `promote_idx/demote_idx/copy_counts` 执行 CKV/KPE 从 DRAM 到 HBM sparse slot 的搬运。
5. dense MLA 在 sparse HBM KV 序列上计算 attention。

```mermaid
flowchart TD
    A["当前层 hidden_states"] --> B["生成 q、ckv、kpe、index_k"]
    B --> C["写入 HBM decode KV 与 IndexCache"]
    C --> D["dsa_indexer_score<br/>基于 IndexCache 计算候选分数"]
    D --> E["dsa_index_update<br/>更新 pool 并输出 promote/demote/copy_counts"]
    E --> F["dsa_scatter_h2d<br/>DRAM -> HBM sparse slots"]
    F --> G["dense MLA over sparse HBM KV"]
    G --> H["attention 输出投影与 MLP/MoE"]
```

## 7. 有损与无损更新策略

本节定义两类 sparse budget 更新策略。两者共享同一套外部接口，差异收敛在 `dsa_index_update` 的内部策略和 `copy_counts` 的取值上。

### 7.1 固定 `Tx` 有损更新

固定 `Tx` 策略用于控制每步每层的召回成本。

对每个 batch item：

1. 从当前 HBM sparse budget 中按分数选出最低的 `Tx` 个 slot，作为 demote 集合。
2. 屏蔽已经在 HBM sparse budget 中的 token。
3. 从未缓存的 DRAM 候选 token 中选出分数最高的 `Tx` 个，作为 promote 集合。
4. 更新 `hbm_cached_tokens_pool`。
5. 输出 `copy_counts[b] = Tx`。

该策略不保证 `GT2048` 全覆盖，因此是有损策略。

### 7.2 动态 `Tx` 无损更新

动态 `Tx` 策略用于保证 `GT2048` 全部在 HBM 可见集合中。

对每个 batch item：

1. 对全量候选分数取 top 2048，得到 `GT2048`。
2. 计算 `GT2048` 中当前不在 HBM sparse budget 的集合，作为 promote 集合。
3. 令 `Tx_b = len(promote 集合)`。
4. 从当前 HBM sparse budget 中不属于 `GT2048` 的 layer_kv_token 里，选出分数最低的 `Tx_b` 个 slot，作为 demote 集合。
5. 更新 `hbm_cached_tokens_pool`。
6. 输出 `copy_counts[b] = Tx_b`。

无损策略要求 HBM sparse budget 能容纳 `GT2048` 中落在 prefill 满块范围内的 layer_kv_token。若启用卸载时始终保持 `Ts >= 2048`，则对 prefill 满块候选部分具备无损更新的容量前提。

### 7.3 接口兼容性结论

有损和无损策略对框架的影响应限制在以下位置：

1. `dsa_index_update` 内部如何选择 promote/demote；
2. `dsa_index_update` 输出的 `copy_counts` 是固定值还是动态值；
3. `dsa_scatter_h2d` 根据 `copy_counts` 只搬运每个 batch item 的有效前缀。

只要 `promote_idx/demote_idx` 预留容量为 `Tmax_copy`，并且接口包含 `copy_counts`，框架其余部分无需因有损/无损策略切换而改接口。无损策略下 `Tmax_copy` 定义为 2048；固定 `Tx` 有损策略只使用前 `Tx` 个有效槽位。

## 8. BlockManager 设计

本节定义三类块管理器。拆分的原因是 IndexCache、HBM KVcache 和 DRAM KVcache 的生命周期不同。

### 8.1 IndexBlockManager

职责：

1. 管理 HBM 上的 IndexCache 物理块。
2. 为 prefill 满块、prefill 尾块和 decode 新块分配 IndexCache 块。
3. 支持 prefill 满块前缀复用。
4. 不参与 DRAM 卸载。

hash 语义：

| 阶段 | 是否参与 hash 复用 | 原因 |
|---|---|---|
| prefill 满块 | 是 | 同 prompt 前缀可复用 IndexCache |
| prefill 尾块 | 否 | 非满块不作为稳定前缀块 |
| decode 新块 | 否 | decode 结果依赖采样，不复用 |

### 8.2 HBMBlockManager

职责：

1. 管理 HBM 上的 CKV/KPE 物理块。
2. prefill 阶段临时容纳完整 KVcache。
3. decode 阶段只保留 sparse budget、prefill 尾部非满块和 decode 新块。
4. 当 decode 新 token 填满当前块时，申请新的 HBM KVcache 块。

HBM KVcache 不做前缀复用，因此不需要 hash/refcount 前缀缓存语义。

prefill 完成后的释放规则：

1. Worker 已将重要 prefill layer_kv_token 打包到 `hbm_block_table` 前 `Ns` 个块。
2. Scheduler 调用 HBMBlockManager 释放 prefill 满块中不再常驻的 HBM 块。
3. `hbm_block_table` 保留 sparse MLA 需要的逻辑顺序：sparse budget 块在前，尾块和 decode 块在后。

### 8.3 DramBlockManager

职责：

1. 管理 CPU DRAM 上的 CKV/KPE 物理块。
2. 只保存 prefill 满块。
3. 支持 prefill 满块前缀复用。
4. 不保存 prefill 尾部非满块和 decode 新 token。

DramBlockManager 的 hash/refcount 机制可参考 Qwen KVStar 的 `CPUBlockManager`：

1. 对 prefill 满块做链式 hash；
2. hash 命中后仍校验 token id，避免 hash 碰撞；
3. 物理块使用 refcount 管理多个请求共享；
4. refcount 归零后块回到空闲队列。

### 8.4 前缀命中 `Nc`

前缀复用需要同时命中 IndexCache 和 DRAM KVcache。Scheduler 计算：

```text
Nc = min(IndexBlockManager 命中块数, DramBlockManager 命中块数)
```

只有前 `Nc` 个连续 prefill 满块视为可复用。若其中任一 manager 在某块处 miss，则该块及其后的 prefill 满块都按未命中处理。

### 8.5 PoolEntryManager

PoolEntryManager 管理 `hbm_cached_tokens_pool` 的槽位。每个启用卸载的请求需要独占一个 `hbm_cached_tokens_pool_entry`，也就是运行时符号 `e_b`，用于索引：

```text
hbm_cached_tokens_pool[:, hbm_cached_tokens_pool_entry, :]
```

职责：

1. 请求进入 prefill 调度时分配 entry。
2. 请求结束或被释放时归还 entry。
3. 保证同一时刻处于运行状态的卸载请求不会写同一个 pool 槽位。
4. 为 `prepare_decode()` 提供 `req_pool_entries = tensor[(bs,), int32, npu]`。

## 9. Sequence 设计

本节定义单个请求需要保存的新增元数据。调度器和 Worker 通过这些字段建立请求到三类物理 cache 的映射。

| 字段 | 类型 | 含义 |
|---|---|---|
| `index_block_table` | `list[int]` | IndexCache 的 HBM 物理块号 |
| `hbm_block_table` | `list[int]` | HBM CKV/KPE 的物理块号，decode 阶段使用 sparse 语义 |
| `dram_block_table` | `list[int]` | DRAM CKV/KPE 的物理块号，只覆盖 prefill 满块 |
| `hbm_cached_tokens_pool_entry` | `int` | 该请求在 `hbm_cached_tokens_pool` 中的槽位 |
| `num_prefill_blocks` | `int` | `Nprefill` |
| `num_prefill_full_blocks` | `int` | `Np` |
| `num_prefill_tail_blocks` | `int` | `Nr` |
| `num_prefix_cached_blocks` | `int` | `Nc` |
| `num_sparse_blocks` | `int` | `Ns` |
| `num_sparse_tokens` | `int` | `Ts` |
| `prefill_tail_len` | `int` | `tail_len` |
| `offload_enabled` | `bool` | 是否启用 decode KVcache 卸载 |

`hbm_block_table` 的语义需要明确：

1. prefill 期间，它可临时表示完整 prefill KVcache 的 HBM 块。
2. prefill 完成后，它必须切换到 sparse 语义。
3. decode 期间，MLA、slot mapping 和 `dsa_scatter_h2d` 都使用 sparse 语义的 `hbm_block_table`。

## 10. Scheduler 设计

本节描述调度器如何在 prefill/decode 阶段分配和释放资源。

### 10.1 Prefill 准入

调度一个等待请求进入 prefill 时，需要同时满足：

1. IndexBlockManager 可分配或复用 `Nprefill` 个 IndexCache HBM 块。
2. HBMBlockManager 可分配 `Nprefill` 个 KVcache HBM 块。
3. DramBlockManager 可分配或复用 `Np` 个 DRAM KVcache 满块。
4. 若启用卸载，PoolEntryManager 可分配一个 `hbm_cached_tokens_pool_entry`。
5. 请求长度满足 `max_model_len`、`max_num_batched_tokens` 等引擎限制。

对启用前缀复用的请求，Scheduler 先分别查询 IndexBlockManager 和 DramBlockManager 的前缀命中块数，再按 `Nc = min(index_hits, dram_hits)` 记录可复用前缀。

### 10.2 Prefill 完成后的 HBM 释放

prefill forward 完成并且 Worker 已完成 DRAM offload 和 sparse budget 初始化后，Scheduler 执行：

1. 将请求的 `hbm_block_table` 调整为 sparse 语义。
2. 调用 HBMBlockManager 释放不再常驻的 prefill 满块 HBM 物理块。
3. 保留 sparse budget 块、prefill 尾部非满块和 decode append 需要的块。

### 10.3 Decode 调度

decode 阶段调度时：

1. 如果 decode 常驻 KV 的最后一个 HBM 块写满，向 HBMBlockManager 申请新块。
2. 如果 IndexCache 最后一个块写满，向 IndexBlockManager 申请新块。
3. decode 新 token 不向 DramBlockManager 申请。
4. 调度器需要为当前 batch 准备可 tensor 化的 block table 和长度元数据。

## 11. Worker 与 tensor 化元数据

本节描述 Worker 在进入 model.forward 前需要准备哪些 tensor。原则是：model.forward 内部只消费 tensor，不再从 Python 对象逐项取元数据并触发小 H2D。

### 11.1 物理 tensor

Worker 初始化时维护：

| tensor | 形状 | device | 说明 |
|---|---:|---|---|
| `index_cache` | `tensor[(L, Cidx, B, 1, Didx), bf16, npu]` | NPU | DSA indexer 使用 |
| `hbm_ckv_cache` | `tensor[(L, Chbm, 1, B, Dckv), bf16, npu]` | NPU | HBM CKV |
| `hbm_kpe_cache` | `tensor[(L, Chbm, 1, B, Dkpe), bf16, npu]` | NPU | HBM KPE |
| `dram_ckv_cache` | `tensor[(L, Cdram, 1, B, Dckv), bf16, cpu]` | CPU DRAM | DRAM CKV |
| `dram_kpe_cache` | `tensor[(L, Cdram, 1, B, Dkpe), bf16, cpu]` | CPU DRAM | DRAM KPE |
| `hbm_cached_tokens_pool` | `tensor[(L, pool_capacity, Tmax_sparse), int32, npu]` | NPU | sparse slot 到原始 token id 的映射 |

### 11.2 Decode 前 tensor 化元数据

`prepare_decode()` 额外准备：

| tensor | 形状 | dtype | device | 说明 |
|---|---:|---|---|---|
| `index_block_tables` | `tensor[(bs, Nmax_index), int32, npu]` | int32 | NPU | IndexCache block table |
| `hbm_block_tables` | `tensor[(bs, Nmax_hbm), int32, npu]` | int32 | NPU | sparse HBM KVcache block table |
| `dram_block_tables` | `tensor[(bs, Nmax_dram), int32, npu]` | int32 | NPU | DRAM KVcache block table |
| `req_pool_entries` | `tensor[(bs,), int32, npu]` | int32 | NPU | 每个请求对应的 pool entry，即 `e_b` |
| `candidate_lens` | `tensor[(bs,), int32, npu]` | int32 | NPU | 每个请求参与候选的 prefill 满块 token 数，等于 `Tp` |
| `sparse_selected_lens` | `tensor[(bs,), int32, npu]` | int32 | NPU | 每个请求当前 sparse budget 有效长度，等于 `Ts` |
| `prefill_tail_lens` | `tensor[(bs,), int32, npu]` | int32 | NPU | 每个请求 prefill 尾部非满块 token 数，等于 `tail_len` |
| `decode_lens` | `tensor[(bs,), int32, npu]` | int32 | NPU | 每个请求 decode 常驻 token 数，等于 `Sd` |
| `sparse_kv_lens` | `tensor[(bs,), int32, npu]` | int32 | NPU | MLA 使用的 sparse KV 长度，等于 `sparse_kv_len` |

这些 tensor 放入 `Context`，供每层 forward 使用。

## 12. 模型 forward 设计

本节描述 `deepseek_v32.py` 中 prefill 和 decode forward 需要新增的逻辑。

### 12.1 Prefill forward

prefill forward 的主要职责：

1. 写入未命中的 CKV/KPE/IndexCache。
2. 对 prefill 满块的 CKV/KPE 建立 DRAM 副本。
3. 初始化每层 HBM sparse budget。
4. 写入 `hbm_cached_tokens_pool[layer, entry, :Ts]`。

如果存在 `Nc > 0` 的前缀命中，命中部分的 IndexCache 和 DRAM KVcache 由 block table 指向复用物理块。命中的 DRAM KV 块会在 prefill forward 前完整加载到 HBM，参与本次 prefill attention；prefill 结束后，这些临时 HBM 前缀块与未命中块一起释放，只保留 sparse budget、尾块和 decode 后续需要的块。

### 12.2 Decode forward

每层 decode forward 的数据流如下：

```mermaid
flowchart TD
    A["hidden_states"] --> B["生成当前 token 的 q / ckv / kpe / index_k"]
    B --> C["写入 HBM decode 常驻 KVcache 和 IndexCache"]
    C --> D["dsa_indexer_score"]
    D --> E["dsa_index_update<br/>原地更新 hbm_cached_tokens_pool<br/>输出 promote/demote/copy_counts"]
    E --> F["dsa_scatter_h2d<br/>将 promote 的 CKV/KPE 从 DRAM 拷贝到 HBM sparse slot"]
    F --> G["dense MLA over sparse HBM KV"]
    G --> H["后续 attention 输出投影与 MLP/MoE"]
```

MLA 的输入必须使用：

```text
sparse_kv_lens = Ts + tail_len + Sd
```

以及 sparse 语义的 `hbm_block_tables`。

## 13. 算子接口设计

本节定义三个新增算子的稳定接口。

三者在单层 decode 中的输入输出关系如下：

```mermaid
flowchart LR
    Q["query_index<br/>当前 token"] --> Score["dsa_indexer_score"]
    IC["IndexCache<br/>index_block_table"] --> Score
    Score --> SO["score_out"]

    SO --> Update["dsa_index_update"]
    Pool["hbm_cached_tokens_pool<br/>单层切片"] --> Update
    Meta["candidate_lens / selected_lens / req_pool_entries"] --> Update
    Update --> Promote["promote_idx"]
    Update --> Demote["demote_idx"]
    Update --> Counts["copy_counts"]

    Promote --> Scatter["dsa_scatter_h2d"]
    Demote --> Scatter
    Counts --> Scatter
    BT["hbm_block_table + dram_block_table"] --> Scatter
    DKV["DRAM CKV/KPE<br/>单层切片"] --> Scatter
    Scatter --> HKV["HBM sparse slots<br/>单层 CKV/KPE"]
```

### 13.1 `dsa_indexer_score`

功能：根据当前 query 的 index 表征和 HBM IndexCache，为每个候选 prefill layer_kv_token 计算重要性分数。

接口定义：

```python
def dsa_indexer_score(
    query_index,          # tensor[(bs, Hidx, Didx), bf16, npu]
    index_cache,          # tensor[(Cidx, B, 1, Didx), bf16, npu]
    index_weights,        # tensor[(bs, Hidx), bf16, npu]
    index_block_table,    # tensor[(bs, Nmax_index), int32, npu]
    candidate_lens,       # tensor[(bs,), int32, npu]
    score_out,            # tensor[(bs, Tmax_candidate), bf16, npu]
):
    ...
```

输出语义：

```text
score_out[b, t] = 第 b 个请求中原始 prefill 满块 token id 为 t 的 layer_kv_token 分数
```

只有 `0 <= t < candidate_lens[b]` 的位置有效，其中 `candidate_lens[b] = Tp_b`。无效位置应被置为足够小的哨兵值。

### 13.2 `dsa_index_update`

功能：根据分数更新 HBM sparse budget 的索引集合。该算子只更新索引，不搬运 CKV/KPE。

接口定义：

```python
def dsa_index_update(
    score,                     # tensor[(bs, Tmax_candidate), bf16, npu], input/output
    hbm_cached_tokens_pool,    # tensor[(pool_capacity, Tmax_sparse), int32, npu], input/output，单层切片
    promote_idx,               # tensor[(bs, Tmax_copy), int32, npu], output
    demote_idx,                # tensor[(bs, Tmax_copy), int32, npu], output
    copy_counts,               # tensor[(bs,), int32, npu], output
    candidate_lens,            # tensor[(bs,), int32, npu]
    selected_lens,             # tensor[(bs,), int32, npu]
    req_pool_entries,          # tensor[(bs,), int32, npu]
    max_copy_tokens: int,       # 等于 Tmax_copy；无损策略取 2048，固定 Tx 策略可小于 2048
):
    ...
```

索引语义：

| tensor | 语义 |
|---|---|
| `promote_idx[b, i]` | 原始 prefill 满块 token id `t`，范围为 `[0, candidate_lens[b])` |
| `demote_idx[b, i]` | HBM sparse budget 本地 slot id `s`，范围为 `[0, selected_lens[b])` |
| `copy_counts[b]` | 第 `b` 个请求本次有效 promote/demote 对数，即 `Tcopy_b` |

对每个 batch item，仅 `0 <= i < Tcopy_b` 的 `promote_idx/demote_idx` 有效。剩余位置由实现填充任意合法值或 0，`dsa_scatter_h2d` 必须忽略。

固定 `Tx` 策略中，`copy_counts[b] = Tx`。无损动态策略中，`copy_counts[b]` 等于该请求该层该步中 `GT2048` 尚未驻留 HBM 的 layer_kv_token 数。

### 13.3 `dsa_scatter_h2d`

功能：根据 `promote_idx/demote_idx/copy_counts`，把 CKV/KPE 从 DRAM 拷贝到 HBM sparse budget。

接口定义：

```python
def dsa_scatter_h2d(
    promote_idx,          # tensor[(bs, Tmax_copy), int32, npu]
    demote_idx,           # tensor[(bs, Tmax_copy), int32, npu]
    copy_counts,          # tensor[(bs,), int32, npu]
    hbm_block_table,      # tensor[(bs, Nmax_hbm), int32, npu]
    dram_block_table,     # tensor[(bs, Nmax_dram), int32, npu]
    hbm_ckv_cache,        # tensor[(Chbm, 1, B, Dckv), bf16, npu], input/output，单层切片
    hbm_kpe_cache,        # tensor[(Chbm, 1, B, Dkpe), bf16, npu], input/output，单层切片
    dram_ckv_cache,       # tensor[(Cdram, 1, B, Dckv), bf16, cpu]，单层切片
    dram_kpe_cache,       # tensor[(Cdram, 1, B, Dkpe), bf16, cpu]，单层切片
):
    ...
```

地址映射：

```text
源地址：
  t = promote_idx[b, i]
  dram_logical_block = blk(t)
  dram_offset = off(t)
  dram_physical_block = dram_block_table[b, dram_logical_block]

目的地址：
  s = demote_idx[b, i]
  hbm_logical_block = blk(s)
  hbm_offset = off(s)
  hbm_physical_block = hbm_block_table[b, hbm_logical_block]
```

循环范围由 `Tcopy_b = copy_counts[b]` 决定：

```text
for i in range(Tcopy_b):
    copy CKV/KPE from DRAM source to HBM destination
```

`dsa_scatter_h2d` 的最终实现预计由 AIV 发起并行拷贝；设计仍保留 timing 观测点，用于确认实际带宽和 TPOT 影响。

## 14. TP 策略

本节说明 TP 场景下新增逻辑如何保持一致。

所有 TP rank 拥有相同的 `Cidx/Chbm/Cdram` 配置，并在本 rank 上维护本 rank 需要的 CKV/KPE/IndexCache 物理 cache。`dsa_indexer_score`、`dsa_index_update` 和 `dsa_scatter_h2d` 可在各 TP rank 冗余执行。

冗余执行的优点：

1. 不需要在 decode 每层增加 rank0 到其他 rank 的 broadcast。
2. `promote_idx/demote_idx/copy_counts` 由相同输入确定，语义上可保持一致。
3. H2D 搬运仍由各 rank 搬运本 rank 的 CKV/KPE 切片。

后续也可优化为 rank0 计算 `promote_idx/demote_idx/copy_counts` 后广播。是否采用该优化取决于 indexer/update 耗时与广播同步开销的对比。

## 15. 资源规模与观测指标

本节记录关键资源规模和必须观测的指标，用于后续验证设计是否达到目标。

### 15.1 score tensor 规模

`dsa_indexer_score` 输出全量候选分数：

```text
score_out = tensor[(bs, Tmax_candidate), bf16, npu]
```

`Tmax_candidate` 不超过 `max_model_len`。在默认最大序列长度 `max_model_len <= 131072` 下，即使 `bs=256`：

```text
256 * 131072 * 2 bytes = 64 MiB
```

该规模可接受。设计不额外处理用户设置 `max_model_len > 131072` 的情况。

### 15.2 IndexCache 常驻 HBM 容量

IndexCache 常驻 HBM 是本设计的必要代价。典型 `B=128` 时，单层单块 IndexCache 大小为：

```text
B * 128 * sizeof(bf16)
= 128 * 128 * 2 bytes
= 32768 bytes
≈ 32 KiB
```

全 61 层单块约：

```text
61 * 32 KiB ≈ 1.91 MiB
```

如果 `max_model_len=131072` 且 `B=128`，则 `Nmax_prefill=1024`。满长请求需要 `1024` 个 IndexCache block，全 61 层约：

```text
1024 * 1.91 MiB ≈ 1.91 GiB
```

因此 IndexCache 虽显著小于完整 CKV/KPE，但仍会影响 `Cidx` 和长序列 decode batch size。调度器需要把 IndexCache HBM block 作为独立资源管理。

### 15.3 元数据规模

多份 block table 和长度 tensor 的体量远小于 KVcache。关键要求不是减少这些元数据本身，而是在 model.forward 前完成 tensor 化，使 forward 内部能够算子化执行，避免每层每步细碎 H2D。

### 15.4 性能观测项

需要记录以下 timing：

1. prefill 后 CKV/KPE D2H 卸载耗时；
2. sparse budget 初始化耗时；
3. `dsa_indexer_score` 耗时；
4. `dsa_index_update` 耗时；
5. `dsa_scatter_h2d` 耗时和有效带宽；
6. dense MLA 耗时；
7. decode TPOT、decode TPS 和可调度 batch size 上限。

## 16. 正确性验证

本节描述实现完成后的验证方法。

### 16.1 单算子验证

1. `dsa_indexer_score`：对小 batch、小序列，用 PyTorch reference 对比分数。
2. `dsa_index_update`：构造固定 score 和初始 pool，验证 promote/demote/copy_counts 与预期一致。
3. `dsa_scatter_h2d`：构造可辨识 CKV/KPE 值，验证 HBM 目标 slot 等于 DRAM 源 token。

### 16.2 端到端验证

1. 关闭卸载时，输出与当前基线一致。
2. 设置 `Ns=Np` 时，sparse MLA 路径与完整 HBM KVcache 路径接近或一致。
3. 启用静态 sparse budget 但不动态更新时，确认流程可跑通且显存下降。
4. 启用固定 `Tx` 动态更新时，对比输出质量、TPOT 和 batch size。
5. 启用无损动态更新时，验证 `GT2048` 是否全部进入 HBM 可见集合。

### 16.3 输出质量验证

1. short prompts：确认基本语言能力不崩。
2. long prompts：确认长上下文问答仍能抓住关键信息。
3. 同一 prompt 对比 dense MLA baseline、静态 sparse budget、固定 `Tx` 动态更新、无损动态更新。

## 17. 分阶段落地计划

本节把完整设计拆成可逐步验证的实现阶段。这里可以体现阶段性取舍。

### 阶段 0：文档和探针

1. 完成本文档。
2. 编写 `ut_ops` 探针：
   - sparse HBM KV budget 上 dense MLA 的输入格式；
   - `hbm_cached_tokens_pool` 更新正确性；
   - DRAM 到 HBM 单层 CKV/KPE 搬运正确性。

### 阶段 1：拆分元数据，不实际卸载

1. 引入 `index_block_table/hbm_block_table/dram_block_table`。
2. 暂时让三者指向兼容的物理块语义或 mock DRAM。
3. 验证关闭卸载时输出与当前基线一致。

### 阶段 2：prefill 后静态卸载

1. prefill 后将 prefill 满块 CKV/KPE 复制到 DRAM。
2. HBM 只保留固定 sparse budget、prefill 尾块和 decode 新块。
3. sparse budget 初始化先使用简单策略，例如最近 `Ts` 个 layer_kv_token。
4. decode 在 sparse HBM KV 上运行 dense MLA。

该阶段用于验证显存下降和 sparse MLA 路径正确性，允许输出有损。

### 阶段 3：固定 `Tx` 动态更新

1. `dsa_indexer_score`、`dsa_index_update`、`dsa_scatter_h2d` 均先写 PyTorch 原型。
2. `dsa_index_update` 实现固定 `Tx` 有损策略。
3. `copy_counts` 固定填 `Tx`，但接口保持动态兼容。
4. 所有 TP rank 冗余执行 update 和 scatter。
5. 三个新增算子都加入 timing。

### 阶段 4：CANN H2D 搬运算子

1. 将 `dsa_scatter_h2d` 替换为 CANN custom op。
2. 使用 AIV 发起并行拷贝。
3. 对比 PyTorch 原型与 CANN 实现的正确性和带宽。

### 阶段 5：无损动态更新

1. 在不改接口的前提下，将 `dsa_index_update` 内部策略替换为 `GT2048` 覆盖策略。
2. `copy_counts` 变为每请求、每层、每步动态值。
3. 验证 `GT2048` 全覆盖和动态 `Tx` 对 TPOT 的影响。

### 阶段 6：前缀复用

1. DramBlockManager 支持 hash/refcount 前缀复用。
2. IndexBlockManager 支持 prefill 满块 hash/refcount 前缀复用。
3. Scheduler 使用 `Nc = min(IndexBlockManager 命中块数, DramBlockManager 命中块数)`。
4. Worker 支持 prefill 前加载命中前缀，或直接基于命中前缀初始化 sparse budget。
