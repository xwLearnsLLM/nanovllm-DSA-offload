# Ascend 950 nano-vLLM C8 offload 算子

本仓库面向 Ascend 950 / CANN 9.1，设计和实现 GLM-5.1/GLM-5.2 W4A4C8 decode KV-cache 卸载算子。当前已有非 MTP C8 链路，以及 MTP1～3 索引选择与 union 管理的功能优先初版。

Torch namespace 固定为 `nanovllm_dsa`，Python 包名为 `nanovllm_dsa_a5`。

## 实现状态

状态含义：

- **初版已实现**：仓库中已有源码、Torch 注册和单算子/图 UT，但尚未代表 nano-vLLM 整网已经接入。
- **源码初版，待上机**：源码、Torch 注册和 UT 已写入，但还没有在 Ascend 950 上完成编译与功能验收。
- **复用待验证**：算子入口已经存在，但新增用法还没有完成对应语义和图 UT。
- **尚未实现**：本文仅定义接口与行为，仓库中还没有实现。

| 使用场景 | 公开入口 | 状态 | 当前结论 |
| --- | --- | --- | --- |
| 非 MTP 索引选择与管理 | `fused_li_manage_c8` | 单算子初版已实现 | 单个仓内 MIX kernel 融合 A5 C8 LightningIndexer 与 request-pool update；已覆盖 18-bit 边界、乱序 pool、重复更新和图链路 |
| 非 MTP / MTP1～3 KV 搬移与 Attention metadata | `kvcache_scatter_copy_c8` | 初版已实现，统一 ABI 待改造 | 当前源码已验证非 MTP 的真实 DRAM→HBM；需扩展为动态 copy capacity 和 TND 多 query metadata |
| 非 MTP sparse+tail Attention | `sparse_tail_attention_c8` | 初版已实现 | 复用 A5 原生 C8 QSFA；已验证单 query，当前限制 `1 <= Q_HEAD <= 64` |
| MTP1～3 多 query 索引选择与 union 管理 | `fused_li_manage_mtp_c8` | 单算子源码初版，待上机 | 单个仓内 MIX kernel 融合官方 A5 C8 LightningIndexer 语义与 request-pool union/update；优先保证功能正确，尚未做性能优化 |
| MTP1～3 sparse+tail Attention | 复用 `sparse_tail_attention_c8` | 复用待验证 | 现有接口形状已经允许 TND 多 query；还需验证 MTP1/2/3 的右下角因果语义与 graph replay |



## 公共术语与数据布局

- `B`：活跃请求数。
- `K`：speculative token 数，支持 `K=0/1/2/3`。
- `Q_b`：请求 `b` 在 target verification 中的 query 数；MTP1/2/3 分别通常为 `2/3/4`。
- `T=sum(Q_b)`：TND packed query 总行数。非 MTP 时 `T=B` 且 `Q_b=1`。
- `actual_seq_lengths_query`：`int32[B]` 累计 query 行数，最后一个元素等于 `T`。固定 MTP3 batch 的典型值是 `[4,8,...,4B]`。
- `C=cache_tokens[b]`：请求固定的 HBM sparse-cache token 预算。
- `candidate_lens[b]`：参与 LightningIndexer 选择的 prefill 满块 token 数。prompt 末尾非满块、历史 decode token 和本轮 verification token 均不参与 top-K。
- `cache_slots_pool[req_pool_entries[b], token_id]`：source token 到 HBM 逻辑 slot 的持久映射；`-1` 表示未缓存。
- block size 固定为 128，source capacity 当前最大为 `2^18=262144`。
- 一个 packed C8 KV token 固定为 656 bytes：`512 FP8 E4M3 + 64 BF16 RoPE + 4 FP32 scales`。

Indexer 的数值与布局语义对齐官方 A5 `npu_quant_lightning_indexer`：query/key 为 FP8 E4M3，weights 为 BF16，query/key scale 为 FP32，layout 为 `TND/PA_BSND`，`sparse_count=2048`，`sparse_mode=3`。非 MTP `fused_li_manage_c8` 已将 LI 和索引管理合入一个仓内 MIX kernel；Attention 使用原生 `npu_kv_quant_sparse_flash_attention`。

## IndexShare group 共享映射

本方案规定 GLM-5.2 的 shared 层不单独维护 `cache_slots_pool`。每个 IndexShare group 只保存一份映射；full 层执行一次 LightningIndexer 和索引管理，随后本组所有层复用：

- 完整 top-K 对应的 HBM 逻辑 slots；
- union miss source IDs；
- union miss destination slots；
- miss count。

各层只共享 token→slot 布局，不共享实际 KV 数据。每层仍有独立的 DRAM/HBM packed KV tensor，并使用同一批 destination slots 将本层数据写到相同的逻辑位置。

以 `full layer 6 + shared layer 7/8/9` 为例：

1. Layer 6 调用一次 `fused_li_manage*_c8`，更新该 group 的 `cache_slots_pool`。
2. Layer 6/7/8/9 依次使用同一份 miss 和 slot metadata，分别搬移各自的 packed KV。
3. 四层分别用各自 query 和 KV 执行 Attention，但使用相同的 top-K slots。
4. 在整个 group 消费完成前，不得覆盖管理输出 buffer。

同一 group 必须使用相同的 C、candidate source、request-pool 生命周期和 slot geometry。初始化、finish、abort、preemption 与 pool row 复用都必须按 group 原子处理。不同 full 层的 top-K 不同，因此不能跨 IndexShare group 共享映射。

首次 decode 的 eager 初始化也按 group 执行：由 full 层选出初始 top-C source tokens，建立一份映射，并把各层自己的 packed KV 搬到相同逻辑 slots；只有整组初始化成功后才标记 ready。该初始化属于框架慢路径，不新增 steady-decode 算子。

GLM-5.2 的 78 个 target layers 是 21 个 full / 57 个 shared；因此映射状态按 21 个 group 保存，而不是按 78 层保存。MTP draft layer 始终保留完整 Indexer，不加入 target layer 的 IndexShare group。

## 已实现：非 MTP 接口

非 MTP split 链路为：

```text
fused_li_manage_c8
    -> kvcache_scatter_copy_c8
    -> sparse_tail_attention_c8
```

### `fused_li_manage_c8`

该入口是真正的单算子：一个 MIX kernel 内先生成与官方 A5 Quant LightningIndexer 对齐的 top-2048，再原地完成 request-pool update。输出前 `miss_counts[b]` 项为 miss-prefix；全部 2048 个 `destination_slots` 都供 Attention 使用。

```python
torch.ops.nanovllm_dsa.fused_li_manage_c8(
    query,                    # float8_e4m3fn[B,INDEX_HEADS,128]，INDEX_HEADS=32|64
    key,                      # float8_e4m3fn[INDEX_BLOCKS,128,1,128]
    weights,                  # bf16[B,INDEX_HEADS]
    query_dequant_scale,      # fp32[B,INDEX_HEADS]
    key_dequant_scale,        # fp32[INDEX_BLOCKS,128,1]
    actual_seq_lengths_query, # cumulative int32[B]，非 MTP 通常为 [1,...,B]
    req_pool_entries,         # int32[B]
    cache_slots_pool,         # int32[POOL_SIZE,SOURCE_CAPACITY]，in/out
    cache_tokens,             # int32[B]
    candidate_lens,           # int32[B]
    block_table,              # int32[B,INDEX_MAX_BLOCKS]
) -> (
    source_ids,               # int32[B,1,2048]，miss-prefix + hit-suffix
    destination_slots,        # int32[B,1,2048]
    miss_counts,              # int32[B]
    cache_slots_alias,
)
```

### `kvcache_scatter_copy_c8`：当前接口与统一目标接口

目标是让同一个算子统一服务非 MTP 和 MTP1～3：搬移每请求的 miss/union-miss，并根据独立的 `topk_destination_slots` 生成每个 query row 的 sparse+tail slots。需要修改当前实现，但保留公开名称 `kvcache_scatter_copy_c8`。

#### 当前真实接口

当前源码仅支持非 MTP。它把 copy metadata 和 Attention top-K slots 合并在同一组 2048-capacity tensors 中，并假设每请求只有一个 query，即 `T=B`。

```python
torch.ops.nanovllm_dsa.kvcache_scatter_copy_c8(
    hbm_packed_kv_bytes,      # int8[HBM_BLOCKS,128,1,656]，in/out
    dram_packed_kv_bytes,     # swapped-memory int8[DRAM_BLOCKS,128,1,656]
    hbm_block_table,          # int32[B,HBM_MAX_BLOCKS]
    dram_block_table,         # int32[B,DRAM_MAX_BLOCKS]
    source_token_ids,         # int32[B,2048] 或 int32[B,1,2048]
    destination_slots,        # 与 source_token_ids 同形状；同时充当完整 top-K slots
    copy_counts,              # int32[B]；只搬前 copy_counts 项
    cache_tokens,             # int32[B]
    candidate_lens,           # int32[B]
    actual_seq_lengths_kv,    # int32[B]
    max_tail_tokens,          # int
) -> (
    hbm_packed_kv_alias,
    attention_slots,          # int32[B,1,2048+max_tail_tokens]
    resident_seq_lengths,     # int32[B]
)
```

当前限制来自源码，而不是算子语义：C++ wrapper 和 CANN host/kernel 都要求 `COPY_CAP=2048`；metadata 输出第一维固定为 B；算子没有 `actual_seq_lengths_query`，因此无法把 T 个 MTP query rows 映射回 B 个请求。

#### 统一目标接口

统一接口把“需要搬移的 union miss”和“每个 query 的完整 top-K slots”拆成两组输入。第一阶段实现规定 `COPY_CAP=2048`（非 MTP）或 `COPY_CAP=8192`（MTP1～3）；后续如有必要再支持 4096/6144 的紧凑 buffer。

```python
torch.ops.nanovllm_dsa.kvcache_scatter_copy_c8(
    hbm_packed_kv_bytes,      # int8/one-byte view[HBM_BLOCKS,128,1,656]，in/out
    dram_packed_kv_bytes,     # swapped-memory one-byte view[DRAM_BLOCKS,128,1,656]
    hbm_block_table,          # int32[B,HBM_MAX_BLOCKS]
    dram_block_table,         # int32[B,DRAM_MAX_BLOCKS]
    copy_source_ids,          # int32[B,COPY_CAP]，COPY_CAP=2048 或 8192
    copy_destination_slots,   # int32[B,COPY_CAP]
    copy_counts,              # int32[B]，仅前 copy_counts 项有效
    topk_destination_slots,   # int32[T,1,2048]，每个 query row 的完整 top-K slots
    cache_tokens,             # int32[B]
    candidate_lens,           # int32[B]
    actual_seq_lengths_query, # cumulative int32[B]，最后一项为 T
    actual_seq_lengths_kv,    # int32[B]，每请求最后一个 query 的逻辑 KV 长度
    max_tail_tokens,          # int，固定图 tail capacity
) -> (
    hbm_packed_kv_alias,
    attention_slots,          # int32[T,1,2048+max_tail_tokens]
    resident_seq_lengths,     # int32[B]
)
```

非 MTP 时，`T=B`，`COPY_CAP=2048`，`copy_destination_slots` 和 `topk_destination_slots` 可以引用同一个 LIDU destination tensor；虽然在参数表中出现两次，但不会复制数据。MTP 时，copy metadata 是按请求去重后的 union miss `[B,8192]`，而 top-K slots 是逐 query 的 `[T,1,2048]`，二者必须分开。

`C>0` 时，`attention_slots` 为每个 query 自己的 top-2048 slots 加 `[C,C+tail)`；`resident_seq_lengths=C+tail`。`C=0` 时走 dense row，slots 覆盖 `[0,actual_len)`，resident length 为原始 KV 长度。

#### 后续实现修改

1. Torch schema 增加 `topk_destination_slots` 和 `actual_seq_lengths_query`，并正确声明 `hbm_packed_kv_bytes` 的 mutable alias。
2. C++ wrapper 将 copy metadata 标准化为 `int32[B,COPY_CAP]`；非 MTP 的 `[B,1,2048]` 可用无拷贝 view 传入。校验 `COPY_CAP=2048|8192`、`topk_destination_slots=[T,1,2048]`、`actual_seq_lengths_query=[B]` 且最后一项为 T。
3. Meta/Fake 输出从 `[B,1,2048+max_tail_tokens]` 改为 `[T,1,2048+max_tail_tokens]`；`resident_seq_lengths` 仍为 `[B]`。
4. CANN host tiling 删除写死的 `COPY_CAP=2048`，从输入 shape 读取 copy capacity，并分别下发 B、T、copy capacity 和 attention capacity。
5. CANN kernel 的搬移阶段仍按 B 行读取 `copy_counts`，但最多扫描 8192 个 union entries；metadata 阶段按 T 行执行，并通过累计 `actual_seq_lengths_query` 找到每个 query 所属请求。
6. 保留现有非 MTP 行为：`T=B`、`COPY_CAP=2048` 时结果必须逐项等同旧实现。新增 MTP1/2/3、variable `Q_b`、C=0 mixed batch、8192 union、真实 DRAM→HBM 和 graph replay 测试。
7. 完成统一 ABI 后删除所有 MTP 专用 SCATTER 命名设想；框架只调用 `kvcache_scatter_copy_c8`。

### `sparse_tail_attention_c8`

该入口是 Python custom op，内部直接调用 A5 原生 C8 QSFA，没有重复维护 Attention CANN kernel。

```python
torch.ops.nanovllm_dsa.sparse_tail_attention_c8(
    query,                    # bf16/fp16[T,Q_HEAD,576]，1<=Q_HEAD<=64
    packed_kv,                # one-byte packed view[HBM_BLOCKS,128,1,656]
    attention_slots,          # int32[T,1,2048+max_tail_tokens]
    hbm_block_table,          # int32[B,HBM_MAX_BLOCKS]
    actual_seq_lengths_query, # cumulative int32[B]，最后一项为 T
    resident_seq_lengths,     # int32[B]
    scale_value,              # float
) -> attention_out            # bf16/fp16[T,Q_HEAD,512]
```

虽然接口形状允许 `T>B`，当前只验收了非 MTP 单 query。MTP 多 query 是否正确由后续 MTP1～3 causal UT 决定。

## 拟新增：MTP1～3 C8 接口

MTP target verification 使用 `Q_b=K_b+1` 个 query。统一 split 链路为：

```text
full layer:
    fused_li_manage_mtp_c8  # 每个 IndexShare group 只调用一次

each layer in the group:
    kvcache_scatter_copy_c8
        -> sparse_tail_attention_c8
```

shared 层不调用任何索引管理算子，只复用 full 层输出。

### `fused_li_manage_mtp_c8`（单算子功能优先初版，待上机）

该算子在一个仓内 `MIX_AIC_1_2` kernel 中对 TND packed 的 2～4 路 query 执行与官方 A5 C8 LightningIndexer 一致的 `sparse_mode=3` 选择，然后按请求求各路 top-2048 的并集，只更新一次 group-shared `cache_slots_pool`。公开接口一次调用只产生一个设备 kernel launch，便于直接纳入 full-decode-only graph。

```python
torch.ops.nanovllm_dsa.fused_li_manage_mtp_c8(
    query,                    # float8_e4m3fn[T,INDEX_HEADS,128]，INDEX_HEADS=32|64
    key,                      # float8_e4m3fn[INDEX_BLOCKS,128,1,128]
    weights,                  # bf16[T,INDEX_HEADS]
    query_dequant_scale,      # fp32[T,INDEX_HEADS]
    key_dequant_scale,        # fp32[INDEX_BLOCKS,128,1]
    actual_seq_lengths_query, # cumulative int32[B]，每请求增量 Q_b∈[2,4]
    req_pool_entries,         # int32[B]
    cache_slots_pool,         # int32[POOL_SIZE,SOURCE_CAPACITY]，group-shared，in/out
    cache_tokens,             # int32[B]
    candidate_lens,           # int32[B]
    block_table,              # int32[B,INDEX_MAX_BLOCKS]
) -> (
    topk_destination_slots,   # int32[T,1,2048]，每个 verification query 的完整 slots
    miss_source_ids,          # int32[B,8192]，每请求 union unique miss，仅前 miss_counts 有效
    miss_destination_slots,   # int32[B,8192]，与 miss_source_ids 一一对应
    miss_counts,              # int32[B]
    cache_slots_alias,
)
```

不输出逐 query 的 `topk_source_ids`：框架后续只需要完整 `topk_destination_slots` 做 Attention，以及按请求去重后的 union miss 做搬移。测试可以单独调用官方 LightningIndexer 获得 source-index golden。

管理行为必须满足：原 hit 保持 slot、union miss 去重、所有 query 的 top-K 在更新后均能通过共享映射得到 slot、有效 slot 仍唯一覆盖 `[0,C)`、相同输入重复更新时第二次 union miss 为零。

当前实现中，同一 MIX core group 先完成 Quant LightningIndexer，再由 AIV0 完成 union 去重、victim 选择、一次 request-pool 更新和逐 query slot 发布。官方 `npu_quant_lightning_indexer` 仅作为单测 golden，不在公开算子的运行路径中；当前没有时延门禁。

统一 `kvcache_scatter_copy_c8` 的 MTP 用法是：`copy_source_ids/copy_destination_slots` 接收 `[B,8192]` union miss buffers，`topk_destination_slots` 接收 `[T,1,2048]`，输出 `[T,1,2048+max_tail_tokens]`。tail 使用该请求最终可见的连续 HBM tail slots；`sparse_tail_attention_c8` 通过 `actual_seq_lengths_query`、`resident_seq_lengths` 与 `sparse_mode=3` 对较早 verification query 屏蔽未来 token。

### 复用 `sparse_tail_attention_c8`（MTP 语义待验证）

MTP 不新增 Attention 入口。调用方式保持不变，但此时 `T=sum(Q_b)`，`attention_slots` 第一维也是 T：

```python
torch.ops.nanovllm_dsa.sparse_tail_attention_c8(
    query,                    # bf16/fp16[T,Q_HEAD,576]
    packed_kv,                # one-byte view[HBM_BLOCKS,128,1,656]
    attention_slots,          # int32[T,1,2048+max_tail_tokens]
    hbm_block_table,          # int32[B,HBM_MAX_BLOCKS]
    actual_seq_lengths_query, # cumulative int32[B]，增量为 2/3/4
    resident_seq_lengths,     # int32[B]，对应每请求最终 verification KV 长度
    scale_value,
) -> attention_out            # bf16/fp16[T,Q_HEAD,512]
```

必须分别用 MTP1、MTP2、MTP3 与逐 query CPU FP32 causal golden 对比，不能仅凭 native QSFA 接口支持 TND 就判定正确。

## MTP union cache 的硬约束

一次 target verification 先搬完 union miss，再并行计算所有 query 的 Attention，因此所有 query 的 top-K 并集必须同时驻留：

```text
U_b = |union(topK(query_0), ..., topK(query_(Q_b-1)))|
C_b >= U_b
```

为避免运行时 overflow，框架按最坏情况保证：

| 模式 | verification query 数 | 最坏 union | offload 时建议最小 C |
| --- | ---: | ---: | ---: |
| MTP1 | 2 | 4096 | 4096 |
| MTP2 | 3 | 6144 | 6144 |
| MTP3 | 4 | 8192 | 8192 |

如果请求当前档位的 C 不满足该约束，框架必须提高 C，或者让该请求走不卸载/dense fallback；算子不得静默丢弃某一路 top-K。`C=0` 的短请求仍是合法 no-op，并在 Attention 侧走 dense row。

## MTP draft layer 的缓存

MTP draft module 自身包含 Attention，因此有独立的 C8 KV cache 和 C8 IndexCache。它与 target verification 的 78 层缓存不是同一套状态。

当前建议：MTP draft layer 数量很少，优先让其完整 C8 KV/Index cache 常驻 HBM；每个 draft step 调用官方 C8 LightningIndexer 和原生 QSFA，不引入新的卸载算子。若后续确认必须卸载 draft layer，则每次递归都是 `query_len=1`，可直接复用现有非 MTP 三算子链路，并使用独立的 request pool、block tables 与 `cache_slots_pool`。

该部分属于 nano-vLLM 框架接入策略，本仓库目前没有端到端实现。

## MTP 验收要求

- `fused_li_manage_mtp_c8`：覆盖 MTP1/2/3、variable `Q_b`、C=0/C>0 mixed batch、乱序 request pool、零 miss、随机 0～300 miss、最坏 8192 union、重复更新、IndexShare group 复用和 18-bit source boundary。
- `kvcache_scatter_copy_c8` 统一 ABI：同时覆盖 `T=B/COPY_CAP=2048` 和 `T>B/COPY_CAP=8192`；使用真实 `empty_with_swapped_memory`，每次调用前 poison 目标 HBM，并验证 656 bytes 精确搬移、guard、union 去重和零搬移。
- `sparse_tail_attention_c8` MTP 用法：分别覆盖 Q=2/3/4、不同 tail、每个 query 不同 top-K，并与逐行 CPU FP32 右下角因果 golden 对比。
- graph chain：capture 使用零 miss，replay 切换到非零 union miss；验证 cache state、数据搬移、Attention、输出地址和 IndexShare group 共享结果。
- full/shared 对照：共享一份映射的结果必须与“每层从相同初始状态独立执行相同管理”完全一致；shared 层不得再次调用管理算子。

## 编译

从仓库根目录执行：

```bash
unset ASCEND_CUSTOM_OPP_PATH
unset NANOVLLM_A5_INSTALL_OPP_PATH
unset NANOVLLM_CUST_OPAPI_LIB
unset A5_SOC_VERSION
unset SOC_VERSION
unset CANN_INSTALL_PATH
export ASCEND_HOME_PATH=/usr/local/Ascend/ascend-toolkit/latest
export CANN_INSTALL_PATH=$ASCEND_HOME_PATH
source "$ASCEND_HOME_PATH/set_env.sh"
export ASCEND_RT_VISIBLE_DEVICES=0
export ASCEND_LAUNCH_BLOCKING=0
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1
export PYTHONPATH=$PWD/torch_extension:$PYTHONPATH
export SOC_VERSION=ascend950
export NANOVLLM_A5_OPS_PYTHON=python3
export NANOVLLM_A5_OPS_BUILD_JOBS=64
bash build.sh
```

## 当前非 MTP C8 测试

```bash
python3 tests/test_fused_li_manage_c8.py --device npu:0 --heads 32,64 --batch-sizes 32 --source-lens 20096 --cache-tokens 6144 --miss-ranges 0:300 --iters 0 --seed 7
```

```bash
for count in 0 1 100 300 2048; do python3 tests/test_kvcache_scatter_copy_c8.py --device npu:0 --batch-size 32 --source-len 20096 --cache-tokens 6144 --tail-tokens 257 --max-tail-tokens 512 --copy-min "$count" --copy-max "$count" --warmup 3 --iters 10 --seed 7; done
```

```bash
python3 tests/test_sparse_tail_attention_c8.py --device npu:0 --heads 8 --batch-sizes 32 --cache-tokens 6144 --tail-tokens 64 --max-tail-tokens 512 --iters 0 --seed 7
```

```bash
python3 tests/test_c8_graph.py --device npu:0 --case pure-long --batch-size 2 --heads 8 --index-heads 32 --source-len 4096 --cache-tokens 3072 --tail-tokens 64 --max-tail-tokens 256 --miss-min 256 --miss-max 512 --replays 4 --seed 7
python3 tests/test_c8_graph.py --device npu:0 --case mixed --batch-size 2 --heads 8 --index-heads 32 --source-len 4096 --cache-tokens 3072 --tail-tokens 64 --max-tail-tokens 256 --miss-min 256 --miss-max 512 --replays 4 --seed 7
```

## MTP1～3 C8 索引管理测试

以下命令覆盖 32/64 index heads、MTP1/2/3 混合 batch、`C=0`、乱序 request pool、union miss 去重、hit slot 保持、重复零 miss、最坏 8192 union及公开接口单设备 kernel：

```bash
python3 tests/test_fused_li_manage_mtp_c8.py --device npu:0 --batch-size 6 --heads 32,64 --source-len 20096 --queries-per-request 0 --miss-min 0 --miss-max 300 --pool-extra 7 --seed 7
```

可选的 18-bit source boundary 测试：

```bash
python3 tests/test_fused_li_manage_mtp_c8.py --device npu:0 --batch-size 1 --heads 32 --source-len 20096 --queries-per-request 4 --miss-min 0 --miss-max 300 --pool-extra 7 --check-18bit-boundary --seed 7
```

## 当前非 MTP C8 性能矩阵

```bash
python3 tests/test_fused_li_manage_c8.py --device npu:0 --heads 32 --batch-sizes 1,4,8,12,16,24,32 --source-lens 12288,20096,65536,131072 --cache-tokens 6144 --miss-ranges 0:0,0:300,300:300 --warmup 10 --iters 100 --seed 7
```

```bash
for bs in 1 4 8 12 16 24 32; do for len in 12288 20096 65536 131072; do python3 tests/test_kvcache_scatter_copy_c8.py --device npu:0 --batch-size "$bs" --source-len "$len" --cache-tokens 6144 --tail-tokens 64 --max-tail-tokens 512 --copy-min 0 --copy-max 300 --warmup 10 --iters 100 --seed 7; done; done
```

```bash
python3 tests/test_sparse_tail_attention_c8.py --device npu:0 --heads 8 --batch-sizes 1,4,8,12,16,24,32 --cache-tokens 6144 --tail-tokens 64 --max-tail-tokens 512 --warmup 10 --iters 100 --seed 7
```
