# Ascend 950 nano-vLLM offload_split 算子

本仓库是独立的 Ascend 950 / CANN 9.1 decode KV-cache 卸载算子工程。W4A8 路径包含 BF16 `lightning_indexer_decode_update`、`kvcache_scatter_copy` 和 `sparse_and_tail_attention`；W4A4C8 路径使用官方 A5 C8 LightningIndexer、request-pool update、packed-C8 SCATTER 和原生 C8 QSFA。构建与运行不读取任何参考仓源码，也不包含 nano-vLLM 引擎、融合算子或 MTP。

## 来源

- LIDU：基于 `ops_li_update_a5@0362e7e` 的生产 P5 内核，增加 request pool、逐请求 C、动态 pool 行跨度、mutable alias 和 caller-owned out。
- SCATTER：基于 `ops_dsa_offload_a5@01f2065` 已验证的 Ascend 950 swapped-memory DRAM→HBM 路径。
- SFA：基于 `vllm-ascend-v0.23.0-custom@6af99b372` 的官方 Arch35 SFA叠加 sparse+tail 语义。官方接口说明见 [aclnnSparseFlashAttention](https://github.com/vllm-project/vllm-ascend/blob/main/csrc/attention/sparse_flash_attention/docs/aclnnSparseFlashAttention.md)。
- W4A4C8 Indexer：使用 A5 官方 `torch_npu.npu_quant_lightning_indexer`，采用 FP8 E4M3 query/key、BF16 weights、FP32 query/key scale、`TND/PA_BSND`、`sparse_count=2048` 和 `sparse_mode=3`；本仓库的 `A5LiduCacheUpdate` 接续完成 request-pool hit/miss、淘汰和更新。
- W4A4C8 Attention：packed KV ABI 与 A5 原生 `npu_kv_quant_sparse_flash_attention` 对齐。ModelSlim 的 GLM-5.1 W4A4C8 量化说明见 [GLM-5 量化 README](https://gitcode.com/Ascend/msmodelslim/blob/26.1.0/example/GLM-5/README.md)。

## 接口

```python
torch.ops.nanovllm_dsa.lidu_decode_update(
    query,                  # bf16/fp16[B, 32|64, 128]
    key,                    # bf16/fp16[NUM_BLOCKS, 128, 1, 128]
    weights,                # bf16/fp16[B, 32|64]
    req_pool_entries,       # int32[B]
    cache_slots_pool,       # int32[POOL_SIZE, SOURCE_CAPACITY], in/out
    cache_tokens,           # int32[B]
    candidate_lens,         # int32[B]
    block_table,            # int32[B, SOURCE_CAPACITY/128]
) -> (
    source_ids,             # int32[B, 1, 2048], miss-prefix + hit-suffix
    destination_slots,      # int32[B, 1, 2048], 完整 attention slots
    miss_counts,            # int32[B]
    cache_slots_alias,      # cache_slots_pool alias
)

torch.ops.nanovllm_dsa.lidu_decode_update_out(
    query, key, weights, req_pool_entries, cache_slots_pool,
    cache_tokens, candidate_lens, block_table,
    source_ids,             # caller-owned int32[B, 1, 2048], in/out
    destination_slots,      # caller-owned int32[B, 1, 2048], in/out
    miss_counts,            # caller-owned int32[B], in/out
) -> (source_ids_alias, destination_slots_alias, miss_counts_alias, cache_slots_alias)

torch.ops.nanovllm_dsa.lidu_decode_update_c8(
    query,                  # float8_e4m3fn[B,32|64,128]
    key,                    # float8_e4m3fn[NUM_BLOCKS,128,1,128]
    weights,                # bfloat16[B,32|64]
    query_dequant_scale,    # float32[B,32|64]
    key_dequant_scale,      # float32[NUM_BLOCKS,128,1]
    actual_seq_lengths_query, # cumulative int32[B]，decode 为 1..B
    req_pool_entries,       # int32[B]
    cache_slots_pool,       # int32[POOL_SIZE,SOURCE_CAPACITY]，in/out
    cache_tokens,           # int32[B]
    candidate_lens,         # int32[B]
    block_table,            # int32[B,SOURCE_CAPACITY/128]
) -> (
    source_ids,             # int32[B,1,2048]，miss-prefix + hit-suffix
    destination_slots,      # int32[B,1,2048]
    miss_counts,            # int32[B]
    cache_slots_alias,
)

torch.ops.nanovllm_dsa.lidu_decode_update_c8_out(
    query, key, weights, query_dequant_scale, key_dequant_scale,
    actual_seq_lengths_query, req_pool_entries, cache_slots_pool,
    cache_tokens, candidate_lens, block_table,
    source_ids, destination_slots, miss_counts,
) -> (source_ids_alias, destination_slots_alias, miss_counts_alias, cache_slots_alias)

torch.ops.nanovllm_dsa.scatter_copy(
    hbm_kpe,                # bf16/fp16[HBM_BLOCKS, 128, 64], in/out
    hbm_ckv,                # bf16/fp16[HBM_BLOCKS, 128, 512], in/out
    dram_kpe,               # swapped-memory bf16/fp16[DRAM_BLOCKS, 128, 64]
    dram_ckv,               # swapped-memory bf16/fp16[DRAM_BLOCKS, 128, 512]
    hbm_block_table,        # int32[B, HBM_MAX_BLOCKS]
    dram_block_table,       # int32[B, DRAM_MAX_BLOCKS]
    source_token_ids,       # int32[B, COPY_CAP]
    destination_slots,      # int32[B, COPY_CAP]
    copy_counts,            # int32[B]
) -> (hbm_kpe_alias, hbm_ckv_alias)

torch.ops.nanovllm_dsa.sparse_and_tail_attention(
    query,                  # bf16/fp16[B, Q_HEAD, 512], 1 <= Q_HEAD <= 64
    key,                    # bf16/fp16[HBM_BLOCKS, 128, 1, 512]
    value,                  # GLM MLA 要求与 key alias
    sparse_slots,           # int32[B, 1, 2048]
    cache_tokens,           # int32[B]
    block_table,            # int32[B, HBM_MAX_BLOCKS]
    actual_seq_lengths_query, # cumulative int32[B], decode 为 1..B
    actual_seq_lengths_kv,  # int32[B]
    query_rope,             # bf16/fp16[B, Q_HEAD, 64]
    key_rope,               # bf16/fp16[HBM_BLOCKS, 128, 1, 64]
    scale_value,            # float
) -> attention_out          # bf16/fp16[B, Q_HEAD, 512]

torch.ops.nanovllm_dsa.packed_scatter_copy(
    hbm_packed_kv_bytes,    # int8 view[HBM_BLOCKS,128,1,656], in/out
    dram_packed_kv_bytes,   # swapped-memory int8 view[DRAM_BLOCKS,128,1,656]
    hbm_block_table,        # int32[B, HBM_MAX_BLOCKS]
    dram_block_table,       # int32[B, DRAM_MAX_BLOCKS]
    source_token_ids,       # int32[B,1,2048]
    destination_slots,      # int32[B,1,2048]
    copy_counts,            # int32[B]
    cache_tokens,           # int32[B]
    candidate_lens,         # int32[B]
    actual_seq_lengths_kv,  # int32[B]
    max_tail_tokens,        # int，固定图的 tail 容量
) -> (
    hbm_packed_kv_alias,
    attention_slots,        # int32[B,1,2048+max_tail_tokens]
    resident_seq_lengths,   # int32[B]
)

torch.ops.nanovllm_dsa.packed_scatter_copy_out(
    hbm_packed_kv_bytes, dram_packed_kv_bytes,
    hbm_block_table, dram_block_table,
    source_token_ids, destination_slots, copy_counts,
    cache_tokens, candidate_lens, actual_seq_lengths_kv,
    max_tail_tokens,
    attention_slots,        # caller-owned int32[B,1,2048+max_tail_tokens]
    resident_seq_lengths,   # caller-owned int32[B]
) -> (hbm_alias, attention_slots_alias, resident_seq_lengths_alias)

nanovllm_dsa_a5.sparse_and_tail_attention_c8(
    query,                  # bf16/fp16[T,Q_HEAD,576], Q_HEAD<=64
    packed_kv,              # float8_e4m3fn[HBM_BLOCKS,128,1,656]
    attention_slots,        # int32[T,1,2048+max_tail_tokens]
    hbm_block_table,        # int32[B,HBM_MAX_BLOCKS]
    actual_seq_lengths_query, # cumulative int32[B]
    resident_seq_lengths,   # int32[B]
    scale_value,            # float
) -> attention_out          # bf16/fp16[T,Q_HEAD,512]
```

全部接口均提供 Fake/Meta 路径；LIDU 与两种 SCATTER 的 schema 显式声明 mutable alias。C8 LIDU 先调用官方 A5 Quant LightningIndexer 生成真实 C8 top-2048，再调用仓内 AIV update kernel；两段都可被 NPUGraph 捕获。C8 Attention adapter 固定调用 A5 原生 QSFA。

## 约束

- 仅面向 GLM-5.1 W4A8/W4A4C8 decode，`q_seq_len=1`，block size 为 128。
- LIDU 支持 `q_head=32|64`；SFA 仅支持 `q_head=1..64`，门禁使用 8。
- `C=0` 时 LIDU no-op，SFA 计算全部有效 KV；`C>0` 时 SFA 计算 2048 个 sparse slots 与 `[C, actual_kv_len)` tail。
- source token ID 为 18 bit，`SOURCE_CAPACITY <= 262144`；LIDU slot 为 14 bit，block-aligned `C <= 16256`。当前预算 `3072/6144/8192/12288` 均受支持。
- `req_pool_entries` 在活跃 batch 内必须唯一且位于 pool 范围内；非零 C 的 pool 行必须恰有 C 个唯一 slots。
- C8 Indexer 使用独立的 `float8_e4m3fn[blocks,128,1,128]` key cache 和 `float32[blocks,128,1]` scale cache，不再复用 BF16 Indexer。C8 Attention cache 每个 token 的 656 bytes 固定为 `512 FP8 E4M3 + 64 BF16 RoPE + 4 FP32 scales`，SCATTER 必须整行搬运。
- C8 LIDU 的 query/key 接口位于官方 GLM 预处理之后：调用方须先完成 RoPE、归一化 128×128 Hadamard，再执行 FP8 E4M3 动态量化并传入对应 FP32 scale；C8 UT 也按该顺序构造输入。
- `max_tail_tokens` 是 full-decode-only 的静态 capture 容量；实际 tail 为 `actual_seq_lengths_kv-candidate_lens`。`C>0` 时 Attention 索引为 2048 个 LIDU slots 加 `[C,C+tail)`，`C=0` 时为 `[0,actual_len)`。

## 环境与构建

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
```

```bash
bash build.sh
```

## 正确性与图回放

```bash
python3 tests/test_api_meta.py
```

```bash
python3 tests/test_lidu.py --device npu:0 --mode check --heads 32,64 --batch-sizes 24 --source-lens 20096 --cache-tokens 6144 --miss-ranges 0:300 --seed 7
```

该 LIDU 命令还会强制覆盖不同 candidate length、`C=0/2048/3072/6144/8192/12288/16256`、零 miss、2048 miss、乱序 pool entries、重复更新、inactive pool guard 和 caller-owned out。

```bash
for count in 0 1 100 300 2048; do python3 tests/test_scatter_copy.py --device npu:0 --batch-size 24 --source-len 65536 --hbm-slots 4096 --copy-cap 2048 --copy-min "$count" --copy-max "$count" --warmup 3 --iters 10 --seed 7; done
```

SCATTER 使用 `empty_with_swapped_memory` 创建真实 DRAM tensor；每次正确性调用前 poison HBM 目标，并验证 CKV、KPE、随机 block tables 和未触碰 guard。

```bash
python3 tests/test_sparse_and_tail_attention.py --device npu:0 --mode check --heads 8 --batch-sizes 24 --source-lens 20096 --cache-tokens 6144 --tail-tokens 64 --seed 7
```

SFA check 会先验证 2048-token smoke、dense `C=0`、四档 C 和 tail `0/1/64/127/257`，并与独立 CPU FP32 golden 比较。

```bash
python3 tests/test_offload_split_graph.py --device npu:0 --case pure-long --replays 4 --seed 7
```

```bash
python3 tests/test_offload_split_graph.py --device npu:0 --case mixed --replays 4 --seed 7
```

图门禁捕获 `lidu_decode_update_out → scatter_copy → sparse_and_tail_attention`；capture 为零 miss，replay 交替产生非零 miss，并校验输出地址、pool 更新、真实 DRAM→HBM 搬运和 attention golden。

## W4A4C8 门禁

C8 LIDU 直接用官方 A5 C8 LightningIndexer 作为 top-2048 基线，并覆盖 mixed C、乱序 request-pool、零/随机/2048 miss、重复更新和 caller-owned out：

```bash
python3 tests/test_lidu_c8.py --device npu:0 --mode check --heads 32,64 --batch-sizes 24 --source-lens 20096 --cache-tokens 6144 --miss-ranges 0:300 --seed 7
```

packed SCATTER 使用真实 swapped memory；分配式与 caller-owned 两条路径都重新 poison HBM，并校验完整 656 bytes、guard、动态 tail metadata、`resident_seq_lengths` 和输出地址：

```bash
python3 tests/test_packed_scatter_copy_c8.py --device npu:0 --batch-size 24 --source-len 20096 --cache-tokens 6144 --tail-tokens 257 --max-tail-tokens 512 --copy-min 0 --copy-max 300 --warmup 10 --iters 100 --seed 7
```

C8 QSFA 使用独立 CPU FP32 golden 校验 FP8 latent、BF16 RoPE 和 FP32 scales 的 packed layout。首轮门禁固定 `q_head=8`；本仓库不支持 `q_head>64`：

```bash
for C in 3072 6144 8192 12288; do for tail in 0 1 64 127 257; do python3 tests/test_sparse_and_tail_attention_c8.py --device npu:0 --mode check --batch-size 24 --heads 8 --cache-tokens "$C" --tail-tokens "$tail" --max-tail-tokens 512 --seed 7; done; done
```

```bash
python3 tests/test_sparse_and_tail_attention_c8.py --device npu:0 --mode check --batch-size 1 --heads 8 --cache-tokens 0 --tail-tokens 2048 --max-tail-tokens 2048 --seed 7
```

图门禁捕获 `official C8 LightningIndexer → lidu_cache_update_out → packed_scatter_copy_out → native C8 QSFA`；不依赖 capture 阶段执行，先用一次 replay 验证零 miss 和完整输出写回，再恢复初始 pool 进行多次非零 miss replay：

```bash
python3 tests/test_offload_split_c8_graph.py --device npu:0 --case pure-long --batch-size 2 --heads 8 --index-heads 32 --source-len 4096 --cache-tokens 3072 --tail-tokens 64 --max-tail-tokens 256 --miss-min 256 --miss-max 512 --replays 4 --seed 7
```

```bash
python3 tests/test_offload_split_c8_graph.py --device npu:0 --case mixed --batch-size 2 --heads 8 --index-heads 32 --source-len 4096 --cache-tokens 3072 --tail-tokens 64 --max-tail-tokens 256 --miss-min 256 --miss-max 512 --replays 4 --seed 7
```

## 性能矩阵

`q_head=8` 的三算子总时延需与相同输入下的来源版本对照，目标是不超过参考时延的 `1.10x`。

LIDU：

```bash
python3 tests/test_lidu.py --device npu:0 --mode bench --heads 32 --batch-sizes 1,4,8,12,16,24,32 --source-lens 12288,20096,65536,131072 --cache-tokens 6144 --miss-ranges 0:0,0:300,300:300 --warmup 10 --iters 100 --seed 7
```

C8 LIDU：

```bash
python3 tests/test_lidu_c8.py --device npu:0 --mode bench --heads 32 --batch-sizes 1,4,8,12,16,24,32 --source-lens 12288,20096,65536,131072 --cache-tokens 6144 --miss-ranges 0:0,0:300,300:300 --warmup 10 --iters 100 --seed 7
```

SCATTER：

```bash
for bs in 1 4 8 12 16 24 32; do for len in 12288 20096 65536 131072; do python3 tests/test_scatter_copy.py --device npu:0 --batch-size "$bs" --source-len "$len" --hbm-slots 8192 --copy-cap 2048 --copy-min 0 --copy-max 0 --warmup 10 --iters 100 --seed 7; python3 tests/test_scatter_copy.py --device npu:0 --batch-size "$bs" --source-len "$len" --hbm-slots 8192 --copy-cap 2048 --copy-min 0 --copy-max 300 --warmup 10 --iters 100 --seed 7; python3 tests/test_scatter_copy.py --device npu:0 --batch-size "$bs" --source-len "$len" --hbm-slots 8192 --copy-cap 2048 --copy-min 300 --copy-max 300 --warmup 10 --iters 100 --seed 7; done; done
```

SFA：

```bash
python3 tests/test_sparse_and_tail_attention.py --device npu:0 --mode bench --heads 8 --batch-sizes 1,4,8,12,16,24,32 --source-lens 12288,20096,65536,131072 --cache-tokens 6144 --tail-tokens 64 --warmup 10 --iters 100 --seed 7
```

## Profile

```bash
msprof --application="python3 tests/test_lidu.py --device npu:0 --mode profile --heads 32 --batch-sizes 24 --source-lens 20096 --cache-tokens 6144 --miss-ranges 0:300 --profile-replays 4 --seed 7" --output=./profile_lidu
```

```bash
msprof --application="python3 tests/test_lidu_c8.py --device npu:0 --mode profile --heads 32 --batch-sizes 24 --source-lens 20096 --cache-tokens 6144 --miss-ranges 0:300 --profile-replays 4 --seed 7" --output=./profile_lidu_c8
```

```bash
msprof --application="python3 tests/test_scatter_copy.py --device npu:0 --batch-size 24 --source-len 65536 --hbm-slots 8192 --copy-cap 2048 --copy-min 0 --copy-max 300 --warmup 3 --iters 4 --seed 7" --output=./profile_scatter
```

```bash
msprof --application="python3 tests/test_sparse_and_tail_attention.py --device npu:0 --mode profile --heads 8 --batch-sizes 24 --source-lens 65536 --cache-tokens 6144 --tail-tokens 64 --profile-replays 4 --seed 7" --output=./profile_sfa_h8
```
