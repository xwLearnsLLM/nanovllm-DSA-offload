# Ascend 950 nano-vLLM offload_split 算子

本仓库是独立的 Ascend 950 / CANN 9.1 算子工程，只包含 `lightning_indexer_decode_update`、`kvcache_scatter_copy` 和 `sparse_and_tail_attention`。构建与运行不读取任何参考仓源码，也不包含 nano-vLLM 引擎、融合算子或 MTP。

## 来源

- LIDU：基于 `ops_li_update_a5@0362e7e` 的生产 P5 内核，增加 request pool、逐请求 C、动态 pool 行跨度、mutable alias 和 caller-owned out。
- SCATTER：基于 `ops_dsa_offload_a5@01f2065` 已验证的 Ascend 950 swapped-memory DRAM→HBM 路径。
- SFA：基于 `vllm-ascend-v0.23.0-custom@6af99b372` 的官方 Arch35 SFA叠加 sparse+tail 语义。官方接口说明见 [aclnnSparseFlashAttention](https://github.com/vllm-project/vllm-ascend/blob/main/csrc/attention/sparse_flash_attention/docs/aclnnSparseFlashAttention.md)。

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
```

全部接口注册 `PrivateUse1` 和 Meta；Meta 实现也供 FakeTensor/Dynamo 使用。LIDU 与 SCATTER 的 schema 显式声明 mutable alias。

## 约束

- 仅面向 GLM-5.1 W4A8 decode，`q_seq_len=1`，block size 为 128，CKV/KPE 为 512/64。
- LIDU 支持 `q_head=32|64`；SFA 仅支持 `q_head=1..64`，门禁使用 8。
- `C=0` 时 LIDU no-op，SFA 计算全部有效 KV；`C>0` 时 SFA 计算 2048 个 sparse slots 与 `[C, actual_kv_len)` tail。
- source token ID 为 18 bit，`SOURCE_CAPACITY <= 262144`；LIDU slot 为 14 bit，block-aligned `C <= 16256`。当前预算 `3072/6144/8192/12288` 均受支持。
- `req_pool_entries` 在活跃 batch 内必须唯一且位于 pool 范围内；非零 C 的 pool 行必须恰有 C 个唯一 slots。

## 环境与构建

从仓库根目录执行：

```bash
unset ASCEND_CUSTOM_OPP_PATH
unset NANOVLLM_A5_INSTALL_OPP_PATH
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

## 性能矩阵

`q_head=8` 的三算子总时延需与相同输入下的来源版本对照，目标是不超过参考时延的 `1.10x`。

LIDU：

```bash
python3 tests/test_lidu.py --device npu:0 --mode bench --heads 32 --batch-sizes 1,4,8,12,16,24,32 --source-lens 12288,20096,65536,131072 --cache-tokens 6144 --miss-ranges 0:0,0:300,300:300 --warmup 10 --iters 100 --seed 7
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
msprof --application="python3 tests/test_scatter_copy.py --device npu:0 --batch-size 24 --source-len 65536 --hbm-slots 8192 --copy-cap 2048 --copy-min 0 --copy-max 300 --warmup 3 --iters 4 --seed 7" --output=./profile_scatter
```

```bash
msprof --application="python3 tests/test_sparse_and_tail_attention.py --device npu:0 --mode profile --heads 8 --batch-sizes 24 --source-lens 65536 --cache-tokens 6144 --tail-tokens 64 --profile-replays 4 --seed 7" --output=./profile_sfa_h8
```
