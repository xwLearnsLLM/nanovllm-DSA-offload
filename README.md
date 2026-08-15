# Ascend 950 BF16 Decode Offload 算子

本仓库提供 GLM decode KV-cache 卸载所需的 5 个 Ascend 950 单算子。Torch namespace 为 `nanovllm_dsa`，block size 固定为 128。

## 算子

| Torch 入口 | CANN 算子 | 功能 |
| --- | --- | --- |
| `fused_li_manage` | `A5FusedLiManage` | LightningIndexer top-2048 与 request-pool 索引管理 |
| `fused_li_manage_mtp` | `A5FusedLiManageMtp` | packed MTP query 的 top-k union 与一次性缓存更新 |
| `kvcache_scatter_copy` | `A5KvcacheScatterCopy` | swapped-memory DRAM → HBM CKV/KPE 搬运 |
| `sparse_tail_attention` | `A5SparseTailAttention` | top-2048 sparse KV + dense tail MLA |
| `fused_copy_sparse_tail_attention` | `A5FusedCopySparseTailAttention` | 融合 DRAM → HBM 搬运与 sparse+tail MLA |

## 接口

```python
torch.ops.nanovllm_dsa.fused_li_manage(
    query,                # bf16/fp16[B,32|64,128]
    key,                  # bf16/fp16[BLOCKS,128,1,128]
    weights,              # bf16/fp16[B,32|64]
    req_pool_entries,     # int32[B]
    cache_slots_pool,     # int32[POOL_SIZE,SOURCE_CAPACITY], in/out
    cache_tokens,         # int32[B]
    candidate_lens,       # int32[B]
    block_table,          # int32[B,MAX_BLOCKS]
) -> (
    source_ids,           # int32[B,1,2048], miss-prefix + hit-suffix
    destination_slots,    # int32[B,1,2048]
    miss_counts,          # int32[B]
    cache_slots_alias,
)

torch.ops.nanovllm_dsa.fused_li_manage_mtp(
    query,                # bf16/fp16[T,32|64,128]
    key,                  # bf16/fp16[BLOCKS,128,1,128]
    weights,              # bf16/fp16[T,32|64]
    cache_slots,          # int32[B,262144], in/out; C=8192
    actual_seq_lengths_query, # cumulative int32[B]
    actual_seq_lengths_key,   # int32[B]
    block_table,          # int32[B,MAX_BLOCKS]
) -> (
    topk_index,           # int32[T,1,2048]
    topk_slots,           # int32[T,1,2048]
    miss_index,           # int32[B,8192], request-level union misses
    miss_slots,           # int32[B,8192]
    miss_count,           # int32[B]
)

torch.ops.nanovllm_dsa.kvcache_scatter_copy(
    hbm_kpe, hbm_ckv,     # bf16/fp16, in/out
    dram_kpe, dram_ckv,   # swapped-memory bf16/fp16
    hbm_block_table,
    dram_block_table,
    source_token_ids,     # int32[B,COPY_CAP]
    destination_slots,    # int32[B,COPY_CAP]
    copy_counts,          # int32[B]
) -> (hbm_kpe_alias, hbm_ckv_alias)

torch.ops.nanovllm_dsa.sparse_tail_attention(
    query,                # bf16/fp16[B,Q_HEAD,512]
    key, value,           # value 必须 alias key
    sparse_slots,         # int32[B,1,2048]
    cache_tokens,
    block_table,
    actual_seq_lengths_query,
    actual_seq_lengths_kv,
    query_rope, key_rope,
    scale_value,
) -> attention_out        # bf16/fp16[B,Q_HEAD,512]

torch.ops.nanovllm_dsa.fused_copy_sparse_tail_attention(
    query,
    hbm_ckv,              # bf16, in/out
    sparse_slots,
    cache_tokens,
    hbm_block_table,
    actual_seq_lengths_query,
    actual_seq_lengths_kv,
    query_rope,
    hbm_kpe,              # bf16, in/out
    dram_kpe, dram_ckv,   # swapped-memory bf16
    dram_block_table,
    source_token_ids,
    copy_counts,
    scale_value,
    prefetch_rows_per_step=5,
) -> (attention_out, hbm_kpe_alias, hbm_ckv_alias)
```

固定约束：index head dim 128，CKV/KPE dim 512/64，`Q_HEAD<=64`，source token ID 为 18 bit，source capacity 不超过 `262144`。`fused_li_manage` 的 `C=0` 请求严格 no-op；活跃 `req_pool_entries` 必须唯一。

## 编译

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
bash build_bf16.sh
export ASCEND_CUSTOM_OPP_PATH=$PWD/_custom_opp_bf16/vendors/customize
export NANOVLLM_A5_INSTALL_OPP_PATH=$PWD/_custom_opp_bf16
export NANOVLLM_CUST_OPAPI_LIB=$PWD/_custom_opp_bf16/vendors/customize/op_api/lib/libcust_opapi.so
```

## 测试

每个脚本固定执行行为检查和 NPU Event 时延测试；时延统一打印为 `us`，不设置性能门槛。

```bash
python3 tests/test_fused_li_manage.py --device npu:0 --heads 32,64 --batch-sizes 24 --source-lens 20096 --cache-tokens 6144 --miss-ranges 0:300 --warmup 3 --iters 20 --seed 7
python3 tests/test_fused_li_manage_mtp.py --device npu:0 --bs 24 --min-seqlen 32768 --max-seqlen 65536 --q-heads 64 --queries-per-request 0 --min-miss-count 0 --max-miss-count 300 --warmup 3 --iters 20 --seed 7
python3 tests/test_kvcache_scatter_copy.py --device npu:0 --batch-size 24 --source-len 65536 --hbm-slots 4096 --copy-cap 2048 --copy-min 0 --copy-max 300 --warmup 3 --iters 20 --seed 7
python3 tests/test_sparse_tail_attention.py --device npu:0 --heads 8 --batch-sizes 24 --source-lens 20096 --cache-tokens 6144 --tail-tokens 64 --warmup 3 --iters 20 --seed 7
python3 tests/test_fused_copy_sparse_tail_attention.py --device npu:0 --batch-size 24 --heads 8 --source-len 65536 --cache-tokens 8192 --tail-tokens 64 --miss-min 0 --miss-max 300 --warmup 10 --iters 100 --seed 7
```
