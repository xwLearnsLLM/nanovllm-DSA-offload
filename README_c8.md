# Ascend 950 C8 Decode Offload 算子

本仓库提供 GLM-5.1/5.2 W4A4C8 decode KV-cache 卸载所需的 4 个 Ascend 950 单算子。Torch namespace 为 `nanovllm_dsa`，index cache 与 KV cache 均采用 C8 格式。

## 算子

| Torch 入口 | CANN 算子 | 功能 |
| --- | --- | --- |
| `fused_li_manage_c8` | `A5FusedLiManageC8` | C8 LightningIndexer top-2048 与 request-pool 索引管理 |
| `fused_li_manage_mtp_c8` | `A5FusedLiManageMtpC8` | MTP1～3 packed query 的 C8 top-k union 与一次性缓存更新 |
| `kvcache_scatter_copy_c8` | `A5KvcacheScatterCopyC8` | packed C8 KV 的 swapped-memory DRAM → HBM 搬运并生成 attention metadata |
| `sparse_tail_attention_c8` | `A5SparseTailAttentionC8` | packed C8 KV 上的 top-2048 sparse + dense tail MLA |
| `sparse_tail_attention_c8_stage1` | `A5SparseTailAttentionC8Stage1` | 非 MTP hit + tail，输出 FP32 P/M/L |
| `sparse_tail_attention_c8_stage2` | `A5SparseTailAttentionC8Stage2` | 非 MTP miss，并在 UB 中合并 Stage1 P/M/L |
| `sparse_tail_attention_c8_mtp_stage1/2` | `A5SparseTailAttentionC8MtpStage1/2` | 既有 packed-MTP 两阶段实现 |

## 接口

```python
torch.ops.nanovllm_dsa.fused_li_manage_c8(
    query,                  # C8[B,32|64,128]
    key,                    # C8[BLOCKS,128,1,128]
    weights,                # bf16[B,32|64]
    query_dequant_scale,    # fp32[B,32|64]
    key_dequant_scale,      # fp32[BLOCKS,128,1]
    actual_seq_lengths_query, # int32[B]
    req_pool_entries,       # int32[B]
    cache_slots_pool,       # int32[POOL_SIZE,SOURCE_CAPACITY], in/out
    cache_tokens,           # int32[B]
    candidate_lens,         # int32[B]
    block_table,            # int32[B,MAX_BLOCKS]
) -> (
    source_ids,             # int32[B,1,2048], miss-prefix + hit-suffix
    destination_slots,      # int32[B,1,2048]
    miss_counts,            # int32[B]
    cache_slots_alias,
)

torch.ops.nanovllm_dsa.fused_li_manage_mtp_c8(
    query,                  # C8[T,32|64,128], T=sum(query_counts)
    key,                    # C8[BLOCKS,128,1,128]
    weights,                # bf16[T,32|64]
    query_dequant_scale,    # fp32[T,32|64]
    key_dequant_scale,      # fp32[BLOCKS,128,1]
    actual_seq_lengths_query, # cumulative int32[B]
    req_pool_entries,       # int32[B]
    cache_slots_pool,       # int32[POOL_SIZE,SOURCE_CAPACITY], in/out
    cache_tokens,           # int32[B]
    candidate_lens,         # int32[B]
    block_table,            # int32[B,MAX_BLOCKS]
) -> (
    topk_destination_slots, # int32[T,1,2048]
    miss_source_ids,        # int32[B,8192], request-level union misses
    miss_destination_slots, # int32[B,8192]
    miss_counts,            # int32[B]
    cache_slots_alias,
)

torch.ops.nanovllm_dsa.kvcache_scatter_copy_c8(
    hbm_kv_bytes,           # one-byte dtype[HBM_BLOCKS,128,1,656], in/out
    dram_kv_bytes,          # swapped-memory one-byte dtype[DRAM_BLOCKS,128,1,656]
    hbm_block_table,
    dram_block_table,
    source_token_ids,       # int32[B,COPY_CAP]
    destination_slots,      # int32[B,COPY_CAP]
    copy_counts,            # int32[B]
    cache_tokens,
    candidate_lens,
    actual_seq_lengths_kv,
    max_tail_tokens,
) -> (
    hbm_kv_alias,
    attention_slots,        # int32[B,1,2048+max_tail_tokens]
    resident_seq_lengths,   # int32[B]
)

torch.ops.nanovllm_dsa.sparse_tail_attention_c8(
    query,                  # bf16[B,Q_HEAD,576]
    packed_kv,              # C8[HBM_BLOCKS,128,1,656]
    sparse_and_tail_slots,  # int32[B,1,2048+MAX_TAIL]
    block_table,
    actual_seq_lengths_query,
    resident_seq_lengths,
    scale_value,
) -> attention_out          # bf16[B,Q_HEAD,512]

# 两阶段非 MTP 路径复用上面完全相同的 7 个 SFA 参数；只额外传入
# LI 已有的 miss_counts，以及跨阶段 P/M/L。所有输出均由 caller 分配。
torch.ops.nanovllm_dsa.sparse_tail_attention_c8_stage1(
    query, packed_kv, sparse_and_tail_slots, block_table,
    actual_seq_lengths_query, resident_seq_lengths,
    miss_counts, scale_value,
    partial_out,             # fp32[B,Q_HEAD,512]
    softmax_max,             # fp32[1,B,Q_HEAD]
    softmax_sum,             # fp32[1,B,Q_HEAD]
) -> None

torch.ops.nanovllm_dsa.sparse_tail_attention_c8_stage2(
    query, packed_kv, sparse_and_tail_slots, block_table,
    actual_seq_lengths_query, resident_seq_lengths,
    miss_counts, scale_value,
    partial_out, softmax_max, softmax_sum,
    attention_out,           # bf16/fp16[B,Q_HEAD,512]
) -> None
```

非 MTP staged 路径不构造新 slot：两阶段都读取 scatter/copy 生成的完整
`sparse_and_tail_slots`。Stage1 读取 `[miss_count, valid_end)`，Stage2 读取
`[0, miss_count)`；Stage2 在 UB 中执行 stable merge 并直接写最终输出。

固定约束：block size 128，index head dim 128，packed KV row 656 bytes，`Q_HEAD<=64`，单请求 MTP query 数为 2～4，MTP union 容量为 8192，source token ID 为 18 bit，source capacity 不超过 `262144`。两种 manager 的 `C=0` 请求严格 no-op；活跃 `req_pool_entries` 必须唯一。

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
bash build_c8.sh
export ASCEND_CUSTOM_OPP_PATH=$PWD/_custom_opp_c8/vendors/customize
export NANOVLLM_A5_INSTALL_OPP_PATH=$PWD/_custom_opp_c8
export NANOVLLM_CUST_OPAPI_LIB=$PWD/_custom_opp_c8/vendors/customize/op_api/lib/libcust_opapi.so
```

## 测试

每个脚本固定执行行为检查和 NPU Event 时延测试；时延统一打印为 `us`，不设置性能门槛。

```bash
python3 tests/test_fused_li_manage_c8.py --device npu:0 --heads 32,64 --batch-sizes 24 --source-lens 20096 --cache-tokens 6144 --miss-ranges 0:300 --warmup 3 --iters 20 --seed 7
python3 tests/test_fused_li_manage_mtp_c8.py --device npu:0 --batch-size 6 --heads 32,64 --source-len 20096 --queries-per-request 0 --miss-min 0 --miss-max 300 --pool-extra 7 --warmup 3 --iters 20 --seed 7
python3 tests/test_kvcache_scatter_copy_c8.py --device npu:0 --batch-size 24 --source-len 20096 --cache-tokens 6144 --tail-tokens 64 --max-tail-tokens 512 --copy-min 0 --copy-max 300 --warmup 3 --iters 20 --seed 7
python3 tests/test_sparse_tail_attention_c8.py --device npu:0 --heads 8 --batch-sizes 24 --cache-tokens 6144 --tail-tokens 64 --max-tail-tokens 512 --warmup 3 --iters 20 --seed 7
python3 tests/test_sparse_tail_attention_c8_staged_nomtp.py --device npu:0 --heads 8 --batch-sizes 1,4,16,64 --graph --warmup 3 --iters 20 --seed 31
python3 tests/test_offload_split_c8_graph.py --device npu:0 --replays 4 --seed 7
```
