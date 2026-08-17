# sparse_tail_attention_mtp_c8

MTP（多 token 验证）场景下的 A5 打包 C8 稀疏 + 尾部注意力算子。每个请求一步打包
1..4 个验证 query（root + drafts），每个验证位置使用**各自的 top-2048** 稀疏 KV，
加上 **causal dense tail**（第 i 路只看本步新增 KV 的 `C..C+i`），一次 launch 完成
全部验证位置的注意力计算。

底层 kernel 与 `sparse_tail_attention_c8` 完全同一实现（`a5_qsfa` QSFA kernel 族），
MTP 语义全部由调用侧张量编码承载：TND query 布局、按行打包的 slots 行、累计式
`actual_seq_lengths_query`。独立算子名（aclnn `A5SparseTailAttentionMtpC8`）仅用于
aclnn 派发 / tiling 缓存 / torch schema 的独立版本化。

## 调用签名

```python
torch.ops.nanovllm_dsa.sparse_tail_attention_mtp_c8(
    query,                    # bf16/fp16 [T, Q_HEAD, 576]，TND，nope(512)+rope(64) 已拼接
    packed_kv,                # fp8_e4m3fn/uint8 [blocks, 128, 1, 656] 打包 KV
    sparse_and_tail_slots,    # int32 [T, 1, 2048 + max_tail]
    block_table,              # int32 [B, max_blocks]
    actual_seq_lengths_query, # int32 [B] 累计值
    resident_seq_lengths,     # int32 [B]
    scale_value,              # float
) -> attention_out            # bf16/fp16 [T, Q_HEAD, 512]
```

Python 侧等价别名：`nanovllm_dsa_a5.sparse_tail_attention_mtp_c8`。

## 参数说明

| 参数 | 形状/dtype | 说明 |
|---|---|---|
| `query` | `[T, Q_HEAD, 576]` bf16/fp16 | `T = sum(query_counts)`；请求 b 的第 i 路位于全局行 `r = prefix_b + i`；`1 <= Q_HEAD <= 64` |
| `packed_kv` | `[blocks, 128, 1, 656]` fp8/uint8 | 656B 行 = nope 512B FP8 + rope 64B BF16 + 4×FP32 tile scale(16B) + pad |
| `sparse_and_tail_slots` | `[T, 1, 2048+max_tail]` int32 | 每路一行：`[0:2048)` = 该路自己的 top-2048 HBM dst slots（取 `fused_li_manage_mtp_c8` 的 `topk_destination_slots[r]`）；`[2048:2048+i+1)` = causal tail `C, C+1, ..., C+i`；其余 `-1` padding（终止符，kernel 读到即停） |
| `block_table` | `[B, max_blocks]` int32 | HBM block table（按请求组织，不按 query 行） |
| `actual_seq_lengths_query` | `[B]` int32 | TND 累计值，差分 = `query_counts[b] ∈ [1,4]`；如 `q=[3,4]` 时 `[3,7]`；末值必须等于 `T` |
| `resident_seq_lengths` | `[B]` int32 | 每请求最后一路的最终 KV 长度 = `C + query_counts[b]` |
| `scale_value` | float | 注意力 scale，通常 `576^-0.5` |
| **返回** | `[T, Q_HEAD, 512]` | 每路验证位置各自的注意力输出（与 query 同 dtype） |

## 约束条件

- `actual_seq_lengths_query` 严格递增，差分 ∈ [1,4]，末值 == T
- `C = resident_seq_lengths[b] - query_counts[b]` 满足 `C == 0 || (C >= 2048 && C % 128 == 0)`
- `sparse_and_tail_slots.size(2) >= 2048 + max(query_counts)`
- `packed_kv`、metadata 全部 int32/连续/同 NPU device
- 与 `sparse_tail_attention_c8` 相同：`Q_HEAD ∈ [1,64]`、`tile_size=128`、`rope_head_dim=64`、`attention_mode=2`(MLA-absorb)、`quant_scale_repo_mode=1`

## 使用示例

```python
import torch
import nanovllm_dsa_a5  # 自动挂载本地 libcust_opapi.so

# 2 个请求，query_counts=[3, 4] -> T=7, C=6144
T, B, C, heads = 7, 2, 6144, 32
query = torch.randn(T, heads, 576, dtype=torch.bfloat16, device="npu:0")
packed_kv = torch.zeros(128, 128, 1, 656, dtype=torch.float8_e4m3fn, device="npu:0")
slots = torch.full((T, 1, 2048 + 4), -1, dtype=torch.int32, device="npu:0")
# ... 每行填 [topk_destination_slots[r] || C..C+i || -1]
block_table = torch.zeros(B, 64, dtype=torch.int32, device="npu:0")
actual_q = torch.tensor([3, 7], dtype=torch.int32, device="npu:0")
resident = torch.tensor([6147, 6148], dtype=torch.int32, device="npu:0")

out = nanovllm_dsa_a5.sparse_tail_attention_mtp_c8(
    query, packed_kv, slots, block_table, actual_q, resident, 576**-0.5
)  # [7, 32, 512]
```

## 与同族算子的差异

| | `sparse_tail_attention_c8` | `sparse_tail_attention_mtp_c8` |
|---|---|---|
| 每请求 query 行数 | 1 | `query_counts[b] ∈ [1,4]`（可变） |
| top-2048 归属 | 每请求一份 | 每验证位置一份（slots 按 T 行打包） |
| tail 可见范围 | 仅自身（slot C） | 第 i 路看 `C..C+i`（query 位置间 causal） |
| `actual_seq_lengths_query` 差分 | 恒 1 | 任意 ∈ [1,4] |
| kernel 实现 | `a5_qsfa`（同一 kernel，TND s1>1 分支此前未被激活） | 同左 |

## 测试

```bash
python tests/test_sparse_tail_attention_mtp_c8.py --device npu:0
```

覆盖：C=0 dense 4 路、C=2048 稀疏下界、C=6144/12288 生产预算、混合路数批次
`[2,3]`/`[1,4,2]`、每路独立 topk、Meta 形状检查、kernel 元数据注册检查。
