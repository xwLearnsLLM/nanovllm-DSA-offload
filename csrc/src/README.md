# 算子目录

- `fused_li_manage`：W4A8 LIDU，本地 CANN op。
- `kvcache_scatter_copy`：W4A8 DRAM→HBM SCATTER，本地 CANN op。
- `sparse_tail_attention`：W4A8 sparse+tail MLA，本地 CANN op。
- `fused_copy_sparse_tail_attention`：W4A8 BF16 MTE-pipeline 融合 CANN op，由框架公开入口 `fused_copy_sparse_tail_attention` 调用。
- `fused_li_manage_c8`：W4A4C8 非 MTP LIDU；单个本地 MIX kernel 融合 Quant LightningIndexer 与 request-pool update。
- `fused_li_manage_mtp_c8`：W4A4C8 MTP1～3 LIDU；单个本地 MIX kernel 融合多 query Quant LightningIndexer、request union 与 request-pool update。
- `kvcache_scatter_copy_c8`：W4A4C8 packed-KV SCATTER，本地 CANN op。
- `sparse_tail_attention_c8`：W4A4C8 sparse top-2048 + dense tail MLA，本地 CANN MIX op。
- `common`：多个本地 CANN op 共享的头文件，不是框架算子。
