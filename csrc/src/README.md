# 算子目录

- `fused_li_manage`：W4A8 LIDU，本地 CANN op。
- `kvcache_scatter_copy`：W4A8 DRAM→HBM SCATTER，本地 CANN op。
- `sparse_tail_attention`：W4A8 sparse+tail MLA，本地 CANN op。
- `fused_copy_sparse_tail_attention`：W4A8 BF16 MTE-pipeline 融合 CANN op，由框架公开入口 `fused_copy_sparse_tail_attention` 调用。
- `fused_li_manage_c8`：W4A4C8 LIDU；官方 C8 LightningIndexer 由 Torch 适配层调用，本目录保存本地 request-pool update CANN op。
- `kvcache_scatter_copy_c8`：W4A4C8 packed-KV SCATTER，本地 CANN op。
- `sparse_tail_attention_c8`：W4A4C8 sparse+tail MLA，直接复用 A5 原生 QSFA，没有本地 CANN kernel。
- `common`：多个本地 CANN op 共享的头文件，不是框架算子。
