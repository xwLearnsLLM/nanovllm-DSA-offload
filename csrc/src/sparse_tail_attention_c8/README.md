# sparse_tail_attention_c8

仓内 `A5SparseTailAttentionC8` CANN MIX 算子。设备实现以 `vllm-ascend-v0.23.0-custom-a5@f04f86cb9951abd18e74b50ca81d1daa9aebfa15` 的 A5 `KvQuantSparseFlashAttention` 为基线内置并重命名，公开 Torch 接口保持不变；运行时不依赖官方 QSFA 注册。

语义保持为 `sparse top-2048 + dense tail`：`sparse_and_tail_slots` 前 2048 项是 LIDU 选择的 sparse slots，后续有效项是连续 tail slots，`-1` 终止本行；`C=0` 时 slots 中保存全部有效 resident KV。
