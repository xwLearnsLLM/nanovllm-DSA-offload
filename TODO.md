# TODO

## 修复 source-aware COPYSFA-MTP 精度

公共 ABI 不再变化。当前 source-aware 单内核已经满足：

- union miss 的 DRAM→HBM CKV/KPE 搬移结果与 split 路径逐元素一致；
- eager、ACLGraph 和 caller-owned output 接口均可执行；
- 典型性能负载按每请求约 300 个 unique union misses 测试，重点覆盖 B=4/24/32。

尚未满足：COPYSFA-MTP Attention 必须与 `scatter_copy + sparse_tail_attention_mtp` 的 split 路径一致。已记录的 B=4、heads=2、source=20992、C=8192、tail=64 诊断结果为：

- split vs canonical HBM：max_abs=0；
- canonical vs nonzero HBM：max_abs≈0.023926；
- nonzero HBM vs mixed DRAM/HBM：max_abs≈0.060303；
- split vs mixed DRAM/HBM：max_abs≈0.067413。

已验证单纯固定 canonical source tiles、以及在持久 cache 写回前 compact reload actual misses，均不能消除误差。下一步应继续对齐 source-aware gather 与 standalone SFA 的 token 顺序、分块和 softmax merge 顺序；修复过程中必须保持 cache payload 精确、算子 ABI 不变，并保留 split 路径作为逐步 golden。
