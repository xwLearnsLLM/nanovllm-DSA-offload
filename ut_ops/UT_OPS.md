# ut_ops 单测目录

这个目录只保留当前 DSA offload 主线仍有价值的单测和性能探针。历史 SFA/prefill/老 indexer 实验脚本已经删除，避免命令入口过多。

## 目录结构

```text
ut_ops/
  common/            # 单测公共工具：device/sync/bench/format
  gather_selection/  # gather-selection KV 选择/搬运单测
  indexer_project/   # indexer_project 与 query-only TorchAir 单测
  mla/               # MLA/MLAPO 正确性探针
  moe/               # MoE gating/grouped MoE 单测
```

## 当前推荐入口

```bash
python3 ut_ops/gather_selection/probe_pool.py

python3 ut_ops/indexer_project/probe_full.py
python3 ut_ops/indexer_project/probe_post.py
python3 ut_ops/indexer_project/probe_query_only_torchair.py
python3 ut_ops/indexer_project/probe_query_only_torchair_accuracy.py

python3 ut_ops/mla/probe_preprocess.py
python3 ut_ops/mla/probe_kv_permutation.py

python3 ut_ops/moe/compare_gating_topk.py
python3 ut_ops/moe/probe_grouped.py
```
