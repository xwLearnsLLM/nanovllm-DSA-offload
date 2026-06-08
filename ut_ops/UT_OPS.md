# ut_ops 单测目录

这里保留当前 DSA offload 主线仍有价值的单测和性能探针。

```text
ut_ops/
  common/            # 公共工具：device、sync、benchmark、diff 打印
  gather_selection/  # gather-selection KV 选择和搬运
  indexer_project/   # indexer_project 和 DSA 小流水 TorchAir
  mla/               # MLA/MLAPO 正确性探针
  moe/               # MoE gating/grouped MoE 探针
```

推荐入口：

```bash
python3 ut_ops/gather_selection/probe_pool.py

python3 ut_ops/indexer_project/probe_full.py
python3 ut_ops/indexer_project/probe_post.py
python3 ut_ops/indexer_project/probe_dsa_pipeline_torchair.py

python3 ut_ops/mla/probe_preprocess.py
python3 ut_ops/mla/probe_kv_permutation.py

python3 ut_ops/moe/compare_gating_topk.py
python3 ut_ops/moe/probe_grouped.py
```
