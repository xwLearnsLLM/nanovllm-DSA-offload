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

## GatherSelection all-core copy tiling

`test_parallel_copy.py` runs the legacy row-core tiling and the new all-core
copy tiling in separate processes.  Both implementations must pass five
stateful top-k transitions with exact BF16 cache equality, including repeated
long rows with zero misses, plus a short-row no-op check that catches stale
workspace pairs.  The timed section
alternates two top-k sets so every invocation has the configured miss rate.
The full KV source uses `empty_with_swapped_memory`, matching the offload path.

```bash
ASCEND_RT_VISIBLE_DEVICES=0 PYTHONPATH=$PWD:$PYTHONPATH \
python3 ut_ops/gather_selection/test_parallel_copy.py \
  --device npu:0 --batch-size 6 --full-len 32768 \
  --overlap 0.6 --warmup 10 --iters 100
```

The command exits non-zero if either implementation is incorrect or if the
new tiling is not faster.  `--overlap 0.6` means 40% of the 2048 tokens are
copied on every timed transition.  Use `--min-speedup 1.05` to require at least
a 5% speedup.
