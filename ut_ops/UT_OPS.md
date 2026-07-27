# Ascend 算子验收

这里只保留与正式推理路径直接对应的 NPU 测试：

- LIDU + SCATTER 的全部缓存档位、request pool、精确搬运、重复更新和链路时延。
- GLM LIDU 融合 SCATTER + sparse-and-tail Attention 的真实 DRAM 搬移与 Attention 结果。
- GatherSelectionKVCache 的状态迁移、精确搬运、短行跳过、零 miss 和性能。
- GLM 32-head Indexer 投影、interleaved RoPE、LightningIndexer 到 GatherSelection 的组合语义。
- GLM ModelSlim W4A8 routed expert。

修改 C++/AscendC 算子后，先重新编译，再运行对应 UT，最后运行 nano-vLLM 推理。

## LIDU + SCATTER

```bash
cd /home/w00916487/nanovllm-dsa_offload

unset NANOVLLM_GS_MISS_RATE_ON_LAYERS
unset NANOVLLM_PROFILE_DECODE_OUTPUT

export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export ASCEND_LAUNCH_BLOCKING=0
export ASCEND_RT_VISIBLE_DEVICES=4
export PYTHONUNBUFFERED=1
export PYTHONPATH=$PWD:$PYTHONPATH

python3 ut_ops/test_lidu_scatter.py \
  --device npu:0 \
  --heads 32,64 \
  --seed 7 \
  --warmup 2 \
  --iters 10
```

成功标志是 `LIDU_SCATTER_UT_OK`。必须先通过该测试，才运行
`NANOVLLM_OFFLOAD_MODE=lidu` 的完整推理。

## GLM 融合 SCATTER + sparse-and-tail Attention

```bash
unset NANOVLLM_GS_MISS_RATE_ON_LAYERS
unset NANOVLLM_PROFILE_DECODE_OUTPUT
unset NANOVLLM_CUST_OPAPI_LIB
unset ASCEND_CUSTOM_OPP_PATH

export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export ASCEND_LAUNCH_BLOCKING=0
export ASCEND_RT_VISIBLE_DEVICES=4
export PYTHONUNBUFFERED=1
export PYTHONPATH=$PWD:$PYTHONPATH
export ASCEND_HOME_PATH=/usr/local/Ascend/cann-8.5.1

python3 ut_ops/test_fused_attention_scatter.py --device npu:0 --mode check --batch-size 24 --heads 8 --source-len 65536 --cache-tokens 8192 --tail-tokens 64 --miss-min 0 --miss-max 2048 --seed 7
python3 ut_ops/test_fused_attention_scatter.py --device npu:0 --mode all --batch-size 24 --heads 8 --source-len 65536 --cache-tokens 8192 --tail-tokens 64 --miss-min 0 --miss-max 300 --warmup 10 --iters 100 --seed 7
```

该测试使用 `empty_with_swapped_memory` 创建真实 DRAM source，并分别对旧路径和融合路径验证目标 slot 的 poison 覆盖、非目标 guard 不变、DRAM→HBM 精确搬移、CPU Attention golden 和两条路径结果一致。最终成功标志是 `FUSED_SCATTER_ATTENTION_UT_OK`。

## GatherSelectionKVCache

```bash
cd /home/w00916487/nanovllm-dsa_offload

export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export ASCEND_LAUNCH_BLOCKING=0
export ASCEND_RT_VISIBLE_DEVICES=4
export PYTHONUNBUFFERED=1
export PYTHONPATH=$PWD:$PYTHONPATH

python3 ut_ops/gather_selection/test_parallel_copy.py \
  --device npu:0 \
  --batch-size 6 \
  --full-len 32768 \
  --overlap 0.6 \
  --warmup 10 \
  --iters 100
```

成功标志是 `GSKV_PARALLEL_UT_OK`。`overlap=0.6` 表示相邻 top-k 集合有 60% 重叠，即每行每次需要搬运约 819 个 token。输出的 `avg_ms` 是 all-core tiling 的算子平均时延。

## GLM DSA Indexer 与 GatherSelection

```bash
cd /home/w00916487/nanovllm-dsa_offload

export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export ASCEND_LAUNCH_BLOCKING=0
export ASCEND_RT_VISIBLE_DEVICES=4
export PYTHONUNBUFFERED=1
export PYTHONPATH=$PWD:$PYTHONPATH

python3 ut_ops/test_glm_dsa_indexer.py \
  --device npu:0 \
  --batch-size 2 \
  --full-len 4096 \
  --topk 2048 \
  --block-size 128 \
  --seed 7
```

成功标志是 `GLM_DSA_INDEXER_UT_OK`。

## GLM ModelSlim W4A8 MoE

```bash
cd /home/w00916487/nanovllm-dsa_offload

export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export ASCEND_LAUNCH_BLOCKING=0
export ASCEND_RT_VISIBLE_DEVICES=4
export PYTHONUNBUFFERED=1
export PYTHONPATH=$PWD:$PYTHONPATH
export NANOVLLM_MODEL=/mnt/models/GLM-5.1-w4a8/

python3 ut_ops/test_glm_w4a8_moe.py \
  --model "$NANOVLLM_MODEL" \
  --device npu:0 \
  --layer 3 \
  --expert 0 \
  --tokens 2 \
  --warmup 2 \
  --iters 10
```

成功标志是 `GLM_W4A8_MOE_UT_OK`。
