# nano-vllm-ascend DeepSeek V3.2 DSA 卸载说明

当前仓库是扁平目录结构。所有命令默认在仓库根目录运行，也就是能看到
`nanovllm/`、`example/`、`scripts/`、`requirements.txt` 的目录。

不需要 editable install。运行时使用 `PYTHONPATH=$PWD:$PYTHONPATH`，这样
Python 能导入本地 `nanovllm`，`pip show nano-vllm-ascend` 也能看到
`nano_vllm_ascend-0.1.0.dist-info/` 里的本地元数据。

## 构建本地 Ascend 算子

克隆后，或者修改 `csrc/` 后，在昇腾机器上运行一次主 custom op 构建。
纯 Python 的 DSA 卸载修改不需要重新构建主 custom op。

```bash
PYTHONPATH=$PWD:$PYTHONPATH bash scripts/build_nanovllm_ops.sh
ls -lh nanovllm/_C*.so nanovllm/libnanovllm_ascend_kernels.so
```

rm -rf dsa_index_update_op/cann/build dsa_index_update_op/cann/output build/dsa_index_update_ext nanovllm/_dsa_index_update_custom nanovllm/_dsa_index_update_C*.so && mkdir -p runlog && SOC_VERSION=ascend910_9391 bash scripts/build_dsa_index_update_op.sh

如果 CANN custom OPP 包已经构建过，只改了 pybind extension，可以设置
`NANOVLLM_SKIP_CANN_OPP_BUILD=1` 跳过较慢的 OPP 重构建。

如果 worker 不是 `ascend910_9391`，在命令前设置 `SOC_VERSION=...`。
脚本内部会使用两种 SoC 名称：`ascend910_93` 用于 CANN custom OPP 包，
类似 `ascend910_9391` 的详细值用于 AscendC extension 构建。CANN custom
OPP 默认串行构建，方便看日志；需要加速时可以设置
`NANOVLLM_CANN_BUILD_JOBS=8`。

### 构建 standalone `dsa_index_update`

`dsa_index_update` 当前是 standalone CANN 算子，不和 `csrc/` 里的主 custom
op 一起编译。这个真算子仍在调试中；推理时可以用
`NANOVLLM_DSA_INDEX_UPDATE_FORCE_TORCH=1` 临时绕过它，改用 PyTorch 伪算子。

```bash
rm -rf dsa_index_update_op/cann/build dsa_index_update_op/cann/output build/dsa_index_update_ext nanovllm/_dsa_index_update_custom nanovllm/_dsa_index_update_C*.so
SOC_VERSION=ascend910_9391 bash scripts/build_dsa_index_update_op.sh
```

## 通用准备

```bash
mkdir -p runlog
PYTHONPATH=$PWD:$PYTHONPATH python -m pip show nano-vllm-ascend
```

## NANOVLLM 环境变量

| 变量 | 默认值 | 使用位置 | 含义 |
|---|---:|---|---|
| `NANOVLLM_MODEL` | `/home/models/Deepseek-V3.2-Pruned-95B-BF/` | examples | 模型目录。设置为本地 BF16 DeepSeek V3.2 导出路径。 |
| `NANOVLLM_TP_SIZE` | `4` | examples | Tensor parallel world size。当前 TPOT 对比一般使用 `8`。 |
| `NANOVLLM_ENABLE_EXPERT_PARALLEL` | `true` | examples | 是否启用 MoE expert parallel。DeepSeek V3.2 保持开启。 |
| `NANOVLLM_GPU_MEMORY_UTILIZATION` | `0.95` | examples | 可见 NPU 显存中用于估算 KV cache block 的比例。分配失败时调低。 |
| `NANOVLLM_KVCACHE_BLOCK_SIZE` | `128` | examples | Paged KV cache block size。当前 MLA/DSA 卸载路径假设为 `128`。 |
| `NANOVLLM_SKIP_WARMUP` | `true` | examples | `true` 跳过 warmup 以加快启动；`false` 在生成前执行 warmup。 |
| `NANOVLLM_MAX_MODEL_LEN` | `example/test.py` 中的 `max(prompt_lengths) + max_gen_tokens` | `example/test.py` | 初始化 engine 时使用的最大序列长度，不能大于 `NANOVLLM_MAX_BATCHED_TOKENS`。 |
| `NANOVLLM_MAX_BATCHED_TOKENS` | `example/test.py` 中的 `max(sum(prompt_lengths), max_model_len)` | `example/test.py` | 单 batch 最多调度 token 数，需要不小于 `NANOVLLM_MAX_MODEL_LEN`。 |
| `NANOVLLM_MAX_NUM_SEQS` | `example/test.py` 中的 `len(prompt_lengths)` | `example/test.py` | 最大并发序列数。 |
| `NANOVLLM_PROMPT_LENGTHS` | 未设置 | `example/test.py` | 逗号分隔的精确 prompt token 长度。用于确定性短/长/混合 DSA 卸载测试。 |
| `NANOVLLM_TEST_NUM_PROMPTS` / `NANOVLLM_NUM_PROMPTS` | `1` | `example/test.py` | 未设置 `NANOVLLM_PROMPT_LENGTHS` 时，随机精确 token prompt 的数量。 |
| `NANOVLLM_PROMPT_MIN_TOKENS` / `NANOVLLM_MIN_PROMPT_TOKENS` | `128` | `example/test.py` | 随机 prompt token 长度下界。 |
| `NANOVLLM_PROMPT_MAX_TOKENS` / `NANOVLLM_MAX_PROMPT_TOKENS` | 与 min 相同 | `example/test.py` | 随机 prompt token 长度上界。 |
| `NANOVLLM_PROMPT_SEED` | `0` | `example/test.py` | prompt 长度随机种子。 |
| `NANOVLLM_LONG_PROMPT_TOKENS` | `0` | `example/test.py` | 旧版单长度 prompt 设置。当前测试优先使用 `NANOVLLM_PROMPT_LENGTHS`。 |
| `NANOVLLM_USE_DEEPSEEK_CHAT` | `example/test.py` 中为 `true` | examples | `true` 时用 DeepSeek chat template 包装精确 token prompt。 |
| `NANOVLLM_ADD_BOS` | 与 `NANOVLLM_USE_DEEPSEEK_CHAT` 相同 | examples | `true` 时在 tokenizer 支持的情况下添加 BOS。 |
| `NANOVLLM_TEMPERATURE` | `example/test.py` 中为 `0.0` | examples | 采样温度。 |
| `NANOVLLM_MAX_GEN_TOKENS` | 各脚本自定 | examples | 每个请求最多 decode token 数。 |
| `NANOVLLM_IGNORE_EOS` | `example/test.py` 中为 `true` | examples | `true` 时忽略 EOS，持续 decode 到 `max_tokens`。 |
| `NANOVLLM_ENABLE_DECODE_MLAPO` | `true` | `deepseek_v32.py` | `true` 时用 `nanovllm.ops.mla_preprocess` 融合 decode qkv-a、q RMSNorm、q-up、kv RMSNorm、RoPE、cache 写入和 q-nope-up。 |
| `NANOVLLM_MLA_ROPE_NEOX_CACHE` | `true` | `deepseek_v32.py` | `true` 时 MLA RoPE cache 使用 neox basis，decode 后不再转回 interleaved。 |
| `NANOVLLM_DECODE_MLA_FIA_V2` | `true` | `deepseek_v32.py` | `true` 时 dense MLA decode 使用 `torch_npu.npu_fused_infer_attention_score_v2.out`，并复用 workspace 和逐层输出 buffer。 |
| `NANOVLLM_PROFILE_LAYER_IDS` | `0,mid,last` | `deepseek_v32.py` | 选择打印 decode timing 的层。支持逗号分隔层号，以及 `mid`、`last`、`all`、`*`。 |
| `NANOVLLM_LOG_DECODE_LAYER_TIMING` | `false` | `deepseek_v32.py` | `true` 时打印选中 decode layer 的时延，包括 `dsa_indexer_score`、`dsa_index_update`、`dsa_scatter_h2d`。 |
| `NANOVLLM_DECODE_LAYER_TIMING_SYNC` | `true` | `deepseek_v32.py` | `true` 时在 profiling 区间前后同步；`false` 时低开销近似计时。 |
| `NANOVLLM_DSA_INDEX_UPDATE_FORCE_TORCH` | `false` | `dsa_offload_ops.py` | `true` 时绕过真实 `dsa_index_update` custom op，推理中改用 PyTorch 伪算子。只用于正确性隔离，预期会慢很多。 |
| `NANOVLLM_FUSE_QKV_A` | `true` | `deepseek_v32.py` | `true` 时加载权重后融合 `q_a_proj` 和 `kv_a_proj_with_mqa`。 |
| `NANOVLLM_FREE_KV_B_PROJ` | `true` | `deepseek_v32.py` | `true` 时准备好 `w_uk_t/w_uv` 后释放 `kv_b_proj.weight`。 |
| `NANOVLLM_Q_UP_BMM_TRANS_MAX_TOKENS` | `1` | `deepseek_v32.py` | q-up projection 使用本地 `batch_matmul_transpose` 的最大 token 数。`0` 表示完全关闭。 |
| `NANOVLLM_INDEXER_Q_BMM_TRANS_MAX_TOKENS` | `8` | `deepseek_v32.py` | indexer `wq_b(q_c)` 使用本地 `batch_matmul_transpose` 的最大 decode token 数。`0` 表示关闭该优化。 |
| `NANOVLLM_MOE_BACKEND` | `grouped` | `deepseek_v32.py` | `grouped` 打包本地 experts 并使用 `npu_grouped_matmul`；`loop` 保留原逐 expert Python loop。 |

DSA offload decode 已经不再有 `NANOVLLM_DECODE_ATTENTION_BACKEND` 开关。
卸载路径先更新 sparse HBM budget，然后在 sparse budget 加新产生 decode token
上执行 dense MLA。

## 每次同步到昇腾后的运行手册

除非命令显式 `cd` 到别的目录，否则都在 DSA offload 仓库根目录运行。仅改
Python 时不需要 rebuild；修改 `csrc/` 后需要重新构建主 custom op。如果 worker
上的模型在 `/mnt/models`，只替换命令里的 `NANOVLLM_MODEL`。

先跑一个便宜的 Python 语法检查：

```bash
PYTHONPATH=$PWD:$PYTHONPATH python -m py_compile nanovllm/models/deepseek_v32.py nanovllm/engine/model_runner.py nanovllm/utils/context.py
```

DSA offload 主 TPOT/timing 命令。每次同步后优先回传这份日志：

```bash
mkdir -p runlog
PYTHONPATH=$PWD:$PYTHONPATH PYTORCH_NPU_ALLOC_CONF=expandable_segments:True NANOVLLM_GPU_MEMORY_UTILIZATION=0.85 ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 NANOVLLM_MODEL=/home/models/Deepseek-V3.2-Pruned-95B-BF/ NANOVLLM_TP_SIZE=8 NANOVLLM_PROMPT_LENGTHS=256,12288,14000,18000 NANOVLLM_MAX_GEN_TOKENS=5 NANOVLLM_IGNORE_EOS=1 NANOVLLM_LOG_DECODE_LAYER_TIMING=1 NANOVLLM_DECODE_LAYER_TIMING_SYNC=0 NANOVLLM_PROFILE_LAYER_IDS=mid NANOVLLM_SKIP_WARMUP=1 python example/test.py 2>&1 | tee runlog/run_offload.txt
```

baseline TPOT/timing 命令。需要和非卸载 baseline 做同 workload 对比时，在
baseline 仓库运行：

```bash
cd ../nano-vllm-ascend-DSA
mkdir -p runlog
PYTHONPATH=$PWD:$PYTHONPATH PYTORCH_NPU_ALLOC_CONF=expandable_segments:True NANOVLLM_GPU_MEMORY_UTILIZATION=0.85 ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 NANOVLLM_MODEL=/home/models/Deepseek-V3.2-Pruned-95B-BF/ NANOVLLM_TP_SIZE=8 NANOVLLM_DECODE_ATTENTION_BACKEND=mla NANOVLLM_ENABLE_DECODE_MLAPO=1 NANOVLLM_MLA_ROPE_NEOX_CACHE=1 NANOVLLM_DECODE_MLA_FIA_V2=1 NANOVLLM_PROMPT_LENGTHS=256,12288,14000,18000 NANOVLLM_MAX_GEN_TOKENS=5 NANOVLLM_IGNORE_EOS=1 NANOVLLM_LOG_DECODE_LAYER_TIMING=1 NANOVLLM_DECODE_LAYER_TIMING_SYNC=0 NANOVLLM_PROFILE_LAYER_IDS=mid NANOVLLM_SKIP_WARMUP=1 python example/test.py 2>&1 | tee runlog/run_baseline.txt
```

单条短序列。用于覆盖不释放 KV / 不触发重度卸载的情况：

```bash
PYTHONPATH=$PWD:$PYTHONPATH PYTORCH_NPU_ALLOC_CONF=expandable_segments:True NANOVLLM_GPU_MEMORY_UTILIZATION=0.85 ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 NANOVLLM_MODEL=/home/models/Deepseek-V3.2-Pruned-95B-BF/ NANOVLLM_TP_SIZE=8 NANOVLLM_PROMPT_LENGTHS=256 NANOVLLM_MAX_GEN_TOKENS=4 NANOVLLM_IGNORE_EOS=1 NANOVLLM_SKIP_WARMUP=1 python example/test.py 2>&1 | tee runlog/dsa_single_short.txt
```

单条长序列。应触发 DSA KV 卸载：

```bash
PYTHONPATH=$PWD:$PYTHONPATH PYTORCH_NPU_ALLOC_CONF=expandable_segments:True NANOVLLM_GPU_MEMORY_UTILIZATION=0.85 ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 NANOVLLM_MODEL=/home/models/Deepseek-V3.2-Pruned-95B-BF/ NANOVLLM_TP_SIZE=8 NANOVLLM_PROMPT_LENGTHS=12288 NANOVLLM_MAX_GEN_TOKENS=4 NANOVLLM_IGNORE_EOS=1 NANOVLLM_SKIP_WARMUP=1 python example/test.py 2>&1 | tee runlog/dsa_single_long.txt
```

一批短序列：

```bash
PYTHONPATH=$PWD:$PYTHONPATH PYTORCH_NPU_ALLOC_CONF=expandable_segments:True NANOVLLM_GPU_MEMORY_UTILIZATION=0.85 ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 NANOVLLM_MODEL=/home/models/Deepseek-V3.2-Pruned-95B-BF/ NANOVLLM_TP_SIZE=8 NANOVLLM_PROMPT_LENGTHS=128,256,384,512 NANOVLLM_MAX_GEN_TOKENS=4 NANOVLLM_IGNORE_EOS=1 NANOVLLM_SKIP_WARMUP=1 python example/test.py 2>&1 | tee runlog/dsa_batch_short.txt
```

长短混合序列。当前主要功能回归测试：

```bash
PYTHONPATH=$PWD:$PYTHONPATH PYTORCH_NPU_ALLOC_CONF=expandable_segments:True NANOVLLM_GPU_MEMORY_UTILIZATION=0.85 ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 NANOVLLM_MODEL=/home/models/Deepseek-V3.2-Pruned-95B-BF/ NANOVLLM_TP_SIZE=8 NANOVLLM_PROMPT_LENGTHS=256,12288,14000,18000 NANOVLLM_MAX_MODEL_LEN=18016 NANOVLLM_MAX_BATCHED_TOKENS=44544 NANOVLLM_MAX_NUM_SEQS=4 NANOVLLM_MAX_GEN_TOKENS=4 NANOVLLM_IGNORE_EOS=1 NANOVLLM_SKIP_WARMUP=1 python example/test.py 2>&1 | tee runlog/dsa_mixed_functional.txt
```

长短混合序列 + decode timing。用于和非卸载 baseline 对比 TPOT 和逐层时延分解：

```bash
PYTHONPATH=$PWD:$PYTHONPATH PYTORCH_NPU_ALLOC_CONF=expandable_segments:True NANOVLLM_GPU_MEMORY_UTILIZATION=0.85 ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 NANOVLLM_MODEL=/home/models/Deepseek-V3.2-Pruned-95B-BF/ NANOVLLM_TP_SIZE=8 NANOVLLM_PROMPT_LENGTHS=256,12288,14000,18000 NANOVLLM_MAX_MODEL_LEN=18016 NANOVLLM_MAX_BATCHED_TOKENS=44544 NANOVLLM_MAX_NUM_SEQS=4 NANOVLLM_MAX_GEN_TOKENS=16 NANOVLLM_IGNORE_EOS=1 NANOVLLM_LOG_DECODE_LAYER_TIMING=1 NANOVLLM_DECODE_LAYER_TIMING_SYNC=0 NANOVLLM_PROFILE_LAYER_IDS=0,mid,last NANOVLLM_SKIP_WARMUP=1 python example/test.py 2>&1 | tee runlog/dsa_mixed_timing.txt
```

如果 timing 日志太大，保持 `NANOVLLM_PROFILE_LAYER_IDS=0,mid,last`。只有需要
每一层细节时再使用 `NANOVLLM_PROFILE_LAYER_IDS=all`。如果要准确归因 indexer
子步骤，短跑一轮 `NANOVLLM_DECODE_LAYER_TIMING_SYNC=1`；异步 `=0` 适合看低开销
趋势，但时间可能串到相邻算子上。

```bash
PYTHONPATH=$PWD:$PYTHONPATH PYTORCH_NPU_ALLOC_CONF=expandable_segments:True NANOVLLM_GPU_MEMORY_UTILIZATION=0.85 ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 NANOVLLM_MODEL=/home/models/Deepseek-V3.2-Pruned-95B-BF/ NANOVLLM_TP_SIZE=8 NANOVLLM_PROMPT_LENGTHS=256,12288,14000,18000 NANOVLLM_MAX_MODEL_LEN=18016 NANOVLLM_MAX_BATCHED_TOKENS=44544 NANOVLLM_MAX_NUM_SEQS=4 NANOVLLM_MAX_GEN_TOKENS=3 NANOVLLM_IGNORE_EOS=1 NANOVLLM_LOG_DECODE_LAYER_TIMING=1 NANOVLLM_DECODE_LAYER_TIMING_SYNC=1 NANOVLLM_PROFILE_LAYER_IDS=mid NANOVLLM_SKIP_WARMUP=1 python example/test.py 2>&1 | tee runlog/dsa_mixed_indexer_detail_sync.txt
```

## Decode 时延字段

timing 行会按选中的 layer 和 TP rank 打印。日志里的单位是秒；分析时经常换算为
ms/layer。

| 字段 | 含义 |
|---|---|
| `attention_total` | 单个 decoder layer 的 attention block 总时延，从进入 self-attention 到 attention 后 RMSNorm 前。 |
| `dsa_total` | 三个 DSA 卸载伪算子的总和：`dsa_indexer_score + dsa_index_update + dsa_scatter_h2d`。 |
| `dsa_scatter_h2d` | 根据 `promote_idx/copy_counts`，把 promoted KV token 从 DRAM KV cache 拷贝进 HBM KV cache。 |
| `dsa_indexer_score` | 根据当前 decode query/indexer projection 和 `IndexCache` 计算候选 token 分数。 |
| `dsa_index_update` | 更新每个请求的 sparse HBM token budget，并输出 promote/demote token id 和 copy count。 |
| `indexer` | 执行 `DeepseekV32Indexer`，产出 `q_index`、`index_k` 和 DSA score 所需权重。 |
| `mlapo` | 执行 decode MLA preprocess，包括融合 qkv-a/q norm/q-up/kv norm/RoPE/cache-write/q-nope-up。 |
| `index_cache` | 把当前 token 的 `index_k` 写入常驻 HBM 的 `IndexCache`。 |
| `indexer_q_proj` | Indexer query projection：`wq_b(q_c)`。`indexer_q_path` 表示走 `linear` 还是 `bmm_transpose`。 |
| `indexer_k_proj` | Indexer key projection：`wk(hidden_states)`。 |
| `indexer_k_norm` | 对 indexer key projection 执行 LayerNorm。 |
| `indexer_rope` | 对 indexer query/key 应用 RoPE。 |
| `indexer_rotate` | 旧版 Hadamard rotate 计时。BF16 offload scorer 跳过该步骤以对齐 vllm-ascend 的非 C8 indexer 路径，所以应打印 `0.000000s`。 |
| `indexer_weights` | Indexer `weights_proj(hidden_states.float())` 投影和缩放。 |
| `decode_attention_op` | 在当前 sparse HBM KV budget 加新产生 decode token 上执行 dense MLA decode。 |
| `v_up` | 使用 `w_uv` 把 MLA latent output 投影回 hidden dimension。 |
| `o_proj` | attention 后输出投影，包括本地 linear projection 和 tensor-parallel all-reduce。 |
| `attention_gap` | 未被细分字段覆盖的 attention 时间：`attention_total - sum(recorded attention detail fields)`。 |
| `moe_total` | attention 后 MLP/MoE block 的耗时，打印在同一行方便对比。 |

## Indexer 投影说明

vllm-ascend 0.19 中，DSA/SFA 使用 MLAPO 融合 MLA preprocess，并通过
`enable_inner_out=True` 暴露 `q_c`，但没有把 indexer projection 融进 MLAPO。
indexer 路径仍然是单独执行的。

| 步骤 | vllm-ascend 0.19 的处理 | 当前本地含义 |
|---|---|---|
| 产生 `q_c` | MLAPO 返回归一化后的 `q_c`。 | 当前 DSA offload 路径已经对齐。 |
| indexer `q` 投影 | `wq_b(q_c)` 在 MLAPO 后执行。 | 小 decode batch 使用本地 `batch_matmul_transpose` 和预转置 per-head 权重。 |
| indexer `k` 投影 | `wk(hidden_states)` 或 fused `wk_weights_proj(hidden_states)`。 | 可以测试融合 `wk + weights_proj`。 |
| indexer weights | `weights_proj(hidden_states)` 或 `wk_weights_proj` 的 weights slice。 | 融合可能减少一个小 GEMM，需要验证 BF16 vs FP32 精度。 |

改热路径前，先在昇腾上跑下面的 probe。它会比较当前 PyTorch 投影路径和候选融合投影路径，
并打印数值差异和平均耗时。

```bash
PYTHONPATH=$PWD:$PYTHONPATH ASCEND_RT_VISIBLE_DEVICES=0 python ut_ops/probe_indexer_project.py --device npu:0 --tokens 4 --warmup 10 --iters 100
```

结果解读：

- 只有当 `fused_wk_weights_bf16_avg_ms` 明显更快，且 `weights` diff 可接受时，才考虑采用融合路径。早期 A2 结果没有显示有用收益。
- `q_bmm_transpose_cached_avg_ms` 用于隔离当前 decode 中已经使用的优化版 `wq_b(q_c)` 路径；触发条件是 `tokens <= NANOVLLM_INDEXER_Q_BMM_TRANS_MAX_TOKENS`。

## 可选本地算子探针

这些命令不会加载完整模型，只在调试本地 kernel 时使用。

```bash
PYTHONPATH=$PWD:$PYTHONPATH ASCEND_RT_VISIBLE_DEVICES=0,1,2,3 python ut_ops/probe_moe_grouped.py --device npu:0 --tokens 3 --hidden-size 512 --intermediate-size 256 --num-experts 32 --num-local-experts 8 --local-start 0 --topk 8 --topk-dtype int32 --warmup 5 --iters 20
```

```bash
PYTHONPATH=$PWD:$PYTHONPATH ASCEND_RT_VISIBLE_DEVICES=0,1,2,3 python ut_ops/probe_mla_preprocess.py --device npu:0 --tokens 128 --heads 32 --warmup 2 --iters 10
```

```bash
PYTHONPATH=$PWD:$PYTHONPATH ASCEND_RT_VISIBLE_DEVICES=0,1,2,3 python ut_ops/probe_v_up_proj.py --device npu:0 --tokens 7 --heads 32 --warmup 5 --iters 20
```

## 下一次运行命令

下面这些命令下一次在昇腾机器上运行。

先确认源码树包含 AIV-only kernel 修复、偶数 raw block-id 映射、以及 DSA
standalone `libcust_opapi.so` 直连加载。grep 应打印 `KERNEL_TYPE_AIV_ONLY`、
`rawBlockId`、`SetBlockDim(usedCoreNum * 2)`、`DSA_INDEX_UPDATE_CUST_OPAPI_PATH`、
`manual_acl_tensor_aiv_only_v5_even_block_map`：

```bash
grep -n "KERNEL_TYPE_AIV_ONLY\|rawBlockId\|SetBlockDim(usedCoreNum \\* 2)\|DSA_INDEX_UPDATE_CUST_OPAPI_PATH\|manual_acl_tensor_aiv_only_v5_even_block_map\|EXEC_NPU_CMD(aclnnDsaIndexUpdate" dsa_index_update_op/cann/op_kernel/dsa_index_update.cpp dsa_index_update_op/cann/op_host/dsa_index_update_tiling.cpp dsa_index_update_op/torch_extension/dsa_index_update_ext.cpp dsa_index_update_op/torch_extension/CMakeLists.txt nanovllm/models/dsa_index_update_real.py
```

然后清理旧 standalone-op 产物，构建 `dsa_index_update` CANN op 和 Python binding：

```bash
rm -rf dsa_index_update_op/cann/build dsa_index_update_op/cann/output build/dsa_index_update_ext nanovllm/_dsa_index_update_custom nanovllm/_dsa_index_update_C*.so && mkdir -p runlog && SOC_VERSION=ascend910_9391 bash scripts/build_dsa_index_update_op.sh 2>&1 | tee runlog/build_dsa_index_update_op_even_block_map_v5.txt
```

再跑 standalone 正确性/性能 probe：

```bash
mkdir -p runlog && PYTHONPATH=$PWD:$PYTHONPATH ASCEND_RT_VISIBLE_DEVICES=0 python ut_ops/probe_dsa_index_update.py --device npu:0 --batch-size 4 --candidate-lens 256,8192,12288,18000 --selected-lens 256,2048,3712,4096 --max-selected-len 8192 --output-capacity 2048 --max-copy-tokens 64 --warmup 5 --iters 50 2>&1 | tee runlog/probe_dsa_index_update_even_block_map_v5.txt
```

如果需要只绕过真实 `dsa_index_update` 算子，用 PyTorch 伪算子隔离推理链路，运行：

```bash
mkdir -p runlog && PYTHONPATH=$PWD:$PYTHONPATH PYTORCH_NPU_ALLOC_CONF=expandable_segments:True NANOVLLM_DSA_INDEX_UPDATE_FORCE_TORCH=1 NANOVLLM_GPU_MEMORY_UTILIZATION=0.85 ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 NANOVLLM_MODEL=/mnt/models/Deepseek-V3.2-Pruned-95B-BF/ NANOVLLM_TP_SIZE=8 NANOVLLM_PROMPT_LENGTHS=256,12288,14000,18000 NANOVLLM_MAX_MODEL_LEN=18016 NANOVLLM_MAX_BATCHED_TOKENS=44544 NANOVLLM_MAX_NUM_SEQS=4 NANOVLLM_MAX_GEN_TOKENS=3 NANOVLLM_IGNORE_EOS=1 NANOVLLM_LOG_DECODE_LAYER_TIMING=1 NANOVLLM_DECODE_LAYER_TIMING_SYNC=1 NANOVLLM_PROFILE_LAYER_IDS=mid NANOVLLM_SKIP_WARMUP=1 python example/test.py 2>&1 | tee runlog/dsa_mixed_force_torch_index_update_sync.txt
```

如果 probe 打印 `DSA_INDEX_UPDATE_ACCURACY ok=1`，再跑一次 sync timing：

```bash
mkdir -p runlog && PYTHONPATH=$PWD:$PYTHONPATH PYTORCH_NPU_ALLOC_CONF=expandable_segments:True NANOVLLM_GPU_MEMORY_UTILIZATION=0.85 ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 NANOVLLM_MODEL=/mnt/models/Deepseek-V3.2-Pruned-95B-BF/ NANOVLLM_TP_SIZE=8 NANOVLLM_PROMPT_LENGTHS=256,12288,14000,18000 NANOVLLM_MAX_MODEL_LEN=18016 NANOVLLM_MAX_BATCHED_TOKENS=44544 NANOVLLM_MAX_NUM_SEQS=4 NANOVLLM_MAX_GEN_TOKENS=3 NANOVLLM_IGNORE_EOS=1 NANOVLLM_LOG_DECODE_LAYER_TIMING=1 NANOVLLM_DECODE_LAYER_TIMING_SYNC=1 NANOVLLM_PROFILE_LAYER_IDS=mid NANOVLLM_SKIP_WARMUP=1 python example/test.py 2>&1 | tee runlog/dsa_mixed_dsa_index_update_real_sync_even_block_map_v5.txt
```

最后关掉 layer timing 测 TPOT：

```bash
mkdir -p runlog && PYTHONPATH=$PWD:$PYTHONPATH PYTORCH_NPU_ALLOC_CONF=expandable_segments:True NANOVLLM_GPU_MEMORY_UTILIZATION=0.85 ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 NANOVLLM_MODEL=/mnt/models/Deepseek-V3.2-Pruned-95B-BF/ NANOVLLM_TP_SIZE=8 NANOVLLM_PROMPT_LENGTHS=256,12288,14000,18000 NANOVLLM_MAX_MODEL_LEN=18016 NANOVLLM_MAX_BATCHED_TOKENS=44544 NANOVLLM_MAX_NUM_SEQS=4 NANOVLLM_MAX_GEN_TOKENS=16 NANOVLLM_IGNORE_EOS=1 NANOVLLM_SKIP_WARMUP=1 python example/test.py 2>&1 | tee runlog/dsa_mixed_dsa_index_update_real_no_timing_even_block_map_v5.txt
```
