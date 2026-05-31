# nano-vllm-ascend DeepSeek V3.2 DSA 卸载说明

在昇腾上利用DSA原生稀疏机制，在decode阶段将KVcache卸载到DRAM侧，提升decode并发。

# 运行说明

当前仓库是扁平目录结构。所有命令默认在仓库根目录运行。

### 构建昇腾算子

克隆代码仓到昇腾环境上，cd进去，然后运行以下命令来编译所需的算子：

```bash
SOC_VERSION=ascend910_9391 PYTHONPATH=$PWD:$PYTHONPATH bash scripts/build_nanovllm_ops.sh   # 编译
ls -lh nanovllm/_C*.so nanovllm/libnanovllm_ascend_kernels.so                               # 查看编译结果
```

注意事项：

- `SOC_VERSION=ascend910_9391` 请按实际情况设置。这里 `ascend910_9391` 对应的是 910C
- 脚本内部会使用两种 SoC 名称：`ascend910_93` 用于 CANN custom OPP 包，类似 `ascend910_9391` 的详细值用于 AscendC extension 构建。
- CANN custom OPP 默认串行构建，方便看日志；可以用 `NANOVLLM_CANN_BUILD_JOBS=8` 加速编译
- 如果 CANN custom OPP 包已经构建过，只改了 pybind extension，可以设置 `NANOVLLM_SKIP_CANN_OPP_BUILD=1` 跳过较慢的 OPP 重构建。

### 运行模型推理

运行不需要 `pip install -e .` 。只需使用 `PYTHONPATH=$PWD:$PYTHONPATH`，这样 Python 能导入本地 `nanovllm`，`pip show nano-vllm-ascend` 也能看到
`nano_vllm_ascend-0.1.0.dist-info/` 里的本地元数据。

跑混合长短序列，随机tokens (并打印时延分解) ：

```bash
PYTHONPATH=$PWD:$PYTHONPATH PYTORCH_NPU_ALLOC_CONF=expandable_segments:True ASCEND_LAUNCH_BLOCKING=0 ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 NANOVLLM_MODEL=/mnt/models/Deepseek-V3.2-Pruned-95B-BF/ NANOVLLM_GPU_MEMORY_UTILIZATION=0.85 NANOVLLM_TP_SIZE=8 NANOVLLM_MAX_MODEL_LEN=65536 NANOVLLM_MAX_BATCHED_TOKENS=65536 NANOVLLM_PROMPT_LENGTHS=256,12288,14000,18000 NANOVLLM_MAX_NUM_SEQS=4 NANOVLLM_MAX_GEN_TOKENS=5 NANOVLLM_IGNORE_EOS=1 NANOVLLM_LOG_DECODE_LAYER_TIMING=1 NANOVLLM_DECODE_LAYER_TIMING_SYNC=0 NANOVLLM_PROFILE_LAYER_IDS=mid NANOVLLM_SKIP_WARMUP=1 python3 example/test.py
```

跑混合长短序列，随机tokens ：

```bash
PYTHONPATH=$PWD:$PYTHONPATH PYTORCH_NPU_ALLOC_CONF=expandable_segments:True ASCEND_LAUNCH_BLOCKING=0 ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 NANOVLLM_MODEL=/mnt/models/Deepseek-V3.2-Pruned-95B-BF/ NANOVLLM_GPU_MEMORY_UTILIZATION=0.85 NANOVLLM_TP_SIZE=8 NANOVLLM_MAX_MODEL_LEN=65536 NANOVLLM_MAX_BATCHED_TOKENS=65536 NANOVLLM_PROMPT_LENGTHS=256,12288,14000,18000 NANOVLLM_MAX_NUM_SEQS=4 NANOVLLM_MAX_GEN_TOKENS=5 NANOVLLM_IGNORE_EOS=1 NANOVLLM_SKIP_WARMUP=1 python3 example/test.py
```

跑一批真实短请求 ：

```bash
PYTHONPATH=$PWD:$PYTHONPATH PYTORCH_NPU_ALLOC_CONF=expandable_segments:True ASCEND_LAUNCH_BLOCKING=0 ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 NANOVLLM_MODEL=/mnt/models/Deepseek-V3.2-Pruned-95B-BF/ NANOVLLM_GPU_MEMORY_UTILIZATION=0.85 NANOVLLM_TP_SIZE=8 NANOVLLM_MAX_MODEL_LEN=65536 NANOVLLM_MAX_BATCHED_TOKENS=65536 NANOVLLM_MAX_NUM_SEQS=4 NANOVLLM_MAX_GEN_TOKENS=16 NANOVLLM_IGNORE_EOS=1 NANOVLLM_SKIP_WARMUP=1 python3 example/short_prompts.py
```

跑一批真实长请求 ：

```bash
PYTHONPATH=$PWD:$PYTHONPATH PYTORCH_NPU_ALLOC_CONF=expandable_segments:True ASCEND_LAUNCH_BLOCKING=0 ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 NANOVLLM_MODEL=/mnt/models/Deepseek-V3.2-Pruned-95B-BF/ NANOVLLM_GPU_MEMORY_UTILIZATION=0.85 NANOVLLM_TP_SIZE=8 NANOVLLM_MAX_MODEL_LEN=65536 NANOVLLM_MAX_BATCHED_TOKENS=65536 NANOVLLM_MAX_NUM_SEQS=4 NANOVLLM_MAX_GEN_TOKENS=16 NANOVLLM_IGNORE_EOS=1 NANOVLLM_SKIP_WARMUP=1 python3 example/long_prompts.py
```



# 运行环境变量

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
| `NANOVLLM_FUSE_QKV_A` | `true` | `deepseek_v32.py` | `true` 时加载权重后融合 `q_a_proj` 和 `kv_a_proj_with_mqa`。 |
| `NANOVLLM_FREE_KV_B_PROJ` | `true` | `deepseek_v32.py` | `true` 时准备好 `w_uk_t/w_uv` 后释放 `kv_b_proj.weight`。 |
| `NANOVLLM_Q_UP_BMM_TRANS_MAX_TOKENS` | `1` | `deepseek_v32.py` | q-up projection 使用本地 `batch_matmul_transpose` 的最大 token 数。`0` 表示完全关闭。 |
| `NANOVLLM_MOE_BACKEND` | `grouped` | `deepseek_v32.py` | `grouped` 打包本地 experts 并使用 `npu_grouped_matmul`；`loop` 保留原逐 expert Python loop。 |



# Decode 时延字段

timing 行会按选中的 layer 和 TP rank 打印。日志里的单位是秒

| 字段 | 含义 |
|---|---|
| `attention_total` | 单个 decoder layer 的 attention block 总时延，从进入 self-attention 到 attention 后 RMSNorm 前。 |
| `dsa_total` | 三个 DSA 卸载伪算子的总和：`dsa_indexer_score + dsa_index_update + dsa_scatter_h2d`。 |
| `dsa_scatter_h2d` | 根据 `promote_idx/copy_counts`，把 promoted KV token 从 DRAM KV cache 拷贝进 HBM KV cache。 |
| `dsa_indexer_score` | 根据当前 decode query/indexer projection 和 `IndexCache` 计算候选 token 分数。 |
| `dsa_index_update` | 更新每个请求的 sparse HBM token budget，并输出 promote/demote token id 和 copy count。 |
| `indexer_project` | 执行 `DeepseekV32Indexer` / `dsa_indexer_project`，产出 `q_index`、`index_k` 和 DSA score 所需权重。 |
| `mlapo` | 执行 decode MLA preprocess，包括融合 qkv-a/q norm/q-up/kv norm/RoPE/cache-write/q-nope-up。 |
| `index_cache` | 把当前 token 的 `index_k` 写入常驻 HBM 的 `IndexCache`。 |
| `indexer_q_proj` | Indexer query projection：`wq_b(q_c)`。`indexer_q_path=linear+ascendc_post` 表示已走到 AscendC post 算子。 |
| `indexer_k_proj` | Indexer key projection：`wk(hidden_states)`。 |
| `indexer_k_norm` | 对 indexer key projection 执行 LayerNorm。 |
| `indexer_rope` | 对 indexer query/key 应用 RoPE。 |
| `indexer_weights` | Indexer `weights_proj(hidden_states.float())` 投影和缩放。 |
| `decode_attention_op` | 在当前 sparse HBM KV budget 加新产生 decode token 上执行 dense MLA decode。 |
| `v_up` | 使用 `w_uv` 把 MLA latent output 投影回 hidden dimension。 |
| `o_proj` | attention 后输出投影，包括本地 linear projection 和 tensor-parallel all-reduce。 |
| `attention_gap` | 未被细分字段覆盖的 attention 时间：`attention_total - sum(recorded attention detail fields)`。 |
| `moe_total` | attention 后 MLP/MoE block 的耗时，打印在同一行方便对比。 |



# Indexer projection 说明

vllm-ascend 0.19 中，DSA/SFA 使用 MLAPO 融合 MLA preprocess，并通过`enable_inner_out=True` 暴露 `q_c`，但没有把 indexer projection 融进 MLAPO。indexer projection 路径仍然是单独执行的。

| 步骤 | vllm-ascend 0.19 的处理 | 当前本地含义 |
|---|---|---|
| 产生 `q_c` | MLAPO 返回归一化后的 `q_c`。 | 当前 DSA offload 路径已经对齐。 |
| indexer `q` 投影 | `wq_b(q_c)` 在 MLAPO 后执行。 | 当前由 `dsa_indexer_project` Python wrapper 调用成熟 GEMM。 |
| indexer `k` 投影 | `wk(hidden_states)`。 | 当前由 `dsa_indexer_project` Python wrapper 调用成熟 GEMM。 |
| indexer weights | `weights_proj(hidden_states.float())`。 | 保持 FP32 权重投影，避免 BF16 融合路径引入精度差异。 |

改热路径后，先在昇腾上跑下面的 probe。它会比较 `dsa_indexer_project`
和 model reference 的数值差异，并打印平均耗时。

```bash
PYTHONPATH=$PWD:$PYTHONPATH ASCEND_RT_VISIBLE_DEVICES=0 python ut_ops/probe_indexer_project.py --device npu:0 --tokens 4 --warmup 10 --iters 100
```

结果解读：

- `dsa_indexer_project_vs_model_ref` 的 `q/k/weights` diff 应当通过阈值检查。
- `INDEXER_DETAIL` 会拆出 `q_proj/k_proj/k_norm/rope/weights_proj`，用于观察 wrapper 和 AscendC post 子算子的收益。
