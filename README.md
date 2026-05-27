# nano-vllm-ascend DeepSeek V3.2 DSA Offload Notes

This repository is flattened: run commands from this repository root, where
`nanovllm/`, `example/`, `scripts/`, and `requirements.txt` live.

No editable install is required. Use `PYTHONPATH=$PWD:$PYTHONPATH` so Python can
import `nanovllm` and `pip show nano-vllm-ascend` can see the local metadata in
`nano_vllm_ascend-0.1.0.dist-info/`.

## Build Local Ascend Ops

Run this once on the Ascend machine after cloning or after changing `csrc/`.
Python-only DSA offload changes do not need a rebuild.

```bash
PYTHONPATH=$PWD:$PYTHONPATH bash scripts/build_nanovllm_ops.sh
ls -lh nanovllm/_C*.so nanovllm/libnanovllm_ascend_kernels.so
```

If the CANN custom OPP package was already built and only the pybind extension
changed, use `NANOVLLM_SKIP_CANN_OPP_BUILD=1` to skip the slow OPP rebuild.

Set `SOC_VERSION=...` before the command if the worker is not `ascend910_9391`.
The script uses two SoC names internally: `ascend910_93` for the CANN custom OPP
package and the detailed value, such as `ascend910_9391`, for the AscendC
extension build. The CANN custom OPP build is serial by default for readable
logs; set `NANOVLLM_CANN_BUILD_JOBS=8` for a faster local rebuild.

## Common Setup

```bash
mkdir -p runlog
PYTHONPATH=$PWD:$PYTHONPATH python -m pip show nano-vllm-ascend
```

## NANOVLLM Environment Variables

| Variable | Default | Used by | Meaning |
|---|---:|---|---|
| `NANOVLLM_MODEL` | `/home/models/Deepseek-V3.2-Pruned-95B-BF/` | examples | Model directory. Set this to the local BF16 DeepSeek V3.2 export path. |
| `NANOVLLM_TP_SIZE` | `4` | examples | Tensor parallel world size. Use `8` for the current TPOT comparison runs. |
| `NANOVLLM_ENABLE_EXPERT_PARALLEL` | `true` | examples | Enables expert parallel execution for routed MoE layers. Keep enabled for DeepSeek V3.2. |
| `NANOVLLM_GPU_MEMORY_UTILIZATION` | `0.95` | examples | Fraction of visible NPU memory used to size KV cache blocks. Lower it if allocation fails. |
| `NANOVLLM_KVCACHE_BLOCK_SIZE` | `128` | examples | Paged KV cache block size. The current MLA/DSA offload paths expect `128`. |
| `NANOVLLM_SKIP_WARMUP` | `true` | examples | `true` skips warmup for faster startup; `false` runs warmup before generation. |
| `NANOVLLM_MAX_MODEL_LEN` | `max(prompt_lengths) + max_gen_tokens` in `example/test.py` | `example/test.py` | Max sequence length used to initialize the engine. Must be no larger than `NANOVLLM_MAX_BATCHED_TOKENS`. |
| `NANOVLLM_MAX_BATCHED_TOKENS` | `max(sum(prompt_lengths), max_model_len)` in `example/test.py` | `example/test.py` | Max tokens scheduled in one batch. Keep this >= `NANOVLLM_MAX_MODEL_LEN`. |
| `NANOVLLM_MAX_NUM_SEQS` | `len(prompt_lengths)` in `example/test.py` | `example/test.py` | Max concurrent sequences. |
| `NANOVLLM_PROMPT_LENGTHS` | unset | `example/test.py` | Comma-separated exact prompt token lengths. Use this for deterministic short/long/mixed DSA offload tests. |
| `NANOVLLM_TEST_NUM_PROMPTS` / `NANOVLLM_NUM_PROMPTS` | `1` | `example/test.py` | Number of random exact-token prompts when `NANOVLLM_PROMPT_LENGTHS` is unset. |
| `NANOVLLM_PROMPT_MIN_TOKENS` / `NANOVLLM_MIN_PROMPT_TOKENS` | `128` | `example/test.py` | Lower bound for random prompt token length. |
| `NANOVLLM_PROMPT_MAX_TOKENS` / `NANOVLLM_MAX_PROMPT_TOKENS` | same as min | `example/test.py` | Upper bound for random prompt token length. |
| `NANOVLLM_PROMPT_SEED` | `0` | `example/test.py` | Random seed for prompt lengths. |
| `NANOVLLM_LONG_PROMPT_TOKENS` | `0` | `example/test.py` | Legacy single-length prompt setting. Prefer `NANOVLLM_PROMPT_LENGTHS` for current tests. |
| `NANOVLLM_USE_DEEPSEEK_CHAT` | `true` for `example/test.py` | examples | `true` wraps exact-token prompts in the DeepSeek chat template. |
| `NANOVLLM_ADD_BOS` | same as `NANOVLLM_USE_DEEPSEEK_CHAT` | examples | `true` prepends tokenizer BOS when available. |
| `NANOVLLM_TEMPERATURE` | `0.0` in `example/test.py` | examples | Sampling temperature. |
| `NANOVLLM_MAX_GEN_TOKENS` | script-specific | examples | Max decode tokens per request. |
| `NANOVLLM_IGNORE_EOS` | `true` in `example/test.py` | examples | `true` keeps decoding until `max_tokens`. |
| `NANOVLLM_ENABLE_DECODE_MLAPO` | `true` | `deepseek_v32.py` | `true` fuses decode qkv-a, q RMSNorm, q-up, kv RMSNorm, RoPE, cache write, and q-nope-up into `nanovllm.ops.mla_preprocess`. |
| `NANOVLLM_MLA_ROPE_NEOX_CACHE` | `true` | `deepseek_v32.py` | `true` stores MLA RoPE cache in neox basis and skips decode post-conversion back to interleaved. |
| `NANOVLLM_DECODE_MLA_FIA_V2` | `true` | `deepseek_v32.py` | `true` uses `torch_npu.npu_fused_infer_attention_score_v2.out` for dense MLA decode, with shared workspace cache and per-layer output buffers. |
| `NANOVLLM_PROFILE_LAYER_IDS` | `0,mid,last` | `deepseek_v32.py` | Selects layers for decode timing. Accepts comma-separated ids plus `mid`, `last`, `all`, or `*`. |
| `NANOVLLM_LOG_DECODE_LAYER_TIMING` | `false` | `deepseek_v32.py` | `true` logs per selected decode layer, including `dsa_indexer_score`, `dsa_index_update`, and `dsa_scatter_h2d`. |
| `NANOVLLM_DECODE_LAYER_TIMING_SYNC` | `true` | `deepseek_v32.py` | `true` synchronizes around profiled regions; `false` prints lower-overhead approximate timing. |
| `NANOVLLM_FUSE_QKV_A` | `true` | `deepseek_v32.py` | `true` fuses `q_a_proj` and `kv_a_proj_with_mqa` into one projection after loading. |
| `NANOVLLM_FREE_KV_B_PROJ` | `true` | `deepseek_v32.py` | `true` frees `kv_b_proj.weight` after preparing `w_uk_t/w_uv`. |
| `NANOVLLM_Q_UP_BMM_TRANS_MAX_TOKENS` | `1` | `deepseek_v32.py` | Max token count using local `batch_matmul_transpose` for q-up projection. `0` disables it fully. |
| `NANOVLLM_MOE_BACKEND` | `grouped` | `deepseek_v32.py` | `grouped` packs local experts and uses `npu_grouped_matmul`; `loop` keeps the original per-expert Python loop. |

DSA offload decode no longer has a `NANOVLLM_DECODE_ATTENTION_BACKEND` switch.
The offload path updates the sparse HBM budget, then runs dense MLA over that
sparse budget plus newly generated decode tokens.

## Runbook After Each Ascend Sync

Run these from the DSA offload repo unless the command explicitly changes
directory. No rebuild is needed for Python-only changes unless `csrc/` changed.
If the model is under `/mnt/models` on a worker, replace only `NANOVLLM_MODEL`.

First run a cheap Python sanity check:

```bash
PYTHONPATH=$PWD:$PYTHONPATH python -m py_compile nanovllm/models/deepseek_v32.py nanovllm/engine/model_runner.py nanovllm/utils/context.py
```

Primary DSA offload TPOT/timing run. This is the first log to send back after
each code sync:

```bash
mkdir -p runlog
PYTHONPATH=$PWD:$PYTHONPATH PYTORCH_NPU_ALLOC_CONF=expandable_segments:True NANOVLLM_GPU_MEMORY_UTILIZATION=0.85 ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 NANOVLLM_MODEL=/home/models/Deepseek-V3.2-Pruned-95B-BF/ NANOVLLM_TP_SIZE=8 NANOVLLM_PROMPT_LENGTHS=256,12288,14000,18000 NANOVLLM_MAX_GEN_TOKENS=5 NANOVLLM_IGNORE_EOS=1 NANOVLLM_LOG_DECODE_LAYER_TIMING=1 NANOVLLM_DECODE_LAYER_TIMING_SYNC=0 NANOVLLM_PROFILE_LAYER_IDS=mid NANOVLLM_SKIP_WARMUP=1 python example/test.py 2>&1 | tee runlog/run_offload.txt
```

Baseline TPOT/timing run. Run this from the non-offload baseline repo with the
same workload when we need an apples-to-apples comparison:

```bash
cd ../nano-vllm-ascend-DSA
mkdir -p runlog
PYTHONPATH=$PWD:$PYTHONPATH PYTORCH_NPU_ALLOC_CONF=expandable_segments:True NANOVLLM_GPU_MEMORY_UTILIZATION=0.85 ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 NANOVLLM_MODEL=/home/models/Deepseek-V3.2-Pruned-95B-BF/ NANOVLLM_TP_SIZE=8 NANOVLLM_DECODE_ATTENTION_BACKEND=mla NANOVLLM_ENABLE_DECODE_MLAPO=1 NANOVLLM_MLA_ROPE_NEOX_CACHE=1 NANOVLLM_DECODE_MLA_FIA_V2=1 NANOVLLM_PROMPT_LENGTHS=256,12288,14000,18000 NANOVLLM_MAX_GEN_TOKENS=5 NANOVLLM_IGNORE_EOS=1 NANOVLLM_LOG_DECODE_LAYER_TIMING=1 NANOVLLM_DECODE_LAYER_TIMING_SYNC=0 NANOVLLM_PROFILE_LAYER_IDS=mid NANOVLLM_SKIP_WARMUP=1 python example/test.py 2>&1 | tee runlog/run_baseline.txt
```

Single short sequence. This should exercise the no-release / no-heavy-offload
case:

```bash
PYTHONPATH=$PWD:$PYTHONPATH PYTORCH_NPU_ALLOC_CONF=expandable_segments:True NANOVLLM_GPU_MEMORY_UTILIZATION=0.85 ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 NANOVLLM_MODEL=/home/models/Deepseek-V3.2-Pruned-95B-BF/ NANOVLLM_TP_SIZE=8 NANOVLLM_PROMPT_LENGTHS=256 NANOVLLM_MAX_GEN_TOKENS=4 NANOVLLM_IGNORE_EOS=1 NANOVLLM_SKIP_WARMUP=1 python example/test.py 2>&1 | tee runlog/dsa_single_short.txt
```

Single long sequence. This should trigger DSA KV offload:

```bash
PYTHONPATH=$PWD:$PYTHONPATH PYTORCH_NPU_ALLOC_CONF=expandable_segments:True NANOVLLM_GPU_MEMORY_UTILIZATION=0.85 ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 NANOVLLM_MODEL=/home/models/Deepseek-V3.2-Pruned-95B-BF/ NANOVLLM_TP_SIZE=8 NANOVLLM_PROMPT_LENGTHS=12288 NANOVLLM_MAX_GEN_TOKENS=4 NANOVLLM_IGNORE_EOS=1 NANOVLLM_SKIP_WARMUP=1 python example/test.py 2>&1 | tee runlog/dsa_single_long.txt
```

Batch of short sequences:

```bash
PYTHONPATH=$PWD:$PYTHONPATH PYTORCH_NPU_ALLOC_CONF=expandable_segments:True NANOVLLM_GPU_MEMORY_UTILIZATION=0.85 ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 NANOVLLM_MODEL=/home/models/Deepseek-V3.2-Pruned-95B-BF/ NANOVLLM_TP_SIZE=8 NANOVLLM_PROMPT_LENGTHS=128,256,384,512 NANOVLLM_MAX_GEN_TOKENS=4 NANOVLLM_IGNORE_EOS=1 NANOVLLM_SKIP_WARMUP=1 python example/test.py 2>&1 | tee runlog/dsa_batch_short.txt
```

Mixed short and long sequences. This is the main functional regression test:

```bash
PYTHONPATH=$PWD:$PYTHONPATH PYTORCH_NPU_ALLOC_CONF=expandable_segments:True NANOVLLM_GPU_MEMORY_UTILIZATION=0.85 ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 NANOVLLM_MODEL=/home/models/Deepseek-V3.2-Pruned-95B-BF/ NANOVLLM_TP_SIZE=8 NANOVLLM_PROMPT_LENGTHS=256,12288,14000,18000 NANOVLLM_MAX_MODEL_LEN=18016 NANOVLLM_MAX_BATCHED_TOKENS=44544 NANOVLLM_MAX_NUM_SEQS=4 NANOVLLM_MAX_GEN_TOKENS=4 NANOVLLM_IGNORE_EOS=1 NANOVLLM_SKIP_WARMUP=1 python example/test.py 2>&1 | tee runlog/dsa_mixed_functional.txt
```

Mixed short and long sequences with decode timing. Use this for TPOT and layer
breakdown comparison against the non-offload baseline:

```bash
PYTHONPATH=$PWD:$PYTHONPATH PYTORCH_NPU_ALLOC_CONF=expandable_segments:True NANOVLLM_GPU_MEMORY_UTILIZATION=0.85 ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 NANOVLLM_MODEL=/home/models/Deepseek-V3.2-Pruned-95B-BF/ NANOVLLM_TP_SIZE=8 NANOVLLM_PROMPT_LENGTHS=256,12288,14000,18000 NANOVLLM_MAX_MODEL_LEN=18016 NANOVLLM_MAX_BATCHED_TOKENS=44544 NANOVLLM_MAX_NUM_SEQS=4 NANOVLLM_MAX_GEN_TOKENS=16 NANOVLLM_IGNORE_EOS=1 NANOVLLM_LOG_DECODE_LAYER_TIMING=1 NANOVLLM_DECODE_LAYER_TIMING_SYNC=0 NANOVLLM_PROFILE_LAYER_IDS=0,mid,last NANOVLLM_SKIP_WARMUP=1 python example/test.py 2>&1 | tee runlog/dsa_mixed_timing.txt
```

If the timing log is too large, keep `NANOVLLM_PROFILE_LAYER_IDS=0,mid,last`.
Use `NANOVLLM_PROFILE_LAYER_IDS=all` only when we need every layer.

## Decode Timing Fields

The timing line is printed per selected layer and per TP rank. Units in the log
are seconds; the notes below often discuss them as ms/layer.

| Field | Meaning |
|---|---|
| `attention_total` | Total time of one decoder layer's attention block, from entering self-attention to before post-attention RMSNorm. |
| `dsa_total` | Sum of the three DSA offload pseudo ops: `dsa_indexer_score + dsa_index_update + dsa_scatter_h2d`. |
| `dsa_scatter_h2d` | Copies promoted KV tokens from DRAM KV cache into HBM KV cache according to `promote_idx/copy_counts`. |
| `dsa_indexer_score` | Computes per-candidate sparse-token scores from current decode query/indexer projection and `IndexCache`. |
| `dsa_index_update` | Updates the per-request sparse HBM token budget and emits promote/demote token ids plus copy counts. |
| `indexer` | Runs `DeepseekV32Indexer`, producing `q_index`, `index_k`, and indexer weights for DSA scoring. |
| `mlapo` | Runs decode MLA preprocess, including fused qkv-a/q norm/q-up/kv norm/RoPE/cache-write/q-nope-up work. |
| `index_cache` | Writes the current token's `index_k` into the HBM resident `IndexCache`. |
| `decode_attention_op` | Runs dense MLA decode over the current sparse HBM KV budget plus newly generated decode tokens. |
| `v_up` | Projects MLA latent output back to hidden dimension using `w_uv`. |
| `o_proj` | Output projection after attention, including local linear projection and tensor-parallel all-reduce. |
| `attention_gap` | Residual unaccounted attention time: `attention_total - sum(recorded attention detail fields)`. |
| `moe_total` | Time spent in the layer MLP/MoE block after attention, printed on the same line for comparison. |

## Indexer Projection Notes

In vllm-ascend 0.19, DSA/SFA uses MLAPO to fuse the MLA preprocess and expose
`q_c` via `enable_inner_out=True`, but it does not fuse the indexer projection
into MLAPO. The indexer path is still separate:

| Step | vllm-ascend 0.19 handling | Local implication |
|---|---|---|
| `q_c` production | MLAPO returns normalized `q_c`. | Already matched by the DSA offload path. |
| indexer `q` projection | `wq_b(q_c)` runs after MLAPO. | Still contributes to `indexer` timing. |
| indexer `k` projection | `wk(hidden_states)` or fused `wk_weights_proj(hidden_states)`. | We can test fusing `wk + weights_proj`. |
| indexer weights | `weights_proj(hidden_states)` or the weights slice of `wk_weights_proj`. | Fusing may reduce one small GEMM, with BF16-vs-FP32 accuracy to verify. |

Before changing the hot path, run the probe below on Ascend. It compares the
current PyTorch projection path with candidate fused projection paths and prints
both numerical differences and average latency.

```bash
PYTHONPATH=$PWD:$PYTHONPATH ASCEND_RT_VISIBLE_DEVICES=0 python ut_ops/probe_indexer_project.py --device npu:0 --tokens 4 --warmup 10 --iters 100
```

## Optional Local Op Probes

These do not load the full model. Use them only when debugging local kernels.

```bash
PYTHONPATH=$PWD:$PYTHONPATH ASCEND_RT_VISIBLE_DEVICES=0,1,2,3 python ut_ops/probe_moe_grouped.py --device npu:0 --tokens 3 --hidden-size 512 --intermediate-size 256 --num-experts 32 --num-local-experts 8 --local-start 0 --topk 8 --topk-dtype int32 --warmup 5 --iters 20
```

```bash
PYTHONPATH=$PWD:$PYTHONPATH ASCEND_RT_VISIBLE_DEVICES=0,1,2,3 python ut_ops/probe_mla_preprocess.py --device npu:0 --tokens 128 --heads 32 --warmup 2 --iters 10
```

```bash
PYTHONPATH=$PWD:$PYTHONPATH ASCEND_RT_VISIBLE_DEVICES=0,1,2,3 python ut_ops/probe_v_up_proj.py --device npu:0 --tokens 7 --heads 32 --warmup 5 --iters 20
```
