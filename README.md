# nano-vllm-ascend DeepSeek V3.2 Notes

This repository is flattened: run commands from this repository root, where
`nanovllm/`, `example/`, `scripts/`, and `requirements.txt` live.

No editable install is required. Use `PYTHONPATH=$PWD:$PYTHONPATH` so Python can
import `nanovllm` and `pip show nano-vllm-ascend` can see the local metadata in
`nano_vllm_ascend-0.1.0.dist-info/`.

## Build Local Ascend Ops

Run this once on the Ascend machine after cloning or after changing `csrc/`.
It builds `nanovllm._C` and installs the local CANN custom OPP package under
`nanovllm/_cann_ops_custom/`.

```bash
PYTHONPATH=$PWD:$PYTHONPATH bash scripts/build_nanovllm_ops.sh
ls -lh nanovllm/_C*.so nanovllm/libnanovllm_ascend_kernels.so
```

`nanovllm/ops/` is only the Python wrapper package. The compiled extension is
loaded from `nanovllm/_C*.so`.
If the CANN custom OPP package was already built and only the pybind extension
changed, use `NANOVLLM_SKIP_CANN_OPP_BUILD=1` to skip the slow OPP rebuild.

Set `SOC_VERSION=...` before the command if the worker is not `ascend910_9391`.
The script uses two SoC names internally: `ascend910_93` for the CANN custom OPP
package and the detailed value, such as `ascend910_9391`, for the AscendC
extension build. The CANN custom OPP build is serial by default for readable
logs; set `NANOVLLM_CANN_BUILD_JOBS=8` if you want a faster local rebuild.

## Common Setup

```bash
mkdir -p logs/sfa_manual
PYTHONPATH=$PWD:$PYTHONPATH python -m pip show nano-vllm-ascend
```

## NANOVLLM Environment Variables

| Variable | Default | Used by | Meaning |
|---|---:|---|---|
| `NANOVLLM_MODEL` | `/home/models/Deepseek-V3.2-Pruned-95B-BF/` | examples | Model directory. Set this to the local BF16 DeepSeek V3.2 export path. |
| `NANOVLLM_TP_SIZE` | `4` | examples | Tensor parallel world size. The current 95B pruned model normally uses `4` cards. |
| `NANOVLLM_ENABLE_EXPERT_PARALLEL` | `true` | examples | Enables expert parallel execution for routed MoE layers. Keep enabled for DeepSeek V3.2. |
| `NANOVLLM_GPU_MEMORY_UTILIZATION` | `0.95` | examples | Fraction of visible NPU memory used to size KV cache blocks. Lower it if allocation fails. |
| `NANOVLLM_KVCACHE_BLOCK_SIZE` | `128` | examples | Paged KV cache block size. The current SFA/MLA paths expect `128`. |
| `NANOVLLM_SKIP_WARMUP` | `true` | examples | `true` skips warmup for faster startup; `false` runs warmup before generation. |
| `NANOVLLM_MAX_MODEL_LEN` | `256` in `example/test.py` | `example/test.py` | Max sequence length used to initialize the engine. Must be no larger than `NANOVLLM_MAX_BATCHED_TOKENS`. |
| `NANOVLLM_MAX_BATCHED_TOKENS` | `NANOVLLM_MAX_MODEL_LEN` in `example/test.py` | `example/test.py` | Max tokens scheduled in one batch. Keep this >= `NANOVLLM_MAX_MODEL_LEN` for the current Nano scheduler. |
| `NANOVLLM_MAX_NUM_SEQS` | `1` in `example/test.py` | `example/test.py` | Max concurrent sequences. |
| `NANOVLLM_LONG_PROMPT_TOKENS` | `0` | `example/test.py` | `0` uses short built-in prompts; `>0` builds one exact-length token prompt for prefill/decode tests. |
| `NANOVLLM_USE_DEEPSEEK_CHAT` | `false` normally, `true` for exact-token test prompt | examples, `LLM.generate` string prompts | `true` wraps string prompts as `<｜User｜>...<｜Assistant｜>`; `false` uses raw prompt text. |
| `NANOVLLM_ADD_BOS` | same as `NANOVLLM_USE_DEEPSEEK_CHAT` | examples, `LLM.generate` string prompts | `true` prepends tokenizer BOS when available; `false` does not. |
| `NANOVLLM_TEMPERATURE` | `0.02` in prompt examples, `0.0` in `example/test.py` | examples | Sampling temperature. |
| `NANOVLLM_MAX_GEN_TOKENS` | script-specific | examples | Max decode tokens per request. Overrides the script default. |
| `NANOVLLM_IGNORE_EOS` | `true` in `example/test.py`, `false` in prompt examples | examples | `true` keeps decoding until `max_tokens`; `false` stops on EOS. `example/test.py` defaults to `true` so exact decode-length tests are not cut short by EOS. |
| `NANOVLLM_DECODE_ATTENTION_BACKEND` | `mla` | `deepseek_v32.py` | Decode attention backend. `mla` uses dense paged MLA; `torch` uses the slow PyTorch sparse reference; `sfa` uses `nanovllm.ops.npu_sparse_flash_attention`. |
| `NANOVLLM_ENABLE_DECODE_MLAPO` | `false` | `deepseek_v32.py` | `true` fuses decode qkv-a, q RMSNorm, q-up, kv RMSNorm, RoPE, cache write, and q-nope-up into `nanovllm.ops.mla_preprocess`. It needs extra NZ attention weights, so TP8 is safer for first tests. |
| `NANOVLLM_ENABLE_NPU_SFA_DECODE` | `false` | `deepseek_v32.py` | Legacy override. `true` forces decode backend to SFA even if `NANOVLLM_DECODE_ATTENTION_BACKEND` is set. This currently reproduces the SFA decode crash. |
| `NANOVLLM_COMPARE_NPU_SFA_DECODE` | `false` | `deepseek_v32.py` | When SFA decode is enabled, also computes the PyTorch sparse reference and logs the max difference. |
| `NANOVLLM_PROFILE_LAYER_IDS` | `0,mid,last` | `deepseek_v32.py` | Selects layers for timing/logging/dumps. Accepts comma-separated ids plus `mid`, `last`, `all`, or `*`. |
| `NANOVLLM_LOG_NPU_SFA_TIMING` | `false` | `deepseek_v32.py` | Logs prefill MLA timing and SFA timing for selected layers. Useful for low-level attention profiling. |
| `NANOVLLM_LOG_NPU_SFA_INPUTS` | `false` | `deepseek_v32.py` | Logs tensor shape/dtype/stride summaries for SFA input tensors on selected phases. |
| `NANOVLLM_DUMP_NPU_SFA_INPUTS` | unset | `deepseek_v32.py` | Directory used to dump SFA-style attention inputs as `.pt` files for replay/debug. |
| `NANOVLLM_DUMP_NPU_SFA_MAX_CALLS` | `1` | `deepseek_v32.py` | Max number of SFA input dumps per attention module. |
| `NANOVLLM_LOG_DECODE_LAYER_TIMING` | `false` | `deepseek_v32.py` | `true` logs per selected decode layer: broad attention time, qkv/q_b, kv split/norm/rotary, cache/q_up/attention/v_up, o linear/all-reduce, and MoE/MLP time. Combine with `NANOVLLM_PROFILE_LAYER_IDS=all` for all layers. |
| `NANOVLLM_DECODE_LAYER_TIMING_SYNC` | `true` | `deepseek_v32.py` | `true` synchronizes before/after profiled regions for accurate layer timing; `false` prints lower-overhead approximate timing. |
| `NANOVLLM_FUSE_QKV_A` | `true` | `deepseek_v32.py` | `true` fuses `q_a_proj` and `kv_a_proj_with_mqa` into one projection after loading; `false` keeps the original two projections. |
| `NANOVLLM_FREE_KV_B_PROJ` | `true` | `deepseek_v32.py` | `true` frees `kv_b_proj.weight` after preparing `w_uk_t/w_uv`; `false` keeps the original weight for debugging. |
| `NANOVLLM_Q_UP_BMM_TRANS_MAX_TOKENS` | `1` | `deepseek_v32.py` | Max token count using local `batch_matmul_transpose` for q-up projection. The local op is only used when `num_tokens == 1`; batched q-up is ignored because it can trigger an AICore MTE fault. `0` disables it fully. |
| `NANOVLLM_MOE_BACKEND` | `grouped` | `deepseek_v32.py` | `grouped` packs local experts and uses `npu_grouped_matmul`; `loop` keeps the original per-expert Python loop for comparison. |

## 33. Long Prefill, Dense MLA Path

This path uses paged dense MLA for prefill and dense MLA decode. It does not
enable the unstable decode SFA kernel.

```bash
PYTHONPATH=$PWD:$PYTHONPATH PYTORCH_NPU_ALLOC_CONF=expandable_segments:True ASCEND_RT_VISIBLE_DEVICES=0,1,2,3 NANOVLLM_MODEL=/home/models/Deepseek-V3.2-Pruned-95B-BF/ NANOVLLM_TP_SIZE=4 NANOVLLM_DECODE_ATTENTION_BACKEND=mla NANOVLLM_LOG_NPU_SFA_TIMING=1 NANOVLLM_PROFILE_LAYER_IDS=0 NANOVLLM_LONG_PROMPT_TOKENS=3072 NANOVLLM_MAX_MODEL_LEN=3200 NANOVLLM_MAX_BATCHED_TOKENS=3200 NANOVLLM_MAX_NUM_SEQS=1 NANOVLLM_MAX_GEN_TOKENS=1 NANOVLLM_IGNORE_EOS=1 NANOVLLM_SKIP_WARMUP=1 python example/test.py
```

## 34. Short Decode, Dense MLA Decode

This uses dense MLA for decode and should avoid the currently unstable SFA
decode kernel.

```bash
PYTHONPATH=$PWD:$PYTHONPATH PYTORCH_NPU_ALLOC_CONF=expandable_segments:True ASCEND_RT_VISIBLE_DEVICES=0,1,2,3 NANOVLLM_MODEL=/home/models/Deepseek-V3.2-Pruned-95B-BF/ NANOVLLM_TP_SIZE=4 NANOVLLM_DECODE_ATTENTION_BACKEND=mla NANOVLLM_LOG_NPU_SFA_TIMING=1 NANOVLLM_PROFILE_LAYER_IDS=0 NANOVLLM_LONG_PROMPT_TOKENS=129 NANOVLLM_MAX_MODEL_LEN=256 NANOVLLM_MAX_BATCHED_TOKENS=256 NANOVLLM_MAX_NUM_SEQS=1 NANOVLLM_MAX_GEN_TOKENS=3 NANOVLLM_IGNORE_EOS=1 NANOVLLM_SKIP_WARMUP=1 python example/test.py
```

## 35. SFA Decode Stress

This is expected to reproduce the decode crash if the SFA decode kernel issue
is still present.

```bash
PYTHONPATH=$PWD:$PYTHONPATH PYTORCH_NPU_ALLOC_CONF=expandable_segments:True ASCEND_RT_VISIBLE_DEVICES=0,1,2,3 NANOVLLM_MODEL=/home/models/Deepseek-V3.2-Pruned-95B-BF/ NANOVLLM_TP_SIZE=4 NANOVLLM_ENABLE_NPU_SFA_DECODE=1 NANOVLLM_LOG_NPU_SFA_TIMING=1 NANOVLLM_PROFILE_LAYER_IDS=0 NANOVLLM_LONG_PROMPT_TOKENS=129 NANOVLLM_MAX_MODEL_LEN=256 NANOVLLM_MAX_BATCHED_TOKENS=256 NANOVLLM_MAX_NUM_SEQS=1 NANOVLLM_MAX_GEN_TOKENS=3 NANOVLLM_IGNORE_EOS=1 NANOVLLM_SKIP_WARMUP=1 python example/test.py
```

## 36. Short Prompts

Runs the short prompt set with dense MLA decode and limits completion to 16
tokens.

```bash
PYTHONPATH=$PWD:$PYTHONPATH PYTORCH_NPU_ALLOC_CONF=expandable_segments:True ASCEND_RT_VISIBLE_DEVICES=0,1,2,3 NANOVLLM_MODEL=/home/models/Deepseek-V3.2-Pruned-95B-BF/ NANOVLLM_TP_SIZE=4 NANOVLLM_DECODE_ATTENTION_BACKEND=mla NANOVLLM_MAX_GEN_TOKENS=16 NANOVLLM_SKIP_WARMUP=1 python example/short_prompts.py
```

## 37. Long Prompts

Runs three hard-coded English long QA prompts with dense MLA decode.

```bash
PYTHONPATH=$PWD:$PYTHONPATH PYTORCH_NPU_ALLOC_CONF=expandable_segments:True ASCEND_RT_VISIBLE_DEVICES=0,1,2,3 NANOVLLM_MODEL=/home/models/Deepseek-V3.2-Pruned-95B-BF/ NANOVLLM_TP_SIZE=4 NANOVLLM_DECODE_ATTENTION_BACKEND=mla NANOVLLM_MAX_GEN_TOKENS=32 NANOVLLM_SKIP_WARMUP=1 python example/long_prompts.py
```

## 38. MoE Grouped Matmul Probe

Checks whether the Ascend grouped MoE route matches the current local-expert
reference path. This does not load the model.

```bash
PYTHONPATH=$PWD:$PYTHONPATH ASCEND_RT_VISIBLE_DEVICES=0,1,2,3 python ut_ops/probe_moe_grouped.py --device npu:0 --tokens 128 --hidden-size 512 --intermediate-size 256 --num-experts 32 --num-local-experts 8 --local-start 0 --topk 8 --topk-dtype int32
```

For a decode-shaped micro-batch:

```bash
PYTHONPATH=$PWD:$PYTHONPATH ASCEND_RT_VISIBLE_DEVICES=0,1,2,3 python ut_ops/probe_moe_grouped.py --device npu:0 --tokens 3 --hidden-size 512 --intermediate-size 256 --num-experts 32 --num-local-experts 8 --local-start 0 --topk 8 --topk-dtype int32 --warmup 5 --iters 20
```

## 39. Short Decode With Layer Timing

Runs the model with grouped MoE enabled by default and prints per-layer decode
timing for selected layers.

```bash
PYTHONPATH=$PWD:$PYTHONPATH PYTORCH_NPU_ALLOC_CONF=expandable_segments:True ASCEND_RT_VISIBLE_DEVICES=0,1,2,3 NANOVLLM_MODEL=/home/models/Deepseek-V3.2-Pruned-95B-BF/ NANOVLLM_TP_SIZE=4 NANOVLLM_DECODE_ATTENTION_BACKEND=mla NANOVLLM_LOG_DECODE_LAYER_TIMING=1 NANOVLLM_PROFILE_LAYER_IDS=0,mid,last NANOVLLM_LONG_PROMPT_TOKENS=129 NANOVLLM_MAX_MODEL_LEN=256 NANOVLLM_MAX_BATCHED_TOKENS=256 NANOVLLM_MAX_NUM_SEQS=1 NANOVLLM_MAX_GEN_TOKENS=8 NANOVLLM_IGNORE_EOS=1 NANOVLLM_SKIP_WARMUP=1 python example/test.py
```

## 40. Decode Optimization Sweep

Recommended TP8 short-prompt decode without per-layer timing:

```bash
PYTHONPATH=$PWD:$PYTHONPATH PYTORCH_NPU_ALLOC_CONF=expandable_segments:True NANOVLLM_GPU_MEMORY_UTILIZATION=0.8 ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 NANOVLLM_MODEL=/home/models/Deepseek-V3.2-Pruned-95B-BF/ NANOVLLM_TP_SIZE=8 NANOVLLM_DECODE_ATTENTION_BACKEND=mla NANOVLLM_MAX_GEN_TOKENS=16 NANOVLLM_SKIP_WARMUP=1 python example/short_prompts.py
```

Batched q-up through the local `batch_matmul_transpose` op is currently disabled
for `num_tokens > 1` because it reproduced an AICore MTE fault. To compare with
the local q-up op fully disabled, run:

```bash
PYTHONPATH=$PWD:$PYTHONPATH PYTORCH_NPU_ALLOC_CONF=expandable_segments:True NANOVLLM_GPU_MEMORY_UTILIZATION=0.8 ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 NANOVLLM_MODEL=/home/models/Deepseek-V3.2-Pruned-95B-BF/ NANOVLLM_TP_SIZE=8 NANOVLLM_DECODE_ATTENTION_BACKEND=mla NANOVLLM_Q_UP_BMM_TRANS_MAX_TOKENS=0 NANOVLLM_MAX_GEN_TOKENS=16 NANOVLLM_SKIP_WARMUP=1 python example/short_prompts.py
```

Timing with less synchronization overhead:

```bash
PYTHONPATH=$PWD:$PYTHONPATH PYTORCH_NPU_ALLOC_CONF=expandable_segments:True NANOVLLM_GPU_MEMORY_UTILIZATION=0.8 ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 NANOVLLM_MODEL=/home/models/Deepseek-V3.2-Pruned-95B-BF/ NANOVLLM_TP_SIZE=8 NANOVLLM_DECODE_ATTENTION_BACKEND=mla NANOVLLM_LOG_DECODE_LAYER_TIMING=1 NANOVLLM_DECODE_LAYER_TIMING_SYNC=0 NANOVLLM_PROFILE_LAYER_IDS=0,mid,last NANOVLLM_MAX_GEN_TOKENS=16 NANOVLLM_SKIP_WARMUP=1 python example/short_prompts.py
```

## 41. MLAPO Preprocess Probe

After rebuilding local Ascend ops, run this to compare the new
`nanovllm.ops.mla_preprocess` binding against the split PyTorch reference on
the same synthetic BF16 inputs. It checks `ql_nope`, `q_pe`, `ckv_cache`, and
`kpe_cache`. BF16 no-quant MLAPO uses an NZ weight column block size of 16,
which is the probe default.

```bash
PYTHONPATH=$PWD:$PYTHONPATH bash scripts/build_nanovllm_ops.sh
PYTHONPATH=$PWD:$PYTHONPATH ASCEND_RT_VISIBLE_DEVICES=0,1,2,3 python ut_ops/probe_mla_preprocess.py --device npu:0 --tokens 7 --heads 32 --warmup 2 --iters 10
```

For a larger decode-shaped micro-batch:

```bash
PYTHONPATH=$PWD:$PYTHONPATH ASCEND_RT_VISIBLE_DEVICES=0,1,2,3 python ut_ops/probe_mla_preprocess.py --device npu:0 --tokens 128 --heads 32 --warmup 2 --iters 10
```

## 42. MLAPO Preprocess Diagnostics

Use these when section 41 still shows large `MLAPO_DIFF` values. They isolate
whether the issue is basic output writing, `npu_format_cast`, or the no-quant
tiling key.

All-zero inputs should produce all-zero outputs:

```bash
PYTHONPATH=$PWD:$PYTHONPATH ASCEND_RT_VISIBLE_DEVICES=0,1,2,3 python ut_ops/probe_mla_preprocess.py --device npu:0 --tokens 7 --heads 32 --init-scale 0 --warmup 0 --iters 1
```

Use explicit `transdata` storage without `torch_npu.npu_format_cast(..., 29)`:

```bash
PYTHONPATH=$PWD:$PYTHONPATH ASCEND_RT_VISIBLE_DEVICES=0,1,2,3 python ut_ops/probe_mla_preprocess.py --device npu:0 --tokens 7 --heads 32 --no-format-cast --warmup 2 --iters 10
```

For comparison only, try the vllm-ascend W8A8-style weight blocking. This is
expected to be wrong for BF16 no-quant, but is useful to confirm the block-size
diagnosis:

```bash
PYTHONPATH=$PWD:$PYTHONPATH ASCEND_RT_VISIBLE_DEVICES=0,1,2,3 python ut_ops/probe_mla_preprocess.py --device npu:0 --tokens 7 --heads 32 --weight-block-cols 32 --warmup 2 --iters 10
```

## 43. Decode MLAPO Path

This enables the fused decode MLA preprocess path. Start with TP8 because the
path keeps extra NZ-format attention weights for now.

```bash
PYTHONPATH=$PWD:$PYTHONPATH PYTORCH_NPU_ALLOC_CONF=expandable_segments:True NANOVLLM_GPU_MEMORY_UTILIZATION=0.8 ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 NANOVLLM_MODEL=/home/models/Deepseek-V3.2-Pruned-95B-BF/ NANOVLLM_TP_SIZE=8 NANOVLLM_DECODE_ATTENTION_BACKEND=mla NANOVLLM_ENABLE_DECODE_MLAPO=1 NANOVLLM_LOG_DECODE_LAYER_TIMING=1 NANOVLLM_DECODE_LAYER_TIMING_SYNC=0 NANOVLLM_PROFILE_LAYER_IDS=0,mid,last NANOVLLM_MAX_GEN_TOKENS=16 NANOVLLM_SKIP_WARMUP=1 python example/short_prompts.py
```

## 44. Decode V-Up Transpose BatchMatmul Check

No rebuild is needed for this change. The decode v-up projection now uses
`torch_npu.npu_transpose_batchmatmul`, matching the vllm-ascend MLA path more
closely. Run this and compare TPOT plus the per-layer `v_up` timing against the
previous MLAPO run.

First, check the v-up micro-kernel against the previous `torch.bmm` reference:

```bash
PYTHONPATH=$PWD:$PYTHONPATH ASCEND_RT_VISIBLE_DEVICES=0,1,2,3 python ut_ops/probe_v_up_proj.py --device npu:0 --tokens 7 --heads 32 --warmup 5 --iters 20
```

Then run the model path:

```bash
PYTHONPATH=$PWD:$PYTHONPATH PYTORCH_NPU_ALLOC_CONF=expandable_segments:True NANOVLLM_GPU_MEMORY_UTILIZATION=0.8 ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 NANOVLLM_MODEL=/home/models/Deepseek-V3.2-Pruned-95B-BF/ NANOVLLM_TP_SIZE=8 NANOVLLM_DECODE_ATTENTION_BACKEND=mla NANOVLLM_ENABLE_DECODE_MLAPO=1 NANOVLLM_LOG_DECODE_LAYER_TIMING=1 NANOVLLM_DECODE_LAYER_TIMING_SYNC=0 NANOVLLM_PROFILE_LAYER_IDS=0,mid,last NANOVLLM_MAX_GEN_TOKENS=16 NANOVLLM_SKIP_WARMUP=1 python example/short_prompts.py
```

## 45. Decode Small-Op Cache Check

No rebuild is needed. This version reuses decode-step MLAPO `cos/sin`, prepares
int32 flat slot ids once in `prepare_decode`, and reuses per-layer MLAPO output
scratch buffers. Compare TPOT and the per-layer `mlapo` / `attention_gap`
timings against section 44.

```bash
PYTHONPATH=$PWD:$PYTHONPATH PYTORCH_NPU_ALLOC_CONF=expandable_segments:True NANOVLLM_GPU_MEMORY_UTILIZATION=0.8 ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 NANOVLLM_MODEL=/home/models/Deepseek-V3.2-Pruned-95B-BF/ NANOVLLM_TP_SIZE=8 NANOVLLM_DECODE_ATTENTION_BACKEND=mla NANOVLLM_ENABLE_DECODE_MLAPO=1 NANOVLLM_LOG_DECODE_LAYER_TIMING=1 NANOVLLM_DECODE_LAYER_TIMING_SYNC=0 NANOVLLM_PROFILE_LAYER_IDS=0,mid,last NANOVLLM_MAX_GEN_TOKENS=16 NANOVLLM_SKIP_WARMUP=1 python example/short_prompts.py
```
