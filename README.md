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
| `NANOVLLM_IGNORE_EOS` | `false` | examples | `true` keeps decoding until `max_tokens`; `false` stops on EOS. |
| `NANOVLLM_DECODE_ATTENTION_BACKEND` | `mla` | `deepseek_v32.py` | Decode attention backend. `mla` uses dense paged MLA; `torch` uses the slow PyTorch sparse reference; `sfa` uses `nanovllm.ops.npu_sparse_flash_attention`. |
| `NANOVLLM_ENABLE_NPU_SFA_DECODE` | `false` | `deepseek_v32.py` | Legacy override. `true` forces decode backend to SFA even if `NANOVLLM_DECODE_ATTENTION_BACKEND` is set. This currently reproduces the SFA decode crash. |
| `NANOVLLM_COMPARE_NPU_SFA_DECODE` | `false` | `deepseek_v32.py` | When SFA decode is enabled, also computes the PyTorch sparse reference and logs the max difference. |
| `NANOVLLM_PROFILE_LAYER_IDS` | `0,mid,last` | `deepseek_v32.py` | Selects layers for timing/logging/dumps. Accepts comma-separated ids plus `mid`, `last`, `all`, or `*`. |
| `NANOVLLM_LOG_NPU_SFA_TIMING` | `false` | `deepseek_v32.py` | Logs prefill MLA timing and SFA timing for selected layers. Useful for low-level attention profiling. |
| `NANOVLLM_LOG_NPU_SFA_INPUTS` | `false` | `deepseek_v32.py` | Logs tensor shape/dtype/stride summaries for SFA input tensors on selected phases. |
| `NANOVLLM_DUMP_NPU_SFA_INPUTS` | unset | `deepseek_v32.py` | Directory used to dump SFA-style attention inputs as `.pt` files for replay/debug. |
| `NANOVLLM_DUMP_NPU_SFA_MAX_CALLS` | `1` | `deepseek_v32.py` | Max number of SFA input dumps per attention module. |
| `NANOVLLM_LOG_DECODE_LAYER_TIMING` | `false` | `deepseek_v32.py` | `true` logs per selected decode layer: broad attention time, narrow decode attention op time, and MoE/MLP time. Combine with `NANOVLLM_PROFILE_LAYER_IDS=all` for all layers. |

## 33. Long Prefill, Dense MLA Path

This path uses paged dense MLA for prefill and dense MLA decode. It does not
enable the unstable decode SFA kernel.

```bash
PYTHONPATH=$PWD:$PYTHONPATH PYTORCH_NPU_ALLOC_CONF=expandable_segments:True ASCEND_RT_VISIBLE_DEVICES=0,1,2,3 NANOVLLM_MODEL=/home/models/Deepseek-V3.2-Pruned-95B-BF/ NANOVLLM_TP_SIZE=4 NANOVLLM_DECODE_ATTENTION_BACKEND=mla NANOVLLM_LOG_NPU_SFA_TIMING=1 NANOVLLM_PROFILE_LAYER_IDS=0 NANOVLLM_LONG_PROMPT_TOKENS=3072 NANOVLLM_MAX_MODEL_LEN=3200 NANOVLLM_MAX_BATCHED_TOKENS=3200 NANOVLLM_MAX_NUM_SEQS=1 NANOVLLM_MAX_GEN_TOKENS=1 NANOVLLM_SKIP_WARMUP=1 python example/test.py
```

## 34. Short Decode, Dense MLA Decode

This uses dense MLA for decode and should avoid the currently unstable SFA
decode kernel.

```bash
PYTHONPATH=$PWD:$PYTHONPATH PYTORCH_NPU_ALLOC_CONF=expandable_segments:True ASCEND_RT_VISIBLE_DEVICES=0,1,2,3 NANOVLLM_MODEL=/home/models/Deepseek-V3.2-Pruned-95B-BF/ NANOVLLM_TP_SIZE=4 NANOVLLM_DECODE_ATTENTION_BACKEND=mla NANOVLLM_LOG_NPU_SFA_TIMING=1 NANOVLLM_PROFILE_LAYER_IDS=0 NANOVLLM_LONG_PROMPT_TOKENS=129 NANOVLLM_MAX_MODEL_LEN=256 NANOVLLM_MAX_BATCHED_TOKENS=256 NANOVLLM_MAX_NUM_SEQS=1 NANOVLLM_MAX_GEN_TOKENS=3 NANOVLLM_SKIP_WARMUP=1 python example/test.py
```

## 35. SFA Decode Stress

This is expected to reproduce the decode crash if the SFA decode kernel issue
is still present.

```bash
PYTHONPATH=$PWD:$PYTHONPATH PYTORCH_NPU_ALLOC_CONF=expandable_segments:True ASCEND_RT_VISIBLE_DEVICES=0,1,2,3 NANOVLLM_MODEL=/home/models/Deepseek-V3.2-Pruned-95B-BF/ NANOVLLM_TP_SIZE=4 NANOVLLM_ENABLE_NPU_SFA_DECODE=1 NANOVLLM_LOG_NPU_SFA_TIMING=1 NANOVLLM_PROFILE_LAYER_IDS=0 NANOVLLM_LONG_PROMPT_TOKENS=129 NANOVLLM_MAX_MODEL_LEN=256 NANOVLLM_MAX_BATCHED_TOKENS=256 NANOVLLM_MAX_NUM_SEQS=1 NANOVLLM_MAX_GEN_TOKENS=3 NANOVLLM_SKIP_WARMUP=1 python example/test.py
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
