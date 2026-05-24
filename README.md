# nano-vllm-ascend DeepSeek V3.2 Notes

This repository is flattened: run commands from this repository root, where
`nanovllm/`, `example/`, `scripts/`, and `requirements.txt` live.

No editable install is required. Use `PYTHONPATH=$PWD:$PYTHONPATH` so Python can
import `nanovllm` and `pip show nano-vllm-ascend` can see the local metadata in
`nano_vllm_ascend-0.1.0.dist-info/`.

## Common Setup

```bash
mkdir -p logs/sfa_manual
PYTHONPATH=$PWD:$PYTHONPATH python -m pip show nano-vllm-ascend
```

## 33. Long Prefill, Dense MLA Path

This path uses paged dense MLA for prefill, NPU MoE gating, NPU indexer, and
fixed sparse count 2048.

```bash
PYTHONPATH=$PWD:$PYTHONPATH PYTORCH_NPU_ALLOC_CONF=expandable_segments:True ASCEND_RT_VISIBLE_DEVICES=0,1,2,3 NANOVLLM_MODEL=/home/models/Deepseek-V3.2-Pruned-95B-BF/ NANOVLLM_TP_SIZE=4 NANOVLLM_ENABLE_NPU_SFA_DECODE=1 NANOVLLM_LOG_NPU_SFA_TIMING=1 NANOVLLM_PROFILE_LAYER_IDS=0 NANOVLLM_LONG_PROMPT_TOKENS=3072 NANOVLLM_MAX_MODEL_LEN=3200 NANOVLLM_MAX_BATCHED_TOKENS=3200 NANOVLLM_MAX_NUM_SEQS=1 NANOVLLM_MAX_GEN_TOKENS=1 NANOVLLM_SKIP_WARMUP=1 python example/test.py
```

## 34. Short Decode, Torch Sparse Decode

This keeps SFA decode disabled and should avoid the currently unstable SFA
decode kernel.

```bash
PYTHONPATH=$PWD:$PYTHONPATH PYTORCH_NPU_ALLOC_CONF=expandable_segments:True ASCEND_RT_VISIBLE_DEVICES=0,1,2,3 NANOVLLM_MODEL=/home/models/Deepseek-V3.2-Pruned-95B-BF/ NANOVLLM_TP_SIZE=4 NANOVLLM_LOG_NPU_SFA_TIMING=1 NANOVLLM_PROFILE_LAYER_IDS=0 NANOVLLM_LONG_PROMPT_TOKENS=129 NANOVLLM_MAX_MODEL_LEN=256 NANOVLLM_MAX_BATCHED_TOKENS=256 NANOVLLM_MAX_NUM_SEQS=1 NANOVLLM_MAX_GEN_TOKENS=3 NANOVLLM_SKIP_WARMUP=1 python example/test.py
```

## 35. SFA Decode Stress

This is expected to reproduce the decode crash if the SFA decode kernel issue
is still present.

```bash
PYTHONPATH=$PWD:$PYTHONPATH PYTORCH_NPU_ALLOC_CONF=expandable_segments:True ASCEND_RT_VISIBLE_DEVICES=0,1,2,3 NANOVLLM_MODEL=/home/models/Deepseek-V3.2-Pruned-95B-BF/ NANOVLLM_TP_SIZE=4 NANOVLLM_ENABLE_NPU_SFA_DECODE=1 NANOVLLM_LOG_NPU_SFA_TIMING=1 NANOVLLM_PROFILE_LAYER_IDS=0 NANOVLLM_LONG_PROMPT_TOKENS=129 NANOVLLM_MAX_MODEL_LEN=256 NANOVLLM_MAX_BATCHED_TOKENS=256 NANOVLLM_MAX_NUM_SEQS=1 NANOVLLM_MAX_GEN_TOKENS=3 NANOVLLM_SKIP_WARMUP=1 python example/test.py
```

## 36. Short Prompts

Runs the short prompt set with stable decode and limits completion to 16 tokens.

```bash
PYTHONPATH=$PWD:$PYTHONPATH PYTORCH_NPU_ALLOC_CONF=expandable_segments:True ASCEND_RT_VISIBLE_DEVICES=0,1,2,3 NANOVLLM_MODEL=/home/models/Deepseek-V3.2-Pruned-95B-BF/ NANOVLLM_TP_SIZE=4 NANOVLLM_MAX_GEN_TOKENS=16 NANOVLLM_SKIP_WARMUP=1 python example/short_prompts.py
```

## 37. Long Prompts

Runs three hard-coded English long QA prompts with stable decode.

```bash
PYTHONPATH=$PWD:$PYTHONPATH PYTORCH_NPU_ALLOC_CONF=expandable_segments:True ASCEND_RT_VISIBLE_DEVICES=0,1,2,3 NANOVLLM_MODEL=/home/models/Deepseek-V3.2-Pruned-95B-BF/ NANOVLLM_TP_SIZE=4 NANOVLLM_MAX_GEN_TOKENS=32 NANOVLLM_SKIP_WARMUP=1 python example/long_prompts.py
```

## 38. Exact 3072-Token Prefill Test

```bash
PYTHONPATH=$PWD:$PYTHONPATH PYTORCH_NPU_ALLOC_CONF=expandable_segments:True ASCEND_RT_VISIBLE_DEVICES=0,1,2,3 NANOVLLM_MODEL=/home/models/Deepseek-V3.2-Pruned-95B-BF/ NANOVLLM_TP_SIZE=4 NANOVLLM_LONG_PROMPT_TOKENS=3072 NANOVLLM_MAX_MODEL_LEN=3200 NANOVLLM_MAX_BATCHED_TOKENS=3200 NANOVLLM_MAX_NUM_SEQS=1 NANOVLLM_MAX_GEN_TOKENS=1 NANOVLLM_SKIP_WARMUP=1 python example/test.py
```

## 39. Exact Short Prompt With Multi-Token Decode

```bash
PYTHONPATH=$PWD:$PYTHONPATH PYTORCH_NPU_ALLOC_CONF=expandable_segments:True ASCEND_RT_VISIBLE_DEVICES=0,1,2,3 NANOVLLM_MODEL=/home/models/Deepseek-V3.2-Pruned-95B-BF/ NANOVLLM_TP_SIZE=4 NANOVLLM_LONG_PROMPT_TOKENS=129 NANOVLLM_MAX_MODEL_LEN=256 NANOVLLM_MAX_BATCHED_TOKENS=256 NANOVLLM_MAX_NUM_SEQS=1 NANOVLLM_MAX_GEN_TOKENS=10 NANOVLLM_SKIP_WARMUP=1 python example/test.py
```

Copy the terminal output back after running the command you want to test.
