# GLM vLLM-Ascend native DSA correctness comparison

`compare_glm_native_dsa.py` feeds vLLM-Ascend the same meaningful 8200-token
prompt constructed by `example/test.py`. It intentionally uses explicit token
IDs instead of asking vLLM to apply a chat template.

This is a correctness diagnostic, not a performance/profile run. The default
is eager execution, one request, and two generated tokens. The second token is
the first token produced by a decode forward, so it directly distinguishes the
dense reference `[39, 672]` (`Haw`) from the observed nano-vLLM DSA failure
`[39, 0]` (`H!`).

## Run

Run from the nano-vLLM repository root on the machine that has vLLM 0.19 and
vLLM-Ascend 0.19 installed:

```bash
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export ASCEND_LAUNCH_BLOCKING=0
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
# Match nano-vLLM's first-decode diagnostic and avoid folding a separate
# MLAPO difference into this native Indexer comparison.
export VLLM_ASCEND_ENABLE_MLAPO=0

export VLLM_MODEL=/mnt/models/GLM-5.1-w4a8/
export VLLM_TP_SIZE=16
export VLLM_ENABLE_EXPERT_PARALLEL=1
export VLLM_ENFORCE_EAGER=1
export VLLM_KVCACHE_BLOCK_SIZE=128
export VLLM_GPU_MEMORY_UTILIZATION=0.95

export VLLM_PROMPT_LENGTH=8200
export VLLM_MAX_GEN_TOKENS=2
export VLLM_MAX_NUM_BATCHED_TOKENS=1024

PYTHONUNBUFFERED=1 \
PYTHONPATH=$PWD:$PYTHONPATH \
python3 compare_to_vllm/compare_glm_native_dsa.py
```

The script refuses to run unless the loaded Hugging Face config reports
`model_type=glm_moe_dsa` and `index_topk=2048`. In the output, check:

```text
token_ids : [39, 672]
VLLM_GLM_DSA_RESULT=matches_dense_reference_[39,672]
```

or:

```text
token_ids : [39, 0]
VLLM_GLM_DSA_RESULT=matches_nanovllm_failure_[39,0]
```

`exact prompt` also prints the prompt length, first/last token IDs, and a
SHA-256 fingerprint of all token IDs. This makes accidental tokenizer or
prompt-construction drift visible in the run log.
