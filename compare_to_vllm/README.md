# GLM vLLM-Ascend 对照

`compare_glm_native_dsa.py` 用 vLLM-Ascend 0.19 对同一个 8200-token Hawthorn prompt 做原生 GLM DSA eager 推理。脚本显式传 token IDs，并打印完整 token 序列的 SHA-256，便于排除 prompt 或 tokenizer 漂移。

在已安装 vLLM 0.19 和 vLLM-Ascend 0.19 的机器上，从 nano-vLLM 仓库根目录运行：

```bash
cd /home/w00916487/nanovllm-dsa_offload

export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export ASCEND_LAUNCH_BLOCKING=0
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
export PYTHONUNBUFFERED=1
export PYTHONPATH=$PWD:$PYTHONPATH

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

python3 compare_to_vllm/compare_glm_native_dsa.py
```

预期输出前缀：

```text
token_ids : [39, 672]
VLLM_GLM_DSA_RESULT=expected_prefix_[39,672]
```

这是正确性对照脚本，不用于 TPOT 或 profile 测量。
