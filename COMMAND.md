## 推32专家残障模型（8卡910C）的公共配置

```bash
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export ASCEND_LAUNCH_BLOCKING=0
export NANOVLLM_MODEL=/mnt/models/Deepseek-V3.2-Pruned-95B-BF/  # 模型路径
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7                # 8 卡
export NANOVLLM_TP_SIZE=8                                       # TP8 
export NANOVLLM_ENABLE_EXPERT_PARALLEL=1
export NANOVLLM_KVCACHE_BLOCK_SIZE=128
export NANOVLLM_HBM_NUM_BLOCKS=500                              # 500个HBM blocks
export NANOVLLM_DRAM_NUM_BLOCKS=2000                            # 2000个DRAM blocks 以及 2000个HBM IndexCache Blocks
export NANOVLLM_MAX_MODEL_LEN=65536
export NANOVLLM_MAX_PREFILL_SEQS_PER_STEP=1                     # prefill最大batch-size设为1，避免爆显存
export NANOVLLM_MAX_DECODE_SEQS_PER_STEP=256                    # decode最大batch-size设为256
export NANOVLLM_IGNORE_EOS=1
export NANOVLLM_DSA_OFFLOAD_FIXED_TX=128   # 每请求每个decode step 每层换入的token数量
```

## 2026-06-01 14:11:12：增强 MLAPO 与 torch 对齐诊断 probe

下一次请在昇腾上先跑这个：

```bash
PYTHONPATH=$PWD:$PYTHONPATH ASCEND_RT_VISIBLE_DEVICES=0 python3 ut_ops/probe_mla_preprocess.py --device npu:0 --tokens 3 --heads 8 --hidden-size 7168 --q-lora-rank 1536 --kv-lora-rank 512 --nope-dim 128 --rope-dim 64 --position-offset 2404 --slot-offset 2404 --blocks 20 --warmup 3 --iters 10 --fail-on-diff
```

如果上面能过，再跑 inner-out 路径：

```bash
PYTHONPATH=$PWD:$PYTHONPATH ASCEND_RT_VISIBLE_DEVICES=0 python3 ut_ops/probe_mla_preprocess.py --device npu:0 --tokens 3 --heads 8 --hidden-size 7168 --q-lora-rank 1536 --kv-lora-rank 512 --nope-dim 128 --rope-dim 64 --position-offset 2404 --slot-offset 2404 --blocks 20 --warmup 3 --iters 10 --enable-inner-out --fail-on-diff
```

## 2026-06-01 14:43:35：推理绕开 MLAPO enable_inner_out=True 分支

下一次请在昇腾上先跑这个：

```bash
PYTHONPATH=$PWD:$PYTHONPATH NANOVLLM_MAX_GEN_TOKENS=16 NANOVLLM_ENABLE_DECODE_MLAPO=1 python3 example/long_prompts.py
```

## 2026-06-01 21:11:14：验证首个 decode fallback 后的长短混合 DSA 路径

下一次请在昇腾上先跑这个：

```bash
PYTHONPATH=$PWD:$PYTHONPATH NANOVLLM_MAX_GEN_TOKENS=16 NANOVLLM_ENABLE_DECODE_MLAPO=1 NANOVLLM_PROMPT_LENGTHS=256,12288,14000,18000 python3 example/test.py
```

如果长序列输出仍然异常，再跑一个 MLAPO 关闭对照，用来区分是 MLAPO 问题还是 DSA/offload budget 问题：

```bash
PYTHONPATH=$PWD:$PYTHONPATH NANOVLLM_MAX_GEN_TOKENS=16 NANOVLLM_ENABLE_DECODE_MLAPO=0 NANOVLLM_PROMPT_LENGTHS=256,12288,14000,18000 python3 example/test.py
```

## 2026-06-01 21:41:24：切分长序列异常来自初始 sparse budget 还是 decode update/scatter

下一次请在昇腾上先跑这个：

```bash
PYTHONPATH=$PWD:$PYTHONPATH NANOVLLM_MAX_GEN_TOKENS=16 NANOVLLM_ENABLE_DECODE_MLAPO=0 NANOVLLM_DSA_OFFLOAD_FIXED_TX=0 NANOVLLM_PROMPT_LENGTHS=12288 python3 example/test.py
```

再跑一个打开换入的单长序列对照：

```bash
PYTHONPATH=$PWD:$PYTHONPATH NANOVLLM_MAX_GEN_TOKENS=16 NANOVLLM_ENABLE_DECODE_MLAPO=0 NANOVLLM_DSA_OFFLOAD_FIXED_TX=128 NANOVLLM_PROMPT_LENGTHS=12288 python3 example/test.py
```

最后跑一个 offload 阈值边界对照，8064 不释放 HBM prefill 块，8192 会释放并进入 sparse budget：

```bash
PYTHONPATH=$PWD:$PYTHONPATH NANOVLLM_MAX_GEN_TOKENS=16 NANOVLLM_ENABLE_DECODE_MLAPO=0 NANOVLLM_DSA_OFFLOAD_FIXED_TX=0 NANOVLLM_PROMPT_LENGTHS=8064,8192 python3 example/test.py
```

再跑一个会触发 DSA decode update 的长序列：

```bash
PYTHONPATH=$PWD:$PYTHONPATH NANOVLLM_MAX_GEN_TOKENS=16 NANOVLLM_ENABLE_DECODE_MLAPO=1 NANOVLLM_PROMPT_LENGTHS=12288 python3 example/long_prompts.py
```

## 2026-06-01 22:33:41：prefill sparse budget 诊断切分与 MLA KV 顺序单测

下一次请在昇腾上先跑 MLA KV 顺序交换单测：

```bash
PYTHONPATH=$PWD:$PYTHONPATH ASCEND_RT_VISIBLE_DEVICES=0 python3 ut_ops/probe_mla_kv_permutation.py --device npu:0 --backend v2 --kv-len 4096 --block-size 128 --heads 8 --kv-lora-rank 512 --rope-dim 64 --warmup 5 --iters 20 --fail-on-diff
```

再跑 score 选 token 的 prefill sparse budget 诊断：

```bash
PYTHONPATH=$PWD:$PYTHONPATH NANOVLLM_MAX_GEN_TOKENS=16 NANOVLLM_ENABLE_DECODE_MLAPO=0 NANOVLLM_DSA_OFFLOAD_FIXED_TX=0 NANOVLLM_DSA_PREFILL_BUDGET_MODE=score NANOVLLM_DSA_DEBUG_PREFILL_BUDGET=1 NANOVLLM_PROFILE_LAYER_IDS=0,mid,last NANOVLLM_PROMPT_LENGTHS=8064,8192 python3 example/test.py
```

最后跑 suffix budget 对照。如果 suffix 正常而 score 异常，说明主要是 token 选择策略问题；如果 suffix 也异常，说明重点查 materialize / block_table / context_lens：

```bash
PYTHONPATH=$PWD:$PYTHONPATH NANOVLLM_MAX_GEN_TOKENS=16 NANOVLLM_ENABLE_DECODE_MLAPO=0 NANOVLLM_DSA_OFFLOAD_FIXED_TX=0 NANOVLLM_DSA_PREFILL_BUDGET_MODE=suffix NANOVLLM_DSA_DEBUG_PREFILL_BUDGET=1 NANOVLLM_PROFILE_LAYER_IDS=0,mid,last NANOVLLM_PROMPT_LENGTHS=8064,8192 python3 example/test.py
```

## 2026-06-02 08:55:46：修正 MLA KV 顺序单测的 FIA v2 参数，并切分 FIA v2/v1 , prefill sparse budget 增加 prefix+suffix sink 保留模式

下一次请在昇腾上先重新跑修正后的 MLA KV 顺序交换单测：

```bash
PYTHONPATH=$PWD:$PYTHONPATH ASCEND_RT_VISIBLE_DEVICES=0 python3 ut_ops/probe_mla_kv_permutation.py --device npu:0 --backend v2 --kv-len 4096 --block-size 128 --heads 8 --kv-lora-rank 512 --rope-dim 64 --warmup 5 --iters 20 --fail-on-diff
```

再跑 FIA v1 decode MLA 对照，判断 8192 sparse 后异常是否只出现在 FIA v2：

```bash
PYTHONPATH=$PWD:$PYTHONPATH NANOVLLM_MAX_GEN_TOKENS=16 NANOVLLM_ENABLE_DECODE_MLAPO=0 NANOVLLM_DECODE_MLA_FIA_V2=0 NANOVLLM_DSA_OFFLOAD_FIXED_TX=0 NANOVLLM_DSA_PREFILL_BUDGET_MODE=suffix NANOVLLM_PROMPT_LENGTHS=8064,8192 python3 example/test.py
```

下一次请在昇腾上先跑 prefix+suffix，默认保留前 128 个 sink token：

```bash
PYTHONPATH=$PWD:$PYTHONPATH NANOVLLM_MAX_GEN_TOKENS=16 NANOVLLM_ENABLE_DECODE_MLAPO=0 NANOVLLM_DSA_OFFLOAD_FIXED_TX=0 NANOVLLM_DSA_PREFILL_BUDGET_MODE=prefix_suffix NANOVLLM_DSA_PREFILL_PREFIX_TOKENS=128 NANOVLLM_DSA_DEBUG_PREFILL_BUDGET=1 NANOVLLM_PROFILE_LAYER_IDS=0,mid,last NANOVLLM_PROMPT_LENGTHS=8064,8192 python3 example/test.py
```

如果 128 个 prefix token 仍然不够，再跑一个 512 个 sink token 对照：

```bash
PYTHONPATH=$PWD:$PYTHONPATH NANOVLLM_MAX_GEN_TOKENS=16 NANOVLLM_ENABLE_DECODE_MLAPO=0 NANOVLLM_DSA_OFFLOAD_FIXED_TX=0 NANOVLLM_DSA_PREFILL_BUDGET_MODE=prefix_suffix NANOVLLM_DSA_PREFILL_PREFIX_TOKENS=512 NANOVLLM_DSA_DEBUG_PREFILL_BUDGET=1 NANOVLLM_PROFILE_LAYER_IDS=0,mid,last NANOVLLM_PROMPT_LENGTHS=8064,8192 python3 example/test.py
```

## 2026-06-02 14:26:10：prefill sparse budget materialize 全量强校验

下一次请在昇腾上先跑这个：

```bash
PYTHONPATH=$PWD:$PYTHONPATH NANOVLLM_MAX_GEN_TOKENS=16 NANOVLLM_ENABLE_DECODE_MLAPO=0 NANOVLLM_DSA_OFFLOAD_FIXED_TX=0 NANOVLLM_DSA_PREFILL_BUDGET_MODE=prefix_suffix NANOVLLM_DSA_PREFILL_PREFIX_TOKENS=128 NANOVLLM_DSA_DEBUG_PREFILL_BUDGET=1 NANOVLLM_DSA_DEBUG_PREFILL_VERIFY_ALL=1 NANOVLLM_DSA_DEBUG_PREFILL_BUDGET_RANK=0 NANOVLLM_PROFILE_LAYER_IDS=0,mid,last NANOVLLM_PROMPT_LENGTHS=8064,8192,9000 python3 example/test.py
```

## 2026-06-02 14:46:15：Tx=0 时完全绕过 decode 动态 DSA update

下一次请在昇腾上先跑这个：

```bash
PYTHONPATH=$PWD:$PYTHONPATH NANOVLLM_MAX_GEN_TOKENS=16 NANOVLLM_ENABLE_DECODE_MLAPO=0 NANOVLLM_DSA_OFFLOAD_FIXED_TX=0 NANOVLLM_DSA_PREFILL_BUDGET_MODE=prefix_suffix NANOVLLM_DSA_PREFILL_PREFIX_TOKENS=128 NANOVLLM_DSA_DEBUG_PREFILL_BUDGET=1 NANOVLLM_DSA_DEBUG_PREFILL_VERIFY_ALL=1 NANOVLLM_DSA_DEBUG_PREFILL_BUDGET_RANK=0 NANOVLLM_PROFILE_LAYER_IDS=0,mid,last NANOVLLM_PROMPT_LENGTHS=8064,8192,9000 python3 example/test.py
```
