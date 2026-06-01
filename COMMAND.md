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

## 2026-06-01 15:30:37：增加真实推理下 MLAPO 与 torch decode 路径对齐诊断

下一次请在昇腾上先跑这个：

```bash
PYTHONPATH=$PWD:$PYTHONPATH NANOVLLM_MAX_GEN_TOKENS=4 NANOVLLM_ENABLE_DECODE_MLAPO=1 NANOVLLM_DEBUG_DECODE_MLAPO_COMPARE=1 NANOVLLM_DEBUG_DECODE_MLAPO_COMPARE_LIMIT=2 NANOVLLM_PROFILE_LAYER_IDS=0,mid,last python3 example/long_prompts.py
```

再跑一个排查 RoPE cache 排列是否相关的对照：

```bash
PYTHONPATH=$PWD:$PYTHONPATH NANOVLLM_MAX_GEN_TOKENS=16 NANOVLLM_ENABLE_DECODE_MLAPO=1 NANOVLLM_MLA_ROPE_NEOX_CACHE=0 python3 example/long_prompts.py
```

## 2026-06-01 20:43:45：prefill 后第一个 decode step 暂时绕开 MLAPO

下一次请在昇腾上先跑这个：

```bash
PYTHONPATH=$PWD:$PYTHONPATH NANOVLLM_MAX_GEN_TOKENS=16 NANOVLLM_ENABLE_DECODE_MLAPO=1 NANOVLLM_MLA_ROPE_NEOX_CACHE=0 python3 example/long_prompts.py
```

再跑默认 RoPE neox cache 路径对照：

```bash
PYTHONPATH=$PWD:$PYTHONPATH NANOVLLM_MAX_GEN_TOKENS=16 NANOVLLM_ENABLE_DECODE_MLAPO=1 python3 example/long_prompts.py
```
