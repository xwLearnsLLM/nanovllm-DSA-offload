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

再跑一个会触发 DSA decode update 的长序列：

```bash
PYTHONPATH=$PWD:$PYTHONPATH NANOVLLM_MAX_GEN_TOKENS=16 NANOVLLM_ENABLE_DECODE_MLAPO=1 NANOVLLM_PROMPT_LENGTHS=12288 python3 example/long_prompts.py
```
