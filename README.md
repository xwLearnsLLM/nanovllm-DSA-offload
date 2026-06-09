# nano-vllm-ascend DeepSeek V3.2 DSA 卸载说明

昇腾上做 DSA 模型的 decode 阶段 KVcache offload ，节省显存，提升 batch-size

　

##  编译算子

在昇腾机器的仓库根目录执行：

```bash
rm -rf build/nanovllm_ascend_ops     # 清除旧的编译
NANOVLLM_CANN_BUILD_JOBS=64 NANOVLLM_EXT_BUILD_JOBS=1 SOC_VERSION=ascend910_9391 PYTHONPATH=$PWD:$PYTHONPATH bash scripts/build_nanovllm_ops.sh
```

说明：

- `SOC_VERSION=ascend910_9391` 按机器实际 SoC 设置。
- 如果只改了 pybind extension，可以设置 `NANOVLLM_SKIP_CANN_OPP_BUILD=1` 跳过较慢的 OPP 重建。

　

## 准备模型

当前这一版 nano-vllm-ascend 只支持 BF16 的 deepseek_v32 系列的模型。因为BF16非常占显存，所以不建议跑满血 256 专家的原版 DeepSeek-V3.2 ，而是跑 ：

- **32专家残障版 deepseek_v32** ：https://www.modelscope.cn/models/xwLearnsLLM/Deepseek-V3.2-Pruned-95B 。注意，需要先把模型下载下来，然后按照它的 README 的指示，把模型权重文件从 FP8 转成 BF16 。该模型在nanovllm上需要使用 4~8 张昇腾 910C 就能拉起（每张卡 64GB显存）。
- **cerebras公司裁剪128专家版的 deepseek_v32** ： https://www.modelscope.cn/models/cerebras/DeepSeek-V3.2-REAP-345B-A37B 。注意，需要先把模型下载下来，然后借用 [这里](https://www.modelscope.cn/models/xwLearnsLLM/Deepseek-V3.2-Pruned-95B) 的python脚本来把模型权重文件从 FP8 转成 BF16。该模型在nanovllm上需要使用 16 张昇腾 910C 就能拉起（每张卡 64GB显存）。

　

## 添加prof
```bash
NANOVLLM_NPU_PROFILE = 1
NANOVLLM_NPU_PROFILE_DIR = xx
```

　

## 推128专家模型（16卡910C）准备工作

先进行一些公用配置：

```bash
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export ASCEND_LAUNCH_BLOCKING=0
export NANOVLLM_MODEL=/var/models/DeepSeek-V3.2-REAP-345B-A37B-BF16/   # 模型路径
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15 # 16卡
export NANOVLLM_TP_SIZE=16                                      # TP16
export NANOVLLM_ENABLE_EXPERT_PARALLEL=1
export NANOVLLM_KVCACHE_BLOCK_SIZE=128
export NANOVLLM_HBM_NUM_BLOCKS=200                              # 200个HBM blocks
export NANOVLLM_DRAM_NUM_BLOCKS=800                             # 800个DRAM blocks 以及 800个HBM IndexCache Blocks
export NANOVLLM_MAX_MODEL_LEN=65536
export NANOVLLM_MAX_PREFILL_SEQS_PER_STEP=1                     # prefill最大batch-size设为1，避免爆显存
export NANOVLLM_MAX_DECODE_SEQS_PER_STEP=256                    # decode最大batch-size设为256
export NANOVLLM_IGNORE_EOS=1
export NANOVLLM_ENABLE_DECODE_MLAPO=1
```

　

## 推32专家残障模型（8卡910C）准备工作

先进行一些公用配置：

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
export NANOVLLM_ENABLE_DECODE_MLAPO=1
```

　

## 开启或者关闭时延分解打印

```
export NANOVLLM_LOG_DECODE_LAYER_TIMING=0    # 是否打印时延分解
export NANOVLLM_DECODE_LAYER_TIMING_SYNC=0   # 计时时是否 sync 一下
export NANOVLLM_PROFILE_LAYER_IDS=mid        # 打印的层
```

　

## 运行推理

然后进入目录，不需要 `pip install -e .` ，直接推：

运行推理（不组图，走 current 路径）：

```
PYTHONPATH=$PWD:$PYTHONPATH NANOVLLM_DSA_QUERY_ONLY_BACKEND=current NANOVLLM_MAX_GEN_TOKENS=16 NANOVLLM_PROMPT_LENGTHS=11000,11100,11200,11300,11400,11500,11600,11700,11800,11900,12000,12100,12200,12300,12400,12500 python3 example/test.py
```

运行推理（组图，走 DSA 小流水 TorchAir 路径）：

```
PYTHONPATH=$PWD:$PYTHONPATH NANOVLLM_DSA_QUERY_ONLY_BACKEND=torchair NANOVLLM_MAX_GEN_TOKENS=16 NANOVLLM_PROMPT_LENGTHS=11000,11100,11200,11300,11400,11500,11600,11700,11800,11900,12000,12100,12200,12300,12400,12500 python3 example/test.py
```

　

## 运行结果

不组图的结果如下：

```
[root@worker-53232 nano-vllm-ascend-DeepseekV32-dev_dsa_offload_gs_graph]# export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
[root@worker-53232 nano-vllm-ascend-DeepseekV32-dev_dsa_offload_gs_graph]# PYTHONPATH=$PWD:$PYTHONPATH NANOVLLM_DSA_QUERY_ONLY_BACKEND=current NANOVLLM_MAX_GEN_TOKENS=16 NANOVLLM_PROMPT_LENGTHS=11000,11100,11200,11300,11400,11500,11600,11700,11800,11900,12000,12100,12200,12300,12400,12500 python3 example/test.py
`torch_dtype` is deprecated! Use `dtype` instead!
INFO 06-09 09:29:38,588 llm_engine.py:57] config: Config(model=/mnt/models/Deepseek-V3.2-Pruned-95B-BF/, max_num_prefill_seqs_per_step=1, max_num_decode_seqs_per_step=256, max_model_len=65536, tensor_parallel_size=8, enable_expert_parallel=True, enforce_eager=True, eos=1, kvcache_block_size=128, num_kvcache_blocks=500, num_index_cache_blocks=2000, num_hbm_kvcache_blocks=500, num_dram_kvcache_blocks=2000, dsa_offload_max_sparse_tokens=-1, hccl_port=28000, trust_remote_code=True, dsa_offload_pool_capacity=256, hf_config=DeepseekV32Config(...))
`torch_dtype` is deprecated! Use `dtype` instead!
`torch_dtype` is deprecated! Use `dtype` instead!
`torch_dtype` is deprecated! Use `dtype` instead!
`torch_dtype` is deprecated! Use `dtype` instead!
`torch_dtype` is deprecated! Use `dtype` instead!
`torch_dtype` is deprecated! Use `dtype` instead!
`torch_dtype` is deprecated! Use `dtype` instead!
INFO 06-09 09:31:56,569 model_runner.py:191] Using explicit DSA cache blocks: hbm=500, dram=2000, index=2000, max_sparse_tokens=2048
INFO 06-09 09:31:56,570 model_runner.py:205] Single HBM KV Block Size: 8.58 MB
INFO 06-09 09:31:56,570 model_runner.py:214] DeepSeek CKV cache allocated successfully shape: (61, 500, 128, 1, 512)
INFO 06-09 09:31:56,570 model_runner.py:214] DeepSeek KPE cache allocated successfully shape: (61, 500, 128, 1, 64)
INFO 06-09 09:31:56,570 model_runner.py:214] DeepSeek index cache allocated successfully shape: (61, 2000, 128, 1, 128)
INFO 06-09 09:31:56,570 model_runner.py:214] DeepSeek DRAM CKV cache allocated successfully shape: (61, 2000, 128, 1, 512)
INFO 06-09 09:31:56,570 model_runner.py:214] DeepSeek DRAM KPE cache allocated successfully shape: (61, 2000, 128, 1, 64)
INFO 06-09 09:31:56,570 model_runner.py:214] DeepSeek gather selection status allocated successfully shape: (256, 1, 1, 2049)
INFO 06-09 09:31:57,222 model_runner.py:191] Using explicit DSA cache blocks: hbm=500, dram=2000, index=2000, max_sparse_tokens=2048
INFO 06-09 09:31:57,222 model_runner.py:205] Single HBM KV Block Size: 8.58 MB
INFO 06-09 09:31:57,222 model_runner.py:214] DeepSeek CKV cache allocated successfully shape: (61, 500, 128, 1, 512)
INFO 06-09 09:31:57,222 model_runner.py:214] DeepSeek KPE cache allocated successfully shape: (61, 500, 128, 1, 64)
INFO 06-09 09:31:57,222 model_runner.py:214] DeepSeek index cache allocated successfully shape: (61, 2000, 128, 1, 128)
INFO 06-09 09:31:57,222 model_runner.py:214] DeepSeek DRAM CKV cache allocated successfully shape: (61, 2000, 128, 1, 512)
INFO 06-09 09:31:57,222 model_runner.py:214] DeepSeek DRAM KPE cache allocated successfully shape: (61, 2000, 128, 1, 64)
INFO 06-09 09:31:57,222 model_runner.py:214] DeepSeek gather selection status allocated successfully shape: (256, 1, 1, 2049)
INFO 06-09 09:31:57,437 model_runner.py:62] config: Config(model=/mnt/models/Deepseek-V3.2-Pruned-95B-BF/, max_num_prefill_seqs_per_step=1, max_num_decode_seqs_per_step=256, max_model_len=65536, tensor_parallel_size=8, enable_expert_parallel=True, enforce_eager=True, eos=1, kvcache_block_size=128, num_kvcache_blocks=500, num_index_cache_blocks=2000, num_hbm_kvcache_blocks=500, num_dram_kvcache_blocks=2000, dsa_offload_max_sparse_tokens=2048, hccl_port=28000, trust_remote_code=True, dsa_offload_pool_capacity=256, hf_config=DeepseekV32Config(...))
INFO 06-09 09:31:58,052 model_runner.py:62] config: Config(model=/mnt/models/Deepseek-V3.2-Pruned-95B-BF/, max_num_prefill_seqs_per_step=1, max_num_decode_seqs_per_step=256, max_model_len=65536, tensor_parallel_size=8, enable_expert_parallel=True, enforce_eager=True, eos=1, kvcache_block_size=128, num_kvcache_blocks=500, num_index_cache_blocks=2000, num_hbm_kvcache_blocks=500, num_dram_kvcache_blocks=2000, dsa_offload_max_sparse_tokens=2048, hccl_port=28000, trust_remote_code=True, dsa_offload_pool_capacity=256, hf_config=DeepseekV32Config(...))
INFO 06-09 09:31:59,154 model_runner.py:191] Using explicit DSA cache blocks: hbm=500, dram=2000, index=2000, max_sparse_tokens=2048
INFO 06-09 09:31:59,154 model_runner.py:205] Single HBM KV Block Size: 8.58 MB
INFO 06-09 09:31:59,154 model_runner.py:214] DeepSeek CKV cache allocated successfully shape: (61, 500, 128, 1, 512)
INFO 06-09 09:31:59,154 model_runner.py:214] DeepSeek KPE cache allocated successfully shape: (61, 500, 128, 1, 64)
INFO 06-09 09:31:59,154 model_runner.py:214] DeepSeek index cache allocated successfully shape: (61, 2000, 128, 1, 128)
INFO 06-09 09:31:59,154 model_runner.py:214] DeepSeek DRAM CKV cache allocated successfully shape: (61, 2000, 128, 1, 512)
INFO 06-09 09:31:59,154 model_runner.py:214] DeepSeek DRAM KPE cache allocated successfully shape: (61, 2000, 128, 1, 64)
INFO 06-09 09:31:59,154 model_runner.py:214] DeepSeek gather selection status allocated successfully shape: (256, 1, 1, 2049)
INFO 06-09 09:32:00,034 model_runner.py:62] config: Config(model=/mnt/models/Deepseek-V3.2-Pruned-95B-BF/, max_num_prefill_seqs_per_step=1, max_num_decode_seqs_per_step=256, max_model_len=65536, tensor_parallel_size=8, enable_expert_parallel=True, enforce_eager=True, eos=1, kvcache_block_size=128, num_kvcache_blocks=500, num_index_cache_blocks=2000, num_hbm_kvcache_blocks=500, num_dram_kvcache_blocks=2000, dsa_offload_max_sparse_tokens=2048, hccl_port=28000, trust_remote_code=True, dsa_offload_pool_capacity=256, hf_config=DeepseekV32Config(...))
INFO 06-09 09:32:08,121 model_runner.py:191] Using explicit DSA cache blocks: hbm=500, dram=2000, index=2000, max_sparse_tokens=2048
INFO 06-09 09:32:08,122 model_runner.py:205] Single HBM KV Block Size: 8.58 MB
INFO 06-09 09:32:08,122 model_runner.py:214] DeepSeek CKV cache allocated successfully shape: (61, 500, 128, 1, 512)
INFO 06-09 09:32:08,122 model_runner.py:214] DeepSeek KPE cache allocated successfully shape: (61, 500, 128, 1, 64)
INFO 06-09 09:32:08,122 model_runner.py:214] DeepSeek index cache allocated successfully shape: (61, 2000, 128, 1, 128)
INFO 06-09 09:32:08,122 model_runner.py:214] DeepSeek DRAM CKV cache allocated successfully shape: (61, 2000, 128, 1, 512)
INFO 06-09 09:32:08,122 model_runner.py:214] DeepSeek DRAM KPE cache allocated successfully shape: (61, 2000, 128, 1, 64)
INFO 06-09 09:32:08,122 model_runner.py:214] DeepSeek gather selection status allocated successfully shape: (256, 1, 1, 2049)
INFO 06-09 09:32:08,933 model_runner.py:62] config: Config(model=/mnt/models/Deepseek-V3.2-Pruned-95B-BF/, max_num_prefill_seqs_per_step=1, max_num_decode_seqs_per_step=256, max_model_len=65536, tensor_parallel_size=8, enable_expert_parallel=True, enforce_eager=True, eos=1, kvcache_block_size=128, num_kvcache_blocks=500, num_index_cache_blocks=2000, num_hbm_kvcache_blocks=500, num_dram_kvcache_blocks=2000, dsa_offload_max_sparse_tokens=2048, hccl_port=28000, trust_remote_code=True, dsa_offload_pool_capacity=256, hf_config=DeepseekV32Config(...))
INFO 06-09 09:32:12,077 model_runner.py:191] Using explicit DSA cache blocks: hbm=500, dram=2000, index=2000, max_sparse_tokens=2048
INFO 06-09 09:32:12,077 model_runner.py:205] Single HBM KV Block Size: 8.58 MB
INFO 06-09 09:32:12,077 model_runner.py:214] DeepSeek CKV cache allocated successfully shape: (61, 500, 128, 1, 512)
INFO 06-09 09:32:12,078 model_runner.py:214] DeepSeek KPE cache allocated successfully shape: (61, 500, 128, 1, 64)
INFO 06-09 09:32:12,078 model_runner.py:214] DeepSeek index cache allocated successfully shape: (61, 2000, 128, 1, 128)
INFO 06-09 09:32:12,078 model_runner.py:214] DeepSeek DRAM CKV cache allocated successfully shape: (61, 2000, 128, 1, 512)
INFO 06-09 09:32:12,078 model_runner.py:214] DeepSeek DRAM KPE cache allocated successfully shape: (61, 2000, 128, 1, 64)
INFO 06-09 09:32:12,078 model_runner.py:214] DeepSeek gather selection status allocated successfully shape: (256, 1, 1, 2049)
INFO 06-09 09:32:12,621 model_runner.py:191] Using explicit DSA cache blocks: hbm=500, dram=2000, index=2000, max_sparse_tokens=2048
INFO 06-09 09:32:12,622 model_runner.py:205] Single HBM KV Block Size: 8.58 MB
INFO 06-09 09:32:12,622 model_runner.py:214] DeepSeek CKV cache allocated successfully shape: (61, 500, 128, 1, 512)
INFO 06-09 09:32:12,622 model_runner.py:214] DeepSeek KPE cache allocated successfully shape: (61, 500, 128, 1, 64)
INFO 06-09 09:32:12,622 model_runner.py:214] DeepSeek index cache allocated successfully shape: (61, 2000, 128, 1, 128)
INFO 06-09 09:32:12,622 model_runner.py:214] DeepSeek DRAM CKV cache allocated successfully shape: (61, 2000, 128, 1, 512)
INFO 06-09 09:32:12,622 model_runner.py:214] DeepSeek DRAM KPE cache allocated successfully shape: (61, 2000, 128, 1, 64)
INFO 06-09 09:32:12,622 model_runner.py:214] DeepSeek gather selection status allocated successfully shape: (256, 1, 1, 2049)
INFO 06-09 09:32:12,959 model_runner.py:62] config: Config(model=/mnt/models/Deepseek-V3.2-Pruned-95B-BF/, max_num_prefill_seqs_per_step=1, max_num_decode_seqs_per_step=256, max_model_len=65536, tensor_parallel_size=8, enable_expert_parallel=True, enforce_eager=True, eos=1, kvcache_block_size=128, num_kvcache_blocks=500, num_index_cache_blocks=2000, num_hbm_kvcache_blocks=500, num_dram_kvcache_blocks=2000, dsa_offload_max_sparse_tokens=2048, hccl_port=28000, trust_remote_code=True, dsa_offload_pool_capacity=256, hf_config=DeepseekV32Config(...))
INFO 06-09 09:32:13,517 model_runner.py:62] config: Config(model=/mnt/models/Deepseek-V3.2-Pruned-95B-BF/, max_num_prefill_seqs_per_step=1, max_num_decode_seqs_per_step=256, max_model_len=65536, tensor_parallel_size=8, enable_expert_parallel=True, enforce_eager=True, eos=1, kvcache_block_size=128, num_kvcache_blocks=500, num_index_cache_blocks=2000, num_hbm_kvcache_blocks=500, num_dram_kvcache_blocks=2000, dsa_offload_max_sparse_tokens=2048, hccl_port=28000, trust_remote_code=True, dsa_offload_pool_capacity=256, hf_config=DeepseekV32Config(...))
INFO 06-09 09:32:14,605 model_runner.py:191] Using explicit DSA cache blocks: hbm=500, dram=2000, index=2000, max_sparse_tokens=2048
INFO 06-09 09:32:14,606 model_runner.py:205] Single HBM KV Block Size: 8.58 MB
INFO 06-09 09:32:14,606 model_runner.py:214] DeepSeek CKV cache allocated successfully shape: (61, 500, 128, 1, 512)
INFO 06-09 09:32:14,606 model_runner.py:214] DeepSeek KPE cache allocated successfully shape: (61, 500, 128, 1, 64)
INFO 06-09 09:32:14,606 model_runner.py:214] DeepSeek index cache allocated successfully shape: (61, 2000, 128, 1, 128)
INFO 06-09 09:32:14,606 model_runner.py:214] DeepSeek DRAM CKV cache allocated successfully shape: (61, 2000, 128, 1, 512)
INFO 06-09 09:32:14,606 model_runner.py:214] DeepSeek DRAM KPE cache allocated successfully shape: (61, 2000, 128, 1, 64)
INFO 06-09 09:32:14,606 model_runner.py:214] DeepSeek gather selection status allocated successfully shape: (256, 1, 1, 2049)
INFO 06-09 09:32:15,426 model_runner.py:62] config: Config(model=/mnt/models/Deepseek-V3.2-Pruned-95B-BF/, max_num_prefill_seqs_per_step=1, max_num_decode_seqs_per_step=256, max_model_len=65536, tensor_parallel_size=8, enable_expert_parallel=True, enforce_eager=True, eos=1, kvcache_block_size=128, num_kvcache_blocks=500, num_index_cache_blocks=2000, num_hbm_kvcache_blocks=500, num_dram_kvcache_blocks=2000, dsa_offload_max_sparse_tokens=2048, hccl_port=28000, trust_remote_code=True, dsa_offload_pool_capacity=256, hf_config=DeepseekV32Config(...))
/usr/local/python3.11.14/lib/python3.11/site-packages/torch/distributed/distributed_c10d.py:4876: UserWarning: barrier(): using the device under current context. You can specify `device_id` in `init_process_group` to mute this warning.
  warnings.warn(  # warn only once
INFO 06-09 09:32:38,959 model_runner.py:191] Using explicit DSA cache blocks: hbm=500, dram=2000, index=2000, max_sparse_tokens=2048
INFO 06-09 09:32:38,959 model_runner.py:205] Single HBM KV Block Size: 8.58 MB
INFO 06-09 09:32:38,959 model_runner.py:214] DeepSeek CKV cache allocated successfully shape: (61, 500, 128, 1, 512)
INFO 06-09 09:32:38,959 model_runner.py:214] DeepSeek KPE cache allocated successfully shape: (61, 500, 128, 1, 64)
INFO 06-09 09:32:38,959 model_runner.py:214] DeepSeek index cache allocated successfully shape: (61, 2000, 128, 1, 128)
INFO 06-09 09:32:38,960 model_runner.py:214] DeepSeek DRAM CKV cache allocated successfully shape: (61, 2000, 128, 1, 512)
INFO 06-09 09:32:38,960 model_runner.py:214] DeepSeek DRAM KPE cache allocated successfully shape: (61, 2000, 128, 1, 64)
INFO 06-09 09:32:38,960 model_runner.py:214] DeepSeek gather selection status allocated successfully shape: (256, 1, 1, 2049)
INFO 06-09 09:32:39,878 model_runner.py:62] config: Config(model=/mnt/models/Deepseek-V3.2-Pruned-95B-BF/, max_num_prefill_seqs_per_step=1, max_num_decode_seqs_per_step=256, max_model_len=65536, tensor_parallel_size=8, enable_expert_parallel=True, enforce_eager=True, eos=1, kvcache_block_size=128, num_kvcache_blocks=500, num_index_cache_blocks=2000, num_hbm_kvcache_blocks=500, num_dram_kvcache_blocks=2000, dsa_offload_max_sparse_tokens=2048, hccl_port=28000, trust_remote_code=True, dsa_offload_pool_capacity=256, hf_config=DeepseekV32Config(...))
test config: num_prompts=16, prompt_min=11000, prompt_max=12500, max_model_len=65536, max_num_prefill_seqs_per_step=1, max_num_decode_seqs_per_step=256, max_gen_tokens=16, prompt_style=meaningful, meaningful_base_tokens=10000
prompt plan:
  prompt 1: target_len=11000, full_blocks=85, sparse_blocks=16, release_blocks=69
  prompt 2: target_len=11100, full_blocks=86, sparse_blocks=16, release_blocks=70
  prompt 3: target_len=11200, full_blocks=87, sparse_blocks=16, release_blocks=71
  prompt 4: target_len=11300, full_blocks=88, sparse_blocks=16, release_blocks=72
  prompt 5: target_len=11400, full_blocks=89, sparse_blocks=16, release_blocks=73
  prompt 6: target_len=11500, full_blocks=89, sparse_blocks=16, release_blocks=73
  prompt 7: target_len=11600, full_blocks=90, sparse_blocks=16, release_blocks=74
  prompt 8: target_len=11700, full_blocks=91, sparse_blocks=16, release_blocks=75
  prompt 9: target_len=11800, full_blocks=92, sparse_blocks=16, release_blocks=76
  prompt 10: target_len=11900, full_blocks=92, sparse_blocks=16, release_blocks=76
  prompt 11: target_len=12000, full_blocks=93, sparse_blocks=16, release_blocks=77
  prompt 12: target_len=12100, full_blocks=94, sparse_blocks=16, release_blocks=78
  prompt 13: target_len=12200, full_blocks=95, sparse_blocks=16, release_blocks=79
  prompt 14: target_len=12300, full_blocks=96, sparse_blocks=16, release_blocks=80
  prompt 15: target_len=12400, full_blocks=96, sparse_blocks=16, release_blocks=80
  prompt 16: target_len=12500, full_blocks=97, sparse_blocks=16, release_blocks=81
prompt 1 token_len=11000 first_ids=[455, 1957, 15986, 515, 9306, 201, 117495, 97721, 25923, 1754, 78019, 97721, 9693, 10461, 270, 28865]
prompt 2 token_len=11100 first_ids=[6006, 412, 260, 15292, 2831, 362, 201, 30230, 29645, 16, 56518, 65021, 9026, 270, 15292, 86508]
prompt 3 token_len=11200 first_ids=[4951, 14, 40685, 57, 8840, 73857, 4914, 15492, 9259, 59, 5329, 3973, 989, 11013, 57, 8840]
prompt 4 token_len=11300 first_ids=[8398, 260, 37754, 11624, 7155, 59344, 201, 17774, 359, 25923, 24570, 270, 4087, 25314, 16, 455]
prompt 5 token_len=11400 first_ids=[295, 270, 2662, 10281, 16, 455, 201, 74304, 4869, 3920, 270, 10346, 91355, 14, 3920, 270]
prompt 6 token_len=11500 first_ids=[2205, 47156, 14, 31434, 40968, 14, 305, 260, 45076, 4090, 16, 455, 1957, 15986, 515, 9306]
prompt 7 token_len=11600 first_ids=[201, 114830, 2599, 4868, 28, 455, 25686, 2709, 44674, 2237, 2775, 6006, 412, 260, 15292, 2831]
prompt 8 token_len=11700 first_ids=[4868, 28, 40685, 57, 8840, 73857, 15, 13455, 17640, 5770, 72546, 4951, 14, 40685, 57, 8840]
prompt 9 token_len=11800 first_ids=[387, 6057, 1066, 339, 14446, 110865, 4868, 28, 334, 3859, 52717, 8398, 260, 37754, 11624, 7155]
prompt 10 token_len=11900 first_ids=[25314, 339, 1124, 64276, 4868, 28, 455, 4087, 25314, 515, 5607, 295, 270, 2662, 10281, 16]
prompt 11 token_len=12000 first_ids=[28, 455, 15389, 1539, 7379, 260, 1957, 18658, 15986, 418, 201, 2205, 47156, 14, 31434, 40968]
prompt 12 token_len=12100 first_ids=[15986, 2329, 538, 270, 201, 85, 6057, 4087, 25314, 9575, 603, 201, 114830, 2599, 4868, 28]
prompt 13 token_len=12200 first_ids=[25923, 4365, 304, 270, 201, 41551, 25314, 4245, 339, 36892, 279, 4868, 28, 40685, 57, 8840]
prompt 14 token_len=12300 first_ids=[56518, 65021, 14023, 270, 9575, 14, 305, 71595, 83386, 201, 10499, 387, 6057, 1066, 339, 14446]
prompt 15 token_len=12400 first_ids=[515, 270, 6690, 18658, 201, 38206, 367, 28865, 2184, 270, 4087, 25314, 339, 1124, 64276, 4868]
prompt 16 token_len=12500 first_ids=[4332, 13256, 201, 265, 270, 23616, 6403, 339, 2167, 18486, 4868, 28, 455, 15389, 1539, 7379]
llm.generate 16 start ...
[step   1 Prefill] bsz=1, num_tokens=11001, HBM_KV=17/500, DRAM_KV=85/2000, HBM_INDEX=86/2000, TTFT=13.6445 sec, TPS=806.26 tok/s
[step   2 Prefill] bsz=1, num_tokens=11101, HBM_KV=34/500, DRAM_KV=171/2000, HBM_INDEX=173/2000, TTFT=6.0875 sec, TPS=1823.56 tok/s
[step   3 Prefill] bsz=1, num_tokens=11201, HBM_KV=51/500, DRAM_KV=258/2000, HBM_INDEX=261/2000, TTFT=5.0113 sec, TPS=2235.15 tok/s
[step   4 Prefill] bsz=1, num_tokens=11301, HBM_KV=68/500, DRAM_KV=346/2000, HBM_INDEX=350/2000, TTFT=6.3466 sec, TPS=1780.63 tok/s
[step   5 Prefill] bsz=1, num_tokens=11401, HBM_KV=85/500, DRAM_KV=435/2000, HBM_INDEX=440/2000, TTFT=3.2910 sec, TPS=3464.32 tok/s
[step   6 Prefill] bsz=1, num_tokens=11501, HBM_KV=102/500, DRAM_KV=524/2000, HBM_INDEX=530/2000, TTFT=3.3589 sec, TPS=3423.99 tok/s
[step   7 Prefill] bsz=1, num_tokens=11601, HBM_KV=119/500, DRAM_KV=614/2000, HBM_INDEX=621/2000, TTFT=3.2109 sec, TPS=3612.98 tok/s
[step   8 Prefill] bsz=1, num_tokens=11701, HBM_KV=136/500, DRAM_KV=705/2000, HBM_INDEX=713/2000, TTFT=3.4165 sec, TPS=3424.81 tok/s
[step   9 Prefill] bsz=1, num_tokens=11801, HBM_KV=153/500, DRAM_KV=797/2000, HBM_INDEX=806/2000, TTFT=3.3324 sec, TPS=3541.34 tok/s
[step  10 Prefill] bsz=1, num_tokens=11901, HBM_KV=170/500, DRAM_KV=889/2000, HBM_INDEX=899/2000, TTFT=3.5957 sec, TPS=3309.83 tok/s
[step  11 Prefill] bsz=1, num_tokens=12001, HBM_KV=187/500, DRAM_KV=982/2000, HBM_INDEX=993/2000, TTFT=3.4128 sec, TPS=3516.47 tok/s
[step  12 Prefill] bsz=1, num_tokens=12101, HBM_KV=204/500, DRAM_KV=1076/2000, HBM_INDEX=1088/2000, TTFT=3.3812 sec, TPS=3578.95 tok/s
[step  13 Prefill] bsz=1, num_tokens=12201, HBM_KV=221/500, DRAM_KV=1171/2000, HBM_INDEX=1184/2000, TTFT=3.5117 sec, TPS=3474.37 tok/s
[step  14 Prefill] bsz=1, num_tokens=12301, HBM_KV=238/500, DRAM_KV=1267/2000, HBM_INDEX=1281/2000, TTFT=3.4380 sec, TPS=3577.96 tok/s
[step  15 Prefill] bsz=1, num_tokens=12401, HBM_KV=255/500, DRAM_KV=1363/2000, HBM_INDEX=1378/2000, TTFT=3.4639 sec, TPS=3580.03 tok/s
[step  16 Prefill] bsz=1, num_tokens=12501, HBM_KV=272/500, DRAM_KV=1460/2000, HBM_INDEX=1476/2000, TTFT=3.5386 sec, TPS=3532.76 tok/s
[step  17  Decode] bsz=16, num_tokens=16, HBM_KV=272/500, DRAM_KV=1460/2000, HBM_INDEX=1476/2000, TPOT=0.3563 sec, TPS=44.91 tok/s
[step  18  Decode] bsz=16, num_tokens=16, HBM_KV=272/500, DRAM_KV=1460/2000, HBM_INDEX=1476/2000, TPOT=0.2922 sec, TPS=54.75 tok/s
[step  19  Decode] bsz=16, num_tokens=16, HBM_KV=272/500, DRAM_KV=1460/2000, HBM_INDEX=1476/2000, TPOT=0.1797 sec, TPS=89.06 tok/s
[step  20  Decode] bsz=16, num_tokens=16, HBM_KV=272/500, DRAM_KV=1460/2000, HBM_INDEX=1476/2000, TPOT=0.1864 sec, TPS=85.82 tok/s
[step  21  Decode] bsz=16, num_tokens=16, HBM_KV=273/500, DRAM_KV=1460/2000, HBM_INDEX=1477/2000, TPOT=0.1864 sec, TPS=85.86 tok/s
[step  22  Decode] bsz=16, num_tokens=16, HBM_KV=273/500, DRAM_KV=1460/2000, HBM_INDEX=1477/2000, TPOT=0.1865 sec, TPS=85.79 tok/s
[step  23  Decode] bsz=16, num_tokens=16, HBM_KV=273/500, DRAM_KV=1460/2000, HBM_INDEX=1477/2000, TPOT=0.1861 sec, TPS=85.99 tok/s
[step  24  Decode] bsz=16, num_tokens=16, HBM_KV=273/500, DRAM_KV=1460/2000, HBM_INDEX=1477/2000, TPOT=0.1883 sec, TPS=84.95 tok/s
[step  25  Decode] bsz=16, num_tokens=16, HBM_KV=274/500, DRAM_KV=1460/2000, HBM_INDEX=1478/2000, TPOT=0.1743 sec, TPS=91.81 tok/s
[step  26  Decode] bsz=16, num_tokens=16, HBM_KV=274/500, DRAM_KV=1460/2000, HBM_INDEX=1478/2000, TPOT=0.1790 sec, TPS=89.36 tok/s
[step  27  Decode] bsz=16, num_tokens=16, HBM_KV=274/500, DRAM_KV=1460/2000, HBM_INDEX=1478/2000, TPOT=0.1920 sec, TPS=83.33 tok/s
[step  28  Decode] bsz=16, num_tokens=16, HBM_KV=274/500, DRAM_KV=1460/2000, HBM_INDEX=1478/2000, TPOT=0.1930 sec, TPS=82.90 tok/s
[step  29  Decode] bsz=16, num_tokens=16, HBM_KV=274/500, DRAM_KV=1460/2000, HBM_INDEX=1478/2000, TPOT=0.1914 sec, TPS=83.60 tok/s
[step  30  Decode] bsz=16, num_tokens=16, HBM_KV=274/500, DRAM_KV=1460/2000, HBM_INDEX=1478/2000, TPOT=0.1784 sec, TPS=89.66 tok/s
[step  31  Decode] bsz=16, num_tokens=16, HBM_KV=0/500, DRAM_KV=0/2000, HBM_INDEX=0/2000, TPOT=0.1801 sec, TPS=88.83 tok/s
llm.generate 16 requests in 31 steps, e2e latency = 75.10 sec
    prefill steps = 16, prefill tokens = 188016, total prefill time = 72.0416 sec, prefill TPS = 2609.83 tok/s
    decode steps = 15, decode tokens = 240, total decode time = 3.0502 sec, decode mean TPOT = 0.2033 sec, decode TPS = 78.68 tok/s
    e2e input TPS = 2503.48 tok/s, e2e output TPS = 3.41 tok/s
    TTFT/TPOT are per-step request latencies printed above.
prompt_len: 11000
prompt    : <meaningful long-QA suffix prompt 1: target_len=11000, tail='g. 3. The signed final inspection notes identify the temporary bridge as Hawthorn Bridge. 4. The later Willow Bridge margin note is explicitly marked as a mistake. Question: What was the name of the temporary bridge used during the final i
response  : '\nRead<｜begin▁of▁sentence｜><｜begin▁of▁sentence｜><｜begin▁of▁sentence｜><｜begin▁of▁sentence｜><｜begin▁of▁sentence｜><｜begin▁of▁sentence｜><｜begin▁of▁sentence｜><｜begin▁of▁sentence｜><｜begin▁of▁sentence｜><｜begin▁of▁sentence｜><｜begin▁of▁sentence｜><｜begin▁of▁sentence｜><｜begin▁of▁sentence｜><｜begin▁of▁sentence｜>'
token_ids : [201, 8158, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

prompt_len: 11100
prompt    : <meaningful long-QA suffix prompt 2: target_len=11100, tail='g. 3. The signed final inspection notes identify the temporary bridge as Hawthorn Bridge. 4. The later Willow Bridge margin note is explicitly marked as a mistake. Question: What was the name of the temporary bridge used during the final i
response  : '\nThe<｜begin▁of▁sentence｜><｜begin▁of▁sentence｜><｜begin▁of▁sentence｜><｜begin▁of▁sentence｜><｜begin▁of▁sentence｜><｜begin▁of▁sentence｜><｜begin▁of▁sentence｜><｜begin▁of▁sentence｜><｜begin▁of▁sentence｜><｜begin▁of▁sentence｜><｜begin▁of▁sentence｜><｜begin▁of▁sentence｜><｜begin▁of▁sentence｜><｜begin▁of▁sentence｜>'
token_ids : [201, 671, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

prompt_len: 11200
prompt    : <meaningful long-QA suffix prompt 3: target_len=11200, tail='g. 3. The signed final inspection notes identify the temporary bridge as Hawthorn Bridge. 4. The later Willow Bridge margin note is explicitly marked as a mistake. Question: What was the name of the temporary bridge used during the final i
response  : " \\'<｜begin▁of▁sentence｜><｜begin▁of▁sentence｜><｜begin▁of▁sentence｜><｜begin▁of▁sentence｜><｜begin▁of▁sentence｜><｜begin▁of▁sentence｜><｜begin▁of▁sentence｜><｜begin▁of▁sentence｜><｜begin▁of▁sentence｜><｜begin▁of▁sentence｜><｜begin▁of▁sentence｜><｜begin▁of▁sentence｜><｜begin▁of▁sentence｜><｜begin▁of▁sentence｜>"
token_ids : [874, 9, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

prompt_len: 11300
prompt    : <meaningful long-QA suffix prompt 4: target_len=11300, tail='g. 3. The signed final inspection notes identify the temporary bridge as Hawthorn Bridge. 4. The later Willow Bridge margin note is explicitly marked as a mistake. Question: What was the name of the temporary bridge used during the final i
response  : ' <｜end▁of▁sentence｜><｜begin▁of▁sentence｜><｜begin▁of▁sentence｜><｜begin▁of▁sentence｜><｜begin▁of▁sentence｜><｜begin▁of▁sentence｜><｜begin▁of▁sentence｜><｜begin▁of▁sentence｜><｜begin▁of▁sentence｜><｜begin▁of▁sentence｜><｜begin▁of▁sentence｜><｜begin▁of▁sentence｜><｜begin▁of▁sentence｜><｜begin▁of▁sentence｜><｜begin▁of▁sentence｜>'
token_ids : [223, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

prompt_len: 11400
prompt    : <meaningful long-QA suffix prompt 5: target_len=11400, tail='g. 3. The signed final inspection notes identify the temporary bridge as Hawthorn Bridge. 4. The later Willow Bridge margin note is explicitly marked as a mistake. Question: What was the name of the temporary bridge used during the final i
response  : " 'The<｜begin▁of▁sentence｜><｜begin▁of▁sentence｜><｜begin▁of▁sentence｜><｜begin▁of▁sentence｜><｜begin▁of▁sentence｜><｜begin▁of▁sentence｜><｜begin▁of▁sentence｜><｜begin▁of▁sentence｜><｜begin▁of▁sentence｜><｜begin▁of▁sentence｜><｜begin▁of▁sentence｜><｜begin▁of▁sentence｜><｜begin▁of▁sentence｜><｜begin▁of▁sentence｜>"
token_ids : [905, 671, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

prompt_len: 11500
prompt    : <meaningful long-QA suffix prompt 6: target_len=11500, tail='g. 3. The signed final inspection notes identify the temporary bridge as Hawthorn Bridge. 4. The later Willow Bridge margin note is explicitly marked as a mistake. Question: What was the name of the temporary bridge used during the final i
response  : '<｜end▁of▁sentence｜><｜end▁of▁sentence｜><｜begin▁of▁sentence｜><｜begin▁of▁sentence｜><｜begin▁of▁sentence｜><｜begin▁of▁sentence｜><｜begin▁of▁sentence｜><｜begin▁of▁sentence｜><｜begin▁of▁sentence｜><｜begin▁of▁sentence｜><｜begin▁of▁sentence｜><｜begin▁of▁sentence｜><｜begin▁of▁sentence｜><｜begin▁of▁sentence｜><｜begin▁of▁sentence｜><｜begin▁of▁sentence｜>'
token_ids : [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

prompt_len: 11600
prompt    : <meaningful long-QA suffix prompt 7: target_len=11600, tail='g. 3. The signed final inspection notes identify the temporary bridge as Hawthorn Bridge. 4. The later Willow Bridge margin note is explicitly marked as a mistake. Question: What was the name of the temporary bridge used during the final i
response  : '\tThe<｜begin▁of▁sentence｜><｜begin▁of▁sentence｜><｜begin▁of▁sentence｜><｜begin▁of▁sentence｜><｜begin▁of▁sentence｜><｜begin▁of▁sentence｜><｜begin▁of▁sentence｜><｜begin▁of▁sentence｜><｜begin▁of▁sentence｜><｜begin▁of▁sentence｜><｜begin▁of▁sentence｜><｜begin▁of▁sentence｜><｜begin▁of▁sentence｜><｜begin▁of▁sentence｜>'
token_ids : [200, 671, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

prompt_len: 11700
prompt    : <meaningful long-QA suffix prompt 8: target_len=11700, tail='g. 3. The signed final inspection notes identify the temporary bridge as Hawthorn Bridge. 4. The later Willow Bridge margin note is explicitly marked as a mistake. Question: What was the name of the temporary bridge used during the final i
response  : '\nRead<｜begin▁of▁sentence｜><｜begin▁of▁sentence｜><｜begin▁of▁sentence｜><｜begin▁of▁sentence｜><｜begin▁of▁sentence｜><｜begin▁of▁sentence｜><｜begin▁of▁sentence｜><｜begin▁of▁sentence｜><｜begin▁of▁sentence｜><｜begin▁of▁sentence｜><｜begin▁of▁sentence｜><｜begin▁of▁sentence｜><｜begin▁of▁sentence｜><｜begin▁of▁sentence｜>'
token_ids : [201, 8158, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

prompt_len: 11800
prompt    : <meaningful long-QA suffix prompt 9: target_len=11800, tail='g. 3. The signed final inspection notes identify the temporary bridge as Hawthorn Bridge. 4. The later Willow Bridge margin note is explicitly marked as a mistake. Question: What was the name of the temporary bridge used during the final i
response  : 'The bridge<｜begin▁of▁sentence｜><｜begin▁of▁sentence｜><｜begin▁of▁sentence｜><｜begin▁of▁sentence｜><｜begin▁of▁sentence｜><｜begin▁of▁sentence｜><｜begin▁of▁sentence｜><｜begin▁of▁sentence｜><｜begin▁of▁sentence｜><｜begin▁of▁sentence｜><｜begin▁of▁sentence｜><｜begin▁of▁sentence｜><｜begin▁of▁sentence｜><｜begin▁of▁sentence｜>'
token_ids : [671, 15986, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

prompt_len: 11900
prompt    : <meaningful long-QA suffix prompt 10: target_len=11900, tail='g. 3. The signed final inspection notes identify the temporary bridge as Hawthorn Bridge. 4. The later Willow Bridge margin note is explicitly marked as a mistake. Question: What was the name of the temporary bridge used during the final
response  : '\n```\n\n<｜begin▁of▁sentence｜><｜begin▁of▁sentence｜><｜begin▁of▁sentence｜><｜begin▁of▁sentence｜><｜begin▁of▁sentence｜><｜begin▁of▁sentence｜><｜begin▁of▁sentence｜><｜begin▁of▁sentence｜><｜begin▁of▁sentence｜><｜begin▁of▁sentence｜><｜begin▁of▁sentence｜><｜begin▁of▁sentence｜><｜begin▁of▁sentence｜><｜begin▁of▁sentence｜>'
token_ids : [201, 20759, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

prompt_len: 12000
prompt    : <meaningful long-QA suffix prompt 11: target_len=12000, tail='g. 3. The signed final inspection notes identify the temporary bridge as Hawthorn Bridge. 4. The later Willow Bridge margin note is explicitly marked as a mistake. Question: What was the name of the temporary bridge used during the final
response  : ' <｜end▁of▁sentence｜><｜begin▁of▁sentence｜><｜begin▁of▁sentence｜><｜begin▁of▁sentence｜><｜begin▁of▁sentence｜><｜begin▁of▁sentence｜><｜begin▁of▁sentence｜><｜begin▁of▁sentence｜><｜begin▁of▁sentence｜><｜begin▁of▁sentence｜><｜begin▁of▁sentence｜><｜begin▁of▁sentence｜><｜begin▁of▁sentence｜><｜begin▁of▁sentence｜><｜begin▁of▁sentence｜>'
token_ids : [223, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

prompt_len: 12100
prompt    : <meaningful long-QA suffix prompt 12: target_len=12100, tail='g. 3. The signed final inspection notes identify the temporary bridge as Hawthorn Bridge. 4. The later Willow Bridge margin note is explicitly marked as a mistake. Question: What was the name of the temporary bridge used during the final
response  : '<｜end▁of▁sentence｜>\n<｜begin▁of▁sentence｜><｜begin▁of▁sentence｜><｜begin▁of▁sentence｜><｜begin▁of▁sentence｜><｜begin▁of▁sentence｜><｜begin▁of▁sentence｜><｜begin▁of▁sentence｜><｜begin▁of▁sentence｜><｜begin▁of▁sentence｜><｜begin▁of▁sentence｜><｜begin▁of▁sentence｜><｜begin▁of▁sentence｜><｜begin▁of▁sentence｜><｜begin▁of▁sentence｜>'
token_ids : [1, 201, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

prompt_len: 12200
prompt    : <meaningful long-QA suffix prompt 13: target_len=12200, tail='g. 3. The signed final inspection notes identify the temporary bridge as Hawthorn Bridge. 4. The later Willow Bridge margin note is explicitly marked as a mistake. Question: What was the name of the temporary bridge used during the final
response  : '<｜end▁of▁sentence｜>The<｜begin▁of▁sentence｜><｜begin▁of▁sentence｜><｜begin▁of▁sentence｜><｜begin▁of▁sentence｜><｜begin▁of▁sentence｜><｜begin▁of▁sentence｜><｜begin▁of▁sentence｜><｜begin▁of▁sentence｜><｜begin▁of▁sentence｜><｜begin▁of▁sentence｜><｜begin▁of▁sentence｜><｜begin▁of▁sentence｜><｜begin▁of▁sentence｜><｜begin▁of▁sentence｜>'
token_ids : [1, 671, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

prompt_len: 12300
prompt    : <meaningful long-QA suffix prompt 14: target_len=12300, tail='g. 3. The signed final inspection notes identify the temporary bridge as Hawthorn Bridge. 4. The later Willow Bridge margin note is explicitly marked as a mistake. Question: What was the name of the temporary bridge used during the final
response  : " ',\n<｜begin▁of▁sentence｜><｜begin▁of▁sentence｜><｜begin▁of▁sentence｜><｜begin▁of▁sentence｜><｜begin▁of▁sentence｜><｜begin▁of▁sentence｜><｜begin▁of▁sentence｜><｜begin▁of▁sentence｜><｜begin▁of▁sentence｜><｜begin▁of▁sentence｜><｜begin▁of▁sentence｜><｜begin▁of▁sentence｜><｜begin▁of▁sentence｜><｜begin▁of▁sentence｜>"
token_ids : [905, 989, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

prompt_len: 12400
prompt    : <meaningful long-QA suffix prompt 15: target_len=12400, tail='g. 3. The signed final inspection notes identify the temporary bridge as Hawthorn Bridge. 4. The later Willow Bridge margin note is explicitly marked as a mistake. Question: What was the name of the temporary bridge used during the final
response  : ' <｜end▁of▁sentence｜><｜begin▁of▁sentence｜><｜begin▁of▁sentence｜><｜begin▁of▁sentence｜><｜begin▁of▁sentence｜><｜begin▁of▁sentence｜><｜begin▁of▁sentence｜><｜begin▁of▁sentence｜><｜begin▁of▁sentence｜><｜begin▁of▁sentence｜><｜begin▁of▁sentence｜><｜begin▁of▁sentence｜><｜begin▁of▁sentence｜><｜begin▁of▁sentence｜><｜begin▁of▁sentence｜>'
token_ids : [223, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

prompt_len: 12500
prompt    : <meaningful long-QA suffix prompt 16: target_len=12500, tail='g. 3. The signed final inspection notes identify the temporary bridge as Hawthorn Bridge. 4. The later Willow Bridge margin note is explicitly marked as a mistake. Question: What was the name of the temporary bridge used during the final
response  : ' <｜end▁of▁sentence｜><｜begin▁of▁sentence｜><｜begin▁of▁sentence｜><｜begin▁of▁sentence｜><｜begin▁of▁sentence｜><｜begin▁of▁sentence｜><｜begin▁of▁sentence｜><｜begin▁of▁sentence｜><｜begin▁of▁sentence｜><｜begin▁of▁sentence｜><｜begin▁of▁sentence｜><｜begin▁of▁sentence｜><｜begin▁of▁sentence｜><｜begin▁of▁sentence｜><｜begin▁of▁sentence｜>'
token_ids : [223, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
```

组图的结果如下：

```
[root@worker-53232 nano-vllm-ascend-DeepseekV32-dev_dsa_offload_gs_graph]# PYTHONPATH=$PWD:$PYTHONPATH NANOVLLM_DSA_QUERY_ONLY_BACKEND=torchair  NANOVLLM_MAX_GEN_TOKENS=16 NANOVLLM_PROMPT_LENGTHS=11000,11100,11200,11300,11400,11500,11600,11700,11800,11900,12000,12100,12200,12300,12400,12500 python3 example/test.py
`torch_dtype` is deprecated! Use `dtype` instead!
INFO 06-09 09:46:04,492 llm_engine.py:57] config: Config(model=/mnt/models/Deepseek-V3.2-Pruned-95B-BF/, max_num_prefill_seqs_per_step=1, max_num_decode_seqs_per_step=256, max_model_len=65536, tensor_parallel_size=8, enable_expert_parallel=True, enforce_eager=True, eos=1, kvcache_block_size=128, num_kvcache_blocks=500, num_index_cache_blocks=2000, num_hbm_kvcache_blocks=500, num_dram_kvcache_blocks=2000, dsa_offload_max_sparse_tokens=-1, hccl_port=28000, trust_remote_code=True, dsa_offload_pool_capacity=256, hf_config=DeepseekV32Config(...))
`torch_dtype` is deprecated! Use `dtype` instead!
`torch_dtype` is deprecated! Use `dtype` instead!
`torch_dtype` is deprecated! Use `dtype` instead!
`torch_dtype` is deprecated! Use `dtype` instead!
`torch_dtype` is deprecated! Use `dtype` instead!
`torch_dtype` is deprecated! Use `dtype` instead!
`torch_dtype` is deprecated! Use `dtype` instead!
INFO 06-09 09:48:24,122 model_runner.py:191] Using explicit DSA cache blocks: hbm=500, dram=2000, index=2000, max_sparse_tokens=2048
INFO 06-09 09:48:24,123 model_runner.py:205] Single HBM KV Block Size: 8.58 MB
INFO 06-09 09:48:24,123 model_runner.py:214] DeepSeek CKV cache allocated successfully shape: (61, 500, 128, 1, 512)
INFO 06-09 09:48:24,123 model_runner.py:214] DeepSeek KPE cache allocated successfully shape: (61, 500, 128, 1, 64)
INFO 06-09 09:48:24,123 model_runner.py:214] DeepSeek index cache allocated successfully shape: (61, 2000, 128, 1, 128)
INFO 06-09 09:48:24,123 model_runner.py:214] DeepSeek DRAM CKV cache allocated successfully shape: (61, 2000, 128, 1, 512)
INFO 06-09 09:48:24,123 model_runner.py:214] DeepSeek DRAM KPE cache allocated successfully shape: (61, 2000, 128, 1, 64)
INFO 06-09 09:48:24,123 model_runner.py:214] DeepSeek gather selection status allocated successfully shape: (256, 1, 1, 2049)
INFO 06-09 09:48:24,943 model_runner.py:62] config: Config(model=/mnt/models/Deepseek-V3.2-Pruned-95B-BF/, max_num_prefill_seqs_per_step=1, max_num_decode_seqs_per_step=256, max_model_len=65536, tensor_parallel_size=8, enable_expert_parallel=True, enforce_eager=True, eos=1, kvcache_block_size=128, num_kvcache_blocks=500, num_index_cache_blocks=2000, num_hbm_kvcache_blocks=500, num_dram_kvcache_blocks=2000, dsa_offload_max_sparse_tokens=2048, hccl_port=28000, trust_remote_code=True, dsa_offload_pool_capacity=256, hf_config=DeepseekV32Config(...))
INFO 06-09 09:48:27,914 model_runner.py:191] Using explicit DSA cache blocks: hbm=500, dram=2000, index=2000, max_sparse_tokens=2048
INFO 06-09 09:48:27,915 model_runner.py:205] Single HBM KV Block Size: 8.58 MB
INFO 06-09 09:48:27,915 model_runner.py:214] DeepSeek CKV cache allocated successfully shape: (61, 500, 128, 1, 512)
INFO 06-09 09:48:27,915 model_runner.py:214] DeepSeek KPE cache allocated successfully shape: (61, 500, 128, 1, 64)
INFO 06-09 09:48:27,915 model_runner.py:214] DeepSeek index cache allocated successfully shape: (61, 2000, 128, 1, 128)
INFO 06-09 09:48:27,915 model_runner.py:214] DeepSeek DRAM CKV cache allocated successfully shape: (61, 2000, 128, 1, 512)
INFO 06-09 09:48:27,915 model_runner.py:214] DeepSeek DRAM KPE cache allocated successfully shape: (61, 2000, 128, 1, 64)
INFO 06-09 09:48:27,915 model_runner.py:214] DeepSeek gather selection status allocated successfully shape: (256, 1, 1, 2049)
INFO 06-09 09:48:28,814 model_runner.py:62] config: Config(model=/mnt/models/Deepseek-V3.2-Pruned-95B-BF/, max_num_prefill_seqs_per_step=1, max_num_decode_seqs_per_step=256, max_model_len=65536, tensor_parallel_size=8, enable_expert_parallel=True, enforce_eager=True, eos=1, kvcache_block_size=128, num_kvcache_blocks=500, num_index_cache_blocks=2000, num_hbm_kvcache_blocks=500, num_dram_kvcache_blocks=2000, dsa_offload_max_sparse_tokens=2048, hccl_port=28000, trust_remote_code=True, dsa_offload_pool_capacity=256, hf_config=DeepseekV32Config(...))
INFO 06-09 09:48:36,668 model_runner.py:191] Using explicit DSA cache blocks: hbm=500, dram=2000, index=2000, max_sparse_tokens=2048
INFO 06-09 09:48:36,668 model_runner.py:205] Single HBM KV Block Size: 8.58 MB
INFO 06-09 09:48:36,668 model_runner.py:214] DeepSeek CKV cache allocated successfully shape: (61, 500, 128, 1, 512)
INFO 06-09 09:48:36,668 model_runner.py:214] DeepSeek KPE cache allocated successfully shape: (61, 500, 128, 1, 64)
INFO 06-09 09:48:36,668 model_runner.py:214] DeepSeek index cache allocated successfully shape: (61, 2000, 128, 1, 128)
INFO 06-09 09:48:36,668 model_runner.py:214] DeepSeek DRAM CKV cache allocated successfully shape: (61, 2000, 128, 1, 512)
INFO 06-09 09:48:36,668 model_runner.py:214] DeepSeek DRAM KPE cache allocated successfully shape: (61, 2000, 128, 1, 64)
INFO 06-09 09:48:36,668 model_runner.py:214] DeepSeek gather selection status allocated successfully shape: (256, 1, 1, 2049)
INFO 06-09 09:48:37,300 model_runner.py:191] Using explicit DSA cache blocks: hbm=500, dram=2000, index=2000, max_sparse_tokens=2048
INFO 06-09 09:48:37,300 model_runner.py:205] Single HBM KV Block Size: 8.58 MB
INFO 06-09 09:48:37,301 model_runner.py:214] DeepSeek CKV cache allocated successfully shape: (61, 500, 128, 1, 512)
INFO 06-09 09:48:37,301 model_runner.py:214] DeepSeek KPE cache allocated successfully shape: (61, 500, 128, 1, 64)
INFO 06-09 09:48:37,301 model_runner.py:214] DeepSeek index cache allocated successfully shape: (61, 2000, 128, 1, 128)
INFO 06-09 09:48:37,301 model_runner.py:214] DeepSeek DRAM CKV cache allocated successfully shape: (61, 2000, 128, 1, 512)
INFO 06-09 09:48:37,301 model_runner.py:214] DeepSeek DRAM KPE cache allocated successfully shape: (61, 2000, 128, 1, 64)
INFO 06-09 09:48:37,301 model_runner.py:214] DeepSeek gather selection status allocated successfully shape: (256, 1, 1, 2049)
INFO 06-09 09:48:37,493 model_runner.py:62] config: Config(model=/mnt/models/Deepseek-V3.2-Pruned-95B-BF/, max_num_prefill_seqs_per_step=1, max_num_decode_seqs_per_step=256, max_model_len=65536, tensor_parallel_size=8, enable_expert_parallel=True, enforce_eager=True, eos=1, kvcache_block_size=128, num_kvcache_blocks=500, num_index_cache_blocks=2000, num_hbm_kvcache_blocks=500, num_dram_kvcache_blocks=2000, dsa_offload_max_sparse_tokens=2048, hccl_port=28000, trust_remote_code=True, dsa_offload_pool_capacity=256, hf_config=DeepseekV32Config(...))
INFO 06-09 09:48:38,091 model_runner.py:62] config: Config(model=/mnt/models/Deepseek-V3.2-Pruned-95B-BF/, max_num_prefill_seqs_per_step=1, max_num_decode_seqs_per_step=256, max_model_len=65536, tensor_parallel_size=8, enable_expert_parallel=True, enforce_eager=True, eos=1, kvcache_block_size=128, num_kvcache_blocks=500, num_index_cache_blocks=2000, num_hbm_kvcache_blocks=500, num_dram_kvcache_blocks=2000, dsa_offload_max_sparse_tokens=2048, hccl_port=28000, trust_remote_code=True, dsa_offload_pool_capacity=256, hf_config=DeepseekV32Config(...))
INFO 06-09 09:48:39,096 model_runner.py:191] Using explicit DSA cache blocks: hbm=500, dram=2000, index=2000, max_sparse_tokens=2048
INFO 06-09 09:48:39,096 model_runner.py:205] Single HBM KV Block Size: 8.58 MB
INFO 06-09 09:48:39,096 model_runner.py:214] DeepSeek CKV cache allocated successfully shape: (61, 500, 128, 1, 512)
INFO 06-09 09:48:39,096 model_runner.py:214] DeepSeek KPE cache allocated successfully shape: (61, 500, 128, 1, 64)
INFO 06-09 09:48:39,096 model_runner.py:214] DeepSeek index cache allocated successfully shape: (61, 2000, 128, 1, 128)
INFO 06-09 09:48:39,096 model_runner.py:214] DeepSeek DRAM CKV cache allocated successfully shape: (61, 2000, 128, 1, 512)
INFO 06-09 09:48:39,096 model_runner.py:214] DeepSeek DRAM KPE cache allocated successfully shape: (61, 2000, 128, 1, 64)
INFO 06-09 09:48:39,096 model_runner.py:214] DeepSeek gather selection status allocated successfully shape: (256, 1, 1, 2049)
INFO 06-09 09:48:39,988 model_runner.py:62] config: Config(model=/mnt/models/Deepseek-V3.2-Pruned-95B-BF/, max_num_prefill_seqs_per_step=1, max_num_decode_seqs_per_step=256, max_model_len=65536, tensor_parallel_size=8, enable_expert_parallel=True, enforce_eager=True, eos=1, kvcache_block_size=128, num_kvcache_blocks=500, num_index_cache_blocks=2000, num_hbm_kvcache_blocks=500, num_dram_kvcache_blocks=2000, dsa_offload_max_sparse_tokens=2048, hccl_port=28000, trust_remote_code=True, dsa_offload_pool_capacity=256, hf_config=DeepseekV32Config(...))
INFO 06-09 09:48:45,811 model_runner.py:191] Using explicit DSA cache blocks: hbm=500, dram=2000, index=2000, max_sparse_tokens=2048
INFO 06-09 09:48:45,811 model_runner.py:205] Single HBM KV Block Size: 8.58 MB
INFO 06-09 09:48:45,811 model_runner.py:214] DeepSeek CKV cache allocated successfully shape: (61, 500, 128, 1, 512)
INFO 06-09 09:48:45,811 model_runner.py:214] DeepSeek KPE cache allocated successfully shape: (61, 500, 128, 1, 64)
INFO 06-09 09:48:45,811 model_runner.py:214] DeepSeek index cache allocated successfully shape: (61, 2000, 128, 1, 128)
INFO 06-09 09:48:45,811 model_runner.py:214] DeepSeek DRAM CKV cache allocated successfully shape: (61, 2000, 128, 1, 512)
INFO 06-09 09:48:45,811 model_runner.py:214] DeepSeek DRAM KPE cache allocated successfully shape: (61, 2000, 128, 1, 64)
INFO 06-09 09:48:45,811 model_runner.py:214] DeepSeek gather selection status allocated successfully shape: (256, 1, 1, 2049)
INFO 06-09 09:48:46,623 model_runner.py:62] config: Config(model=/mnt/models/Deepseek-V3.2-Pruned-95B-BF/, max_num_prefill_seqs_per_step=1, max_num_decode_seqs_per_step=256, max_model_len=65536, tensor_parallel_size=8, enable_expert_parallel=True, enforce_eager=True, eos=1, kvcache_block_size=128, num_kvcache_blocks=500, num_index_cache_blocks=2000, num_hbm_kvcache_blocks=500, num_dram_kvcache_blocks=2000, dsa_offload_max_sparse_tokens=2048, hccl_port=28000, trust_remote_code=True, dsa_offload_pool_capacity=256, hf_config=DeepseekV32Config(...))
INFO 06-09 09:48:49,680 model_runner.py:191] Using explicit DSA cache blocks: hbm=500, dram=2000, index=2000, max_sparse_tokens=2048
INFO 06-09 09:48:49,681 model_runner.py:205] Single HBM KV Block Size: 8.58 MB
INFO 06-09 09:48:49,681 model_runner.py:214] DeepSeek CKV cache allocated successfully shape: (61, 500, 128, 1, 512)
INFO 06-09 09:48:49,681 model_runner.py:214] DeepSeek KPE cache allocated successfully shape: (61, 500, 128, 1, 64)
INFO 06-09 09:48:49,681 model_runner.py:214] DeepSeek index cache allocated successfully shape: (61, 2000, 128, 1, 128)
INFO 06-09 09:48:49,681 model_runner.py:214] DeepSeek DRAM CKV cache allocated successfully shape: (61, 2000, 128, 1, 512)
INFO 06-09 09:48:49,681 model_runner.py:214] DeepSeek DRAM KPE cache allocated successfully shape: (61, 2000, 128, 1, 64)
INFO 06-09 09:48:49,681 model_runner.py:214] DeepSeek gather selection status allocated successfully shape: (256, 1, 1, 2049)
INFO 06-09 09:48:50,534 model_runner.py:62] config: Config(model=/mnt/models/Deepseek-V3.2-Pruned-95B-BF/, max_num_prefill_seqs_per_step=1, max_num_decode_seqs_per_step=256, max_model_len=65536, tensor_parallel_size=8, enable_expert_parallel=True, enforce_eager=True, eos=1, kvcache_block_size=128, num_kvcache_blocks=500, num_index_cache_blocks=2000, num_hbm_kvcache_blocks=500, num_dram_kvcache_blocks=2000, dsa_offload_max_sparse_tokens=2048, hccl_port=28000, trust_remote_code=True, dsa_offload_pool_capacity=256, hf_config=DeepseekV32Config(...))
/usr/local/python3.11.14/lib/python3.11/site-packages/torch/distributed/distributed_c10d.py:4876: UserWarning: barrier(): using the device under current context. You can specify `device_id` in `init_process_group` to mute this warning.
  warnings.warn(  # warn only once
INFO 06-09 09:49:01,090 model_runner.py:191] Using explicit DSA cache blocks: hbm=500, dram=2000, index=2000, max_sparse_tokens=2048
INFO 06-09 09:49:01,090 model_runner.py:205] Single HBM KV Block Size: 8.58 MB
INFO 06-09 09:49:01,090 model_runner.py:214] DeepSeek CKV cache allocated successfully shape: (61, 500, 128, 1, 512)
INFO 06-09 09:49:01,091 model_runner.py:214] DeepSeek KPE cache allocated successfully shape: (61, 500, 128, 1, 64)
INFO 06-09 09:49:01,091 model_runner.py:214] DeepSeek index cache allocated successfully shape: (61, 2000, 128, 1, 128)
INFO 06-09 09:49:01,091 model_runner.py:214] DeepSeek DRAM CKV cache allocated successfully shape: (61, 2000, 128, 1, 512)
INFO 06-09 09:49:01,091 model_runner.py:214] DeepSeek DRAM KPE cache allocated successfully shape: (61, 2000, 128, 1, 64)
INFO 06-09 09:49:01,091 model_runner.py:214] DeepSeek gather selection status allocated successfully shape: (256, 1, 1, 2049)
INFO 06-09 09:49:01,932 model_runner.py:62] config: Config(model=/mnt/models/Deepseek-V3.2-Pruned-95B-BF/, max_num_prefill_seqs_per_step=1, max_num_decode_seqs_per_step=256, max_model_len=65536, tensor_parallel_size=8, enable_expert_parallel=True, enforce_eager=True, eos=1, kvcache_block_size=128, num_kvcache_blocks=500, num_index_cache_blocks=2000, num_hbm_kvcache_blocks=500, num_dram_kvcache_blocks=2000, dsa_offload_max_sparse_tokens=2048, hccl_port=28000, trust_remote_code=True, dsa_offload_pool_capacity=256, hf_config=DeepseekV32Config(...))
test config: num_prompts=16, prompt_min=11000, prompt_max=12500, max_model_len=65536, max_num_prefill_seqs_per_step=1, max_num_decode_seqs_per_step=256, max_gen_tokens=16, prompt_style=meaningful, meaningful_base_tokens=10000
prompt plan:
  prompt 1: target_len=11000, full_blocks=85, sparse_blocks=16, release_blocks=69
  prompt 2: target_len=11100, full_blocks=86, sparse_blocks=16, release_blocks=70
  prompt 3: target_len=11200, full_blocks=87, sparse_blocks=16, release_blocks=71
  prompt 4: target_len=11300, full_blocks=88, sparse_blocks=16, release_blocks=72
  prompt 5: target_len=11400, full_blocks=89, sparse_blocks=16, release_blocks=73
  prompt 6: target_len=11500, full_blocks=89, sparse_blocks=16, release_blocks=73
  prompt 7: target_len=11600, full_blocks=90, sparse_blocks=16, release_blocks=74
  prompt 8: target_len=11700, full_blocks=91, sparse_blocks=16, release_blocks=75
  prompt 9: target_len=11800, full_blocks=92, sparse_blocks=16, release_blocks=76
  prompt 10: target_len=11900, full_blocks=92, sparse_blocks=16, release_blocks=76
  prompt 11: target_len=12000, full_blocks=93, sparse_blocks=16, release_blocks=77
  prompt 12: target_len=12100, full_blocks=94, sparse_blocks=16, release_blocks=78
  prompt 13: target_len=12200, full_blocks=95, sparse_blocks=16, release_blocks=79
  prompt 14: target_len=12300, full_blocks=96, sparse_blocks=16, release_blocks=80
  prompt 15: target_len=12400, full_blocks=96, sparse_blocks=16, release_blocks=80
  prompt 16: target_len=12500, full_blocks=97, sparse_blocks=16, release_blocks=81
prompt 1 token_len=11000 first_ids=[455, 1957, 15986, 515, 9306, 201, 117495, 97721, 25923, 1754, 78019, 97721, 9693, 10461, 270, 28865]
prompt 2 token_len=11100 first_ids=[6006, 412, 260, 15292, 2831, 362, 201, 30230, 29645, 16, 56518, 65021, 9026, 270, 15292, 86508]
prompt 3 token_len=11200 first_ids=[4951, 14, 40685, 57, 8840, 73857, 4914, 15492, 9259, 59, 5329, 3973, 989, 11013, 57, 8840]
prompt 4 token_len=11300 first_ids=[8398, 260, 37754, 11624, 7155, 59344, 201, 17774, 359, 25923, 24570, 270, 4087, 25314, 16, 455]
prompt 5 token_len=11400 first_ids=[295, 270, 2662, 10281, 16, 455, 201, 74304, 4869, 3920, 270, 10346, 91355, 14, 3920, 270]
prompt 6 token_len=11500 first_ids=[2205, 47156, 14, 31434, 40968, 14, 305, 260, 45076, 4090, 16, 455, 1957, 15986, 515, 9306]
prompt 7 token_len=11600 first_ids=[201, 114830, 2599, 4868, 28, 455, 25686, 2709, 44674, 2237, 2775, 6006, 412, 260, 15292, 2831]
prompt 8 token_len=11700 first_ids=[4868, 28, 40685, 57, 8840, 73857, 15, 13455, 17640, 5770, 72546, 4951, 14, 40685, 57, 8840]
prompt 9 token_len=11800 first_ids=[387, 6057, 1066, 339, 14446, 110865, 4868, 28, 334, 3859, 52717, 8398, 260, 37754, 11624, 7155]
prompt 10 token_len=11900 first_ids=[25314, 339, 1124, 64276, 4868, 28, 455, 4087, 25314, 515, 5607, 295, 270, 2662, 10281, 16]
prompt 11 token_len=12000 first_ids=[28, 455, 15389, 1539, 7379, 260, 1957, 18658, 15986, 418, 201, 2205, 47156, 14, 31434, 40968]
prompt 12 token_len=12100 first_ids=[15986, 2329, 538, 270, 201, 85, 6057, 4087, 25314, 9575, 603, 201, 114830, 2599, 4868, 28]
prompt 13 token_len=12200 first_ids=[25923, 4365, 304, 270, 201, 41551, 25314, 4245, 339, 36892, 279, 4868, 28, 40685, 57, 8840]
prompt 14 token_len=12300 first_ids=[56518, 65021, 14023, 270, 9575, 14, 305, 71595, 83386, 201, 10499, 387, 6057, 1066, 339, 14446]
prompt 15 token_len=12400 first_ids=[515, 270, 6690, 18658, 201, 38206, 367, 28865, 2184, 270, 4087, 25314, 339, 1124, 64276, 4868]
prompt 16 token_len=12500 first_ids=[4332, 13256, 201, 265, 270, 23616, 6403, 339, 2167, 18486, 4868, 28, 455, 15389, 1539, 7379]
llm.generate 16 start ...
[step   1 Prefill] bsz=1, num_tokens=11001, HBM_KV=17/500, DRAM_KV=85/2000, HBM_INDEX=86/2000, TTFT=6.3176 sec, TPS=1741.32 tok/s
[step   2 Prefill] bsz=1, num_tokens=11101, HBM_KV=34/500, DRAM_KV=171/2000, HBM_INDEX=173/2000, TTFT=3.1028 sec, TPS=3577.74 tok/s
[step   3 Prefill] bsz=1, num_tokens=11201, HBM_KV=51/500, DRAM_KV=258/2000, HBM_INDEX=261/2000, TTFT=3.1690 sec, TPS=3534.51 tok/s
[step   4 Prefill] bsz=1, num_tokens=11301, HBM_KV=68/500, DRAM_KV=346/2000, HBM_INDEX=350/2000, TTFT=3.1370 sec, TPS=3602.49 tok/s
[step   5 Prefill] bsz=1, num_tokens=11401, HBM_KV=85/500, DRAM_KV=435/2000, HBM_INDEX=440/2000, TTFT=3.2514 sec, TPS=3506.54 tok/s
[step   6 Prefill] bsz=1, num_tokens=11501, HBM_KV=102/500, DRAM_KV=524/2000, HBM_INDEX=530/2000, TTFT=3.2875 sec, TPS=3498.39 tok/s
[step   7 Prefill] bsz=1, num_tokens=11601, HBM_KV=119/500, DRAM_KV=614/2000, HBM_INDEX=621/2000, TTFT=3.3433 sec, TPS=3469.91 tok/s
[step   8 Prefill] bsz=1, num_tokens=11701, HBM_KV=136/500, DRAM_KV=705/2000, HBM_INDEX=713/2000, TTFT=3.2719 sec, TPS=3576.26 tok/s
[step   9 Prefill] bsz=1, num_tokens=11801, HBM_KV=153/500, DRAM_KV=797/2000, HBM_INDEX=806/2000, TTFT=3.4892 sec, TPS=3382.14 tok/s
[step  10 Prefill] bsz=1, num_tokens=11901, HBM_KV=170/500, DRAM_KV=889/2000, HBM_INDEX=899/2000, TTFT=3.3252 sec, TPS=3579.00 tok/s
[step  11 Prefill] bsz=1, num_tokens=12001, HBM_KV=187/500, DRAM_KV=982/2000, HBM_INDEX=993/2000, TTFT=3.3966 sec, TPS=3533.21 tok/s
[step  12 Prefill] bsz=1, num_tokens=12101, HBM_KV=204/500, DRAM_KV=1076/2000, HBM_INDEX=1088/2000, TTFT=3.3856 sec, TPS=3574.23 tok/s
[step  13 Prefill] bsz=1, num_tokens=12201, HBM_KV=221/500, DRAM_KV=1171/2000, HBM_INDEX=1184/2000, TTFT=3.4861 sec, TPS=3499.89 tok/s
[step  14 Prefill] bsz=1, num_tokens=12301, HBM_KV=238/500, DRAM_KV=1267/2000, HBM_INDEX=1281/2000, TTFT=3.5032 sec, TPS=3511.34 tok/s
[step  15 Prefill] bsz=1, num_tokens=12401, HBM_KV=255/500, DRAM_KV=1363/2000, HBM_INDEX=1378/2000, TTFT=3.5127 sec, TPS=3530.32 tok/s
[step  16 Prefill] bsz=1, num_tokens=12501, HBM_KV=272/500, DRAM_KV=1460/2000, HBM_INDEX=1476/2000, TTFT=3.5294 sec, TPS=3541.95 tok/s
[step  17  Decode] bsz=16, num_tokens=16, HBM_KV=272/500, DRAM_KV=1460/2000, HBM_INDEX=1476/2000, TPOT=0.3564 sec, TPS=44.89 tok/s
........[step  18  Decode] bsz=16, num_tokens=16, HBM_KV=272/500, DRAM_KV=1460/2000, HBM_INDEX=1476/2000, TPOT=1.1140 sec, TPS=14.36 tok/s
[step  19  Decode] bsz=16, num_tokens=16, HBM_KV=272/500, DRAM_KV=1460/2000, HBM_INDEX=1476/2000, TPOT=0.1721 sec, TPS=92.97 tok/s
[step  20  Decode] bsz=16, num_tokens=16, HBM_KV=272/500, DRAM_KV=1460/2000, HBM_INDEX=1476/2000, TPOT=0.1689 sec, TPS=94.74 tok/s
[step  21  Decode] bsz=16, num_tokens=16, HBM_KV=273/500, DRAM_KV=1460/2000, HBM_INDEX=1477/2000, TPOT=0.1715 sec, TPS=93.30 tok/s
[step  22  Decode] bsz=16, num_tokens=16, HBM_KV=273/500, DRAM_KV=1460/2000, HBM_INDEX=1477/2000, TPOT=0.1695 sec, TPS=94.41 tok/s
[step  23  Decode] bsz=16, num_tokens=16, HBM_KV=273/500, DRAM_KV=1460/2000, HBM_INDEX=1477/2000, TPOT=0.1679 sec, TPS=95.28 tok/s
[step  24  Decode] bsz=16, num_tokens=16, HBM_KV=273/500, DRAM_KV=1460/2000, HBM_INDEX=1477/2000, TPOT=0.1684 sec, TPS=95.03 tok/s
[step  25  Decode] bsz=16, num_tokens=16, HBM_KV=274/500, DRAM_KV=1460/2000, HBM_INDEX=1478/2000, TPOT=0.1679 sec, TPS=95.32 tok/s
[step  26  Decode] bsz=16, num_tokens=16, HBM_KV=274/500, DRAM_KV=1460/2000, HBM_INDEX=1478/2000, TPOT=0.1732 sec, TPS=92.38 tok/s
[step  27  Decode] bsz=16, num_tokens=16, HBM_KV=274/500, DRAM_KV=1460/2000, HBM_INDEX=1478/2000, TPOT=0.1747 sec, TPS=91.59 tok/s
[step  28  Decode] bsz=16, num_tokens=16, HBM_KV=274/500, DRAM_KV=1460/2000, HBM_INDEX=1478/2000, TPOT=0.1833 sec, TPS=87.30 tok/s
[step  29  Decode] bsz=16, num_tokens=16, HBM_KV=274/500, DRAM_KV=1460/2000, HBM_INDEX=1478/2000, TPOT=0.1702 sec, TPS=94.03 tok/s
[step  30  Decode] bsz=16, num_tokens=16, HBM_KV=274/500, DRAM_KV=1460/2000, HBM_INDEX=1478/2000, TPOT=0.1655 sec, TPS=96.67 tok/s
[step  31  Decode] bsz=16, num_tokens=16, HBM_KV=0/500, DRAM_KV=0/2000, HBM_INDEX=0/2000, TPOT=0.1716 sec, TPS=93.24 tok/s
llm.generate 16 requests in 31 steps, e2e latency = 60.21 sec
    prefill steps = 16, prefill tokens = 188016, total prefill time = 56.5086 sec, prefill TPS = 3327.21 tok/s
    decode steps = 15, decode tokens = 240, total decode time = 3.6949 sec, decode mean TPOT = 0.2463 sec, decode TPS = 64.95 tok/s
    e2e input TPS = 3122.63 tok/s, e2e output TPS = 4.25 tok/s
    TTFT/TPOT are per-step request latencies printed above.
prompt_len: 11000
prompt    : <meaningful long-QA suffix prompt 1: target_len=11000, tail='g. 3. The signed final inspection notes identify the temporary bridge as Hawthorn Bridge. 4. The later Willow Bridge margin note is explicitly marked as a mistake. Question: What was the name of the temporary bridge used during the final i
response  : '\nRead the following long archive packet and answer the final question with one\nshort'
token_ids : [201, 8158, 270, 2502, 1606, 41273, 23648, 305, 3287, 270, 4087, 3417, 418, 834, 201, 31150]

prompt_len: 11100
prompt    : <meaningful long-QA suffix prompt 2: target_len=11100, tail='g. 3. The signed final inspection notes identify the temporary bridge as Hawthorn Bridge. 4. The later Willow Bridge margin note is explicitly marked as a mistake. Question: What was the name of the temporary bridge used during the final i
response  : '\nThe final inspection notes are the same as the repair ledger. Elena Ruiz supervised'
token_ids : [201, 671, 4087, 25314, 9575, 477, 270, 1975, 412, 270, 15292, 86508, 16, 71595, 83386, 52671]

prompt_len: 11200
prompt    : <meaningful long-QA suffix prompt 3: target_len=11200, tail='g. 3. The signed final inspection notes identify the temporary bridge as Hawthorn Bridge. 4. The later Willow Bridge margin note is explicitly marked as a mistake. Question: What was the name of the temporary bridge used during the final i
response  : " \\' '\n\nRead the following long archive packet and answer the final question with"
token_ids : [874, 9, 905, 271, 8158, 270, 2502, 1606, 41273, 23648, 305, 3287, 270, 4087, 3417, 418]

prompt_len: 11300
prompt    : <meaningful long-QA suffix prompt 4: target_len=11300, tail='g. 3. The signed final inspection notes identify the temporary bridge as Hawthorn Bridge. 4. The later Willow Bridge margin note is explicitly marked as a mistake. Question: What was the name of the temporary bridge used during the final i
response  : ' <｜end▁of▁sentence｜>\nRead the following long archive packet and answer the final question with one'
token_ids : [223, 1, 201, 8158, 270, 2502, 1606, 41273, 23648, 305, 3287, 270, 4087, 3417, 418, 834]

prompt_len: 11400
prompt    : <meaningful long-QA suffix prompt 5: target_len=11400, tail='g. 3. The signed final inspection notes identify the temporary bridge as Hawthorn Bridge. 4. The later Willow Bridge margin note is explicitly marked as a mistake. Question: What was the name of the temporary bridge used during the final i
response  : " 'The bridge name is Willow Bridge.'\n\nThe packet describes a valley repair project that"
token_ids : [905, 671, 15986, 2329, 344, 88754, 25923, 27457, 671, 23648, 13308, 260, 30795, 15292, 2775, 396]

prompt_len: 11500
prompt    : <meaningful long-QA suffix prompt 6: target_len=11500, tail='g. 3. The signed final inspection notes identify the temporary bridge as Hawthorn Bridge. 4. The later Willow Bridge margin note is explicitly marked as a mistake. Question: What was the name of the temporary bridge used during the final i
response  : '<｜end▁of▁sentence｜><｜end▁of▁sentence｜><｜end▁of▁sentence｜><｜end▁of▁sentence｜><｜end▁of▁sentence｜><｜end▁of▁sentence｜>\n<｜end▁of▁sentence｜> the bridge name in those signed inspection notes'
token_ids : [1, 1, 1, 1, 1, 1, 201, 1, 270, 15986, 2329, 295, 1948, 14023, 25314, 9575]

prompt_len: 11600
prompt    : <meaningful long-QA suffix prompt 7: target_len=11600, tail='g. 3. The signed final inspection notes identify the temporary bridge as Hawthorn Bridge. 4. The later Willow Bridge margin note is explicitly marked as a mistake. Question: What was the name of the temporary bridge used during the final i
response  : '\tThe bridge name is Willow Bridge.\n\nThe bridge name is Willow Bridge.\n\nThe'
token_ids : [200, 671, 15986, 2329, 344, 88754, 25923, 339, 671, 15986, 2329, 344, 88754, 25923, 339, 671]

prompt_len: 11700
prompt    : <meaningful long-QA suffix prompt 8: target_len=11700, tail='g. 3. The signed final inspection notes identify the temporary bridge as Hawthorn Bridge. 4. The later Willow Bridge margin note is explicitly marked as a mistake. Question: What was the name of the temporary bridge used during the final i
response  : '\nRead the following long archive packet and answer the final question with one\nshort'
token_ids : [201, 8158, 270, 2502, 1606, 41273, 23648, 305, 3287, 270, 4087, 3417, 418, 834, 201, 31150]

prompt_len: 11800
prompt    : <meaningful long-QA suffix prompt 9: target_len=11800, tail='g. 3. The signed final inspection notes identify the temporary bridge as Hawthorn Bridge. 4. The later Willow Bridge margin note is explicitly marked as a mistake. Question: What was the name of the temporary bridge used during the final i
response  : 'The bridge name is Willow Bridge.\n\nThe signed inspection notes are the final inspection notes'
token_ids : [671, 15986, 2329, 344, 88754, 25923, 339, 671, 14023, 25314, 9575, 477, 270, 4087, 25314, 9575]

prompt_len: 11900
prompt    : <meaningful long-QA suffix prompt 10: target_len=11900, tail='g. 3. The signed final inspection notes identify the temporary bridge as Hawthorn Bridge. 4. The later Willow Bridge margin note is explicitly marked as a mistake. Question: What was the name of the temporary bridge used during the final
response  : '\n```\n\n### 2.1.1.1.1.1.'
token_ids : [201, 20759, 795, 223, 20, 16, 19, 16, 19, 16, 19, 16, 19, 16, 19, 16]

prompt_len: 12000
prompt    : <meaningful long-QA suffix prompt 11: target_len=12000, tail='g. 3. The signed final inspection notes identify the temporary bridge as Hawthorn Bridge. 4. The later Willow Bridge margin note is explicitly marked as a mistake. Question: What was the name of the temporary bridge used during the final
response  : ' <｜end▁of▁sentence｜>\nRead the following long archive packet and answer the final question with one'
token_ids : [223, 1, 201, 8158, 270, 2502, 1606, 41273, 23648, 305, 3287, 270, 4087, 3417, 418, 834]

prompt_len: 12100
prompt    : <meaningful long-QA suffix prompt 12: target_len=12100, tail='g. 3. The signed final inspection notes identify the temporary bridge as Hawthorn Bridge. 4. The later Willow Bridge margin note is explicitly marked as a mistake. Question: What was the name of the temporary bridge used during the final
response  : '<｜end▁of▁sentence｜>\n<｜end▁of▁sentence｜>\n<｜end▁of▁sentence｜>\n<｜end▁of▁sentence｜>\n<｜end▁of▁sentence｜>\n<｜end▁of▁sentence｜>\n##\n##\n'
token_ids : [1, 201, 1, 201, 1, 201, 1, 201, 1, 201, 1, 201, 372, 201, 372, 201]

prompt_len: 12200
prompt    : <meaningful long-QA suffix prompt 13: target_len=12200, tail='g. 3. The signed final inspection notes identify the temporary bridge as Hawthorn Bridge. 4. The later Willow Bridge margin note is explicitly marked as a mistake. Question: What was the name of the temporary bridge used during the final
response  : '<｜end▁of▁sentence｜>The bridge name is Willow Bridge.\n\nThe bridge name is Willow Bridge.\n\nThe'
token_ids : [1, 671, 15986, 2329, 344, 88754, 25923, 339, 671, 15986, 2329, 344, 88754, 25923, 339, 671]

prompt_len: 12300
prompt    : <meaningful long-QA suffix prompt 14: target_len=12300, tail='g. 3. The signed final inspection notes identify the temporary bridge as Hawthorn Bridge. 4. The later Willow Bridge margin note is explicitly marked as a mistake. Question: What was the name of the temporary bridge used during the final
response  : " ',\n        'The bridge name in the signed final inspection notes is Willow Bridge"
token_ids : [905, 989, 528, 905, 671, 15986, 2329, 295, 270, 14023, 4087, 25314, 9575, 344, 88754, 25923]

prompt_len: 12400
prompt    : <meaningful long-QA suffix prompt 15: target_len=12400, tail='g. 3. The signed final inspection notes identify the temporary bridge as Hawthorn Bridge. 4. The later Willow Bridge margin note is explicitly marked as a mistake. Question: What was the name of the temporary bridge used during the final
response  : ' <｜end▁of▁sentence｜>\nRead the following long archive packet and answer the final question with one'
token_ids : [223, 1, 201, 8158, 270, 2502, 1606, 41273, 23648, 305, 3287, 270, 4087, 3417, 418, 834]

prompt_len: 12500
prompt    : <meaningful long-QA suffix prompt 16: target_len=12500, tail='g. 3. The signed final inspection notes identify the temporary bridge as Hawthorn Bridge. 4. The later Willow Bridge margin note is explicitly marked as a mistake. Question: What was the name of the temporary bridge used during the final
response  : ' <｜end▁of▁sentence｜>\n<｜end▁of▁sentence｜>\n<｜end▁of▁sentence｜>\n<｜end▁of▁sentence｜>\nThe bridge name in the signed final'
token_ids : [223, 1, 201, 1, 201, 1, 201, 1, 201, 671, 15986, 2329, 295, 270, 14023, 4087]
```

　

　

## 主要环境变量含义

| 变量 | 默认值 | 含义 |
|---|---:|---|
| `NANOVLLM_MODEL` | `/home/models/Deepseek-V3.2-Pruned-95B-BF/` | 模型目录。 |
| `NANOVLLM_TP_SIZE` | `4` | Tensor parallel world size。 |
| `NANOVLLM_ENABLE_EXPERT_PARALLEL` | `true` | 是否启用 MoE expert parallel。 |
| `NANOVLLM_KVCACHE_BLOCK_SIZE` | `128` | Paged KV cache block size。 |
| `NANOVLLM_HBM_NUM_BLOCKS` | 必填 | HBM KV cache block 数量。 |
| `NANOVLLM_DRAM_NUM_BLOCKS` | 必填 | DRAM KV cache 和 IndexCache block 数量。 |
| `NANOVLLM_MAX_MODEL_LEN` | `65536` | engine 最大序列长度。 |
| `NANOVLLM_MAX_PREFILL_SEQS_PER_STEP` | `1` | 单次 prefill step 最多调度多少个新请求。 |
| `NANOVLLM_MAX_DECODE_SEQS_PER_STEP` | example 自推导 | running 队列容量上限和 decode batch size 上限。 |
| `NANOVLLM_PROMPT_LENGTHS` | 未设置 | 逗号分隔的精确 prompt token 长度。 |
| `NANOVLLM_MAX_GEN_TOKENS` | 脚本自定义 | 每个请求最大 decode token 数。 |
| `NANOVLLM_IGNORE_EOS` | `false` | 是否忽略 EOS，持续 decode 到 `max_tokens`。 |
| `NANOVLLM_LOG_DECODE_LAYER_TIMING` | `false` | 是否打印 decode layer timing。 |
| `NANOVLLM_DECODE_LAYER_TIMING_SYNC` | `true` | timing 前后是否同步。 |
| `NANOVLLM_PROFILE_LAYER_IDS` | `0,mid,last` | 打印 timing 的层。 |
| `NANOVLLM_DSA_QUERY_ONLY_BACKEND` | `current` | DSA decode 后端：`current` 为不组图路径，`torchair` 为 DSA 小流水组图路径。 |

　

## Decode 时延分解字段含义

| 字段 | 含义 |
|---|---|
| `attention_total` | 单层 attention block 总耗时。 |
| `indexer_project` | 生成 `q_index`、`index_k` 和 DSA score 权重。 |
| `index_cache` | 把当前 token 的 `index_k` 写入 HBM IndexCache。 |
| `dsa_total` | `dsa_lightning_indexer + dsa_gather_selection` 总和。 |
| `dsa_lightning_indexer` | 基于 query 和 IndexCache 选择 top sparse token。 |
| `dsa_gather_selection` | 根据 top sparse token 把 KV 从 DRAM gather 到 HBM sparse budget。 |
| `decode_attention_op` | 在 sparse HBM KV budget 上执行 decode MLA。 |
| `moe_total` | attention 后 MLP/MoE block 耗时。 |
