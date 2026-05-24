import os
import time
import numpy as np
import torch
import torch_npu
import nanovllm.ops as ascend_ops


# 性能测试配置 --------------------------------------------
BATCH_SIZES = [1]               # 不同的 batch size
SEQ_LENGTHS = [7] # 不同的序列长度
NUM_WARMUP = 10                                                # 预热次数
NUM_ITERATIONS = 100                                           # 正式测试迭代次数

# 固定参数 --------------------------------------------
HEAD_DIM = 128
NUM_HEADS = 64
BLOCK_SIZE = 128  # vLLM 默认 block size
SPARSE_COUNT = 4096
VLLM_MAX_NUM_BLOCKS = (max(SEQ_LENGTHS) + BLOCK_SIZE - 1) // BLOCK_SIZE

# sfa参数
ACTUAL_HEADS    = 128//4  # 8
KV_LATENT_DIM   = 512
ROPE_DIM        = 64
SCALE_VALUE     = 0.1352337788608801

device = 'npu:0'


def generate_block_tables(batch_size, seq_len):
    num_blocks = (seq_len + BLOCK_SIZE - 1) // BLOCK_SIZE
    block_tables = torch.zeros((batch_size, VLLM_MAX_NUM_BLOCKS), dtype=torch.int32, device=device)
    global_block_id = 1
    for b in range(batch_size):
        for i in range(num_blocks):
            block_tables[b, i] = global_block_id
            global_block_id += 1
    return block_tables

def generate_topk_indices(batch_size, seq_len, block_tables):
    num_blocks_per_seq = (seq_len + BLOCK_SIZE - 1) // BLOCK_SIZE
    topk_indices = torch.zeros((batch_size, 1, SPARSE_COUNT), dtype=torch.int32, device=device)
    for b in range(batch_size):
        # 拿到当前序列在 block_tables 中分配的所有有效物理块 ID
        valid_blocks = block_tables[b, :num_blocks_per_seq]
        
        if num_blocks_per_seq >= SPARSE_COUNT:
            # 场景 A: 序列很长，块很多。从中随机抽取 SPARSE_COUNT 个块
            # 使用 randperm 模拟真实的、非连续的稀疏访问模式
            indices = torch.randperm(num_blocks_per_seq, device=device)[:SPARSE_COUNT]
            topk_indices[b, 0, :] = valid_blocks[indices]
        else:
            # 场景 B: 序列较短，总块数不足 SPARSE_COUNT。
            # 为了保证测试压力，通过 repeat 填满 SPARSE_COUNT
            # 这样 NPU 每一轮循环都会执行真实的 Load 和 Compute
            repeat_count = (SPARSE_COUNT // num_blocks_per_seq) + 1
            extended_blocks = valid_blocks.repeat(repeat_count)
            topk_indices[b, 0, :] = extended_blocks[:SPARSE_COUNT]
    return topk_indices

def generate_test_data(batch_size, seq_len):
    blocks_per_seq   = (seq_len + BLOCK_SIZE - 1) // BLOCK_SIZE
    total_req_blocks = batch_size * blocks_per_seq
    q_nope       = torch.randn(batch_size, ACTUAL_HEADS, KV_LATENT_DIM, dtype=torch.bfloat16, device=device)
    kv           = torch.randn(total_req_blocks, BLOCK_SIZE, 1, KV_LATENT_DIM, dtype=torch.bfloat16, device=device)
    q_pe         = torch.randn(batch_size, ACTUAL_HEADS, ROPE_DIM, dtype=torch.bfloat16, device=device)
    k_pe         = torch.randn(total_req_blocks, BLOCK_SIZE, 1, ROPE_DIM, dtype=torch.bfloat16, device=device)
    block_tables = generate_block_tables(batch_size, seq_len)
    seq_len_q    = torch.arange(batch_size, dtype=torch.int32, device=device) + 1
    seq_len_k    = torch.full((batch_size,), seq_len, dtype=torch.int32, device=device)
    topk_indices = generate_topk_indices(batch_size, seq_len, block_tables)
    return q_nope, kv, q_pe, k_pe, topk_indices, seq_len_q, seq_len_k, block_tables



def run_sfa(q_nope, kv, q_pe, k_pe, topk_indices, seq_len_q, seq_len_k, block_tables, seq_len, verify=False):
    attn_output = ascend_ops.npu_sparse_flash_attention(
        query = q_nope,
        key = kv,
        value = kv,
        sparse_indices = topk_indices,
        scale_value = SCALE_VALUE,
        sparse_block_size = 1,
        actual_seq_lengths_query = seq_len_q,
        actual_seq_lengths_kv   = seq_len_k,
        block_table = block_tables,
        query_rope = q_pe,
        key_rope = k_pe,
        layout_query = "TND",
        layout_kv = "PA_BSND",
        sparse_mode = 3,
    )
    torch.npu.synchronize()
    if verify :
        pass
    return attn_output



def benchmark_config(batch_size, seq_len):
    # 生成测试数据
    ql_nope, kv, q_pe, k_pe, topk_indices, seq_len_q, seq_len_k, block_tables = generate_test_data(batch_size, seq_len)
    
    # 预热
    for _ in range(NUM_WARMUP):
        run_sfa(ql_nope, kv, q_pe, k_pe, topk_indices, seq_len_q, seq_len_k, block_tables, seq_len, verify=True)
    
    # 正式测试
    torch.npu.synchronize()
    start_event = torch.npu.Event(enable_timing=True)
    end_event   = torch.npu.Event(enable_timing=True)
    
    latencies_us = []  # 存储微秒为单位的时延
    for _ in range(NUM_ITERATIONS):
        start_event.record()
        run_sfa(ql_nope, kv, q_pe, k_pe, topk_indices, seq_len_q, seq_len_k, block_tables, seq_len)
        end_event.record()
        torch.npu.synchronize()
        elapsed_ms = start_event.elapsed_time(end_event)  # PyTorch Event 返回的是毫秒，转换为微秒
        elapsed_us = elapsed_ms * 1000.0
        latencies_us.append(elapsed_us)
    
    avg_latency_us = np.mean(latencies_us)
    std_latency_us = np.std(latencies_us)
    min_latency_us = np.min(latencies_us)
    max_latency_us = np.max(latencies_us)
    
    return {
        'avg_us': avg_latency_us,
        'std_us': std_latency_us,
        'min_us': min_latency_us,
        'max_us': max_latency_us,
        'p99_us': np.percentile(latencies_us, 99)
    }


def print_results(results):
    """打印测试结果表格"""
    print("\n" + "="*110)
    print("npu_sparse_flash_attention 性能测试结果 (单位: 微秒 μs)")
    print("="*110)
    print(f"{'Batch Size':<12} {'Seq Len':<12} {'Avg (μs)':<14} {'Std (μs)':<14} {'Min (μs)':<14} {'Max (μs)':<14} {'P99 (μs)':<14}")
    print("-"*110)
    
    for result in results:
        print(f"{result['batch_size']:<12} {result['seq_len']:<12} "
              f"{result['avg_us']:<14.2f} {result['std_us']:<14.2f} "
              f"{result['min_us']:<14.2f} {result['max_us']:<14.2f} "
              f"{result['p99_us']:<14.2f}")
    
    print("="*110)





if __name__ == '__main__':
    print("验证算子是否加载成功...")
    print(ascend_ops.npu_sparse_flash_attention)
    print(f"\n设备: {device}")
    print(f"预热次数: {NUM_WARMUP}")
    print(f"测试迭代: {NUM_ITERATIONS}")
    print(f"Head Dim: {HEAD_DIM}, Num Heads: {NUM_HEADS}, Sparse Count: {SPARSE_COUNT}")
    print("-" * 80)
    
    all_results = []
    
    # 遍历所有配置进行测试
    for seq_len in SEQ_LENGTHS:
        for batch_size in BATCH_SIZES:
            print(f"\n测试配置: batch_size={batch_size}, seq_len={seq_len}")
            
            try:
                metrics = benchmark_config(batch_size, seq_len)
                
                result = {
                    'batch_size': batch_size,
                    'seq_len': seq_len,
                    **metrics
                }
                all_results.append(result)
                
                print(f"  平均时延: {metrics['avg_us']:.2f} μs, "
                      f"标准差: {metrics['std_us']:.2f} μs, "
                      f"P99: {metrics['p99_us']:.2f} μs")
                
            except Exception as e:
                print(f"  测试失败: {e}")
                import traceback
                traceback.print_exc()
    
    # 打印汇总结果
    print_results(all_results)
