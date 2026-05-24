import os
import time
import numpy as np
import torch
import torch_npu
import nanovllm.ops as ascend_ops


# 性能测试配置 --------------------------------------------
BATCH_SIZES = [1, 8, 16, 32, 64, 128, 256]               # 不同的 batch size
SEQ_LENGTHS = [16384, 32768, 65536] # 不同的序列长度
NUM_WARMUP = 10                                                # 预热次数
NUM_ITERATIONS = 100                                           # 正式测试迭代次数

# 固定参数 --------------------------------------------
HEAD_DIM = 128
NUM_HEADS = 64
BLOCK_SIZE = 128  # vLLM 默认 block size
SPARSE_COUNT = 2048
VLLM_MAX_NUM_BLOCKS = (max(SEQ_LENGTHS) + BLOCK_SIZE - 1) // BLOCK_SIZE

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



# 按最大可能的block数量来生成 krep 物理块
krep_blocks = torch.randn(
    (max(BATCH_SIZES)*VLLM_MAX_NUM_BLOCKS), BLOCK_SIZE, 1, HEAD_DIM,   # block数量 = 最大batchsize*每个请求最大的block数量
    dtype=torch.bfloat16, device=device
)
print(f'k表征的容量是 {krep_blocks.numel()} 个元素')



def generate_test_data(batch_size, seq_len):
    q = torch.randn(batch_size, NUM_HEADS, HEAD_DIM, dtype=torch.bfloat16, device=device)
    block_tables = generate_block_tables(batch_size, seq_len)
    weights = torch.randn(batch_size, NUM_HEADS, dtype=torch.bfloat16, device=device)
    seq_len_q = torch.arange(batch_size, dtype=torch.int32, device=device) + 1
    seq_len_k = torch.full((batch_size,), seq_len, dtype=torch.int32, device=device)
    return q, weights, seq_len_q, seq_len_k, block_tables



def run_indexer(q, weights, seq_len_q, seq_len_k, block_tables, seq_len, verify=False):
    topk_indices = ascend_ops.npu_lightning_indexer(
        query=q,
        key=krep_blocks,
        weights=weights,
        actual_seq_lengths_query=seq_len_q,
        actual_seq_lengths_key=seq_len_k,
        block_table=block_tables,
        layout_query="TND",
        layout_key="PA_BSND",
        sparse_count=SPARSE_COUNT,
        sparse_mode=3,
    )
    torch.npu.synchronize()
    if verify :
        min_index = topk_indices.min().item()
        max_index = topk_indices.max().item()
        assert min_index >= 0
        assert min_index < seq_len
        assert max_index >= 0
        assert max_index < seq_len
    return topk_indices



def benchmark_config(batch_size, seq_len):
    # 生成测试数据
    q, weights, seq_len_q, seq_len_k, block_tables = generate_test_data(batch_size, seq_len)
    
    # 预热
    for _ in range(NUM_WARMUP):
        run_indexer(q, weights, seq_len_q, seq_len_k, block_tables, seq_len, verify=True)
    
    # 正式测试
    torch.npu.synchronize()
    start_event = torch.npu.Event(enable_timing=True)
    end_event   = torch.npu.Event(enable_timing=True)
    
    latencies_us = []  # 存储微秒为单位的时延
    for _ in range(NUM_ITERATIONS):
        start_event.record()
        run_indexer(q, weights, seq_len_q, seq_len_k, block_tables, seq_len)
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
    print("npu_lightning_indexer 性能测试结果 (单位: 微秒 μs)")
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
    print(ascend_ops.npu_lightning_indexer)
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
