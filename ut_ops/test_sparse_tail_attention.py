"""Semantic and latency UT for bundled sparse_tail_attention."""

from __future__ import annotations

import argparse
import math
from time import perf_counter

import torch
import torch_npu  # type: ignore

import nanovllm.ops  # noqa: F401  # Load repository-local custom operators.
from ut_ops._op_utils import require_local_opapi


BLOCK_SIZE = 128
CKV_DIM = 512
KPE_DIM = 64
TOPK = 2048


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate NanovllmSparseTailAttention against CPU golden and dense MLA."
    )
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--heads", type=int, default=4, choices=(4, 8, 128))
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--cache-tokens", type=int, default=5120)
    parser.add_argument("--tail-tokens", type=int, default=64)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--skip-performance",
        action="store_true",
        help="Run the CPU-golden semantic checks only.",
    )
    return parser.parse_args()


def random_block_table(
    batch_size: int,
    blocks_per_request: int,
    generator: torch.Generator,
) -> tuple[torch.Tensor, int]:
    total_blocks = batch_size * blocks_per_request
    table = torch.randperm(total_blocks, generator=generator).to(torch.int32)
    return table.view(batch_size, blocks_per_request).contiguous(), total_blocks


def sparse_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    sparse_slots: torch.Tensor,
    cache_tokens: torch.Tensor,
    block_table: torch.Tensor,
    actual_q: torch.Tensor,
    actual_kv: torch.Tensor,
    query_rope: torch.Tensor,
    key_rope: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    attention_out = torch.empty_like(query)
    torch.ops.nanovllm_dsa.sparse_tail_attention.default(
        query_rope,
        query,
        actual_q,
        actual_kv,
        cache_tokens,
        sparse_slots,
        block_table,
        key_rope,
        key,
        scale,
        attention_out,
    )
    return attention_out


def logical_rows(
    cache: torch.Tensor,
    block_table: torch.Tensor,
    row: int,
    logical_slots: torch.Tensor,
) -> torch.Tensor:
    logical_slots = logical_slots.to(torch.int64)
    physical_blocks = block_table[row, logical_slots // BLOCK_SIZE].to(
        torch.int64
    )
    return cache[physical_blocks, logical_slots % BLOCK_SIZE, 0]


def run_semantic_check(
    device: torch.device,
    heads: int,
    seed: int,
) -> None:
    # C=0 exercises mixed short requests.  Tails of 129/257/1025 prove the
    # implementation is not limited to one 128-token block.
    cases = (
        (0, 1),
        (0, 129),
        (0, 2049),
        (2048, 0),
        (3072, 257),
        (5120, 1025),
    )
    generator = torch.Generator().manual_seed(seed)
    batch_size = len(cases)
    actual_lens = [cache_tokens + tail_tokens for cache_tokens, tail_tokens in cases]
    max_blocks = math.ceil(max(actual_lens) / BLOCK_SIZE)
    block_table_cpu, total_blocks = random_block_table(
        batch_size, max_blocks, generator
    )
    query_cpu = (
        torch.randn(
            batch_size, heads, CKV_DIM, generator=generator, dtype=torch.float32
        )
        .mul_(0.25)
        .to(torch.bfloat16)
    )
    query_rope_cpu = (
        torch.randn(
            batch_size, heads, KPE_DIM, generator=generator, dtype=torch.float32
        )
        .mul_(0.25)
        .to(torch.bfloat16)
    )
    key_cpu = (
        torch.randn(
            total_blocks,
            BLOCK_SIZE,
            1,
            CKV_DIM,
            generator=generator,
            dtype=torch.float32,
        )
        .mul_(0.25)
        .to(torch.bfloat16)
    )
    key_rope_cpu = (
        torch.randn(
            total_blocks,
            BLOCK_SIZE,
            1,
            KPE_DIM,
            generator=generator,
            dtype=torch.float32,
        )
        .mul_(0.25)
        .to(torch.bfloat16)
    )
    sparse_slots_cpu = torch.zeros(
        batch_size, 1, TOPK, dtype=torch.int32
    )
    selected_slots: list[torch.Tensor] = []
    for row, (cache_tokens, _) in enumerate(cases):
        if cache_tokens == 0:
            selected = torch.empty(0, dtype=torch.int64)
        else:
            selected = torch.randperm(
                cache_tokens, generator=generator
            )[:TOPK]
            sparse_slots_cpu[row, 0] = selected.to(torch.int32)
        selected_slots.append(selected)

    cache_tokens_cpu = torch.tensor(
        [case[0] for case in cases], dtype=torch.int32
    )
    actual_kv_cpu = torch.tensor(actual_lens, dtype=torch.int32)
    actual_q = torch.arange(
        1, batch_size + 1, dtype=torch.int32, device=device
    )
    scale = 1.0 / math.sqrt(CKV_DIM + KPE_DIM)
    output = sparse_attention(
        query_cpu.to(device),
        key_cpu.to(device),
        sparse_slots_cpu.to(device),
        cache_tokens_cpu.to(device),
        block_table_cpu.to(device),
        actual_q,
        actual_kv_cpu.to(device),
        query_rope_cpu.to(device),
        key_rope_cpu.to(device),
        scale,
    )
    torch.npu.synchronize()

    golden_rows = []
    for row, (cache_tokens, tail_tokens) in enumerate(cases):
        logical_slots = selected_slots[row]
        if cache_tokens == 0:
            logical_slots = torch.arange(tail_tokens, dtype=torch.int64)
        elif tail_tokens:
            logical_slots = torch.cat(
                (
                    logical_slots,
                    torch.arange(
                        cache_tokens,
                        cache_tokens + tail_tokens,
                        dtype=torch.int64,
                    ),
                )
            )
        key = logical_rows(
            key_cpu, block_table_cpu, row, logical_slots
        ).float()
        key_rope = logical_rows(
            key_rope_cpu, block_table_cpu, row, logical_slots
        ).float()
        query = query_cpu[row].float()
        query_rope = query_rope_cpu[row].float()
        scores = (query @ key.T + query_rope @ key_rope.T) * scale
        golden_rows.append(torch.softmax(scores, dim=-1) @ key)
    golden = torch.stack(golden_rows)
    actual = output.float().cpu()
    torch.testing.assert_close(actual, golden, rtol=0.08, atol=0.08)
    max_abs = float((actual - golden).abs().max())
    print(
        "SPARSE_TAIL_ATTENTION_CHECK "
        f"heads={heads} cache_tail_cases={list(cases)} "
        f"max_abs={max_abs:.6f} ok=1"
    )


def capture_graph(launch) -> torch.npu.NPUGraph:
    launch()
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    pool = torch.npu.graph_pool_handle()
    with torch.npu.graph(graph, pool=pool):
        launch()
    torch.npu.synchronize()
    return graph


def benchmark_graph(
    graph: torch.npu.NPUGraph,
    warmup: int,
    iters: int,
) -> float:
    for _ in range(warmup):
        graph.replay()
    torch.npu.synchronize()
    start = perf_counter()
    for _ in range(iters):
        graph.replay()
    torch.npu.synchronize()
    return (perf_counter() - start) * 1000.0 / iters


def run_performance(
    device: torch.device,
    *,
    heads: int,
    batch_size: int,
    cache_tokens: int,
    tail_tokens: int,
    warmup: int,
    iters: int,
    seed: int,
) -> None:
    actual_len = cache_tokens + tail_tokens
    blocks_per_request = math.ceil(actual_len / BLOCK_SIZE)
    generator = torch.Generator().manual_seed(seed + 1000)
    block_table_cpu, total_blocks = random_block_table(
        batch_size, blocks_per_request, generator
    )
    query = torch.randn(
        batch_size, heads, CKV_DIM, dtype=torch.bfloat16, device=device
    )
    query_rope = torch.randn(
        batch_size, heads, KPE_DIM, dtype=torch.bfloat16, device=device
    )
    key = torch.randn(
        total_blocks,
        BLOCK_SIZE,
        1,
        CKV_DIM,
        dtype=torch.bfloat16,
        device=device,
    )
    key_rope = torch.randn(
        total_blocks,
        BLOCK_SIZE,
        1,
        KPE_DIM,
        dtype=torch.bfloat16,
        device=device,
    )
    sparse_slots_cpu = torch.empty(
        batch_size, 1, TOPK, dtype=torch.int32
    )
    for row in range(batch_size):
        sparse_slots_cpu[row, 0] = torch.randperm(
            cache_tokens, generator=generator
        )[:TOPK].to(torch.int32)
    sparse_slots = sparse_slots_cpu.to(device)
    cache_tokens_tensor = torch.full(
        (batch_size,), cache_tokens, dtype=torch.int32, device=device
    )
    block_table = block_table_cpu.to(device)
    actual_q = torch.arange(
        1, batch_size + 1, dtype=torch.int32, device=device
    )
    actual_kv = torch.full(
        (batch_size,), actual_len, dtype=torch.int32, device=device
    )
    actual_kv_list = [actual_len] * batch_size
    scale = 1.0 / math.sqrt(CKV_DIM + KPE_DIM)

    sparse_outputs: list[torch.Tensor] = []

    def launch_sparse() -> None:
        sparse_outputs[:] = [
            sparse_attention(
                query,
                key,
                sparse_slots,
                cache_tokens_tensor,
                block_table,
                actual_q,
                actual_kv,
                query_rope,
                key_rope,
                scale,
            )
        ]

    query_v2 = query.view(batch_size, heads, 1, CKV_DIM)
    query_rope_v2 = query_rope.view(batch_size, heads, 1, KPE_DIM)
    key_v2 = key.view(-1, 1, BLOCK_SIZE, CKV_DIM)
    key_rope_v2 = key_rope.view(-1, 1, BLOCK_SIZE, KPE_DIM)
    dense_kwargs = {
        "query_rope": query_rope_v2,
        "key_rope": key_rope_v2,
        "num_query_heads": heads,
        "num_key_value_heads": 1,
        "input_layout": "BNSD_NBSD",
        "atten_mask": None,
        "sparse_mode": 0,
        "softmax_scale": scale,
        "block_table": block_table,
        "block_size": BLOCK_SIZE,
        "actual_seq_qlen": None,
        "actual_seq_kvlen": actual_kv_list,
    }
    dense_out = torch.empty(
        heads,
        batch_size,
        1,
        CKV_DIM,
        dtype=query.dtype,
        device=device,
    )
    dense_lse = torch.empty(batch_size, dtype=query.dtype, device=device)
    dense_workspace = (
        torch_npu._npu_fused_infer_attention_score_v2_get_max_workspace(
            query_v2, key_v2, key_v2, **dense_kwargs
        )
    )

    def launch_dense() -> None:
        torch_npu.npu_fused_infer_attention_score_v2.out(
            query_v2,
            key_v2,
            key_v2,
            **dense_kwargs,
            workspace=dense_workspace,
            out=[dense_out, dense_lse],
        )

    sparse_graph = capture_graph(launch_sparse)
    dense_graph = capture_graph(launch_dense)
    sparse_ms = benchmark_graph(sparse_graph, warmup, iters)
    dense_ms = benchmark_graph(dense_graph, warmup, iters)
    speedup = dense_ms / max(sparse_ms, 1e-9)
    print(
        "SPARSE_TAIL_ATTENTION_RESULT "
        f"batch={batch_size} heads={heads} cache_tokens={cache_tokens} "
        f"tail_tokens={tail_tokens} attended_tokens={TOPK + tail_tokens} "
        f"dense_tokens={actual_len} sparse_ms={sparse_ms:.6f} "
        f"dense_mla_ms={dense_ms:.6f} speedup={speedup:.4f} "
        f"warmup={warmup} iters={iters}"
    )


def main() -> None:
    args = parse_args()
    if (
        args.batch_size <= 0
        or args.cache_tokens < TOPK
        or args.tail_tokens < 0
        or args.warmup < 0
        or args.iters <= 0
    ):
        raise ValueError(
            "batch/cache/iters must be positive, cache_tokens >= 2048, and "
            "tail/warmup >= 0."
        )
    device = torch.device(args.device)
    if device.type != "npu":
        raise ValueError("--device must select one NPU, for example npu:0.")
    torch.npu.set_device(device)
    torch.npu.config.allow_internal_format = False
    opapi_path = require_local_opapi()
    print(f"SPARSE_TAIL_ATTENTION_OPAPI path={opapi_path} local=1")
    run_semantic_check(device, args.heads, args.seed)
    if not args.skip_performance:
        run_performance(
            device,
            heads=args.heads,
            batch_size=args.batch_size,
            cache_tokens=args.cache_tokens,
            tail_tokens=args.tail_tokens,
            warmup=args.warmup,
            iters=args.iters,
            seed=args.seed,
        )
    print("SPARSE_TAIL_ATTENTION_UT_OK")


if __name__ == "__main__":
    main()
