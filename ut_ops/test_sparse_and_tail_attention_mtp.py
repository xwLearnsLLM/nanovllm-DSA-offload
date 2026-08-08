"""Semantic, graph and latency UT for MTP3 sparse-and-tail MLA Attention."""

from __future__ import annotations

import argparse
import math
import os
from time import perf_counter

import torch
import torch_npu  # type: ignore

import nanovllm.ops  # noqa: F401  # Load repository-local custom operators.


BLOCK_SIZE = 128
CKV_DIM = 512
KPE_DIM = 64
QUERY_COUNT = 4
TOPK = 2048


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate NanovllmSparseAndTailAttentionMtp against a CPU "
            "golden and four serial single-query launches."
        )
    )
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--heads", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--cache-tokens", type=int, default=8192)
    parser.add_argument("--tail-tokens", type=int, default=64)
    parser.add_argument("--graph-replays", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--min-speedup", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--skip-performance",
        action="store_true",
        help="Run semantic and graph checks without the latency comparison.",
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


def logical_rows(
    cache: torch.Tensor,
    block_table: torch.Tensor,
    request: int,
    logical_slots: torch.Tensor,
) -> torch.Tensor:
    slots = logical_slots.to(torch.int64)
    blocks = block_table[request, slots // BLOCK_SIZE].to(torch.int64)
    return cache[blocks, slots % BLOCK_SIZE, 0]


def call_mtp(
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
    *,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    args = (
        query,
        key,
        key,
        sparse_slots,
        cache_tokens,
        block_table,
        actual_q,
        actual_kv,
        query_rope,
        key_rope,
        scale,
    )
    if out is None:
        return torch.ops.nanovllm_dsa.sparse_and_tail_attention_mtp.default(
            *args
        )
    return torch.ops.nanovllm_dsa.sparse_and_tail_attention_mtp_out.default(
        *args, out
    )


def call_single(
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
    return torch.ops.nanovllm_dsa.sparse_and_tail_attention.default(
        query,
        key,
        key,
        sparse_slots,
        cache_tokens,
        block_table,
        actual_q,
        actual_kv,
        query_rope,
        key_rope,
        scale,
    )


def make_sparse_slots(
    cache_tokens: list[int],
    generator: torch.Generator,
) -> torch.Tensor:
    slots = torch.zeros(
        len(cache_tokens) * QUERY_COUNT, 1, TOPK, dtype=torch.int32
    )
    for request, cache_count in enumerate(cache_tokens):
        if cache_count == 0:
            continue
        if cache_count < TOPK:
            raise ValueError("nonzero cache_tokens must be at least 2048")
        for query_idx in range(QUERY_COUNT):
            row = request * QUERY_COUNT + query_idx
            slots[row, 0] = torch.randperm(
                cache_count, generator=generator
            )[:TOPK].to(torch.int32)
    return slots


def meta_check() -> None:
    batch_size = 2
    query = torch.empty(
        batch_size * QUERY_COUNT, 2, CKV_DIM,
        dtype=torch.bfloat16,
        device="meta",
    )
    key = torch.empty(
        2, BLOCK_SIZE, 1, CKV_DIM,
        dtype=query.dtype,
        device="meta",
    )
    sparse = torch.empty(
        batch_size * QUERY_COUNT, 1, TOPK,
        dtype=torch.int32,
        device="meta",
    )
    cache_tokens = torch.empty(batch_size, dtype=torch.int32, device="meta")
    block_table = torch.empty(batch_size, 2, dtype=torch.int32, device="meta")
    lengths = torch.empty(batch_size, dtype=torch.int32, device="meta")
    q_rope = torch.empty(
        batch_size * QUERY_COUNT, 2, KPE_DIM,
        dtype=query.dtype,
        device="meta",
    )
    k_rope = torch.empty(
        2, BLOCK_SIZE, 1, KPE_DIM,
        dtype=query.dtype,
        device="meta",
    )
    output = call_mtp(
        query, key, sparse, cache_tokens, block_table, lengths, lengths,
        q_rope, k_rope, 1.0,
    )
    out_buffer = torch.empty_like(query)
    out_result = call_mtp(
        query, key, sparse, cache_tokens, block_table, lengths, lengths,
        q_rope, k_rope, 1.0, out=out_buffer,
    )
    if (
        output.shape != query.shape
        or not torch._C._is_alias_of(out_result, out_buffer)
    ):
        raise AssertionError("MTP Attention Meta/Fake output contract is invalid")
    print("MTP_SPARSE_TAIL_META_CHECK alloc=1 out=1 alias=1 ok=1")


def run_semantic_check(
    device: torch.device,
    heads: int,
    graph_replays: int,
    seed: int,
) -> None:
    # tail_tokens excludes the four verification rows already appended to KV.
    cases = (
        (0, 1),
        (0, 129),
        (2048, 0),
        (4096, 257),
        (8192, 1025),
    )
    generator = torch.Generator().manual_seed(seed)
    batch_size = len(cases)
    cache_counts = [case[0] for case in cases]
    final_lens = [cache + tail + QUERY_COUNT for cache, tail in cases]
    blocks_per_request = math.ceil(max(final_lens) / BLOCK_SIZE)
    block_table_cpu, total_blocks = random_block_table(
        batch_size, blocks_per_request, generator
    )

    query_cpu = (
        torch.randn(
            batch_size * QUERY_COUNT,
            heads,
            CKV_DIM,
            generator=generator,
            dtype=torch.float32,
        )
        .mul_(0.25)
        .to(torch.bfloat16)
    )
    query_rope_cpu = (
        torch.randn(
            batch_size * QUERY_COUNT,
            heads,
            KPE_DIM,
            generator=generator,
            dtype=torch.float32,
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
    sparse_cpu = make_sparse_slots(cache_counts, generator)

    query = query_cpu.to(device)
    query_rope = query_rope_cpu.to(device)
    key = key_cpu.to(device)
    key_rope = key_rope_cpu.to(device)
    sparse = sparse_cpu.to(device)
    cache_tokens = torch.tensor(cache_counts, dtype=torch.int32, device=device)
    block_table = block_table_cpu.to(device)
    actual_q = torch.arange(
        QUERY_COUNT,
        batch_size * QUERY_COUNT + 1,
        QUERY_COUNT,
        dtype=torch.int32,
        device=device,
    )
    actual_kv = torch.tensor(final_lens, dtype=torch.int32, device=device)
    scale = 1.0 / math.sqrt(CKV_DIM + KPE_DIM)

    output = call_mtp(
        query, key, sparse, cache_tokens, block_table, actual_q, actual_kv,
        query_rope, key_rope, scale,
    )
    output_buffer = torch.empty_like(query)
    output_from_out = call_mtp(
        query, key, sparse, cache_tokens, block_table, actual_q, actual_kv,
        query_rope, key_rope, scale, out=output_buffer,
    )
    torch.npu.synchronize()
    if output_from_out.data_ptr() != output_buffer.data_ptr():
        raise AssertionError("MTP Attention _out did not return its output buffer")
    torch.testing.assert_close(output, output_from_out, rtol=0, atol=0)

    golden_rows: list[torch.Tensor] = []
    for request, (cache_count, tail_count) in enumerate(cases):
        for query_idx in range(QUERY_COUNT):
            row = request * QUERY_COUNT + query_idx
            visible_len = cache_count + tail_count + query_idx + 1
            if cache_count == 0:
                logical_slots = torch.arange(visible_len, dtype=torch.int64)
            else:
                logical_slots = torch.cat(
                    (
                        sparse_cpu[row, 0].to(torch.int64),
                        torch.arange(
                            cache_count, visible_len, dtype=torch.int64
                        ),
                    )
                )
            selected_key = logical_rows(
                key_cpu, block_table_cpu, request, logical_slots
            ).float()
            selected_rope = logical_rows(
                key_rope_cpu, block_table_cpu, request, logical_slots
            ).float()
            scores = (
                query_cpu[row].float() @ selected_key.T
                + query_rope_cpu[row].float() @ selected_rope.T
            ) * scale
            golden_rows.append(torch.softmax(scores, dim=-1) @ selected_key)
    golden = torch.stack(golden_rows)
    actual = output.float().cpu()
    torch.testing.assert_close(actual, golden, rtol=0.08, atol=0.08)
    max_abs = float((actual - golden).abs().max())
    print(
        "MTP_SPARSE_TAIL_SEMANTIC_CHECK "
        f"heads={heads} cases={list(cases)} max_abs={max_abs:.6f} "
        "alloc_out_equal=1 causal_rows=4 ok=1"
    )

    graph_output = torch.empty_like(query)

    def graph_launch() -> None:
        call_mtp(
            query, key, sparse, cache_tokens, block_table, actual_q, actual_kv,
            query_rope, key_rope, scale, out=graph_output,
        )

    graph_launch()
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    pool = torch.npu.graph_pool_handle()
    with torch.npu.graph(graph, pool=pool):
        graph_launch()
    torch.npu.synchronize()

    base_actual_kv = torch.tensor(final_lens, dtype=torch.int32, device=device)
    for replay in range(graph_replays):
        query.copy_(torch.randn_like(query).mul_(0.25))
        query_rope.copy_(torch.randn_like(query_rope).mul_(0.25))
        sparse.copy_(torch.roll(sparse, shifts=replay + 1, dims=2))
        block_table.copy_(
            torch.roll(block_table, shifts=replay + 1, dims=1)
        )
        dynamic_lens = list(final_lens)
        for request, (_, tail_count) in enumerate(cases):
            if tail_count > 0 and (request + replay) % 2:
                dynamic_lens[request] -= 1
        actual_kv.copy_(
            torch.tensor(dynamic_lens, dtype=torch.int32, device=device)
        )
        eager = call_mtp(
            query, key, sparse, cache_tokens, block_table, actual_q, actual_kv,
            query_rope, key_rope, scale,
        )
        graph.replay()
        torch.npu.synchronize()
        torch.testing.assert_close(graph_output, eager, rtol=0, atol=0)
    actual_kv.copy_(base_actual_kv)
    print(
        "MTP_SPARSE_TAIL_GRAPH_CHECK "
        f"replays={graph_replays} dynamic_query=1 dynamic_sparse_slots=1 "
        "dynamic_kvlen=1 dynamic_block_table=1 out_buffer=1 ok=1"
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
    cache_count: int,
    tail_count: int,
    warmup: int,
    iters: int,
    min_speedup: float,
    seed: int,
) -> None:
    final_len = cache_count + tail_count + QUERY_COUNT
    blocks_per_request = math.ceil(final_len / BLOCK_SIZE)
    generator = torch.Generator().manual_seed(seed + 1000)
    block_table_cpu, total_blocks = random_block_table(
        batch_size, blocks_per_request, generator
    )
    query_rows = [
        torch.randn(
            batch_size, heads, CKV_DIM,
            dtype=torch.bfloat16,
            device=device,
        )
        for _ in range(QUERY_COUNT)
    ]
    query_rope_rows = [
        torch.randn(
            batch_size, heads, KPE_DIM,
            dtype=torch.bfloat16,
            device=device,
        )
        for _ in range(QUERY_COUNT)
    ]
    query = torch.stack(query_rows, dim=1).reshape(
        batch_size * QUERY_COUNT, heads, CKV_DIM
    )
    query_rope = torch.stack(query_rope_rows, dim=1).reshape(
        batch_size * QUERY_COUNT, heads, KPE_DIM
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
    sparse_cpu = make_sparse_slots([cache_count] * batch_size, generator)
    sparse = sparse_cpu.to(device)
    sparse_rows = [
        sparse.view(batch_size, QUERY_COUNT, 1, TOPK)[:, query_idx]
        .contiguous()
        for query_idx in range(QUERY_COUNT)
    ]
    cache_tokens = torch.full(
        (batch_size,), cache_count, dtype=torch.int32, device=device
    )
    block_table = block_table_cpu.to(device)
    actual_q_mtp = torch.arange(
        QUERY_COUNT,
        batch_size * QUERY_COUNT + 1,
        QUERY_COUNT,
        dtype=torch.int32,
        device=device,
    )
    actual_q_single = torch.arange(
        1, batch_size + 1, dtype=torch.int32, device=device
    )
    actual_kv_mtp = torch.full(
        (batch_size,), final_len, dtype=torch.int32, device=device
    )
    actual_kv_rows = [
        torch.full(
            (batch_size,),
            cache_count + tail_count + query_idx + 1,
            dtype=torch.int32,
            device=device,
        )
        for query_idx in range(QUERY_COUNT)
    ]
    scale = 1.0 / math.sqrt(CKV_DIM + KPE_DIM)

    mtp_output = torch.empty_like(query)

    def launch_mtp() -> None:
        call_mtp(
            query, key, sparse, cache_tokens, block_table,
            actual_q_mtp, actual_kv_mtp, query_rope, key_rope, scale,
            out=mtp_output,
        )

    serial_outputs: list[torch.Tensor] = []

    def launch_serial() -> None:
        serial_outputs[:] = [
            call_single(
                query_rows[query_idx],
                key,
                sparse_rows[query_idx],
                cache_tokens,
                block_table,
                actual_q_single,
                actual_kv_rows[query_idx],
                query_rope_rows[query_idx],
                key_rope,
                scale,
            )
            for query_idx in range(QUERY_COUNT)
        ]

    launch_mtp()
    launch_serial()
    torch.npu.synchronize()
    serial_packed = torch.stack(serial_outputs, dim=1).reshape_as(mtp_output)
    torch.testing.assert_close(mtp_output, serial_packed, rtol=0, atol=0)

    mtp_graph = capture_graph(launch_mtp)
    serial_graph = capture_graph(launch_serial)
    mtp_ms = benchmark_graph(mtp_graph, warmup, iters)
    serial_ms = benchmark_graph(serial_graph, warmup, iters)
    speedup = serial_ms / max(mtp_ms, 1e-9)
    print(
        "MTP_SPARSE_TAIL_PERF_RESULT "
        f"batch={batch_size} heads={heads} cache_tokens={cache_count} "
        f"tail_tokens={tail_count} max_attended={TOPK + tail_count + 4} "
        f"mtp_ms={mtp_ms:.6f} four_serial_ms={serial_ms:.6f} "
        f"speedup={speedup:.4f} min_speedup={min_speedup:.4f} "
        f"warmup={warmup} iters={iters}"
    )
    if speedup < min_speedup:
        raise AssertionError(
            f"MTP Attention speedup={speedup:.4f} is below {min_speedup:.4f}"
        )


def main() -> None:
    args = parse_args()
    if args.heads <= 0 or args.heads > 128 or args.heads & (args.heads - 1):
        raise ValueError("--heads must be a power of two in [1, 128]")
    if (
        args.batch_size <= 0
        or args.cache_tokens < TOPK
        or args.tail_tokens < 0
        or args.graph_replays < 1
        or args.warmup < 0
        or args.iters <= 0
        or args.min_speedup < 0
    ):
        raise ValueError("invalid batch/cache/tail/replay/performance arguments")
    device = torch.device(args.device)
    if device.type != "npu":
        raise ValueError("--device must select one NPU, for example npu:0")
    torch.npu.set_device(device)
    torch.npu.config.allow_internal_format = False
    opapi_path = os.environ.get("NANOVLLM_CUST_OPAPI_LIB", "")
    if not opapi_path or not os.path.isfile(opapi_path):
        raise RuntimeError(
            "Repository-local libcust_opapi.so was not selected; rebuild "
            "with `bash scripts/build_nanovllm_ops.sh`."
        )
    print(f"MTP_SPARSE_TAIL_OPAPI path={opapi_path} local=1")
    print(
        "MTP_SPARSE_TAIL_CONFIG "
        f"device={device} query_len={QUERY_COUNT} heads={args.heads} "
        f"topk={TOPK} batch={args.batch_size} "
        f"cache_tokens={args.cache_tokens} tail_tokens={args.tail_tokens} "
        f"seed={args.seed}"
    )
    meta_check()
    run_semantic_check(
        device, args.heads, args.graph_replays, args.seed
    )
    if not args.skip_performance:
        run_performance(
            device,
            heads=args.heads,
            batch_size=args.batch_size,
            cache_count=args.cache_tokens,
            tail_count=args.tail_tokens,
            warmup=args.warmup,
            iters=args.iters,
            min_speedup=args.min_speedup,
            seed=args.seed,
        )
    print("MTP_SPARSE_TAIL_ATTENTION_UT_OK")


if __name__ == "__main__":
    main()
