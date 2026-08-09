"""Correctness and fixed-miss benchmark for bundled KV-cache SCATTER."""

from __future__ import annotations

import argparse
import math
from time import perf_counter

import torch
import torch_npu  # type: ignore

import nanovllm.ops  # noqa: F401  # Load repository-local custom operators.
from ut_ops._op_utils import require_local_opapi


BLOCK_SIZE = 128
KPE_DIM = 64
CKV_DIM = 512
BYTES_PER_TOKEN = (KPE_DIM + CKV_DIM) * 2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate and benchmark NanovllmKvcacheScatterCopy."
    )
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--source-len", type=int, default=20992)
    parser.add_argument("--cache-tokens", type=int, default=5120)
    parser.add_argument("--copy-cap", type=int, default=2048)
    parser.add_argument(
        "--miss-counts",
        default="0,256,512,1024,1536,2048",
        help="Comma-separated uniform per-request copy counts.",
    )
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--skip-performance",
        action="store_true",
        help="Run semantic checks only.",
    )
    return parser.parse_args()


def parse_miss_counts(value: str, copy_cap: int) -> tuple[int, ...]:
    counts = tuple(dict.fromkeys(int(item.strip()) for item in value.split(",")))
    if not counts or any(count < 0 or count > copy_cap for count in counts):
        raise ValueError(f"--miss-counts values must be in [0, {copy_cap}].")
    return counts


def random_block_table(
    batch_size: int,
    blocks_per_request: int,
    generator: torch.Generator,
) -> tuple[torch.Tensor, int]:
    total_blocks = batch_size * blocks_per_request
    table = torch.randperm(total_blocks, generator=generator).to(torch.int32)
    return table.view(batch_size, blocks_per_request).contiguous(), total_blocks


def swapped_from_cpu(cpu: torch.Tensor, device: torch.device) -> torch.Tensor:
    tensor = torch_npu.empty_with_swapped_memory(
        cpu.shape,
        dtype=cpu.dtype,
        device=device,
    )
    tensor.fill_(0)
    tensor.add_(cpu.to(device))
    return tensor


def scatter(
    hbm_kpe: torch.Tensor,
    hbm_ckv: torch.Tensor,
    dram_kpe: torch.Tensor,
    dram_ckv: torch.Tensor,
    hbm_table: torch.Tensor,
    dram_table: torch.Tensor,
    source_ids: torch.Tensor,
    destination_slots: torch.Tensor,
    copy_counts: torch.Tensor,
) -> None:
    torch.ops.nanovllm_dsa.scatter_copy.default(
        source_ids,
        destination_slots,
        copy_counts,
        hbm_table,
        dram_table,
        hbm_kpe,
        hbm_ckv,
        dram_kpe,
        dram_ckv,
    )


def make_metadata(
    batch_size: int,
    copy_cap: int,
    source_len: int,
    cache_tokens: int,
    counts: tuple[int, ...],
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    source_ids = torch.full((batch_size, copy_cap), -1, dtype=torch.int32)
    destination_slots = torch.full_like(source_ids, -1)
    for row, count in enumerate(counts):
        if count == 0:
            continue
        source_ids[row, :count] = torch.randperm(
            source_len, generator=generator
        )[:count].to(torch.int32)
        destination_slots[row, :count] = torch.randperm(
            cache_tokens, generator=generator
        )[:count].to(torch.int32)
    return (
        source_ids.contiguous(),
        destination_slots.contiguous(),
        torch.tensor(counts, dtype=torch.int32),
    )


def apply_reference(
    expected_kpe: torch.Tensor,
    expected_ckv: torch.Tensor,
    dram_kpe: torch.Tensor,
    dram_ckv: torch.Tensor,
    hbm_table: torch.Tensor,
    dram_table: torch.Tensor,
    source_ids: torch.Tensor,
    destination_slots: torch.Tensor,
    counts: torch.Tensor,
) -> None:
    for row, count_value in enumerate(counts.tolist()):
        count = int(count_value)
        if count == 0:
            continue
        sources = source_ids[row, :count].to(torch.int64)
        destinations = destination_slots[row, :count].to(torch.int64)
        src_blocks = dram_table[row, sources // BLOCK_SIZE].to(torch.int64)
        src_offsets = sources % BLOCK_SIZE
        dst_blocks = hbm_table[row, destinations // BLOCK_SIZE].to(torch.int64)
        dst_offsets = destinations % BLOCK_SIZE
        expected_kpe[dst_blocks, dst_offsets] = dram_kpe[src_blocks, src_offsets]
        expected_ckv[dst_blocks, dst_offsets] = dram_ckv[src_blocks, src_offsets]


def run_correctness_case(
    device: torch.device,
    *,
    label: str,
    batch_size: int,
    source_len: int,
    cache_tokens: int,
    copy_cap: int,
    rounds: tuple[tuple[int, ...], ...],
    seed: int,
) -> None:
    if any(len(row_counts) != batch_size for row_counts in rounds):
        raise ValueError("every SCATTER correctness round must match batch_size")
    generator = torch.Generator().manual_seed(seed)
    dram_table_cpu, dram_blocks = random_block_table(
        batch_size, source_len // BLOCK_SIZE, generator
    )
    hbm_table_cpu, hbm_blocks = random_block_table(
        batch_size, cache_tokens // BLOCK_SIZE, generator
    )
    dram_kpe_cpu = torch.randn(
        dram_blocks,
        BLOCK_SIZE,
        KPE_DIM,
        generator=generator,
        dtype=torch.float32,
    ).to(torch.bfloat16)
    dram_ckv_cpu = torch.randn(
        dram_blocks,
        BLOCK_SIZE,
        CKV_DIM,
        generator=generator,
        dtype=torch.float32,
    ).to(torch.bfloat16)
    expected_kpe = torch.zeros(
        hbm_blocks, BLOCK_SIZE, KPE_DIM, dtype=torch.bfloat16
    )
    expected_ckv = torch.zeros(
        hbm_blocks, BLOCK_SIZE, CKV_DIM, dtype=torch.bfloat16
    )

    dram_kpe = swapped_from_cpu(dram_kpe_cpu, device)
    dram_ckv = swapped_from_cpu(dram_ckv_cpu, device)
    hbm_kpe = torch.zeros_like(expected_kpe, device=device)
    hbm_ckv = torch.zeros_like(expected_ckv, device=device)
    dram_table = dram_table_cpu.to(device)
    hbm_table = hbm_table_cpu.to(device)

    for step, row_counts in enumerate(rounds):
        source_ids_cpu, slots_cpu, counts_cpu = make_metadata(
            batch_size,
            copy_cap,
            source_len,
            cache_tokens,
            row_counts,
            generator,
        )
        scatter(
            hbm_kpe,
            hbm_ckv,
            dram_kpe,
            dram_ckv,
            hbm_table,
            dram_table,
            source_ids_cpu.to(device),
            slots_cpu.to(device),
            counts_cpu.to(device),
        )
        torch.npu.synchronize()
        apply_reference(
            expected_kpe,
            expected_ckv,
            dram_kpe_cpu,
            dram_ckv_cpu,
            hbm_table_cpu,
            dram_table_cpu,
            source_ids_cpu,
            slots_cpu,
            counts_cpu,
        )
        if not torch.equal(hbm_kpe.cpu(), expected_kpe):
            raise AssertionError(f"SCATTER KPE mismatch at update step {step}")
        if not torch.equal(hbm_ckv.cpu(), expected_ckv):
            raise AssertionError(f"SCATTER CKV mismatch at update step {step}")
        print(
            f"SCATTER_COPY_CHECK case={label} copy_cap={copy_cap} "
            f"step={step} counts={list(row_counts)} "
            "random_block_tables=1 ok=1"
        )

    zero_sources = torch.full(
        (batch_size, copy_cap), -1, dtype=torch.int32, device=device
    )
    zero_counts = torch.zeros(batch_size, dtype=torch.int32, device=device)
    scatter(
        hbm_kpe,
        hbm_ckv,
        dram_kpe,
        dram_ckv,
        hbm_table,
        dram_table,
        zero_sources,
        zero_sources,
        zero_counts,
    )
    torch.npu.synchronize()
    if not torch.equal(hbm_kpe.cpu(), expected_kpe) or not torch.equal(
        hbm_ckv.cpu(), expected_ckv
    ):
        raise AssertionError("zero-count SCATTER modified the HBM cache")
    print(
        f"SCATTER_COPY_ZERO_COUNT_CHECK case={label} "
        f"copy_cap={copy_cap} ok=1"
    )

    # Capture with zero copies, then replay the same fixed-shape graph after
    # refreshing only copy_counts.  This is the contract used when an earlier
    # MTP LIM writes a variable-length union miss list in the graph.
    graph_sources_cpu, graph_slots_cpu, graph_counts_cpu = make_metadata(
        batch_size,
        copy_cap,
        source_len,
        cache_tokens,
        rounds[-1],
        generator,
    )
    graph_sources = graph_sources_cpu.to(device)
    graph_slots = graph_slots_cpu.to(device)
    graph_counts = torch.zeros(batch_size, dtype=torch.int32, device=device)
    graph_kpe = torch.zeros_like(hbm_kpe)
    graph_ckv = torch.zeros_like(hbm_ckv)
    graph = torch.npu.NPUGraph()
    pool = torch.npu.graph_pool_handle()
    with torch.npu.graph(graph, pool=pool):
        scatter(
            graph_kpe,
            graph_ckv,
            dram_kpe,
            dram_ckv,
            hbm_table,
            dram_table,
            graph_sources,
            graph_slots,
            graph_counts,
        )
    torch.npu.synchronize()

    graph_kpe.zero_()
    graph_ckv.zero_()
    graph_counts.copy_(graph_counts_cpu.to(device))
    torch.npu.synchronize()
    graph.replay()
    torch.npu.synchronize()
    expected_graph_kpe = torch.zeros_like(expected_kpe)
    expected_graph_ckv = torch.zeros_like(expected_ckv)
    apply_reference(
        expected_graph_kpe,
        expected_graph_ckv,
        dram_kpe_cpu,
        dram_ckv_cpu,
        hbm_table_cpu,
        dram_table_cpu,
        graph_sources_cpu,
        graph_slots_cpu,
        graph_counts_cpu,
    )
    if not torch.equal(graph_kpe.cpu(), expected_graph_kpe):
        raise AssertionError(f"{label}: graph replay KPE mismatch")
    if not torch.equal(graph_ckv.cpu(), expected_graph_ckv):
        raise AssertionError(f"{label}: graph replay CKV mismatch")
    print(
        f"SCATTER_COPY_GRAPH_CHECK case={label} copy_cap={copy_cap} "
        f"counts={graph_counts_cpu.tolist()} dynamic_counts=1 ok=1"
    )

    del (
        dram_kpe,
        dram_ckv,
        hbm_kpe,
        hbm_ckv,
        graph_kpe,
        graph_ckv,
        graph,
    )
    torch.npu.empty_cache()


def run_correctness(device: torch.device, seed: int) -> None:
    run_correctness_case(
        device,
        label="legacy_2048",
        batch_size=6,
        source_len=4096,
        cache_tokens=2048,
        copy_cap=2048,
        rounds=(
            (0, 1, 47, 256, 1024, 2048),
            (2048, 1024, 257, 31, 1, 0),
        ),
        seed=seed,
    )
    run_correctness_case(
        device,
        label="mtp_union_8192",
        batch_size=6,
        source_len=16384,
        cache_tokens=8192,
        copy_cap=8192,
        rounds=(
            (0, 1, 2047, 2048, 4097, 8192),
            (8192, 6145, 4096, 2049, 1, 0),
        ),
        seed=seed + 100,
    )


def run_performance(
    device: torch.device,
    *,
    batch_size: int,
    source_len: int,
    cache_tokens: int,
    copy_cap: int,
    miss_counts: tuple[int, ...],
    warmup: int,
    iters: int,
    seed: int,
) -> None:
    generator = torch.Generator().manual_seed(seed + 1000)
    source_blocks = math.ceil(source_len / BLOCK_SIZE)
    hbm_blocks_per_request = math.ceil(cache_tokens / BLOCK_SIZE)
    dram_table_cpu, dram_blocks = random_block_table(
        batch_size, source_blocks, generator
    )
    hbm_table_cpu, hbm_blocks = random_block_table(
        batch_size, hbm_blocks_per_request, generator
    )
    dram_kpe = torch_npu.empty_with_swapped_memory(
        (dram_blocks, BLOCK_SIZE, KPE_DIM),
        dtype=torch.bfloat16,
        device=device,
    )
    dram_ckv = torch_npu.empty_with_swapped_memory(
        (dram_blocks, BLOCK_SIZE, CKV_DIM),
        dtype=torch.bfloat16,
        device=device,
    )
    hbm_kpe = torch.empty(
        hbm_blocks, BLOCK_SIZE, KPE_DIM, dtype=torch.bfloat16, device=device
    )
    hbm_ckv = torch.empty(
        hbm_blocks, BLOCK_SIZE, CKV_DIM, dtype=torch.bfloat16, device=device
    )
    dram_table = dram_table_cpu.to(device)
    hbm_table = hbm_table_cpu.to(device)
    source_ids_cpu, slots_cpu, _ = make_metadata(
        batch_size,
        copy_cap,
        source_len,
        cache_tokens,
        (copy_cap,) * batch_size,
        generator,
    )
    source_ids = source_ids_cpu.to(device)
    destination_slots = slots_cpu.to(device)
    copy_counts = torch.zeros(batch_size, dtype=torch.int32, device=device)

    def launch() -> None:
        scatter(
            hbm_kpe,
            hbm_ckv,
            dram_kpe,
            dram_ckv,
            hbm_table,
            dram_table,
            source_ids,
            destination_slots,
            copy_counts,
        )

    launch()
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    pool = torch.npu.graph_pool_handle()
    with torch.npu.graph(graph, pool=pool):
        launch()
    torch.npu.synchronize()

    for count in miss_counts:
        copy_counts.fill_(count)
        torch.npu.synchronize()
        for _ in range(warmup):
            graph.replay()
        torch.npu.synchronize()
        start = perf_counter()
        for _ in range(iters):
            graph.replay()
        torch.npu.synchronize()
        avg_ms = (perf_counter() - start) * 1000.0 / iters
        payload_gbps = (
            batch_size * count * BYTES_PER_TOKEN / max(avg_ms, 1e-9) / 1e6
        )
        print(
            f"SCATTER_COPY_RESULT batch={batch_size} source_len={source_len} "
            f"cache_tokens={cache_tokens} copy_cap={copy_cap} "
            f"misses_per_row={count} total_misses={batch_size * count} "
            f"graph_avg_ms={avg_ms:.6f} payload_gbps={payload_gbps:.3f} "
            f"warmup={warmup} iters={iters}"
        )


def main() -> None:
    args = parse_args()
    if (
        args.batch_size <= 0
        or args.source_len <= 0
        or args.cache_tokens <= 0
        or args.copy_cap <= 0
        or args.copy_cap > 65536
        or args.copy_cap > args.source_len
        or args.copy_cap > args.cache_tokens
        or args.warmup < 0
        or args.iters <= 0
    ):
        raise ValueError(
            "batch/source/cache/copy-cap/iters must be positive, copy-cap must "
            "not exceed source/cache or 65536, and warmup must be >= 0."
        )
    miss_counts = parse_miss_counts(args.miss_counts, args.copy_cap)
    device = torch.device(args.device)
    if device.type != "npu":
        raise ValueError("--device must select one NPU, for example npu:0.")
    torch.npu.set_device(device)
    torch.npu.config.allow_internal_format = False
    opapi_path = require_local_opapi()
    print(f"SCATTER_COPY_OPAPI path={opapi_path} local=1")
    run_correctness(device, args.seed)
    if not args.skip_performance:
        run_performance(
            device,
            batch_size=args.batch_size,
            source_len=args.source_len,
            cache_tokens=args.cache_tokens,
            copy_cap=args.copy_cap,
            miss_counts=miss_counts,
            warmup=args.warmup,
            iters=args.iters,
            seed=args.seed,
        )
    print("SCATTER_COPY_UT_OK")


if __name__ == "__main__":
    main()
