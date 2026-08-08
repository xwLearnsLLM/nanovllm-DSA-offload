#!/usr/bin/env python3
"""Compare split and fused BF16 copy+sparse-tail Attention paths."""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass

import torch

# This package selects the repository-local OPP before importing torch_npu.
import nanovllm_dsa_a5
import torch_npu  # noqa: E402,F401

from _utils import require_a5, swapped_from_cpu


BLOCK_SIZE = 128
CKV_DIM = 512
KPE_DIM = 64
SPARSE_COUNT = 2048
DEFAULT_TEST_HEADS = 8
SUPPORTED_TEST_HEADS = (1, 2, 4, 8, 16, 32, 64)


@dataclass
class Case:
    query: torch.Tensor
    query_rope: torch.Tensor
    sparse_slots: torch.Tensor
    cache_tokens: torch.Tensor
    hbm_block_table: torch.Tensor
    dram_block_table: torch.Tensor
    actual_q: torch.Tensor
    actual_kv: torch.Tensor
    source_token_ids: torch.Tensor
    copy_counts: torch.Tensor
    dram_kpe: torch.Tensor
    dram_ckv: torch.Tensor
    dram_kpe_cpu: torch.Tensor
    dram_ckv_cpu: torch.Tensor
    serial_kpe: torch.Tensor
    serial_ckv: torch.Tensor
    fused_kpe: torch.Tensor
    fused_ckv: torch.Tensor
    scale: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="npu:0")
    parser.add_argument(
        "--mode",
        choices=("all", "check", "bench", "profile"),
        default="all",
    )
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument(
        "--heads",
        type=int,
        choices=SUPPORTED_TEST_HEADS,
        default=DEFAULT_TEST_HEADS,
    )
    parser.add_argument("--source-len", type=int, default=65536)
    parser.add_argument("--cache-tokens", type=int, default=8192)
    parser.add_argument("--tail-tokens", type=int, default=64)
    parser.add_argument("--miss-min", type=int, default=0)
    parser.add_argument("--miss-max", type=int, default=300)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--profile-replays", type=int, default=4)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--allow-non-a5", action="store_true")
    return parser.parse_args()


def random_bf16(shape: tuple[int, ...], device: torch.device) -> torch.Tensor:
    return torch.randn(shape, device=device, dtype=torch.bfloat16)


def random_cpu_bf16(
    shape: tuple[int, ...], generator: torch.Generator
) -> torch.Tensor:
    return torch.randn(
        shape, generator=generator, dtype=torch.float32
    ).to(torch.bfloat16)


def assert_random_scatter_addresses(
    hbm_table: torch.Tensor,
    dram_table: torch.Tensor,
    destination_slots: torch.Tensor,
    source_token_ids: torch.Tensor,
    copy_counts: torch.Tensor,
) -> None:
    """Reject a benchmark that accidentally turns scatter into a linear copy."""

    def physical(
        table: torch.Tensor, logical: torch.Tensor
    ) -> torch.Tensor:
        blocks = table[logical // BLOCK_SIZE].to(torch.int64)
        return blocks * BLOCK_SIZE + logical.remainder(BLOCK_SIZE)

    for row, count_value in enumerate(copy_counts.tolist()):
        count = int(count_value)
        if count == 0:
            continue
        source = source_token_ids[row, :count].to(torch.int64)
        destination = destination_slots[row, :count].to(torch.int64)
        source_physical = physical(dram_table[row], source)
        destination_physical = physical(hbm_table[row], destination)
        for name, addresses in (
            ("source", source_physical),
            ("destination", destination_physical),
        ):
            if torch.unique(addresses).numel() != count:
                raise AssertionError(
                    f"Active {name} addresses must be unique: "
                    f"row={row}, count={count}."
                )
            if count < 4:
                continue
            unit_stride_pairs = int(
                (addresses[1:].sub(addresses[:-1]).abs() == 1).sum()
            )
            max_random_unit_stride_pairs = max(1, (count - 1) // 8)
            if unit_stride_pairs > max_random_unit_stride_pairs:
                raise AssertionError(
                    f"Active {name} addresses are insufficiently scattered: "
                    f"row={row}, count={count}, "
                    f"unit_stride_pairs={unit_stride_pairs}."
                )


def make_case(args: argparse.Namespace) -> Case:
    if args.batch_size <= 0:
        raise ValueError("batch-size must be positive.")
    if args.source_len <= 0 or args.source_len % BLOCK_SIZE != 0:
        raise ValueError("source-len must be positive and block aligned.")
    if args.cache_tokens < SPARSE_COUNT or args.cache_tokens % BLOCK_SIZE != 0:
        raise ValueError("cache-tokens must be >=2048 and block aligned.")
    if args.tail_tokens < 0:
        raise ValueError("tail-tokens must be non-negative.")
    if not 0 <= args.miss_min <= args.miss_max <= SPARSE_COUNT:
        raise ValueError(
            f"miss range must satisfy 0 <= min <= max <= {SPARSE_COUNT}."
        )

    torch.manual_seed(args.seed)
    torch.npu.manual_seed_all(args.seed)
    device = torch.device(args.device)
    batch = args.batch_size
    cache_blocks = math.ceil((args.cache_tokens + args.tail_tokens) / BLOCK_SIZE)
    source_blocks = args.source_len // BLOCK_SIZE

    # HBM blocks are private to each request. DRAM is read-only, so all rows
    # may use different permutations of one shared physical source pool.
    hbm_table_cpu = torch.empty((batch, cache_blocks), dtype=torch.int32)
    for row in range(batch):
        physical = torch.arange(
            row * cache_blocks, (row + 1) * cache_blocks, dtype=torch.int32
        )
        hbm_table_cpu[row] = physical[torch.randperm(cache_blocks)]
    dram_table_cpu = torch.stack(
        [
            torch.randperm(source_blocks, dtype=torch.int64).to(torch.int32)
            for _ in range(batch)
        ]
    )

    counts_cpu = torch.randint(
        args.miss_min,
        args.miss_max + 1,
        (batch,),
        dtype=torch.int32,
    )
    counts_cpu[0] = args.miss_min
    if batch > 1:
        counts_cpu[1] = args.miss_max

    slots_cpu = torch.empty((batch, SPARSE_COUNT), dtype=torch.int32)
    source_ids_cpu = torch.full(
        (batch, SPARSE_COUNT), -1, dtype=torch.int32
    )
    for row in range(batch):
        slots_cpu[row] = torch.randperm(
            args.cache_tokens, dtype=torch.int64
        )[:SPARSE_COUNT].to(torch.int32)
        count = int(counts_cpu[row])
        if count:
            source_ids_cpu[row, :count] = torch.randperm(
                args.source_len, dtype=torch.int64
            )[:count].to(torch.int32)
    assert_random_scatter_addresses(
        hbm_table_cpu,
        dram_table_cpu,
        slots_cpu,
        source_ids_cpu,
        counts_cpu,
    )

    total_hbm_blocks = batch * cache_blocks
    initial_kpe = random_bf16(
        (total_hbm_blocks, BLOCK_SIZE, 1, KPE_DIM), device
    )
    initial_ckv = random_bf16(
        (total_hbm_blocks, BLOCK_SIZE, 1, CKV_DIM), device
    )
    query = random_bf16(
        (batch, args.heads, CKV_DIM), device
    )
    query_rope = random_bf16(
        (batch, args.heads, KPE_DIM), device
    )
    dram_generator = torch.Generator().manual_seed(args.seed + 1000)
    dram_kpe_cpu = random_cpu_bf16(
        (source_blocks, BLOCK_SIZE, KPE_DIM), dram_generator
    )
    dram_ckv_cpu = random_cpu_bf16(
        (source_blocks, BLOCK_SIZE, CKV_DIM), dram_generator
    )
    # Keep an independent CPU oracle, while the tensors passed to both
    # kernels are backed by swapped (host DRAM) memory.
    dram_kpe = swapped_from_cpu(dram_kpe_cpu, device)
    dram_ckv = swapped_from_cpu(dram_ckv_cpu, device)
    torch.npu.synchronize()

    return Case(
        query=query,
        query_rope=query_rope,
        sparse_slots=slots_cpu[:, None, :].to(device),
        cache_tokens=torch.full(
            (batch,), args.cache_tokens, dtype=torch.int32, device=device
        ),
        hbm_block_table=hbm_table_cpu.to(device),
        dram_block_table=dram_table_cpu.to(device),
        actual_q=torch.arange(
            1, batch + 1, dtype=torch.int32, device=device
        ),
        actual_kv=torch.full(
            (batch,),
            args.cache_tokens + args.tail_tokens,
            dtype=torch.int32,
            device=device,
        ),
        source_token_ids=source_ids_cpu.to(device),
        copy_counts=counts_cpu.to(device),
        dram_kpe=dram_kpe,
        dram_ckv=dram_ckv,
        dram_kpe_cpu=dram_kpe_cpu,
        dram_ckv_cpu=dram_ckv_cpu,
        serial_kpe=initial_kpe.clone(),
        serial_ckv=initial_ckv.clone(),
        fused_kpe=initial_kpe.clone(),
        fused_ckv=initial_ckv.clone(),
        scale=1.0 / math.sqrt(CKV_DIM + KPE_DIM),
    )


def launch_serial(case: Case) -> torch.Tensor:
    nanovllm_dsa_a5.kvcache_scatter_copy(
        case.serial_kpe.view(
            case.serial_kpe.size(0), BLOCK_SIZE, KPE_DIM
        ),
        case.serial_ckv.view(
            case.serial_ckv.size(0), BLOCK_SIZE, CKV_DIM
        ),
        case.dram_kpe,
        case.dram_ckv,
        case.hbm_block_table,
        case.dram_block_table,
        case.source_token_ids,
        case.sparse_slots.view(case.sparse_slots.size(0), SPARSE_COUNT),
        case.copy_counts,
    )
    return nanovllm_dsa_a5.sparse_tail_attention(
        case.query,
        case.serial_ckv,
        case.serial_ckv,
        case.sparse_slots,
        case.cache_tokens,
        case.hbm_block_table,
        case.actual_q,
        case.actual_kv,
        case.query_rope,
        case.serial_kpe,
        case.scale,
    )


def launch_fused(case: Case) -> torch.Tensor:
    output, _, _ = (
        nanovllm_dsa_a5
        .fused_copy_sparse_tail_attention(
            case.query,
            case.fused_ckv,
            case.sparse_slots,
            case.cache_tokens,
            case.hbm_block_table,
            case.actual_q,
            case.actual_kv,
            case.query_rope,
            case.fused_kpe,
            case.dram_kpe,
            case.dram_ckv,
            case.dram_block_table,
            case.source_token_ids,
            case.copy_counts,
            case.scale,
        )
    )
    return output


def launch_scatter_only(
    case: Case,
) -> tuple[torch.Tensor, torch.Tensor]:
    return nanovllm_dsa_a5.kvcache_scatter_copy(
        case.serial_kpe.view(
            case.serial_kpe.size(0), BLOCK_SIZE, KPE_DIM
        ),
        case.serial_ckv.view(
            case.serial_ckv.size(0), BLOCK_SIZE, CKV_DIM
        ),
        case.dram_kpe,
        case.dram_ckv,
        case.hbm_block_table,
        case.dram_block_table,
        case.source_token_ids,
        case.sparse_slots.view(case.sparse_slots.size(0), SPARSE_COUNT),
        case.copy_counts,
    )


def launch_sfa_only(case: Case) -> torch.Tensor:
    return nanovllm_dsa_a5.sparse_tail_attention(
        case.query,
        case.serial_ckv,
        case.serial_ckv,
        case.sparse_slots,
        case.cache_tokens,
        case.hbm_block_table,
        case.actual_q,
        case.actual_kv,
        case.query_rope,
        case.serial_kpe,
        case.scale,
    )


def active_physical_token_indices(
    block_table: torch.Tensor,
    logical_tokens: torch.Tensor,
    counts: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    table = block_table.cpu()
    tokens = logical_tokens.cpu()
    counts_cpu = counts.cpu()
    rows: list[int] = []
    physical: list[int] = []
    for row in range(tokens.size(0)):
        count = int(counts_cpu[row])
        for token in tokens[row, :count].tolist():
            block = int(table[row, token // BLOCK_SIZE])
            physical.append(block * BLOCK_SIZE + token % BLOCK_SIZE)
            rows.append(row)
    return (
        torch.tensor(rows, dtype=torch.int64),
        torch.tensor(physical, dtype=torch.int64),
    )


def physical_token_rows(
    block_table: torch.Tensor,
    logical_tokens: torch.Tensor,
) -> torch.Tensor:
    table = block_table.cpu().to(torch.int64)
    tokens = logical_tokens.cpu().to(torch.int64)
    blocks = torch.gather(table, 1, tokens // BLOCK_SIZE)
    return blocks * BLOCK_SIZE + tokens % BLOCK_SIZE


def attention_logical_tokens(case: Case) -> torch.Tensor:
    sparse = case.sparse_slots[:, 0, :].cpu().to(torch.int64)
    cache_tokens = case.cache_tokens.cpu()
    actual_kv = case.actual_kv.cpu()
    rows: list[torch.Tensor] = []
    attended_length: int | None = None
    for row in range(sparse.size(0)):
        cache_length = int(cache_tokens[row])
        actual_length = int(actual_kv[row])
        if cache_length == 0:
            logical = torch.arange(
                actual_length, dtype=torch.int64
            )
        else:
            tail = torch.arange(
                cache_length, actual_length, dtype=torch.int64
            )
            logical = torch.cat((sparse[row], tail))
        if attended_length is None:
            attended_length = logical.numel()
        elif logical.numel() != attended_length:
            raise AssertionError(
                "The CPU golden currently requires equal attended lengths."
            )
        rows.append(logical)
    return torch.stack(rows)


def guard_physical_token_indices(case: Case) -> torch.Tensor:
    slots = case.sparse_slots[:, 0, :].cpu().to(torch.int64)
    counts = case.copy_counts.cpu()
    cache_tokens = case.cache_tokens.cpu()
    actual_kv = case.actual_kv.cpu()
    table = case.hbm_block_table.cpu().to(torch.int64)
    guards: list[int] = []
    for row in range(slots.size(0)):
        count = int(counts[row])
        if count < SPARSE_COUNT:
            logical = int(slots[row, count])
        elif int(actual_kv[row]) > int(cache_tokens[row]):
            logical = int(cache_tokens[row])
        else:
            selected = set(slots[row].tolist())
            logical = next(
                (
                    token
                    for token in range(int(cache_tokens[row]))
                    if token not in selected
                ),
                -1,
            )
            if logical < 0:
                continue
        block = int(table[row, logical // BLOCK_SIZE])
        guards.append(block * BLOCK_SIZE + logical % BLOCK_SIZE)
    return torch.tensor(guards, dtype=torch.int64)


def assert_dram_hbm_storage_disjoint(case: Case) -> None:
    pairs = (
        (case.dram_ckv, case.serial_ckv),
        (case.dram_ckv, case.fused_ckv),
        (case.dram_kpe, case.serial_kpe),
        (case.dram_kpe, case.fused_kpe),
    )
    for dram, hbm in pairs:
        if dram.data_ptr() == hbm.data_ptr():
            raise AssertionError("DRAM source aliases an HBM cache allocation.")


def poison_active_destinations(
    case: Case,
    destination_cpu: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if destination_cpu.numel() == 0:
        return (
            torch.empty((0, CKV_DIM), dtype=torch.bfloat16),
            torch.empty((0, KPE_DIM), dtype=torch.bfloat16),
        )
    destination = destination_cpu.to(case.query.device)
    for cache in (case.serial_ckv, case.fused_ckv):
        cache.view(-1, CKV_DIM).index_fill_(0, destination, 37.0)
    for cache in (case.serial_kpe, case.fused_kpe):
        cache.view(-1, KPE_DIM).index_fill_(0, destination, -29.0)
    torch.npu.synchronize()
    poisoned_ckv = case.fused_ckv.view(-1, CKV_DIM)[destination].cpu()
    poisoned_kpe = case.fused_kpe.view(-1, KPE_DIM)[destination].cpu()
    torch.testing.assert_close(
        poisoned_ckv,
        torch.full_like(poisoned_ckv, 37.0),
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        poisoned_kpe,
        torch.full_like(poisoned_kpe, -29.0),
        rtol=0,
        atol=0,
    )
    return poisoned_ckv, poisoned_kpe


def build_cpu_attention_golden(
    case: Case,
    source_physical_cpu: torch.Tensor,
) -> torch.Tensor:
    logical = attention_logical_tokens(case)
    physical = physical_token_rows(case.hbm_block_table, logical)
    device_indices = physical.flatten().to(case.query.device)
    batch, attended = physical.shape
    key = (
        case.fused_ckv.view(-1, CKV_DIM)[device_indices]
        .cpu()
        .view(batch, attended, CKV_DIM)
    )
    key_rope = (
        case.fused_kpe.view(-1, KPE_DIM)[device_indices]
        .cpu()
        .view(batch, attended, KPE_DIM)
    )

    source_ckv = case.dram_ckv_cpu.view(-1, CKV_DIM)
    source_kpe = case.dram_kpe_cpu.view(-1, KPE_DIM)
    counts = case.copy_counts.cpu()
    cursor = 0
    for row, count_value in enumerate(counts.tolist()):
        count = int(count_value)
        if count:
            sources = source_physical_cpu[cursor : cursor + count]
            key[row, :count].copy_(source_ckv[sources])
            key_rope[row, :count].copy_(source_kpe[sources])
        cursor += count
    if cursor != source_physical_cpu.numel():
        raise AssertionError("CPU golden source mapping is inconsistent.")

    query = case.query.cpu().float()
    query_rope = case.query_rope.cpu().float()
    golden_rows: list[torch.Tensor] = []
    for row in range(batch):
        row_key = key[row].float()
        row_key_rope = key_rope[row].float()
        scores = (
            query[row] @ row_key.T
            + query_rope[row] @ row_key_rope.T
        ) * case.scale
        golden_rows.append(torch.softmax(scores, dim=-1) @ row_key)
    return torch.stack(golden_rows)


def assert_attention_close(
    actual: torch.Tensor,
    expected: torch.Tensor,
    *,
    rtol: float,
    atol: float,
) -> tuple[float, float]:
    actual_cpu = actual.detach().cpu().float()
    expected_cpu = expected.detach().cpu().float()
    if not torch.isfinite(actual_cpu).all():
        raise AssertionError("Attention output contains NaN/Inf")
    torch.testing.assert_close(actual_cpu, expected_cpu, rtol=rtol, atol=atol)
    max_abs = float((actual_cpu - expected_cpu).abs().max())
    cosine = float(torch.nn.functional.cosine_similarity(
        actual_cpu.flatten(), expected_cpu.flatten(), dim=0
    ))
    return max_abs, cosine


def check_semantics(case: Case) -> None:
    original_cache_tokens = case.cache_tokens.clone()
    original_actual_kv = case.actual_kv.clone()
    original_counts = case.copy_counts.clone()

    # C=0 must run dense Attention without touching either HBM cache.
    dense_tokens = max(
        1,
        int(original_actual_kv[0] - original_cache_tokens[0]),
    )
    case.cache_tokens.zero_()
    case.actual_kv.fill_(dense_tokens)
    case.copy_counts.zero_()
    dense_logical = attention_logical_tokens(case)
    dense_physical = physical_token_rows(
        case.hbm_block_table, dense_logical
    ).flatten().to(case.query.device)
    dense_snapshots = [
        (cache, width, cache.view(-1, width)[dense_physical].cpu().clone())
        for cache, width in (
            (case.serial_ckv, CKV_DIM),
            (case.serial_kpe, KPE_DIM),
            (case.fused_ckv, CKV_DIM),
            (case.fused_kpe, KPE_DIM),
        )
    ]
    dense_golden = build_cpu_attention_golden(
        case, torch.empty((0,), dtype=torch.int64)
    )
    dense_serial_out = launch_serial(case)
    torch.npu.synchronize()
    dense_fused_out = launch_fused(case)
    torch.npu.synchronize()
    dense_pair_max, _ = assert_attention_close(
        dense_fused_out, dense_serial_out, rtol=0.02, atol=0.01
    )
    assert_attention_close(
        dense_serial_out, dense_golden, rtol=0.08, atol=0.08
    )
    dense_golden_max, dense_cosine = assert_attention_close(
        dense_fused_out, dense_golden, rtol=0.08, atol=0.08
    )
    for cache, width, before in dense_snapshots:
        torch.testing.assert_close(
            cache.view(-1, width)[dense_physical].cpu(), before, rtol=0, atol=0
        )
    print(
        "A5_FUSED_COPY_SPARSE_TAIL_ATTENTION_DENSE_CHECK "
        f"batch={case.query.size(0)} tokens={dense_tokens} "
        f"split_fused_max_abs={dense_pair_max:.9f} "
        f"cpu_max_abs={dense_golden_max:.9f} cosine={dense_cosine:.9f} "
        "cache_unchanged=1 ok=1",
        flush=True,
    )

    case.cache_tokens.copy_(original_cache_tokens)
    case.actual_kv.copy_(original_actual_kv)

    # The metadata remains zero from the dense phase: verify the no-copy path.
    serial_zero_out = launch_serial(case)
    torch.npu.synchronize()
    fused_zero_out = launch_fused(case)
    torch.npu.synchronize()
    zero_max_abs, zero_cosine = assert_attention_close(
        fused_zero_out, serial_zero_out, rtol=0.02, atol=0.01
    )
    print(
        "A5_FUSED_COPY_SPARSE_TAIL_ATTENTION_ZERO_MISS_CHECK "
        f"batch={case.query.size(0)} max_abs={zero_max_abs:.9f} "
        f"cosine={zero_cosine:.9f} ok=1",
        flush=True,
    )
    case.copy_counts.copy_(original_counts)

    _, destination_cpu = active_physical_token_indices(
        case.hbm_block_table,
        case.sparse_slots[:, 0, :],
        case.copy_counts,
    )
    _, source_cpu = active_physical_token_indices(
        case.dram_block_table,
        case.source_token_ids,
        case.copy_counts,
    )
    copied_tokens = int(case.copy_counts.cpu().sum())
    if (
        destination_cpu.numel() != copied_tokens
        or source_cpu.numel() != copied_tokens
    ):
        raise AssertionError("Active copy metadata has an inconsistent size.")
    assert_dram_hbm_storage_disjoint(case)
    # A stale-HBM implementation must fail: every active destination starts
    # with a conspicuous value that is absent from the DRAM CPU oracle.
    poisoned_ckv, poisoned_kpe = poison_active_destinations(
        case, destination_cpu
    )
    expected_ckv_cpu = case.dram_ckv_cpu.view(-1, CKV_DIM)[source_cpu]
    expected_kpe_cpu = case.dram_kpe_cpu.view(-1, KPE_DIM)[source_cpu]
    if copied_tokens and (
        torch.equal(poisoned_ckv, expected_ckv_cpu)
        or torch.equal(poisoned_kpe, expected_kpe_cpu)
    ):
        raise AssertionError(
            "Poisoned HBM destinations unexpectedly equal DRAM source data."
        )

    # Build the independent oracle and guard snapshots before either path runs.
    cpu_golden = build_cpu_attention_golden(case, source_cpu)
    guard_cpu = guard_physical_token_indices(case)
    guard = guard_cpu.to(case.query.device)
    guard_snapshots = [
        (cache, width, cache.view(-1, width)[guard].cpu().clone())
        for cache, width in (
            (case.serial_ckv, CKV_DIM),
            (case.serial_kpe, KPE_DIM),
            (case.fused_ckv, CKV_DIM),
            (case.fused_kpe, KPE_DIM),
        )
    ]

    serial_out = launch_serial(case)
    torch.npu.synchronize()
    fused_out = launch_fused(case)
    torch.npu.synchronize()

    destination = destination_cpu.to(case.query.device)
    for cache, width, expected in (
        (case.serial_ckv, CKV_DIM, expected_ckv_cpu),
        (case.fused_ckv, CKV_DIM, expected_ckv_cpu),
        (case.serial_kpe, KPE_DIM, expected_kpe_cpu),
        (case.fused_kpe, KPE_DIM, expected_kpe_cpu),
    ):
        torch.testing.assert_close(
            cache.view(-1, width)[destination].cpu(), expected, rtol=0, atol=0
        )
    for cache, width, before in guard_snapshots:
        torch.testing.assert_close(
            cache.view(-1, width)[guard].cpu(), before, rtol=0, atol=0
        )

    pair_max_abs, pair_cosine = assert_attention_close(
        fused_out, serial_out, rtol=0.02, atol=0.01
    )
    serial_golden_max_abs, _ = assert_attention_close(
        serial_out, cpu_golden, rtol=0.08, atol=0.08
    )
    fused_golden_max_abs, fused_golden_cosine = assert_attention_close(
        fused_out, cpu_golden, rtol=0.08, atol=0.08
    )
    print(
        "A5_FUSED_COPY_SPARSE_TAIL_ATTENTION_CHECK "
        f"batch={case.query.size(0)} misses={case.copy_counts.cpu().tolist()} "
        f"copied_tokens={copied_tokens} poisoned_destinations={copied_tokens} "
        f"guard_tokens={guard_cpu.numel()} dram_to_hbm_exact=1 "
        f"split_fused_max_abs={pair_max_abs:.9f} "
        f"split_fused_cosine={pair_cosine:.9f} "
        f"split_cpu_max_abs={serial_golden_max_abs:.9f} "
        f"fused_cpu_max_abs={fused_golden_max_abs:.9f} "
        f"fused_cpu_cosine={fused_golden_cosine:.9f} ok=1",
        flush=True,
    )


def event_benchmark(fn, warmup: int, iters: int) -> float:
    for _ in range(warmup):
        fn()
    torch.npu.synchronize()
    start = torch.npu.Event(enable_timing=True)
    end = torch.npu.Event(enable_timing=True)
    outputs: list[object] = []
    start.record()
    for _ in range(iters):
        outputs.append(fn())
    end.record()
    end.synchronize()
    if len(outputs) != iters:
        raise AssertionError("Timed output retention failed.")
    return float(start.elapsed_time(end)) / iters


def benchmark(case: Case, warmup: int, iters: int) -> None:
    batch = case.query.size(0)
    strategy = "a5_mte_pipeline_default_prefetch5"
    serial_ms = event_benchmark(lambda: launch_serial(case), warmup, iters)
    fused_ms = event_benchmark(lambda: launch_fused(case), warmup, iters)
    # Keep serial/fused first so component timing cannot perturb their values.
    scatter_ms = event_benchmark(
        lambda: launch_scatter_only(case), warmup, iters
    )
    sfa_ms = event_benchmark(
        lambda: launch_sfa_only(case), warmup, iters
    )
    speedup = serial_ms / fused_ms
    print(
        "A5_FUSED_COPY_SPARSE_TAIL_ATTENTION_RESULT "
        f"batch={batch} heads={case.query.size(1)} strategy={strategy} "
        f"source_len={case.dram_block_table.size(1) * BLOCK_SIZE} "
        f"attended_tokens={SPARSE_COUNT + int(case.actual_kv[0]) - int(case.cache_tokens[0])} "
        f"miss_min={int(case.copy_counts.min())} "
        f"miss_max={int(case.copy_counts.max())} "
        f"miss_mean={float(case.copy_counts.float().mean()):.3f} "
        f"serial_ms={serial_ms:.6f} "
        f"scatter_ms={scatter_ms:.6f} sfa_ms={sfa_ms:.6f} "
        f"fused_ms={fused_ms:.6f} "
        f"speedup={speedup:.4f} fused_faster={int(fused_ms < serial_ms)} "
        f"warmup={warmup} iters={iters}",
        flush=True,
    )


def profile_calls(case: Case, replays: int) -> None:
    launch_serial(case)
    launch_fused(case)
    torch.npu.synchronize()
    print(f"A5_FUSED_PROFILE_SPLIT_BEGIN replays={replays}", flush=True)
    for _ in range(replays):
        launch_serial(case)
    torch.npu.synchronize()
    print("A5_FUSED_PROFILE_SPLIT_END", flush=True)
    print(f"A5_FUSED_PROFILE_FUSED_BEGIN replays={replays}", flush=True)
    for _ in range(replays):
        launch_fused(case)
    torch.npu.synchronize()
    print("A5_FUSED_PROFILE_FUSED_END", flush=True)


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    if device.type != "npu":
        raise ValueError("--device must select an NPU")
    torch.npu.set_device(device)
    torch.npu.config.allow_internal_format = False
    require_a5(device, args.allow_non_a5)
    case = make_case(args)
    strategy = "a5_mte_pipeline_default_prefetch5"
    print(
        "A5_FUSED_COPY_SPARSE_TAIL_ATTENTION_CONFIG "
        "model=GLM-5.1 tp=16 dtype=bf16 "
        f"mode={args.mode} local_heads={args.heads} "
        f"batch={args.batch_size} source_len={args.source_len} "
        f"cache_tokens={args.cache_tokens} tail_tokens={args.tail_tokens} "
        f"miss_range=[{args.miss_min},{args.miss_max}] "
        "dram_allocator=empty_with_swapped_memory "
        "source_addresses=random_scattered "
        "destination_addresses=random_scattered "
        f"fused_strategy={strategy} "
        f"opapi={nanovllm_dsa_a5.local_opapi_path()}",
        flush=True,
    )
    if args.mode in ("all", "check"):
        check_semantics(case)
    if args.mode in ("all", "bench"):
        benchmark(case, args.warmup, args.iters)
    if args.mode == "profile":
        profile_calls(case, args.profile_replays)
    print("A5_FUSED_COPY_SPARSE_TAIL_ATTENTION_UT_OK", flush=True)


if __name__ == "__main__":
    main()
