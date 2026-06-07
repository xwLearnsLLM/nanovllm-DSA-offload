from __future__ import annotations

import argparse

import torch

import nanovllm.ops as ascend_ops
from ut_ops.common.bench import benchmark_ms
from ut_ops.common.device import set_device, sync_device
from ut_ops.common.format import tensor_desc


def _rand_bf16(shape: tuple[int, ...], device: torch.device) -> torch.Tensor:
    return torch.randn(shape, dtype=torch.float32, device=device).to(torch.bfloat16).contiguous()


def _make_unique_topk(*, batch: int, topk: int, full_len: int, seed: int) -> torch.Tensor:
    gen = torch.Generator().manual_seed(seed)
    topk_cpu = torch.empty(batch, 1, 1, topk, dtype=torch.int32)
    for b in range(batch):
        perm = torch.randperm(full_len, generator=gen, dtype=torch.int64)[:topk].to(torch.int32)
        topk_cpu[b, 0, 0].copy_(perm)
    return topk_cpu


def make_inputs(
    *,
    device: torch.device,
    batch: int,
    pool_capacity: int,
    full_len: int,
    topk: int,
    block_size: int,
    rope_dim: int,
    kv_dim: int,
    seed: int,
    mixed_short: bool,
) -> dict[str, torch.Tensor]:
    torch.manual_seed(seed)
    max_sparse_blocks = (topk + block_size - 1) // block_size
    full_blocks_per_req = (full_len + block_size - 1) // block_size
    full_block_num = batch * full_blocks_per_req
    sparse_block_num = batch * max_sparse_blocks

    selection_k_rope = _rand_bf16((sparse_block_num, block_size, rope_dim), device)
    selection_kv_cache = _rand_bf16((sparse_block_num, block_size, kv_dim), device)
    full_k_rope = _rand_bf16((full_block_num, block_size, rope_dim), device)
    full_kv_cache = _rand_bf16((full_block_num, block_size, kv_dim), device)

    selection_kv_block_table = torch.arange(sparse_block_num, dtype=torch.int32, device=device).view(batch, max_sparse_blocks)
    full_kv_block_table = torch.arange(full_block_num, dtype=torch.int32, device=device).view(batch, full_blocks_per_req)
    selection_kv_block_status_pool = torch.full((pool_capacity, 1, 1, topk + 1), -1, dtype=torch.int32, device=device)

    if mixed_short:
        entries = []
        for b in range(batch):
            entries.append(-1 if b == 1 else (batch - 1 - b) % pool_capacity)
        req_pool_entries = torch.tensor(entries, dtype=torch.int32, device=device)
        full_kv_actual_seq = torch.full((batch,), full_len, dtype=torch.int32, device=device)
        for b in range(batch):
            if b % 2 == 1:
                full_kv_actual_seq[b] = topk
    else:
        req_pool_entries = torch.arange(batch, dtype=torch.int32, device=device) % pool_capacity
        full_kv_actual_seq = torch.full((batch,), full_len, dtype=torch.int32, device=device)

    selection_topk_indices = _make_unique_topk(batch=batch, topk=topk, full_len=full_len, seed=seed + 1).to(device)
    return {
        "selection_k_rope": selection_k_rope,
        "selection_kv_cache": selection_kv_cache,
        "selection_kv_block_table": selection_kv_block_table,
        "selection_kv_block_status_pool": selection_kv_block_status_pool,
        "req_pool_entries": req_pool_entries,
        "selection_topk_indices": selection_topk_indices,
        "full_k_rope": full_k_rope,
        "full_kv_cache": full_kv_cache,
        "full_kv_block_table": full_kv_block_table,
        "full_kv_actual_seq": full_kv_actual_seq,
    }


def call_op(tensors: dict[str, torch.Tensor]) -> None:
    ascend_ops.npu_gather_selection_kv_cache(
        tensors["selection_k_rope"],
        tensors["selection_kv_cache"],
        tensors["selection_kv_block_table"],
        tensors["selection_kv_block_status_pool"],
        tensors["req_pool_entries"],
        tensors["selection_topk_indices"],
        tensors["full_k_rope"],
        tensors["full_kv_cache"],
        tensors["full_kv_block_table"],
        tensors["full_kv_actual_seq"],
    )


def validate(tensors: dict[str, torch.Tensor], before_rope: torch.Tensor, before_kv: torch.Tensor, before_status: torch.Tensor, topk: int, block_size: int) -> None:
    sync_device(tensors["selection_kv_cache"].device)
    status_cpu = tensors["selection_kv_block_status_pool"].detach().cpu()
    req_entries_cpu = tensors["req_pool_entries"].detach().cpu().tolist()
    actual_seq_cpu = tensors["full_kv_actual_seq"].detach().cpu().tolist()
    topk_cpu = tensors["selection_topk_indices"].detach().cpu()

    long_rows = 0
    short_rows = 0
    max_k_rope_diff = 0.0
    max_kv_diff = 0.0
    unique_bad = 0
    set_bad = 0
    seq_bad = 0
    short_status_bad = 0
    short_cache_bad = 0

    for b, actual_seq in enumerate(actual_seq_cpu):
        entry = int(req_entries_cpu[b])
        table = tensors["selection_kv_block_table"][b].to(torch.int64)
        if int(actual_seq) <= topk:
            short_rows += 1
            if entry >= 0 and not torch.equal(tensors["selection_kv_block_status_pool"][entry].detach().cpu(), before_status[entry].detach().cpu()):
                short_status_bad += 1
            if not torch.equal(tensors["selection_k_rope"][table].detach().cpu(), before_rope[table].detach().cpu()):
                short_cache_bad += 1
            if not torch.equal(tensors["selection_kv_cache"][table].detach().cpu(), before_kv[table].detach().cpu()):
                short_cache_bad += 1
            continue

        long_rows += 1
        if entry < 0:
            raise AssertionError(f"long row {b} has invalid req_pool_entry={entry}")
        row_status = status_cpu[entry, 0, 0]
        selected = row_status[:topk].to(torch.int64)
        if torch.unique(selected).numel() != topk:
            unique_bad += 1
        if not torch.equal(torch.sort(selected).values, torch.sort(topk_cpu[b, 0, 0].to(torch.int64)).values):
            set_bad += 1
        if int(row_status[topk].item()) != topk:
            seq_bad += 1

        dst_pos = torch.arange(topk, dtype=torch.int64, device=table.device)
        dst_blocks = table[dst_pos // block_size]
        dst_offsets = dst_pos % block_size
        src_ids = selected.to(table.device)
        src_blocks = tensors["full_kv_block_table"][b].to(torch.int64)[src_ids // block_size]
        src_offsets = src_ids % block_size

        got_rope = tensors["selection_k_rope"][dst_blocks, dst_offsets].float()
        ref_rope = tensors["full_k_rope"][src_blocks, src_offsets].float()
        got_kv = tensors["selection_kv_cache"][dst_blocks, dst_offsets].float()
        ref_kv = tensors["full_kv_cache"][src_blocks, src_offsets].float()
        max_k_rope_diff = max(max_k_rope_diff, float((got_rope - ref_rope).abs().max().item()))
        max_kv_diff = max(max_kv_diff, float((got_kv - ref_kv).abs().max().item()))

    print(
        "GSKV_POOL_CHECK "
        f"long_rows={long_rows} short_rows={short_rows} "
        f"unique_bad={unique_bad} set_bad={set_bad} seq_bad={seq_bad} "
        f"short_status_bad={short_status_bad} short_cache_bad={short_cache_bad} "
        f"max_k_rope_diff={max_k_rope_diff:g} max_kv_diff={max_kv_diff:g}"
    )
    if unique_bad or set_bad or seq_bad or short_status_bad or short_cache_bad or max_k_rope_diff != 0.0 or max_kv_diff != 0.0:
        raise AssertionError("gather_selection pool validation failed")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--pool-capacity", type=int, default=8)
    parser.add_argument("--full-len", type=int, default=4096)
    parser.add_argument("--topk", type=int, default=2048)
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument("--rope-dim", type=int, default=64)
    parser.add_argument("--kv-dim", type=int, default=512)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-mixed-short", action="store_true")
    args = parser.parse_args()

    if args.full_len <= args.topk:
        raise ValueError("--full-len must be greater than --topk so long rows exercise the gather path")
    if args.pool_capacity < args.batch_size:
        raise ValueError("--pool-capacity must be >= --batch-size")
    if args.topk % args.block_size != 0:
        raise ValueError("--topk must be divisible by --block-size for this probe")

    device = set_device(args.device)
    tensors = make_inputs(
        device=device,
        batch=args.batch_size,
        pool_capacity=args.pool_capacity,
        full_len=args.full_len,
        topk=args.topk,
        block_size=args.block_size,
        rope_dim=args.rope_dim,
        kv_dim=args.kv_dim,
        seed=args.seed,
        mixed_short=not args.no_mixed_short,
    )
    for name, tensor in tensors.items():
        print("GSKV_POOL_TENSOR " + tensor_desc(name, tensor))

    before_rope = tensors["selection_k_rope"].clone()
    before_kv = tensors["selection_kv_cache"].clone()
    before_status = tensors["selection_kv_block_status_pool"].clone()
    call_op(tensors)
    validate(tensors, before_rope, before_kv, before_status, args.topk, args.block_size)

    bench_inputs = make_inputs(
        device=device,
        batch=args.batch_size,
        pool_capacity=args.pool_capacity,
        full_len=args.full_len,
        topk=args.topk,
        block_size=args.block_size,
        rope_dim=args.rope_dim,
        kv_dim=args.kv_dim,
        seed=args.seed + 17,
        mixed_short=not args.no_mixed_short,
    )
    avg_ms = benchmark_ms(lambda: call_op(bench_inputs), device, args.warmup, args.iters)
    print(f"GSKV_POOL_BENCH same_topk_avg_ms={avg_ms:.6f} warmup={args.warmup} iters={args.iters}")
    print("GSKV_POOL_OK")


if __name__ == "__main__":
    main()
