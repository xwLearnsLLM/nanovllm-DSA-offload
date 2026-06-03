from __future__ import annotations

import argparse
from time import perf_counter

import torch

try:
    import torch_npu  # type: ignore
except Exception:
    torch_npu = None

from nanovllm.models.dsa_offload_ops import _dsa_index_update_cann, dsa_index_update_torch


def sync(device: torch.device) -> None:
    if device.type == "npu" and torch_npu is not None:
        torch.npu.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize(device)


def make_inputs(
    *,
    device: torch.device,
    batch: int,
    candidate: int,
    selected: int,
    pool_capacity: int,
    k: int,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    torch.manual_seed(seed)
    score = torch.randn(batch, candidate, device=device, dtype=torch.bfloat16).contiguous()
    pool = torch.empty(pool_capacity, selected, device=device, dtype=torch.int32)
    for b in range(batch):
        ids = torch.randperm(candidate, device=device, dtype=torch.int64)[:selected].to(torch.int32)
        pool[b].copy_(ids)
    promote = torch.empty(batch, k, device=device, dtype=torch.int32)
    demote = torch.empty(batch, k, device=device, dtype=torch.int32)
    counts = torch.empty(batch, device=device, dtype=torch.int32)
    candidate_lens = torch.full((batch,), candidate, device=device, dtype=torch.int32)
    selected_lens = torch.full((batch,), selected, device=device, dtype=torch.int32)
    req_pool_entries = torch.arange(batch, device=device, dtype=torch.int32)
    return score, pool, promote, demote, counts, candidate_lens, selected_lens, req_pool_entries


def compare_valid(name: str, actual: torch.Tensor, expected: torch.Tensor, counts: torch.Tensor) -> None:
    counts_cpu = counts.cpu().tolist()
    for b, count in enumerate(counts_cpu):
        count = int(count)
        if count <= 0:
            continue
        if not torch.equal(actual[b, :count], expected[b, :count]):
            raise AssertionError(
                f"{name} mismatch at batch={b} count={count}: "
                f"actual={actual[b, :count].cpu().tolist()} expected={expected[b, :count].cpu().tolist()}"
            )


def bench(fn, device: torch.device, warmup: int, iters: int) -> float:
    for _ in range(warmup):
        fn()
    sync(device)
    start = perf_counter()
    for _ in range(iters):
        fn()
    sync(device)
    return (perf_counter() - start) * 1000.0 / max(iters, 1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--candidate", type=int, default=8192)
    parser.add_argument("--selected", type=int, default=2560)
    parser.add_argument("--pool-capacity", type=int, default=16)
    parser.add_argument("--k", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    device = torch.device(args.device)
    if device.type == "npu":
        if torch_npu is None:
            raise RuntimeError("torch_npu is required for NPU runs")
        torch.npu.set_device(device)

    base = make_inputs(
        device=device,
        batch=args.batch,
        candidate=args.candidate,
        selected=args.selected,
        pool_capacity=max(args.pool_capacity, args.batch),
        k=args.k,
        seed=args.seed,
    )
    score, pool, promote, demote, counts, candidate_lens, selected_lens, req_pool_entries = base

    torch_pool = pool.clone()
    torch_promote = torch.empty_like(promote)
    torch_demote = torch.empty_like(demote)
    torch_counts = torch.empty_like(counts)
    dsa_index_update_torch(score.clone(), torch_pool, torch_promote, torch_demote, torch_counts, candidate_lens, selected_lens, req_pool_entries, args.k)

    cann_pool = pool.clone()
    cann_promote = torch.empty_like(promote)
    cann_demote = torch.empty_like(demote)
    cann_counts = torch.empty_like(counts)
    _dsa_index_update_cann(score.clone(), cann_pool, cann_promote, cann_demote, cann_counts, candidate_lens, selected_lens, req_pool_entries, args.k)
    sync(device)

    if not torch.equal(cann_counts, torch_counts):
        raise AssertionError(f"copy_counts mismatch: cann={cann_counts.cpu().tolist()} torch={torch_counts.cpu().tolist()}")
    compare_valid("promote_idx", cann_promote, torch_promote, torch_counts)
    compare_valid("demote_idx", cann_demote, torch_demote, torch_counts)
    if not torch.equal(cann_pool, torch_pool):
        diff = (cann_pool != torch_pool).nonzero()
        first = diff[0].cpu().tolist() if diff.numel() else []
        raise AssertionError(f"pool mismatch: diff_count={int(diff.shape[0])} first_diff={first}")

    def run_torch():
        dsa_index_update_torch(score.clone(), pool.clone(), torch.empty_like(promote), torch.empty_like(demote), torch.empty_like(counts), candidate_lens, selected_lens, req_pool_entries, args.k)

    def run_cann():
        _dsa_index_update_cann(score.clone(), pool.clone(), torch.empty_like(promote), torch.empty_like(demote), torch.empty_like(counts), candidate_lens, selected_lens, req_pool_entries, args.k)

    torch_ms = bench(run_torch, device, args.warmup, args.iters)
    cann_ms = bench(run_cann, device, args.warmup, args.iters)
    print(
        "DSA_INDEX_UPDATE_PROBE "
        f"device={args.device} batch={args.batch} candidate={args.candidate} "
        f"selected={args.selected} k={args.k} counts={cann_counts.cpu().tolist()} "
        f"torch_avg_ms={torch_ms:.6f} cann_avg_ms={cann_ms:.6f}"
    )


if __name__ == "__main__":
    main()
