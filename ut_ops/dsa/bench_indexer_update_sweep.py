from __future__ import annotations

import argparse
from time import perf_counter

import torch

from nanovllm.models.dsa_offload_ops import _dsa_indexer_update_cann, dsa_indexer_update_torch
from ut_ops.common.device import set_device, sync_device
from ut_ops.common.format import parse_int_list


def make_case_tensors(
    *,
    device: torch.device,
    batch_size: int,
    candidate_len: int,
    selected_len: int,
    k: int,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    torch.manual_seed(seed)
    score = torch.randn((batch_size, candidate_len), dtype=torch.bfloat16, device=device).contiguous()
    selected_ids = torch.arange(selected_len, dtype=torch.int32, device=device)
    pool = selected_ids.unsqueeze(0).expand(batch_size, selected_len).contiguous()
    promote = torch.empty((batch_size, k), dtype=torch.int32, device=device)
    demote = torch.empty((batch_size, k), dtype=torch.int32, device=device)
    copy_counts = torch.full((batch_size,), k, dtype=torch.int32, device=device)
    candidate_lens = torch.full((batch_size,), candidate_len, dtype=torch.int32, device=device)
    selected_lens = torch.full((batch_size,), selected_len, dtype=torch.int32, device=device)
    req_pool_entries = torch.arange(batch_size, dtype=torch.int32, device=device)
    return score, pool, promote, demote, copy_counts, candidate_lens, selected_lens, req_pool_entries


def reset_outputs(
    *,
    pool: torch.Tensor,
    base_pool: torch.Tensor,
    promote: torch.Tensor,
    demote: torch.Tensor,
    copy_counts: torch.Tensor,
    k: int,
) -> None:
    pool.copy_(base_pool)
    promote.zero_()
    demote.zero_()
    copy_counts.fill_(k)


def run_update(
    backend: str,
    score: torch.Tensor,
    pool: torch.Tensor,
    promote: torch.Tensor,
    demote: torch.Tensor,
    copy_counts: torch.Tensor,
    candidate_lens: torch.Tensor,
    selected_lens: torch.Tensor,
    req_pool_entries: torch.Tensor,
    k: int,
) -> None:
    if backend == "cann":
        _dsa_indexer_update_cann(score, pool, promote, demote, copy_counts, candidate_lens, selected_lens, req_pool_entries, k, all_copy_count_k=True, pool_entries_start=0)
    elif backend == "torch":
        dsa_indexer_update_torch(score, pool, promote, demote, copy_counts, candidate_lens, selected_lens, req_pool_entries, k)
    else:
        raise ValueError(f"unsupported backend: {backend}")


def bench_case(
    *,
    backend: str,
    device: torch.device,
    batch_size: int,
    candidate_len: int,
    selected_len: int,
    k: int,
    warmup: int,
    iters: int,
    seed: int,
    reset_each_iter: bool,
) -> tuple[float, list[int]]:
    score, pool, promote, demote, copy_counts, candidate_lens, selected_lens, req_pool_entries = make_case_tensors(
        device=device,
        batch_size=batch_size,
        candidate_len=candidate_len,
        selected_len=selected_len,
        k=k,
        seed=seed,
    )
    base_pool = pool.clone()

    for _ in range(warmup):
        if reset_each_iter:
            reset_outputs(pool=pool, base_pool=base_pool, promote=promote, demote=demote, copy_counts=copy_counts, k=k)
        run_update(backend, score, pool, promote, demote, copy_counts, candidate_lens, selected_lens, req_pool_entries, k)
    sync_device(device)

    start = perf_counter()
    for _ in range(iters):
        if reset_each_iter:
            reset_outputs(pool=pool, base_pool=base_pool, promote=promote, demote=demote, copy_counts=copy_counts, k=k)
        run_update(backend, score, pool, promote, demote, copy_counts, candidate_lens, selected_lens, req_pool_entries, k)
    sync_device(device)
    avg_ms = (perf_counter() - start) * 1000.0 / max(iters, 1)
    counts = [int(x) for x in copy_counts.detach().cpu().tolist()]
    return avg_ms, counts


def selected_len_for_case(candidate_idx: int, candidate_len: int, candidate_lens: list[int], selected_lens: list[int]) -> int:
    if len(selected_lens) == 1:
        selected_len = selected_lens[0]
    else:
        if len(selected_lens) != len(candidate_lens):
            raise ValueError("--selected-lens must contain either 1 value or the same number of values as --candidate-lens.")
        selected_len = selected_lens[candidate_idx]
    return max(1, min(int(selected_len), int(candidate_len)))


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark dsa_indexer_update across batch sizes and sequence lengths.")
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--batch-sizes", default="1,2,4,8,10,16")
    parser.add_argument("--candidate-lens", default="8192,16384,32768,65536")
    parser.add_argument("--selected-lens", default="2560")
    parser.add_argument("--k", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--include-torch", action="store_true")
    parser.add_argument("--reset-each-iter", action="store_true", help="Reset pool/output buffers inside the timed loop. This is useful for correctness-like runs but includes reset copy overhead.")
    args = parser.parse_args()

    if int(args.k) != 128:
        raise ValueError("This benchmark is intended for fixed Tx/k=128.")
    device = set_device(args.device)

    batch_sizes = parse_int_list(args.batch_sizes)
    candidate_lens = parse_int_list(args.candidate_lens)
    selected_lens = parse_int_list(args.selected_lens)
    backends = ["cann"] + (["torch"] if args.include_torch else [])

    print(
        "DSA_INDEXER_UPDATE_SWEEP_CONFIG "
        f"device={args.device} batch_sizes={batch_sizes} candidate_lens={candidate_lens} "
        f"selected_lens={selected_lens} k={args.k} warmup={args.warmup} iters={args.iters} "
        f"reset_each_iter={int(args.reset_each_iter)} backends={backends}"
    )

    for candidate_idx, candidate_len in enumerate(candidate_lens):
        selected_len = selected_len_for_case(candidate_idx, candidate_len, candidate_lens, selected_lens)
        for batch_size in batch_sizes:
            for backend in backends:
                avg_ms, counts = bench_case(
                    backend=backend,
                    device=device,
                    batch_size=batch_size,
                    candidate_len=candidate_len,
                    selected_len=selected_len,
                    k=int(args.k),
                    warmup=int(args.warmup),
                    iters=int(args.iters),
                    seed=int(args.seed) + candidate_len + batch_size,
                    reset_each_iter=bool(args.reset_each_iter),
                )
                count_min = min(counts) if counts else 0
                count_max = max(counts) if counts else 0
                print(
                    "DSA_INDEXER_UPDATE_SWEEP "
                    f"backend={backend} batch={batch_size} candidate={candidate_len} selected={selected_len} "
                    f"k={args.k} avg_ms={avg_ms:.6f} counts_min={count_min} counts_max={count_max} "
                    f"est_61_layers_ms={avg_ms * 61.0:.6f}"
                )


if __name__ == "__main__":
    main()
