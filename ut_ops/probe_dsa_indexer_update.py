from __future__ import annotations

import argparse
from time import perf_counter

import torch

try:
    import torch_npu  # type: ignore
except Exception:
    torch_npu = None

from nanovllm.models.dsa_offload_ops import _dsa_indexer_update_cann, dsa_indexer_update_torch


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


def compare_valid_exact(name: str, actual: torch.Tensor, expected: torch.Tensor, counts: torch.Tensor) -> None:
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


def _assert_unique(name: str, values: list[int]) -> None:
    if len(values) != len(set(values)):
        duplicates = sorted({value for value in values if values.count(value) > 1})
        raise AssertionError(f"{name} is not unique: duplicates={duplicates[:16]} values={values[:256]}")


def validate_hard_invariants(
    *,
    before_pool: torch.Tensor,
    after_pool: torch.Tensor,
    promote: torch.Tensor,
    demote: torch.Tensor,
    counts: torch.Tensor,
    candidate_lens: torch.Tensor,
    selected_lens: torch.Tensor,
    req_pool_entries: torch.Tensor,
    k: int,
) -> None:
    before_cpu = before_pool.detach().cpu()
    after_cpu = after_pool.detach().cpu()
    promote_cpu = promote.detach().cpu()
    demote_cpu = demote.detach().cpu()
    counts_cpu = counts.detach().cpu().tolist()
    candidate_lens_cpu = candidate_lens.detach().cpu().tolist()
    selected_lens_cpu = selected_lens.detach().cpu().tolist()
    req_pool_entries_cpu = req_pool_entries.detach().cpu().tolist()

    for b, count_value in enumerate(counts_cpu):
        entry = int(req_pool_entries_cpu[b])
        candidate_len = int(candidate_lens_cpu[b])
        selected_len = int(selected_lens_cpu[b])
        count = int(count_value)
        uncached_len = max(candidate_len - selected_len, 0)
        expected_count = min(int(k), selected_len, uncached_len)
        if count != expected_count:
            raise AssertionError(
                f"copy_counts mismatch at batch={b}: actual={count} "
                f"expected={expected_count} candidate_len={candidate_len} selected_len={selected_len} k={k}"
            )
        if count <= 0:
            continue

        promote_values = [int(x) for x in promote_cpu[b, :count].tolist()]
        demote_slots = [int(x) for x in demote_cpu[b, :count].tolist()]
        before_values = [int(x) for x in before_cpu[entry, :selected_len].tolist()]
        after_values = [int(x) for x in after_cpu[entry, :selected_len].tolist()]

        _assert_unique(f"promote_idx batch={b}", promote_values)
        _assert_unique(f"demote_idx batch={b}", demote_slots)
        _assert_unique(f"hbm_cached_tokens_pool before batch={b}", before_values)
        _assert_unique(f"hbm_cached_tokens_pool after batch={b}", after_values)

        before_set = set(before_values)
        for token in promote_values:
            if token < 0 or token >= candidate_len:
                raise AssertionError(f"promote_idx out of range at batch={b}: token={token} candidate_len={candidate_len}")
            if token in before_set:
                raise AssertionError(f"promote_idx already cached at batch={b}: token={token}")
        for slot in demote_slots:
            if slot < 0 or slot >= selected_len:
                raise AssertionError(f"demote_idx out of range at batch={b}: slot={slot} selected_len={selected_len}")

        demote_slot_set = set(demote_slots)
        for slot, token in zip(demote_slots, promote_values):
            actual = int(after_cpu[entry, slot].item())
            if actual != token:
                raise AssertionError(
                    f"pool update mismatch at batch={b}: slot={slot} "
                    f"after={actual} promote={token}"
                )
        for slot in range(selected_len):
            if slot in demote_slot_set:
                continue
            before = int(before_cpu[entry, slot].item())
            after = int(after_cpu[entry, slot].item())
            if before != after:
                raise AssertionError(
                    f"non-demoted pool slot changed at batch={b}: "
                    f"slot={slot} before={before} after={after}"
                )
        if after_pool.shape[1] > selected_len and not torch.equal(after_cpu[entry, selected_len:], before_cpu[entry, selected_len:]):
            raise AssertionError(f"pool tail beyond selected_len changed at batch={b} entry={entry}")


def print_overlap_report(
    *,
    cann_pool: torch.Tensor,
    torch_pool: torch.Tensor,
    cann_promote: torch.Tensor,
    torch_promote: torch.Tensor,
    cann_demote: torch.Tensor,
    torch_demote: torch.Tensor,
    counts: torch.Tensor,
    candidate_lens: torch.Tensor,
    selected_lens: torch.Tensor,
    req_pool_entries: torch.Tensor,
) -> None:
    counts_cpu = counts.detach().cpu().tolist()
    candidate_lens_cpu = candidate_lens.detach().cpu().tolist()
    selected_lens_cpu = selected_lens.detach().cpu().tolist()
    req_pool_entries_cpu = req_pool_entries.detach().cpu().tolist()
    cann_promote_cpu = cann_promote.detach().cpu()
    torch_promote_cpu = torch_promote.detach().cpu()
    cann_demote_cpu = cann_demote.detach().cpu()
    torch_demote_cpu = torch_demote.detach().cpu()
    cann_pool_cpu = cann_pool.detach().cpu()
    torch_pool_cpu = torch_pool.detach().cpu()

    promote_ratios: list[float] = []
    demote_ratios: list[float] = []
    pool_ratios: list[float] = []
    cann_promote_max_values: list[int] = []
    cann_promote_ge_half_ratios: list[float] = []
    for b, count_value in enumerate(counts_cpu):
        count = int(count_value)
        candidate_len = int(candidate_lens_cpu[b])
        selected_len = int(selected_lens_cpu[b])
        entry = int(req_pool_entries_cpu[b])
        if count <= 0:
            continue
        cann_promote_set = set(int(x) for x in cann_promote_cpu[b, :count].tolist())
        torch_promote_set = set(int(x) for x in torch_promote_cpu[b, :count].tolist())
        cann_demote_set = set(int(x) for x in cann_demote_cpu[b, :count].tolist())
        torch_demote_set = set(int(x) for x in torch_demote_cpu[b, :count].tolist())
        cann_pool_set = set(int(x) for x in cann_pool_cpu[entry, :selected_len].tolist())
        torch_pool_set = set(int(x) for x in torch_pool_cpu[entry, :selected_len].tolist())
        promote_ratios.append(len(cann_promote_set & torch_promote_set) / max(count, 1))
        demote_ratios.append(len(cann_demote_set & torch_demote_set) / max(count, 1))
        pool_ratios.append(len(cann_pool_set & torch_pool_set) / max(selected_len, 1))
        cann_promote_max_values.append(max(cann_promote_set))
        half = candidate_len // 2
        cann_promote_ge_half_ratios.append(sum(1 for token in cann_promote_set if token >= half) / max(count, 1))

    def fmt(values: list[float]) -> str:
        if not values:
            return "n/a"
        return f"mean={sum(values) / len(values):.6f} min={min(values):.6f} max={max(values):.6f}"

    print(
        "DSA_INDEXER_UPDATE_OVERLAP "
        f"promote={fmt(promote_ratios)} "
        f"demote={fmt(demote_ratios)} "
        f"pool={fmt(pool_ratios)} "
        f"cann_promote_max={cann_promote_max_values} "
        f"cann_promote_ge_half={fmt(cann_promote_ge_half_ratios)}"
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
    parser.add_argument("--batch", "--batch-size", dest="batch", type=int, default=4)
    parser.add_argument("--candidate", "--candidate-len", dest="candidate", type=int, default=8192)
    parser.add_argument("--selected", "--selected-len", dest="selected", type=int, default=2560)
    parser.add_argument("--pool-capacity", type=int, default=16)
    parser.add_argument("--k", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--strict-torch", action="store_true", help="Also require CANN outputs to exactly match the torch prototype.")
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
    dsa_indexer_update_torch(score.clone(), torch_pool, torch_promote, torch_demote, torch_counts, candidate_lens, selected_lens, req_pool_entries, args.k)

    cann_pool = pool.clone()
    cann_promote = torch.empty_like(promote)
    cann_demote = torch.empty_like(demote)
    cann_counts = torch.empty_like(counts)
    _dsa_indexer_update_cann(score.clone(), cann_pool, cann_promote, cann_demote, cann_counts, candidate_lens, selected_lens, req_pool_entries, args.k)
    sync(device)

    validate_hard_invariants(
        before_pool=pool,
        after_pool=cann_pool,
        promote=cann_promote,
        demote=cann_demote,
        counts=cann_counts,
        candidate_lens=candidate_lens,
        selected_lens=selected_lens,
        req_pool_entries=req_pool_entries,
        k=args.k,
    )
    print_overlap_report(
        cann_pool=cann_pool,
        torch_pool=torch_pool,
        cann_promote=cann_promote,
        torch_promote=torch_promote,
        cann_demote=cann_demote,
        torch_demote=torch_demote,
        counts=cann_counts,
        candidate_lens=candidate_lens,
        selected_lens=selected_lens,
        req_pool_entries=req_pool_entries,
    )
    if args.strict_torch:
        if not torch.equal(cann_counts, torch_counts):
            raise AssertionError(f"copy_counts mismatch: cann={cann_counts.cpu().tolist()} torch={torch_counts.cpu().tolist()}")
        compare_valid_exact("promote_idx", cann_promote, torch_promote, torch_counts)
        compare_valid_exact("demote_idx", cann_demote, torch_demote, torch_counts)
        if not torch.equal(cann_pool, torch_pool):
            diff = (cann_pool != torch_pool).nonzero()
            first = diff[0].cpu().tolist() if diff.numel() else []
            raise AssertionError(f"pool mismatch: diff_count={int(diff.shape[0])} first_diff={first}")

    def run_torch():
        dsa_indexer_update_torch(score.clone(), pool.clone(), torch.empty_like(promote), torch.empty_like(demote), torch.empty_like(counts), candidate_lens, selected_lens, req_pool_entries, args.k)

    def run_cann():
        _dsa_indexer_update_cann(score.clone(), pool.clone(), torch.empty_like(promote), torch.empty_like(demote), torch.empty_like(counts), candidate_lens, selected_lens, req_pool_entries, args.k)

    torch_ms = bench(run_torch, device, args.warmup, args.iters)
    cann_ms = bench(run_cann, device, args.warmup, args.iters)
    print(
        "DSA_INDEXER_UPDATE_PROBE "
        f"device={args.device} batch={args.batch} candidate={args.candidate} "
        f"selected={args.selected} k={args.k} counts={cann_counts.cpu().tolist()} "
        f"torch_avg_ms={torch_ms:.6f} cann_avg_ms={cann_ms:.6f}"
    )


if __name__ == "__main__":
    main()
