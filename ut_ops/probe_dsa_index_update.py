from __future__ import annotations

import argparse
from time import perf_counter

import torch

from nanovllm.models.dsa_index_update_real import (
    availability_error,
    binding_version,
    dsa_index_update_real,
    is_available as is_real_available,
)
from nanovllm.models.dsa_offload_ops import dsa_index_update_torch


def _parse_int_list(value: str | None, default: list[int]) -> list[int]:
    if value is None or value.strip() == "":
        return default
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def _sync(device: torch.device) -> None:
    if device.type == "npu":
        torch.npu.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize(device)


def _desc(name: str, tensor: torch.Tensor) -> str:
    return (
        f"{name}=shape={tuple(tensor.shape)} dtype={tensor.dtype} "
        f"device={tensor.device} contiguous={tensor.is_contiguous()} "
        f"stride={tuple(tensor.stride())}"
    )


def _make_lengths(args: argparse.Namespace) -> tuple[list[int], list[int]]:
    default_candidates = [256, 8192, 12288, 18000]
    default_selected = [256, 2048, 3712, 4096]
    candidate_lens = _parse_int_list(args.candidate_lens, default_candidates)
    selected_lens = _parse_int_list(args.selected_lens, default_selected)
    if len(candidate_lens) < args.batch_size:
        candidate_lens = [
            candidate_lens[i % len(candidate_lens)] for i in range(args.batch_size)
        ]
    if len(selected_lens) < args.batch_size:
        selected_lens = [
            selected_lens[i % len(selected_lens)] for i in range(args.batch_size)
        ]
    candidate_lens = candidate_lens[: args.batch_size]
    selected_lens = selected_lens[: args.batch_size]
    selected_lens = [
        min(max(s, 0), max(c, 0), args.max_selected_len)
        for c, s in zip(candidate_lens, selected_lens)
    ]
    return candidate_lens, selected_lens


def _make_case(args: argparse.Namespace, device: torch.device):
    candidate_lens_list, selected_lens_list = _make_lengths(args)
    batch_size = args.batch_size
    max_candidate = max(max(candidate_lens_list), 1)
    pool_capacity = max(args.pool_capacity, batch_size + 3)
    req_entries_list = [(i * 3 + 1) % pool_capacity for i in range(batch_size)]

    generator = torch.Generator()
    generator.manual_seed(args.seed)
    score = torch.randn(
        (batch_size, max_candidate),
        dtype=torch.float32,
        generator=generator,
    ).to(device=device, dtype=torch.bfloat16)

    pool = torch.full(
        (pool_capacity, args.max_selected_len),
        -1,
        dtype=torch.int32,
        device=device,
    )
    for b, (candidate_len, selected_len) in enumerate(
        zip(candidate_lens_list, selected_lens_list)
    ):
        if selected_len <= 0:
            continue
        perm = torch.randperm(candidate_len, generator=generator)
        pool[req_entries_list[b], :selected_len] = perm[:selected_len].to(
            device=device,
            dtype=torch.int32,
        )

    promote = torch.empty(
        (batch_size, args.output_capacity),
        dtype=torch.int32,
        device=device,
    )
    demote = torch.empty_like(promote)
    copy_counts = torch.empty((batch_size,), dtype=torch.int32, device=device)
    candidate_lens = torch.tensor(candidate_lens_list, dtype=torch.int32, device=device)
    selected_lens = torch.tensor(selected_lens_list, dtype=torch.int32, device=device)
    req_entries = torch.tensor(req_entries_list, dtype=torch.int32, device=device)
    return score, pool, promote, demote, copy_counts, candidate_lens, selected_lens, req_entries


def _run_torch(case, max_copy_tokens: int):
    score, pool, promote, demote, copy_counts, candidate_lens, selected_lens, req_entries = case
    dsa_index_update_torch(
        score,
        pool,
        promote,
        demote,
        copy_counts,
        candidate_lens,
        selected_lens,
        req_entries,
        max_copy_tokens,
    )


def _run_real(case, max_copy_tokens: int):
    score, pool, promote, demote, copy_counts, candidate_lens, selected_lens, req_entries = case
    dsa_index_update_real(
        score,
        pool,
        promote,
        demote,
        copy_counts,
        candidate_lens,
        selected_lens,
        req_entries,
        max_copy_tokens,
    )


def _clone_case(case):
    return tuple(t.clone() for t in case)


def _check_equal(name: str, lhs: torch.Tensor, rhs: torch.Tensor) -> bool:
    lhs_cpu = lhs.detach().cpu()
    rhs_cpu = rhs.detach().cpu()
    ok = torch.equal(lhs_cpu, rhs_cpu)
    if not ok:
        diff = (lhs_cpu != rhs_cpu).nonzero()
        first = diff[0].tolist() if diff.numel() else []
        print(
            f"DSA_INDEX_UPDATE_DIFF {name}: mismatch_count={diff.shape[0]} "
            f"first={first}"
        )
    return ok


def _accuracy(args: argparse.Namespace, device: torch.device) -> bool:
    base = _make_case(args, device)
    torch_case = _clone_case(base)
    real_case = _clone_case(base)
    _run_torch(torch_case, args.max_copy_tokens)
    _sync(device)
    _run_real(real_case, args.max_copy_tokens)
    _sync(device)

    _, pool_t, promote_t, demote_t, counts_t, _, selected_lens, req_entries = torch_case
    _, pool_r, promote_r, demote_r, counts_r, _, _, _ = real_case
    ok = _check_equal("copy_counts", counts_r, counts_t)

    counts_cpu = counts_t.detach().cpu().tolist()
    for b, copy_count in enumerate(counts_cpu):
        if copy_count <= 0:
            continue
        ok = _check_equal(
            f"promote_idx[{b},:{copy_count}]",
            promote_r[b, :copy_count],
            promote_t[b, :copy_count],
        ) and ok
        ok = _check_equal(
            f"demote_idx[{b},:{copy_count}]",
            demote_r[b, :copy_count],
            demote_t[b, :copy_count],
        ) and ok

    selected_lens_cpu = selected_lens.detach().cpu().tolist()
    req_entries_cpu = req_entries.detach().cpu().tolist()
    for b, selected_len in enumerate(selected_lens_cpu):
        entry = req_entries_cpu[b]
        ok = _check_equal(
            f"hbm_cached_tokens_pool[{entry},:{selected_len}]",
            pool_r[entry, :selected_len],
            pool_t[entry, :selected_len],
        ) and ok

    print(
        "DSA_INDEX_UPDATE_ACCURACY "
        f"ok={int(ok)} copy_counts={counts_cpu}"
    )
    return ok


def _bench_one(name: str, fn, case, max_copy_tokens: int, warmup: int, iters: int, device: torch.device) -> float:
    for _ in range(warmup):
        fn(case, max_copy_tokens)
    _sync(device)
    start = perf_counter()
    for _ in range(iters):
        fn(case, max_copy_tokens)
    _sync(device)
    elapsed = perf_counter() - start
    avg_ms = elapsed * 1000.0 / max(iters, 1)
    print(f"DSA_INDEX_UPDATE_BENCH {name}_avg_ms={avg_ms:.6f}")
    return avg_ms


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--candidate-lens", default=None)
    parser.add_argument("--selected-lens", default=None)
    parser.add_argument("--pool-capacity", type=int, default=16)
    parser.add_argument("--max-selected-len", type=int, default=8192)
    parser.add_argument("--output-capacity", type=int, default=2048)
    parser.add_argument("--max-copy-tokens", type=int, default=64)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--allow-missing-real", action="store_true")
    args = parser.parse_args()

    device = torch.device(args.device)
    if device.type == "npu":
        import torch_npu  # type: ignore  # noqa: F401

    print(
        "DSA_INDEX_UPDATE_CONFIG "
        f"device={device} batch_size={args.batch_size} "
        f"max_copy_tokens={args.max_copy_tokens} output_capacity={args.output_capacity} "
        f"real_available={int(is_real_available())} "
        f"binding_version={binding_version()}"
    )
    if not is_real_available():
        print(f"DSA_INDEX_UPDATE_REAL_IMPORT_ERROR {availability_error()}")
        if not args.allow_missing_real:
            raise SystemExit(2)

    sample_case = _make_case(args, device)
    print("DSA_INDEX_UPDATE_TENSOR " + _desc("score", sample_case[0]))
    print("DSA_INDEX_UPDATE_TENSOR " + _desc("hbm_cached_tokens_pool", sample_case[1]))
    print("DSA_INDEX_UPDATE_TENSOR " + _desc("promote_idx", sample_case[2]))
    print("DSA_INDEX_UPDATE_TENSOR " + _desc("candidate_lens", sample_case[5]))

    if is_real_available():
        if not _accuracy(args, device):
            raise SystemExit(1)

    torch_case = sample_case
    _bench_one(
        "torch",
        _run_torch,
        torch_case,
        args.max_copy_tokens,
        args.warmup,
        args.iters,
        device,
    )
    if is_real_available():
        real_case = _make_case(args, device)
        _bench_one(
            "real",
            _run_real,
            real_case,
            args.max_copy_tokens,
            args.warmup,
            args.iters,
            device,
        )


if __name__ == "__main__":
    main()
