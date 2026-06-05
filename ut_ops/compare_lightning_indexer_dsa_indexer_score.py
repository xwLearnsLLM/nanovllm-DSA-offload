from __future__ import annotations

import argparse
import math
import time

import torch

from nanovllm.models.dsa_offload_ops import dsa_indexer_score

try:
    import torch_npu  # type: ignore  # noqa: F401
except Exception:  # pragma: no cover - local non-Ascend syntax checks
    torch_npu = None

try:
    import nanovllm.ops as ascend_ops
    ascend_ops_import_error = None
except Exception as exc:  # pragma: no cover
    ascend_ops = None
    ascend_ops_import_error = repr(exc)


def parse_int_list(value: str | None, batch_size: int, default_value: int) -> list[int]:
    if value is None or value.strip() == "":
        return [default_value] * batch_size
    items = [int(item.strip()) for item in value.split(",") if item.strip()]
    if len(items) == 1:
        return items * batch_size
    if len(items) != batch_size:
        raise ValueError(f"candidate-lens expects 1 or batch-size values, got {len(items)} for batch_size={batch_size}")
    return items


def sync(device: torch.device) -> None:
    if device.type == "npu" and torch_npu is not None:
        torch.npu.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize(device)


def tensor_desc(name: str, tensor: torch.Tensor) -> str:
    return (
        f"{name}=shape={tuple(tensor.shape)} dtype={tensor.dtype} "
        f"device={tensor.device} contiguous={tensor.is_contiguous()} stride={tuple(tensor.stride())}"
    )


def make_block_tables(
    candidate_lens: list[int],
    *,
    block_size: int,
    block_base: int,
    device: torch.device,
) -> tuple[torch.Tensor, int]:
    max_blocks = max(max((length + block_size - 1) // block_size for length in candidate_lens), 1)
    block_tables = torch.empty((len(candidate_lens), max_blocks), dtype=torch.int32, device=device)

    next_block = block_base
    safe_fill = max(block_base, 0)
    for batch_idx, candidate_len in enumerate(candidate_lens):
        n_blocks = max((candidate_len + block_size - 1) // block_size, 1)
        block_ids = torch.arange(next_block, next_block + n_blocks, dtype=torch.int32, device=device)
        block_tables[batch_idx, :n_blocks] = block_ids
        if n_blocks < max_blocks:
            block_tables[batch_idx, n_blocks:] = safe_fill
        next_block += n_blocks

    physical_blocks = max(next_block, safe_fill + 1)
    return block_tables, physical_blocks


def dtype_from_name(name: str) -> torch.dtype:
    if name == "bf16":
        return torch.bfloat16
    if name == "fp16":
        return torch.float16
    if name == "fp32":
        return torch.float32
    raise ValueError(f"unsupported dtype: {name}")


def as_list(tensor: torch.Tensor, limit: int) -> list[int]:
    return [int(x) for x in tensor[:limit].detach().cpu().tolist()]


def overlap_ratio(lhs: torch.Tensor, rhs: torch.Tensor, k: int) -> float:
    lhs_set = set(int(x) for x in lhs[:k].detach().cpu().tolist())
    rhs_set = set(int(x) for x in rhs[:k].detach().cpu().tolist())
    if not lhs_set and not rhs_set:
        return 1.0
    return len(lhs_set & rhs_set) / float(k)


def summarize_batch(
    *,
    batch_idx: int,
    candidate_len: int,
    lightning_topk: torch.Tensor,
    score_topk: torch.Tensor,
    score: torch.Tensor,
    topk: int,
    preview: int,
) -> None:
    score_f = score.float()
    finite = torch.isfinite(score_f)
    nan_count = int(torch.isnan(score_f).sum().item())
    posinf_count = int(torch.isposinf(score_f).sum().item())
    neginf_count = int(torch.isneginf(score_f).sum().item())
    invalid_lightning = int(((lightning_topk < 0) | (lightning_topk >= candidate_len)).sum().item())
    invalid_score = int(((score_topk < 0) | (score_topk >= candidate_len)).sum().item())

    if finite.any():
        finite_score = score_f[finite]
        score_min = float(finite_score.min().item())
        score_max = float(finite_score.max().item())
        score_mean = float(finite_score.mean().item())
    else:
        score_min = float("nan")
        score_max = float("nan")
        score_mean = float("nan")

    overlap_marks = [32, 128, 512, 1024, topk]
    overlap_marks = sorted(set(mark for mark in overlap_marks if mark <= topk))
    overlap_text = " ".join(
        f"overlap@{mark}={overlap_ratio(lightning_topk, score_topk, mark):.6f}"
        for mark in overlap_marks
    )

    print(
        "INDEXER_SCORE_COMPARE "
        f"batch={batch_idx} candidate_len={candidate_len} topk={topk} "
        f"{overlap_text} "
        f"invalid_lightning={invalid_lightning} invalid_score={invalid_score} "
        f"score_nan={nan_count} score_posinf={posinf_count} score_neginf={neginf_count} "
        f"score_min={score_min:.6g} score_max={score_max:.6g} score_mean={score_mean:.6g}"
    )
    print(f"  lightning_top{preview}={as_list(lightning_topk, min(preview, topk))}")
    print(f"  dsa_indexer_score_top{preview}={as_list(score_topk, min(preview, topk))}")


def run_once(args: argparse.Namespace) -> None:
    if ascend_ops is None:
        raise RuntimeError(f"nanovllm.ops import failed: {ascend_ops_import_error}")

    device = torch.device(args.device)
    data_dtype = dtype_from_name(args.dtype)
    score_dtype = dtype_from_name(args.score_dtype)
    candidate_lens_list = parse_int_list(args.candidate_lens, args.batch_size, args.seq_len)
    min_candidate = min(candidate_lens_list)
    max_candidate = max(candidate_lens_list)
    topk = min(args.topk, min_candidate)
    if topk <= 0:
        raise ValueError(f"topk must be positive after clipping, got {topk}")

    torch.manual_seed(args.seed)
    if device.type == "npu" and torch_npu is not None:
        torch.npu.manual_seed(args.seed)

    block_tables, physical_blocks = make_block_tables(
        candidate_lens_list,
        block_size=args.block_size,
        block_base=args.block_base,
        device=device,
    )
    query = torch.randn(args.batch_size, args.num_heads, args.head_dim, dtype=data_dtype, device=device)
    index_cache = torch.randn(physical_blocks, args.block_size, 1, args.head_dim, dtype=data_dtype, device=device)
    weights = torch.randn(args.batch_size, args.num_heads, dtype=data_dtype, device=device)
    actual_seq_lengths_query = torch.arange(1, args.batch_size + 1, dtype=torch.int32, device=device)
    candidate_lens = torch.tensor(candidate_lens_list, dtype=torch.int32, device=device)

    max_blocks = int(block_tables.shape[1])
    score_out = torch.empty((args.batch_size, max_candidate), dtype=score_dtype, device=device)

    print(
        "INDEXER_SCORE_COMPARE config "
        f"device={device} batch_size={args.batch_size} candidate_lens={candidate_lens_list} "
        f"topk={topk} requested_topk={args.topk} dtype={data_dtype} score_dtype={score_dtype} "
        f"num_heads={args.num_heads} head_dim={args.head_dim} block_size={args.block_size} "
        f"block_base={args.block_base} physical_blocks={physical_blocks} max_blocks={max_blocks} "
        f"sparse_mode={args.sparse_mode} seed={args.seed}"
    )
    print("INDEXER_SCORE_COMPARE " + tensor_desc("query", query))
    print("INDEXER_SCORE_COMPARE " + tensor_desc("index_cache", index_cache))
    print("INDEXER_SCORE_COMPARE " + tensor_desc("weights", weights))
    print("INDEXER_SCORE_COMPARE " + tensor_desc("block_tables", block_tables))

    lightning_topk = ascend_ops.npu_lightning_indexer(
        query=query.contiguous(),
        key=index_cache,
        weights=weights.contiguous(),
        actual_seq_lengths_query=actual_seq_lengths_query,
        actual_seq_lengths_key=candidate_lens,
        block_table=block_tables.contiguous(),
        layout_query="TND",
        layout_key="PA_BSND",
        sparse_count=topk,
        sparse_mode=args.sparse_mode,
    )
    sync(device)

    dsa_indexer_score(
        query.contiguous(),
        index_cache,
        weights.contiguous(),
        block_tables.contiguous(),
        candidate_lens,
        score_out,
        actual_seq_lengths_query=actual_seq_lengths_query,
    )
    sync(device)

    if lightning_topk.dim() != 3 or lightning_topk.shape[0] != args.batch_size:
        raise RuntimeError(f"unexpected lightning_topk shape: {tuple(lightning_topk.shape)}")

    for batch_idx, candidate_len in enumerate(candidate_lens_list):
        scores = score_out[batch_idx, :candidate_len]
        score_topk = torch.topk(scores.float(), k=topk, largest=True).indices.to(torch.int32)
        summarize_batch(
            batch_idx=batch_idx,
            candidate_len=candidate_len,
            lightning_topk=lightning_topk[batch_idx, 0, :topk].to(torch.int32),
            score_topk=score_topk,
            score=scores,
            topk=topk,
            preview=args.preview,
        )

    if args.iters <= 0:
        return

    for _ in range(args.warmup):
        ascend_ops.npu_lightning_indexer(
            query.contiguous(),
            index_cache,
            weights.contiguous(),
            actual_seq_lengths_query,
            candidate_lens,
            block_tables.contiguous(),
            "TND",
            "PA_BSND",
            topk,
            args.sparse_mode,
        )
        dsa_indexer_score(
            query.contiguous(),
            index_cache,
            weights.contiguous(),
            block_tables.contiguous(),
            candidate_lens,
            score_out,
            actual_seq_lengths_query=actual_seq_lengths_query,
        )
    sync(device)

    t0 = time.perf_counter()
    for _ in range(args.iters):
        ascend_ops.npu_lightning_indexer(
            query.contiguous(),
            index_cache,
            weights.contiguous(),
            actual_seq_lengths_query,
            candidate_lens,
            block_tables.contiguous(),
            "TND",
            "PA_BSND",
            topk,
            args.sparse_mode,
        )
    sync(device)
    lightning_ms = (time.perf_counter() - t0) * 1000.0 / args.iters

    t0 = time.perf_counter()
    for _ in range(args.iters):
        dsa_indexer_score(
            query.contiguous(),
            index_cache,
            weights.contiguous(),
            block_tables.contiguous(),
            candidate_lens,
            score_out,
            actual_seq_lengths_query=actual_seq_lengths_query,
        )
    sync(device)
    dsa_indexer_score_ms = (time.perf_counter() - t0) * 1000.0 / args.iters
    print(
        "INDEXER_SCORE_BENCH "
        f"iters={args.iters} lightning_indexer_avg_ms={lightning_ms:.6f} "
        f"dsa_indexer_score_avg_ms={dsa_indexer_score_ms:.6f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare npu_lightning_indexer topk with dsa_indexer_score + torch.topk.")
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seq-len", type=int, default=8192)
    parser.add_argument("--candidate-lens", default=None, help="Comma separated candidate lengths. One value is broadcast to all batches.")
    parser.add_argument("--topk", type=int, default=2048)
    parser.add_argument("--num-heads", type=int, default=64)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument("--block-base", type=int, default=1)
    parser.add_argument("--sparse-mode", type=int, default=3)
    parser.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--score-dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--preview", type=int, default=32)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=0)
    args = parser.parse_args()
    run_once(args)


if __name__ == "__main__":
    main()
