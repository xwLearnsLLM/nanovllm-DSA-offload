from __future__ import annotations

import argparse
import time

import torch

try:
    import torch_npu  # type: ignore  # noqa: F401
except ImportError:
    torch_npu = None


def sync(device: torch.device) -> None:
    if device.type == "npu" and torch_npu is not None:
        torch.npu.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize(device)


def hadamard_transform(x: torch.Tensor) -> torch.Tensor:
    dim = x.shape[-1]
    if dim == 0 or dim & (dim - 1):
        raise ValueError("last dimension must be a power of two")
    y = x.float().reshape(-1, dim)
    block = 1
    while block < dim:
        y = y.view(-1, dim // (block * 2), 2, block)
        left = y[:, :, 0, :]
        right = y[:, :, 1, :]
        y = torch.cat((left + right, left - right), dim=-1).reshape(-1, dim)
        block *= 2
    return y.reshape_as(x.float()) * (dim**-0.5)


def score(
    query: torch.Tensor,
    key: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    scores = torch.einsum("bhd,btd->bht", query.float(), key.float())
    return (scores * weights.float().unsqueeze(-1)).sum(dim=1)


def diff_report(name: str, actual: torch.Tensor, expected: torch.Tensor) -> str:
    diff = (actual.float() - expected.float()).abs()
    denom = expected.float().abs().clamp_min(1e-6)
    return (
        f"{name}: max_abs={diff.max().item():.6g} "
        f"mean_abs={diff.mean().item():.6g} "
        f"max_rel={(diff / denom).max().item():.6g} "
        f"mean_rel={(diff / denom).mean().item():.6g}"
    )


def bench(fn, warmup: int, iters: int, device: torch.device) -> float:
    for _ in range(warmup):
        fn()
    sync(device)
    start = time.perf_counter()
    for _ in range(iters):
        fn()
    sync(device)
    return (time.perf_counter() - start) * 1000.0 / max(iters, 1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--heads", type=int, default=64)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--tokens", type=int, default=18000)
    parser.add_argument("--topk", type=int, default=512)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    device = torch.device(args.device)
    if device.type == "npu":
        if torch_npu is None:
            raise RuntimeError("torch_npu is required for NPU runs")
        torch.npu.set_device(device)
    torch.manual_seed(args.seed)

    query = torch.randn(
        args.batch,
        args.heads,
        args.head_dim,
        dtype=torch.bfloat16,
        device=device,
    )
    key = torch.randn(
        args.batch,
        args.tokens,
        args.head_dim,
        dtype=torch.bfloat16,
        device=device,
    )
    weights = torch.randn(
        args.batch,
        args.heads,
        dtype=torch.bfloat16,
        device=device,
    )

    raw = score(query, key, weights)
    rotated = score(
        hadamard_transform(query).to(query.dtype),
        hadamard_transform(key).to(key.dtype),
        weights,
    )
    print(
        "HADAMARD_SKIP config "
        + " ".join(f"{k}={v}" for k, v in vars(args).items())
    )
    print("HADAMARD_SKIP " + diff_report("score", raw, rotated))
    topk = min(args.topk, args.tokens)
    raw_topk = raw.topk(topk, dim=-1).indices
    rotated_topk = rotated.topk(topk, dim=-1).indices
    overlaps = []
    for b in range(args.batch):
        overlap = len(set(raw_topk[b].tolist()) & set(rotated_topk[b].tolist()))
        overlaps.append(overlap / topk)
    print(
        "HADAMARD_SKIP topk_overlap "
        + " ".join(f"b{idx}={value:.6f}" for idx, value in enumerate(overlaps))
        + f" avg={sum(overlaps) / len(overlaps):.6f}"
    )

    raw_ms = bench(lambda: score(query, key, weights), args.warmup, args.iters, device)
    rotated_ms = bench(
        lambda: score(
            hadamard_transform(query).to(query.dtype),
            hadamard_transform(key).to(key.dtype),
            weights,
        ),
        args.warmup,
        args.iters,
        device,
    )
    print(
        "HADAMARD_SKIP bench "
        f"raw_score_avg_ms={raw_ms:.6f} rotated_score_avg_ms={rotated_ms:.6f}"
    )


if __name__ == "__main__":
    main()
