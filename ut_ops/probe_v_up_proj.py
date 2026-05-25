from __future__ import annotations

import argparse
import time

import torch
import torch_npu  # type: ignore


def desc(name: str, tensor: torch.Tensor) -> str:
    return (
        f"{name}=shape={tuple(tensor.shape)} dtype={tensor.dtype} "
        f"device={tensor.device} contiguous={tensor.is_contiguous()} "
        f"stride={tuple(tensor.stride())} storage_offset={tensor.storage_offset()}"
    )


def sync(device: torch.device) -> None:
    if device.type == "npu":
        torch.npu.synchronize()


def v_up_ref(latent: torch.Tensor, w_uv: torch.Tensor) -> torch.Tensor:
    num_tokens = latent.shape[0]
    latent_by_head = latent.transpose(0, 1).contiguous()
    out = torch.bmm(latent_by_head, w_uv)
    return out.transpose(0, 1).reshape(num_tokens, -1)


def v_up_npu(latent: torch.Tensor, w_uv: torch.Tensor) -> torch.Tensor:
    num_tokens = latent.shape[0]
    latent_by_head = latent.transpose(0, 1).contiguous()
    out = torch_npu.npu_transpose_batchmatmul(
        latent_by_head,
        w_uv,
        perm_y=(1, 0, 2),
    )
    return out.reshape(num_tokens, -1)


def diff_report(left: torch.Tensor, right: torch.Tensor) -> str:
    left_f = left.float()
    right_f = right.float()
    diff = (left_f - right_f).abs()
    value_range = float((left_f.max() - left_f.min()).item())
    denom = max(value_range, 1e-12)
    return (
        f"max_abs={float(diff.max().item()):.6g} "
        f"mean_abs={float(diff.mean().item()):.6g} "
        f"value_range={value_range:.6g} "
        f"relative_max_error={float(diff.max().item()) / denom:.6g} "
        f"relative_mean_abs_error={float(diff.mean().item()) / denom:.6g}"
    )


def bench(fn, latent: torch.Tensor, w_uv: torch.Tensor, warmup: int, iters: int) -> tuple[torch.Tensor, float]:
    for _ in range(warmup):
        out = fn(latent, w_uv)
    sync(latent.device)
    start = time.perf_counter()
    for _ in range(iters):
        out = fn(latent, w_uv)
    sync(latent.device)
    elapsed_ms = (time.perf_counter() - start) * 1000.0 / max(iters, 1)
    return out, elapsed_ms


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--tokens", type=int, default=7)
    parser.add_argument("--heads", type=int, default=32)
    parser.add_argument("--latent-dim", type=int, default=512)
    parser.add_argument("--value-dim", type=int, default=512)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    device = torch.device(args.device)
    torch.npu.set_device(device)
    torch.manual_seed(args.seed)
    torch.set_default_device(device)

    latent = torch.randn(
        args.tokens,
        args.heads,
        args.latent_dim,
        dtype=torch.bfloat16,
        device=device,
    )
    w_uv = torch.randn(
        args.heads,
        args.latent_dim,
        args.value_dim,
        dtype=torch.bfloat16,
        device=device,
    )

    print("VUP_PROBE op=torch_npu.npu_transpose_batchmatmul")
    print(
        "VUP_PROBE config "
        f"tokens={args.tokens} heads={args.heads} "
        f"latent_dim={args.latent_dim} value_dim={args.value_dim}"
    )
    print("VUP_PROBE " + desc("latent", latent))
    print("VUP_PROBE " + desc("w_uv", w_uv))

    ref, ref_ms = bench(v_up_ref, latent, w_uv, args.warmup, args.iters)
    opt, opt_ms = bench(v_up_npu, latent, w_uv, args.warmup, args.iters)

    print("VUP_PROBE after_ref " + desc("ref", ref))
    print("VUP_PROBE after_npu " + desc("opt", opt))
    print("VUP_DIFF " + diff_report(ref, opt))
    print(f"VUP_BENCH ref_avg_ms={ref_ms:.6f} npu_avg_ms={opt_ms:.6f}")


if __name__ == "__main__":
    main()
