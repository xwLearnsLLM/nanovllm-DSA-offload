from __future__ import annotations

import argparse
import math
import time

import torch
import torch.nn.functional as F

import nanovllm.ops as ascend_ops

try:
    import torch_npu
except ImportError:
    torch_npu = None


ACL_FORMAT_FRACTAL_NZ = 29


def desc(name: str, tensor: torch.Tensor) -> str:
    return (
        f"{name}=shape={tuple(tensor.shape)} dtype={tensor.dtype} "
        f"device={tensor.device} contiguous={tensor.is_contiguous()} "
        f"stride={tuple(tensor.stride())} storage_offset={tensor.storage_offset()}"
    )


def sync(device: torch.device) -> None:
    if device.type == "npu":
        torch.npu.synchronize()


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    y = x.float()
    y = y * torch.rsqrt(y.pow(2).mean(dim=-1, keepdim=True) + eps)
    return (y.to(x.dtype) * weight).contiguous()


def rotate_half_neox(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


def apply_rope_neox(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    if x.dim() == 2:
        cos = cos
        sin = sin
    else:
        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)
    y = x * cos + rotate_half_neox(x.float()).to(x.dtype) * sin
    return y.contiguous()


def round_up(value: int, align: int) -> int:
    return ((value + align - 1) // align) * align


def trans_rope_weight(weight: torch.Tensor, rope_dim: int) -> torch.Tensor:
    if rope_dim == 0:
        return weight.contiguous()
    nope_part = weight[..., :-rope_dim, :]
    rope_part = weight[..., -rope_dim:, :]
    rope_part = torch.cat((rope_part[..., ::2, :], rope_part[..., 1::2, :]), dim=-2)
    return torch.cat((nope_part, rope_part), dim=-2).contiguous()


def transdata(nd_mat: torch.Tensor, block_size: tuple[int, int] = (16, 16)) -> torch.Tensor:
    rows = round_up(nd_mat.shape[0], block_size[0])
    cols = round_up(nd_mat.shape[1], block_size[1])
    row_pad = rows - nd_mat.shape[0]
    col_pad = cols - nd_mat.shape[1]
    nd_mat = F.pad(nd_mat, (0, row_pad, 0, col_pad))
    nz_mat = nd_mat.reshape(
        rows // block_size[0],
        block_size[0],
        cols // block_size[1],
        block_size[1],
    )
    nz_mat = nz_mat.permute(2, 0, 1, 3)
    return nz_mat.reshape(nz_mat.shape[0], nz_mat.shape[1] * nz_mat.shape[2], nz_mat.shape[3])


def to_mlapo_nz_weight(nd_weight: torch.Tensor, block_size: tuple[int, int] = (16, 32)) -> torch.Tensor:
    nz_weight = transdata(nd_weight, block_size=block_size).unsqueeze(0).contiguous()
    if torch_npu is not None and nz_weight.device.type == "npu":
        return torch_npu.npu_format_cast(nz_weight, ACL_FORMAT_FRACTAL_NZ)
    return nz_weight


def make_cos_sin(
    tokens: int,
    rope_dim: int,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    inv_freq = 1.0 / (
        10000.0 ** (torch.arange(0, rope_dim, 2, device=device, dtype=torch.float32) / rope_dim)
    )
    positions = torch.arange(tokens, device=device, dtype=torch.float32)
    freqs = torch.einsum("i,j->ij", positions, inv_freq)
    cos = torch.cat((freqs.cos(), freqs.cos()), dim=-1).to(dtype).contiguous()
    sin = torch.cat((freqs.sin(), freqs.sin()), dim=-1).to(dtype).contiguous()
    return cos, sin


def diff(name: str, actual: torch.Tensor, expected: torch.Tensor) -> None:
    a = actual.float()
    e = expected.float()
    d = (a - e).abs()
    value_range = float(e.max().item() - e.min().item())
    denom = value_range if value_range > 0 else 1.0
    print(
        f"MLAPO_DIFF {name} max_abs={d.max().item():.6g} "
        f"mean_abs={d.mean().item():.6g} "
        f"relative_max_error={d.max().item() / denom:.6g} "
        f"relative_mean_abs_error={d.mean().item() / denom:.6g} "
        f"expected_range={value_range:.6g}"
    )


def build_inputs(args: argparse.Namespace, device: torch.device, dtype: torch.dtype) -> dict[str, torch.Tensor]:
    torch.manual_seed(args.seed)
    hidden = torch.randn(args.tokens, args.hidden_size, device=device, dtype=dtype) * args.init_scale
    kv_weight = torch.randn(
        args.kv_lora_rank + args.rope_dim,
        args.hidden_size,
        device=device,
        dtype=dtype,
    ) * args.init_scale
    q_weight = torch.randn(args.q_lora_rank, args.hidden_size, device=device, dtype=dtype) * args.init_scale
    kv_weight = trans_rope_weight(kv_weight, args.rope_dim)
    wdqkv_ref = torch.cat((kv_weight, q_weight), dim=0).contiguous()
    wuq_ref = torch.randn(
        args.heads * (args.nope_dim + args.rope_dim),
        args.q_lora_rank,
        device=device,
        dtype=dtype,
    ) * args.init_scale
    wuq_ref = trans_rope_weight(
        wuq_ref.view(args.heads, args.nope_dim + args.rope_dim, args.q_lora_rank),
        args.rope_dim,
    ).reshape(args.heads * (args.nope_dim + args.rope_dim), args.q_lora_rank).contiguous()
    wdqkv = to_mlapo_nz_weight(wdqkv_ref, block_size=(16, 32))
    wuq = to_mlapo_nz_weight(wuq_ref, block_size=(16, 32))
    wuk = torch.randn(
        args.heads,
        args.nope_dim,
        args.kv_lora_rank,
        device=device,
        dtype=dtype,
    ) * args.init_scale
    gamma1 = (torch.randn(args.q_lora_rank, device=device, dtype=dtype) * 0.01 + 1.0).contiguous()
    beta1 = torch.zeros(args.q_lora_rank, device=device, dtype=dtype).contiguous()
    gamma2 = (torch.randn(args.kv_lora_rank, device=device, dtype=dtype) * 0.01 + 1.0).contiguous()
    cos, sin = make_cos_sin(args.tokens, args.rope_dim, device, dtype)
    blocks = max(args.blocks, math.ceil(args.tokens / args.block_size))
    ckv_cache = torch.zeros(
        blocks,
        args.block_size,
        1,
        args.kv_lora_rank,
        device=device,
        dtype=dtype,
    )
    kpe_cache = torch.zeros(
        blocks,
        args.block_size,
        1,
        args.rope_dim,
        device=device,
        dtype=dtype,
    )
    slotmapping = torch.arange(args.tokens, device=device, dtype=torch.int32).contiguous()
    return {
        "hidden": hidden.contiguous(),
        "wdqkv": wdqkv,
        "wdqkv_ref": wdqkv_ref,
        "wuq": wuq,
        "wuq_ref": wuq_ref,
        "wuk": wuk.contiguous(),
        "gamma1": gamma1,
        "beta1": beta1,
        "gamma2": gamma2,
        "cos": cos,
        "sin": sin,
        "ckv_cache": ckv_cache.contiguous(),
        "kpe_cache": kpe_cache.contiguous(),
        "slotmapping": slotmapping,
    }


def reference(
    tensors: dict[str, torch.Tensor],
    args: argparse.Namespace,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    qkv = F.linear(tensors["hidden"], tensors["wdqkv_ref"])
    kv, q_c = torch.split(
        qkv,
        [args.kv_lora_rank + args.rope_dim, args.q_lora_rank],
        dim=-1,
    )
    ckv, k_pe = torch.split(kv, [args.kv_lora_rank, args.rope_dim], dim=-1)
    ckv = rms_norm(ckv, tensors["gamma2"], args.rms_eps)
    q_c = rms_norm(q_c, tensors["gamma1"], args.rms_eps)

    q = F.linear(q_c, tensors["wuq_ref"]).view(
        args.tokens,
        args.heads,
        args.nope_dim + args.rope_dim,
    )
    q_nope, q_pe = torch.split(q, [args.nope_dim, args.rope_dim], dim=-1)
    q_pe = apply_rope_neox(q_pe, tensors["cos"], tensors["sin"])
    k_pe = apply_rope_neox(k_pe, tensors["cos"], tensors["sin"])

    ql_nope = torch.bmm(q_nope.transpose(0, 1).contiguous(), tensors["wuk"])
    ql_nope = ql_nope.transpose(0, 1).contiguous()

    ckv_cache = torch.zeros_like(tensors["ckv_cache"])
    kpe_cache = torch.zeros_like(tensors["kpe_cache"])
    slots = tensors["slotmapping"].to(torch.long)
    ckv_cache.view(-1, args.kv_lora_rank).index_copy_(0, slots, ckv)
    kpe_cache.view(-1, args.rope_dim).index_copy_(0, slots, k_pe)
    return ql_nope, q_pe, ckv_cache, kpe_cache


def run_mlapo(
    tensors: dict[str, torch.Tensor],
    args: argparse.Namespace,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    ql_nope = torch.empty(
        args.tokens,
        args.heads,
        args.kv_lora_rank,
        device=tensors["hidden"].device,
        dtype=tensors["hidden"].dtype,
    )
    q_pe = torch.empty(
        args.tokens,
        args.heads,
        args.rope_dim,
        device=tensors["hidden"].device,
        dtype=tensors["hidden"].dtype,
    )
    ckv_cache = torch.zeros_like(tensors["ckv_cache"])
    kpe_cache = torch.zeros_like(tensors["kpe_cache"])
    inner_out = torch.empty(0, device=tensors["hidden"].device, dtype=tensors["hidden"].dtype)
    ascend_ops.mla_preprocess(
        tensors["hidden"],
        tensors["wdqkv"],
        None,
        tensors["gamma1"],
        tensors["beta1"],
        tensors["wuq"],
        None,
        tensors["gamma2"],
        tensors["cos"],
        tensors["sin"],
        tensors["wuk"],
        ckv_cache,
        kpe_cache,
        tensors["slotmapping"],
        q_out0=ql_nope,
        kv_cache_out0=ckv_cache,
        q_out1=q_pe,
        kv_cache_out1=kpe_cache,
        inner_out=inner_out,
        cache_mode=args.cache_mode,
        quant_mode="no_quant",
        enable_inner_out=False,
    )
    return ql_nope, q_pe, ckv_cache, kpe_cache


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--tokens", type=int, default=7)
    parser.add_argument("--heads", type=int, default=32)
    parser.add_argument("--hidden-size", type=int, default=7168)
    parser.add_argument("--q-lora-rank", type=int, default=1536)
    parser.add_argument("--kv-lora-rank", type=int, default=512)
    parser.add_argument("--nope-dim", type=int, default=128)
    parser.add_argument("--rope-dim", type=int, default=64)
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument("--blocks", type=int, default=1)
    parser.add_argument("--rms-eps", type=float, default=1e-6)
    parser.add_argument("--init-scale", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dtype", choices=["bf16", "fp16"], default="bf16")
    parser.add_argument("--cache-mode", choices=["krope_ctkv"], default="krope_ctkv")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iters", type=int, default=10)
    args = parser.parse_args()

    device = torch.device(args.device)
    if device.type == "npu":
        torch.npu.set_device(device)
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    tensors = build_inputs(args, device, dtype)

    print("MLAPO_PROBE op=nanovllm.ops.mla_preprocess")
    print(
        "MLAPO_PROBE config "
        f"tokens={args.tokens} heads={args.heads} hidden={args.hidden_size} "
        f"q_lora={args.q_lora_rank} kv_lora={args.kv_lora_rank} "
        f"nope={args.nope_dim} rope={args.rope_dim} cache_mode={args.cache_mode}"
    )
    for name in (
        "hidden",
        "wdqkv_ref",
        "wdqkv",
        "wuq_ref",
        "wuq",
        "wuk",
        "gamma1",
        "beta1",
        "gamma2",
        "cos",
        "sin",
        "ckv_cache",
        "kpe_cache",
    ):
        print("MLAPO_PROBE " + desc(name, tensors[name]))

    expected = reference(tensors, args)
    sync(device)
    actual = run_mlapo(tensors, args)
    sync(device)
    for name, out in zip(("ql_nope", "q_pe", "ckv_cache", "kpe_cache"), actual):
        print("MLAPO_PROBE after_mlapo " + desc(name, out))
    for name, got, exp in zip(("ql_nope", "q_pe", "ckv_cache", "kpe_cache"), actual, expected):
        diff(name, got, exp)

    for _ in range(args.warmup):
        run_mlapo(tensors, args)
    sync(device)
    times = []
    for _ in range(args.iters):
        start = time.perf_counter()
        run_mlapo(tensors, args)
        sync(device)
        times.append((time.perf_counter() - start) * 1000.0)
    if times:
        times_sorted = sorted(times)
        print(
            f"MLAPO_BENCH avg_ms={sum(times) / len(times):.6f} "
            f"min_ms={times_sorted[0]:.6f} max_ms={times_sorted[-1]:.6f}"
        )


if __name__ == "__main__":
    main()
