from __future__ import annotations

import argparse
import time

import torch
import torch_npu  # type: ignore


def first_tensor(value):
    if isinstance(value, (tuple, list)):
        return value[0]
    return value


def sync(device: torch.device) -> None:
    if device.type == "npu":
        torch.npu.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize(device)


def tensor_desc(name: str, tensor: torch.Tensor) -> str:
    return (
        f"{name}=shape={tuple(tensor.shape)} dtype={tensor.dtype} "
        f"device={tensor.device} contiguous={tensor.is_contiguous()} "
        f"stride={tensor.stride()}"
    )


def make_cache(tokens: torch.Tensor, blocks: int, block_size: int) -> torch.Tensor:
    cache = torch.zeros(
        (blocks, block_size, tokens.shape[1], tokens.shape[2]),
        dtype=tokens.dtype,
        device=tokens.device,
    )
    cache.view(-1, tokens.shape[1], tokens.shape[2])[: tokens.shape[0]].copy_(tokens)
    return cache


def run_mla_v1(
    q_nope: torch.Tensor,
    q_pe: torch.Tensor,
    ckv_cache: torch.Tensor,
    kpe_cache: torch.Tensor,
    block_table: torch.Tensor,
    kv_len: int,
    scale: float,
) -> torch.Tensor:
    out = torch_npu.npu_fused_infer_attention_score(
        q_nope,
        ckv_cache,
        ckv_cache,
        query_rope=q_pe,
        key_rope=kpe_cache,
        num_heads=int(q_nope.shape[1]),
        num_key_value_heads=int(ckv_cache.shape[2]),
        input_layout="TND",
        atten_mask=None,
        sparse_mode=0,
        scale=float(scale),
        antiquant_mode=0,
        antiquant_scale=None,
        block_table=block_table,
        block_size=int(ckv_cache.shape[1]),
        softmax_lse_flag=False,
        actual_seq_lengths=[int(q_nope.shape[0])],
        actual_seq_lengths_kv=[int(kv_len)],
    )
    return first_tensor(out)


def run_mla_v2(
    q_nope: torch.Tensor,
    q_pe: torch.Tensor,
    ckv_cache: torch.Tensor,
    kpe_cache: torch.Tensor,
    block_table: torch.Tensor,
    kv_len: int,
    scale: float,
) -> torch.Tensor:
    ckv_cache_v2 = ckv_cache.transpose(1, 2)
    kpe_cache_v2 = kpe_cache.transpose(1, 2)
    out = torch.empty_like(q_nope)
    lse = torch.empty((int(q_nope.shape[0]), int(q_nope.shape[1]), 1), dtype=torch.float32, device=q_nope.device)
    info = dict(
        head_num=int(q_nope.shape[1]),
        input_layout="TND",
        atten_mask=None,
        scale_value=float(scale),
        pre_tokens=2147483647,
        next_tokens=2147483647,
        sparse_mode=0,
        block_size=int(ckv_cache.shape[1]),
        block_table=block_table,
        actual_seq_qlen=[int(q_nope.shape[0])],
        actual_seq_kvlen=[int(kv_len)],
    )
    workspace = torch_npu._npu_fused_infer_attention_score_v2_get_max_workspace(
        q_nope,
        ckv_cache_v2,
        ckv_cache_v2,
        query_rope=q_pe,
        key_rope=kpe_cache_v2,
        **info,
    )
    torch_npu.npu_fused_infer_attention_score_v2.out(
        q_nope,
        ckv_cache_v2,
        ckv_cache_v2,
        query_rope=q_pe,
        key_rope=kpe_cache_v2,
        attention_out=out,
        softmax_lse=lse,
        workspace=workspace,
        **info,
    )
    return out


def compare(name: str, actual: torch.Tensor, expected: torch.Tensor, atol: float, rtol: float) -> tuple[float, float, int]:
    diff = (actual.float() - expected.float()).abs()
    allowed = float(atol) + float(rtol) * expected.float().abs()
    bad = diff > allowed
    max_abs = float(diff.max().item()) if diff.numel() else 0.0
    mean_abs = float(diff.mean().item()) if diff.numel() else 0.0
    bad_count = int(bad.sum().item()) if bad.numel() else 0
    max_allowed = float(allowed.max().item()) if allowed.numel() else float(atol)
    print(
        f"MLA_KV_PERM_DIFF {name} max_abs={max_abs:.6g} mean_abs={mean_abs:.6g} "
        f"bad_count={bad_count} max_allowed={max_allowed:.6g}"
    )
    return max_abs, mean_abs, bad_count


def bench(fn, warmup: int, iters: int, device: torch.device) -> float:
    for _ in range(warmup):
        fn()
    sync(device)
    times = []
    for _ in range(iters):
        start = time.perf_counter()
        fn()
        sync(device)
        times.append(time.perf_counter() - start)
    return sum(times) / max(1, len(times)) * 1000.0


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify MLA output is almost invariant to KV token physical order.")
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--backend", choices=("v1", "v2"), default="v2")
    parser.add_argument("--kv-len", type=int, default=4096)
    parser.add_argument("--query-len", type=int, default=1)
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--kv-lora-rank", type=int, default=512)
    parser.add_argument("--rope-dim", type=int, default=64)
    parser.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--atol", type=float, default=0.03125)
    parser.add_argument("--rtol", type=float, default=0.01)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--fail-on-diff", action="store_true")
    args = parser.parse_args()

    device = torch.device(args.device)
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]
    torch.manual_seed(args.seed)
    if device.type == "npu":
        torch.npu.set_device(device)

    blocks = (args.kv_len + args.block_size - 1) // args.block_size
    block_table = torch.arange(blocks, dtype=torch.int32, device=device).view(1, blocks)
    scale = float(args.kv_lora_rank) ** -0.5

    q_nope = torch.randn((args.query_len, args.heads, args.kv_lora_rank), dtype=torch.float32, device=device).to(dtype)
    q_pe = torch.randn((args.query_len, args.heads, args.rope_dim), dtype=torch.float32, device=device).to(dtype)
    ckv_tokens = torch.randn((args.kv_len, 1, args.kv_lora_rank), dtype=torch.float32, device=device).to(dtype)
    kpe_tokens = torch.randn((args.kv_len, 1, args.rope_dim), dtype=torch.float32, device=device).to(dtype)
    perm = torch.randperm(args.kv_len, device=device)

    ckv_cache = make_cache(ckv_tokens, blocks, args.block_size).contiguous()
    kpe_cache = make_cache(kpe_tokens, blocks, args.block_size).contiguous()
    ckv_cache_perm = make_cache(ckv_tokens.index_select(0, perm), blocks, args.block_size).contiguous()
    kpe_cache_perm = make_cache(kpe_tokens.index_select(0, perm), blocks, args.block_size).contiguous()

    runner = run_mla_v2 if args.backend == "v2" else run_mla_v1
    print(
        "MLA_KV_PERM config "
        f"backend={args.backend} device={device} kv_len={args.kv_len} query_len={args.query_len} "
        f"blocks={blocks} block_size={args.block_size} heads={args.heads} "
        f"kv_lora_rank={args.kv_lora_rank} rope_dim={args.rope_dim} dtype={dtype} seed={args.seed}"
    )
    print("MLA_KV_PERM " + tensor_desc("q_nope", q_nope))
    print("MLA_KV_PERM " + tensor_desc("ckv_cache", ckv_cache))
    print("MLA_KV_PERM " + tensor_desc("block_table", block_table))

    ref = runner(q_nope, q_pe, ckv_cache, kpe_cache, block_table, args.kv_len, scale)
    permuted = runner(q_nope, q_pe, ckv_cache_perm, kpe_cache_perm, block_table, args.kv_len, scale)
    sync(device)
    _, _, bad_count = compare("original_vs_permuted_kv", permuted, ref, args.atol, args.rtol)

    ref_ms = bench(lambda: runner(q_nope, q_pe, ckv_cache, kpe_cache, block_table, args.kv_len, scale), args.warmup, args.iters, device)
    perm_ms = bench(lambda: runner(q_nope, q_pe, ckv_cache_perm, kpe_cache_perm, block_table, args.kv_len, scale), args.warmup, args.iters, device)
    print(f"MLA_KV_PERM_BENCH original_avg_ms={ref_ms:.6f} permuted_avg_ms={perm_ms:.6f}")

    if args.fail_on_diff and bad_count:
        raise AssertionError("MLA output changed after permuting KV tokens.")


if __name__ == "__main__":
    main()
