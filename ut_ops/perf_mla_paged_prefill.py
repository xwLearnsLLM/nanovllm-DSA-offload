import argparse
import math
import time

import torch
import torch_npu


DEFAULT_SCALE = 0.1352337788608801


def desc(name: str, tensor: torch.Tensor) -> str:
    return (
        f"{name}=shape={tuple(tensor.shape)} dtype={tensor.dtype} "
        f"device={tensor.device} contiguous={tensor.is_contiguous()} "
        f"stride={tuple(tensor.stride())} storage_offset={tensor.storage_offset()}"
    )


def parse_seq_lens(value: str) -> list[int]:
    seq_lens = [int(item) for item in value.split(",") if item.strip()]
    if not seq_lens or any(seq_len <= 0 for seq_len in seq_lens):
        raise argparse.ArgumentTypeError("seq lens must be positive integers")
    return seq_lens


def cumulative(values: list[int]) -> list[int]:
    total = 0
    out = []
    for value in values:
        total += int(value)
        out.append(total)
    return out


def make_block_table(
    seq_lens: list[int],
    block_size: int,
    block_id_base: int,
    device: str,
) -> tuple[torch.Tensor, int]:
    blocks_per_seq = [(seq_len + block_size - 1) // block_size for seq_len in seq_lens]
    max_blocks = max(blocks_per_seq)
    block_table = torch.zeros(len(seq_lens), max_blocks, dtype=torch.int32, device=device)
    next_block = block_id_base
    for seq_idx, num_blocks in enumerate(blocks_per_seq):
        block_table[seq_idx, :num_blocks] = torch.arange(
            next_block,
            next_block + num_blocks,
            dtype=torch.int32,
            device=device,
        )
        next_block += num_blocks
    return block_table, next_block


def make_causal_mask(mask_size: int, device: str) -> torch.Tensor:
    return torch.triu(
        torch.ones(mask_size, mask_size, dtype=torch.int8),
        diagonal=1,
    ).to(device)


def make_inputs(args: argparse.Namespace):
    torch.manual_seed(args.seed)
    total_tokens = sum(args.seq_lens)
    block_table, next_block = make_block_table(
        args.seq_lens,
        args.block_size,
        args.block_id_base,
        args.device,
    )
    num_blocks = args.num_blocks or next_block
    if num_blocks < next_block:
        raise ValueError(
            f"--num-blocks={num_blocks} is too small; need at least {next_block}"
        )

    query = torch.randn(
        total_tokens,
        args.heads,
        args.latent_dim,
        dtype=torch.bfloat16,
        device=args.device,
    )
    key_cache = torch.randn(
        num_blocks,
        args.block_size,
        args.kv_heads,
        args.latent_dim,
        dtype=torch.bfloat16,
        device=args.device,
    )
    value_cache = key_cache if args.share_kv else torch.randn(
        num_blocks,
        args.block_size,
        args.kv_heads,
        args.latent_dim,
        dtype=torch.bfloat16,
        device=args.device,
    )
    query_rope = torch.randn(
        total_tokens,
        args.heads,
        args.rope_dim,
        dtype=torch.bfloat16,
        device=args.device,
    )
    key_rope_cache = torch.randn(
        num_blocks,
        args.block_size,
        args.kv_heads,
        args.rope_dim,
        dtype=torch.bfloat16,
        device=args.device,
    )
    attn_mask = make_causal_mask(args.mask_size, args.device)
    actual_seq_lengths_query = cumulative(args.seq_lens)
    actual_seq_lengths_kv = [int(seq_len) for seq_len in args.seq_lens]
    return (
        query,
        key_cache,
        value_cache,
        query_rope,
        key_rope_cache,
        block_table,
        attn_mask,
        actual_seq_lengths_query,
        actual_seq_lengths_kv,
    )


def run_paged_mla_prefill(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    query_rope: torch.Tensor,
    key_rope_cache: torch.Tensor,
    block_table: torch.Tensor,
    attn_mask: torch.Tensor,
    actual_seq_lengths_query: list[int],
    actual_seq_lengths_kv: list[int],
    args: argparse.Namespace,
):
    out, lse = torch_npu.npu_fused_infer_attention_score(
        query,
        key_cache,
        value_cache,
        query_rope=query_rope,
        key_rope=key_rope_cache,
        num_heads=args.heads,
        num_key_value_heads=args.kv_heads,
        input_layout="TND",
        atten_mask=attn_mask,
        sparse_mode=3,
        scale=args.scale,
        antiquant_mode=0,
        antiquant_scale=None,
        block_table=block_table,
        block_size=args.block_size,
        softmax_lse_flag=True,
        actual_seq_lengths=actual_seq_lengths_query,
        actual_seq_lengths_kv=actual_seq_lengths_kv,
    )
    return out, lse


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, math.ceil(q / 100.0 * len(ordered)) - 1))
    return ordered[index]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Probe dense MLA prefill with paged absorb KV cache.",
    )
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--seq-lens", type=parse_seq_lens, default=[2048])
    parser.add_argument("--heads", type=int, default=32)
    parser.add_argument("--kv-heads", type=int, default=1)
    parser.add_argument("--latent-dim", type=int, default=512)
    parser.add_argument("--rope-dim", type=int, default=64)
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument("--block-id-base", type=int, default=1)
    parser.add_argument("--num-blocks", type=int)
    parser.add_argument("--mask-size", type=int, default=2048)
    parser.add_argument("--scale", type=float, default=DEFAULT_SCALE)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--iters", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--share-kv", action="store_true", default=True)
    args = parser.parse_args()

    device_index = int(str(args.device).split(":")[-1])
    torch.npu.set_device(device_index)
    torch.npu.config.allow_internal_format = True

    (
        query,
        key_cache,
        value_cache,
        query_rope,
        key_rope_cache,
        block_table,
        attn_mask,
        actual_seq_lengths_query,
        actual_seq_lengths_kv,
    ) = make_inputs(args)

    print("PAGED_MLA op=torch_npu.npu_fused_infer_attention_score")
    print(
        "PAGED_MLA config "
        f"seq_lens={args.seq_lens} total_tokens={sum(args.seq_lens)} "
        f"heads={args.heads} kv_heads={args.kv_heads} "
        f"latent_dim={args.latent_dim} rope_dim={args.rope_dim} "
        f"block_size={args.block_size} mask_size={args.mask_size} scale={args.scale}"
    )
    if args.mask_size < max(args.seq_lens):
        print(
            "PAGED_MLA mask_note "
            f"mask_size={args.mask_size} is smaller than max_seq_len={max(args.seq_lens)}; "
            "this follows vllm-ascend's fixed 2048x2048 splitfuse mask.",
            flush=True,
        )
    for name, tensor in (
        ("query", query),
        ("key_cache", key_cache),
        ("value_cache", value_cache),
        ("query_rope", query_rope),
        ("key_rope_cache", key_rope_cache),
        ("block_table", block_table),
        ("attn_mask", attn_mask),
    ):
        print("PAGED_MLA", desc(name, tensor))
    print(
        "PAGED_MLA lengths "
        f"actual_seq_lengths_query={actual_seq_lengths_query} "
        f"actual_seq_lengths_kv={actual_seq_lengths_kv}"
    )

    for _ in range(args.warmup):
        out, lse = run_paged_mla_prefill(
            query,
            key_cache,
            value_cache,
            query_rope,
            key_rope_cache,
            block_table,
            attn_mask,
            actual_seq_lengths_query,
            actual_seq_lengths_kv,
            args,
        )
    torch.npu.synchronize()

    latencies_ms = []
    for _ in range(args.iters):
        start = time.perf_counter()
        out, lse = run_paged_mla_prefill(
            query,
            key_cache,
            value_cache,
            query_rope,
            key_rope_cache,
            block_table,
            attn_mask,
            actual_seq_lengths_query,
            actual_seq_lengths_kv,
            args,
        )
        torch.npu.synchronize()
        latencies_ms.append((time.perf_counter() - start) * 1000.0)

    finite = torch.isfinite(out)
    print("PAGED_MLA after_mla", desc("out", out), desc("lse", lse))
    print(
        "PAGED_MLA out_stats "
        f"finite={finite.sum().item()}/{out.numel()} "
        f"min={out.float().min().item():.6g} max={out.float().max().item():.6g}"
    )
    print(
        "PAGED_MLA_BENCH "
        f"avg_ms={sum(latencies_ms) / len(latencies_ms):.6f} "
        f"min_ms={min(latencies_ms):.6f} max_ms={max(latencies_ms):.6f} "
        f"p99_ms={percentile(latencies_ms, 99):.6f}"
    )


if __name__ == "__main__":
    main()
