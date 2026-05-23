import argparse
import math
import time

import torch
import torch_npu


DEFAULT_SCALE = 0.1352337788608801


def tensor_desc(name: str, tensor: torch.Tensor) -> str:
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


def make_causal_mask(mask_size: int, device: str) -> torch.Tensor:
    return torch.triu(
        torch.ones(mask_size, mask_size, dtype=torch.int8),
        diagonal=1,
    ).to(device)


def make_actual_seq_lengths(seq_lens: list[int]) -> list[int]:
    total = 0
    actual_seq_lengths = []
    for seq_len in seq_lens:
        total += seq_len
        actual_seq_lengths.append(total)
    return actual_seq_lengths


def make_inputs(args: argparse.Namespace):
    total_tokens = sum(args.seq_lens)
    torch.manual_seed(args.seed)

    q_nope = torch.randn(
        total_tokens,
        args.heads,
        args.nope_dim,
        dtype=torch.bfloat16,
        device=args.device,
    )
    k_nope = torch.randn(
        total_tokens,
        args.heads,
        args.nope_dim,
        dtype=torch.bfloat16,
        device=args.device,
    )
    value = torch.randn(
        total_tokens,
        args.heads,
        args.value_dim,
        dtype=torch.bfloat16,
        device=args.device,
    )
    q_pe = torch.randn(
        total_tokens,
        args.heads,
        args.rope_dim,
        dtype=torch.bfloat16,
        device=args.device,
    )
    k_pe = torch.randn(
        total_tokens,
        args.heads,
        args.rope_dim,
        dtype=torch.bfloat16,
        device=args.device,
    )
    attn_mask = make_causal_mask(args.mask_size, args.device)
    actual_seq_lengths = make_actual_seq_lengths(args.seq_lens)
    return q_nope, k_nope, value, q_pe, k_pe, attn_mask, actual_seq_lengths


def run_mla_prefill(
    q_nope: torch.Tensor,
    k_nope: torch.Tensor,
    value: torch.Tensor,
    q_pe: torch.Tensor,
    k_pe: torch.Tensor,
    attn_mask: torch.Tensor,
    actual_seq_lengths: list[int],
    args: argparse.Namespace,
):
    out, lse = torch_npu.npu_fused_infer_attention_score(
        q_nope,
        k_nope,
        value,
        query_rope=q_pe,
        key_rope=k_pe,
        num_heads=args.heads,
        num_key_value_heads=args.heads,
        input_layout="TND",
        atten_mask=attn_mask,
        sparse_mode=3,
        scale=args.scale,
        antiquant_mode=0,
        antiquant_scale=None,
        block_table=None,
        block_size=0,
        softmax_lse_flag=True,
        actual_seq_lengths=actual_seq_lengths,
        actual_seq_lengths_kv=actual_seq_lengths,
    )
    return out, lse


def reference_prefill(
    q_nope: torch.Tensor,
    k_nope: torch.Tensor,
    value: torch.Tensor,
    q_pe: torch.Tensor,
    k_pe: torch.Tensor,
    seq_lens: list[int],
    scale: float,
) -> torch.Tensor:
    q_nope = q_nope.float().cpu()
    k_nope = k_nope.float().cpu()
    value = value.float().cpu()
    q_pe = q_pe.float().cpu()
    k_pe = k_pe.float().cpu()

    outputs = []
    start = 0
    for seq_len in seq_lens:
        end = start + seq_len
        q = q_nope[start:end]
        k = k_nope[start:end]
        v = value[start:end]
        qr = q_pe[start:end]
        kr = k_pe[start:end]

        scores = torch.einsum("thd,shd->hts", q, k)
        scores += torch.einsum("thd,shd->hts", qr, kr)
        scores *= scale
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, dtype=torch.bool),
            diagonal=1,
        )
        scores.masked_fill_(causal_mask.unsqueeze(0), float("-inf"))
        probs = torch.softmax(scores, dim=-1)
        outputs.append(torch.einsum("hts,shd->thd", probs, v))
        start = end
    return torch.cat(outputs, dim=0)


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, math.ceil(q / 100.0 * len(ordered)) - 1))
    return ordered[index]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Probe vLLM-Ascend dense MLA prefill FIA op.",
    )
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--seq-lens", type=parse_seq_lens, default=[128])
    parser.add_argument("--heads", type=int, default=32)
    parser.add_argument("--nope-dim", type=int, default=128)
    parser.add_argument("--rope-dim", type=int, default=64)
    parser.add_argument("--value-dim", type=int, default=128)
    parser.add_argument("--scale", type=float, default=DEFAULT_SCALE)
    parser.add_argument("--mask-size", type=int, default=2048)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--check-reference", action="store_true")
    parser.add_argument("--assert-reference", action="store_true")
    parser.add_argument("--reference-max-tokens", type=int, default=256)
    args = parser.parse_args()

    total_tokens = sum(args.seq_lens)

    device_index = int(str(args.device).split(":")[-1])
    torch.npu.set_device(device_index)
    torch.npu.config.allow_internal_format = True

    q_nope, k_nope, value, q_pe, k_pe, attn_mask, actual_seq_lengths = make_inputs(args)

    print("MLA_PROBE op=torch_npu.npu_fused_infer_attention_score")
    print(
        "MLA_PROBE config "
        f"seq_lens={args.seq_lens} total_tokens={total_tokens} "
        f"heads={args.heads} nope_dim={args.nope_dim} "
        f"rope_dim={args.rope_dim} value_dim={args.value_dim} "
        f"scale={args.scale} mask_size={args.mask_size}"
    )
    if args.mask_size < max(args.seq_lens):
        print(
            "MLA_PROBE mask_note "
            f"mask_size={args.mask_size} is smaller than max_seq_len={max(args.seq_lens)}; "
            "this matches vllm-ascend's fixed 2048x2048 splitfuse MLA mask.",
            flush=True,
        )
    for name, tensor in (
        ("q_nope", q_nope),
        ("k_nope", k_nope),
        ("value", value),
        ("q_pe", q_pe),
        ("k_pe", k_pe),
        ("attn_mask", attn_mask),
    ):
        print("MLA_PROBE", tensor_desc(name, tensor))
    print(f"MLA_PROBE actual_seq_lengths={actual_seq_lengths}")

    for _ in range(args.warmup):
        out, lse = run_mla_prefill(
            q_nope,
            k_nope,
            value,
            q_pe,
            k_pe,
            attn_mask,
            actual_seq_lengths,
            args,
        )
    torch.npu.synchronize()

    latencies_ms = []
    for _ in range(args.iters):
        start = time.perf_counter()
        out, lse = run_mla_prefill(
            q_nope,
            k_nope,
            value,
            q_pe,
            k_pe,
            attn_mask,
            actual_seq_lengths,
            args,
        )
        torch.npu.synchronize()
        latencies_ms.append((time.perf_counter() - start) * 1000.0)

    print("MLA_PROBE after_mla", tensor_desc("out", out), tensor_desc("lse", lse))
    finite = torch.isfinite(out)
    print(
        "MLA_PROBE out_stats "
        f"finite={finite.sum().item()}/{out.numel()} "
        f"min={out.float().min().item():.6g} max={out.float().max().item():.6g}"
    )
    print(
        "MLA_BENCH "
        f"avg_ms={sum(latencies_ms) / len(latencies_ms):.6f} "
        f"min_ms={min(latencies_ms):.6f} max_ms={max(latencies_ms):.6f} "
        f"p99_ms={percentile(latencies_ms, 99):.6f}"
    )

    if args.check_reference:
        if total_tokens > args.reference_max_tokens:
            print(
                "MLA_PROBE reference_skipped "
                f"total_tokens={total_tokens} limit={args.reference_max_tokens}"
            )
            return
        ref = reference_prefill(q_nope, k_nope, value, q_pe, k_pe, args.seq_lens, args.scale)
        diff = (out.float().cpu() - ref).abs()
        max_abs = diff.max().item()
        mean_abs = diff.mean().item()
        ref_abs = ref.abs().clamp_min(1e-6)
        max_rel = (diff / ref_abs).max().item()
        print(
            "MLA_PROBE reference "
            f"max_abs={max_abs:.6g} mean_abs={mean_abs:.6g} max_rel={max_rel:.6g}"
        )
        if args.assert_reference:
            if max_abs > 0.08 and max_rel > 0.08:
                raise AssertionError(
                    f"dense MLA output differs from reference: max_abs={max_abs}, max_rel={max_rel}"
                )


if __name__ == "__main__":
    main()
