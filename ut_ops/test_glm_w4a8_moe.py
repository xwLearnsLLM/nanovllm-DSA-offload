"""Semantic and performance UT for GLM-5.1 ModelSlim W4A8 MoE.

This reads one real routed expert, checks the packed bytes/scales/biases, and
compares both NPU grouped matmuls with an independently unpacked CPU golden.
It also compares the old split SwiGLU + DynamicQuant path with the official
fused torch_npu operator used by the model.
"""

from __future__ import annotations

import argparse
import json
import os

import torch
import torch_npu  # type: ignore
from safetensors import safe_open

from nanovllm.utils.glm_quant import float32_scale_to_int64_bits


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--layer", type=int, default=3)
    parser.add_argument("--expert", type=int, default=0)
    parser.add_argument("--tokens", type=int, default=2)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iters", type=int, default=10)
    return parser.parse_args()


class CheckpointReader:
    def __init__(self, root: str):
        indices = [
            name
            for name in os.listdir(root)
            if name.endswith(".safetensors.index.json")
        ]
        if not indices:
            raise FileNotFoundError("No safetensors index was found.")
        with open(os.path.join(root, sorted(indices)[0]), encoding="utf-8") as file:
            self.weight_map = json.load(file)["weight_map"]
        self.root = root

    def get(self, name: str) -> torch.Tensor:
        shard = self.weight_map.get(name)
        if shard is None:
            raise KeyError(f"Checkpoint tensor {name!r} is missing.")
        with safe_open(os.path.join(self.root, shard), "pt", "cpu") as reader:
            return reader.get_tensor(name)


def unpack_output_packed_int4(weight: torch.Tensor) -> torch.Tensor:
    """[N/2,K] packed bytes -> [K,N] signed INT4 values."""

    if weight.dtype != torch.int8:
        raise TypeError(f"Packed weight must be int8, got {weight.dtype}.")
    packed = weight.transpose(0, 1).contiguous().to(torch.uint8)
    low = (packed & 0x0F).to(torch.int8)
    high = ((packed >> 4) & 0x0F).to(torch.int8)
    low = torch.where(low >= 8, low - 16, low)
    high = torch.where(high >= 8, high - 16, high)
    logical = torch.empty(
        packed.shape[0], packed.shape[1] * 2, dtype=torch.int8
    )
    logical[:, 0::2] = low
    logical[:, 1::2] = high
    return logical


def gmm_golden(
    x_int8: torch.Tensor,
    per_token_scale: torch.Tensor,
    packed_weight: torch.Tensor,
    weight_scale: torch.Tensor,
    scale_bias: torch.Tensor,
) -> torch.Tensor:
    logical_weight = unpack_output_packed_int4(packed_weight).to(torch.float32)
    # The A8W4 kernel splits each INT8 activation into two signed INT4 rows;
    # their recombination represents x-8. scale_bias is the exported +8W
    # correction, already in scaled units.
    mm = (x_int8.to(torch.float32) - 8.0) @ logical_weight
    mm = mm * weight_scale.flatten().to(torch.float32)
    mm = mm + scale_bias.to(torch.float32)
    return mm * per_token_scale.reshape(-1, 1).to(torch.float32)


def prepare_weight(
    packed_weight: torch.Tensor,
    weight_scale: torch.Tensor,
    scale_bias: torch.Tensor,
    device: str,
):
    runtime_weight_i8 = (
        packed_weight.unsqueeze(0)
        .to(device)
        .transpose(1, 2)
        .contiguous()
    )
    runtime_weight = torch_npu.npu_format_cast(
        runtime_weight_i8, 29
    ).view(torch.int32).contiguous()
    runtime_scale = float32_scale_to_int64_bits(
        weight_scale.unsqueeze(0).to(device).transpose(1, 2).contiguous()
    )
    if scale_bias.dim() == 2:
        runtime_bias = (
            scale_bias.unsqueeze(0)
            .to(device)
            .transpose(1, 2)
            .contiguous()
            .sum(dim=1)
        )
    else:
        raise ValueError(f"Unexpected scale_bias rank: {scale_bias.dim()}.")
    return runtime_weight, runtime_scale, runtime_bias


def run_gmm(x, x_scale, weight, scale, bias, group_list):
    return torch_npu.npu_grouped_matmul(
        x=[x],
        weight=[weight],
        scale=[scale],
        bias=[bias],
        per_token_scale=[x_scale],
        split_item=2,
        group_list_type=1,
        group_type=0,
        group_list=group_list,
        output_dtype=torch.bfloat16,
    )[0]


def run_split_swiglu_quant(gate_up):
    return torch_npu.npu_dynamic_quant(torch_npu.npu_swiglu(gate_up))


def run_fused_swiglu_quant(gate_up, group_list):
    return torch_npu.npu_swiglu_quant(
        gate_up,
        group_index=group_list,
        activate_left=True,
        quant_mode=1,
        group_list_type=1,
        dst_type=torch.int8,
    )


def benchmark_npu(fn, warmup: int, iters: int) -> float:
    for _ in range(warmup):
        fn()
    torch.npu.synchronize()
    start = torch.npu.Event(enable_timing=True)
    end = torch.npu.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    end.synchronize()
    return float(start.elapsed_time(end)) / iters


@torch.inference_mode()
def main():
    args = parse_args()
    if args.tokens <= 0:
        raise ValueError("--tokens must be positive.")
    if args.warmup < 0:
        raise ValueError("--warmup must be non-negative.")
    if args.iters <= 0:
        raise ValueError("--iters must be positive.")
    torch.npu.set_device(args.device)
    torch.npu.config.allow_internal_format = True
    reader = CheckpointReader(args.model)
    prefix = (
        f"model.layers.{args.layer}.mlp.experts.{args.expert}"
    )
    gate_w = reader.get(f"{prefix}.gate_proj.weight")
    up_w = reader.get(f"{prefix}.up_proj.weight")
    down_w = reader.get(f"{prefix}.down_proj.weight")
    gate_s = reader.get(f"{prefix}.gate_proj.weight_scale")
    up_s = reader.get(f"{prefix}.up_proj.weight_scale")
    down_s = reader.get(f"{prefix}.down_proj.weight_scale")
    gate_o = reader.get(f"{prefix}.gate_proj.weight_offset")
    up_o = reader.get(f"{prefix}.up_proj.weight_offset")
    down_o = reader.get(f"{prefix}.down_proj.weight_offset")
    gate_b = reader.get(f"{prefix}.gate_proj.scale_bias")
    up_b = reader.get(f"{prefix}.up_proj.scale_bias")
    down_b = reader.get(f"{prefix}.down_proj.scale_bias")
    for name, offset in (
        ("gate", gate_o),
        ("up", up_o),
        ("down", down_o),
    ):
        if torch.count_nonzero(offset).item() != 0:
            raise AssertionError(
                f"{name}_proj weight_offset is non-zero; this UT and "
                "runtime support symmetric W4A8 only."
            )

    hidden_size = gate_w.shape[1]
    intermediate_size = gate_s.shape[0]
    expected = {
        "gate_weight": (intermediate_size // 2, hidden_size),
        "up_weight": (intermediate_size // 2, hidden_size),
        "down_weight": (hidden_size // 2, intermediate_size),
        "gate_scale": (intermediate_size, 1),
        "up_scale": (intermediate_size, 1),
        "down_scale": (hidden_size, 1),
        "gate_bias": (intermediate_size, 1),
        "up_bias": (intermediate_size, 1),
        "down_bias": (hidden_size, 16),
    }
    actual = {
        "gate_weight": tuple(gate_w.shape),
        "up_weight": tuple(up_w.shape),
        "down_weight": tuple(down_w.shape),
        "gate_scale": tuple(gate_s.shape),
        "up_scale": tuple(up_s.shape),
        "down_scale": tuple(down_s.shape),
        "gate_bias": tuple(gate_b.shape),
        "up_bias": tuple(up_b.shape),
        "down_bias": tuple(down_b.shape),
    }
    if actual != expected:
        raise AssertionError(f"Checkpoint shape mismatch: {actual} != {expected}")

    w13 = torch.cat((gate_w, up_w), dim=0)
    s13 = torch.cat((gate_s, up_s), dim=0)
    b13 = torch.cat((gate_b, up_b), dim=0)
    w13_npu, s13_npu, b13_npu = prepare_weight(w13, s13, b13, args.device)
    w2_npu, s2_npu, b2_npu = prepare_weight(
        down_w, down_s, down_b, args.device
    )
    b13_cpu = b13.sum(dim=1)
    b2_cpu = down_b.sum(dim=1)
    torch.manual_seed(7)
    x = (torch.randn(args.tokens, hidden_size) * 0.25).to(
        dtype=torch.bfloat16, device=args.device
    )
    selected_experts = torch.zeros(
        args.tokens, 1, dtype=torch.int32, device=args.device
    )
    x_q, expanded_row_idx, group_list, x_scale = (
        torch_npu.npu_moe_init_routing_v2(
            x,
            selected_experts,
            scale=None,
            active_num=args.tokens,
            expert_num=1,
            expert_tokens_num_type=1,
            expert_tokens_num_flag=True,
            active_expert_range=[0, 1],
            quant_mode=1,
        )
    )
    group_list = group_list.to(torch.int64)
    gate_up = run_gmm(
        x_q, x_scale, w13_npu, s13_npu, b13_npu, group_list
    )
    gate_up_golden = gmm_golden(
        x_q.cpu(), x_scale.cpu(), w13, s13, b13_cpu
    )
    torch.testing.assert_close(
        gate_up.float().cpu(),
        gate_up_golden.to(torch.bfloat16).float(),
        rtol=0.02,
        atol=0.25,
    )

    split_q, split_scale = run_split_swiglu_quant(gate_up)
    fused_q, fused_scale = run_fused_swiglu_quant(gate_up, group_list)
    if split_q.shape != fused_q.shape or split_q.dtype != fused_q.dtype:
        raise AssertionError(
            "Fused SwiGLU-quant output metadata differs from split path: "
            f"split={split_q.shape}/{split_q.dtype}, "
            f"fused={fused_q.shape}/{fused_q.dtype}."
        )
    split_scale_flat = split_scale.reshape(-1).float().cpu()
    fused_scale_flat = fused_scale.reshape(-1).float().cpu()
    torch.testing.assert_close(
        fused_scale_flat,
        split_scale_flat,
        rtol=0.02,
        atol=1e-6,
    )
    int8_max_diff = int(
        (fused_q.to(torch.int16) - split_q.to(torch.int16))
        .abs()
        .max()
        .item()
    )
    if int8_max_diff > 1:
        raise AssertionError(
            "Fused SwiGLU-quant differs from the split INT8 result by more "
            f"than one quantization level: max_diff={int8_max_diff}."
        )
    split_dequant = (
        split_q.float() * split_scale.reshape(-1, 1).float()
    )
    fused_dequant = (
        fused_q.float() * fused_scale.reshape(-1, 1).float()
    )
    torch.testing.assert_close(
        fused_dequant.cpu(),
        split_dequant.cpu(),
        rtol=0.03,
        atol=0.03,
    )

    split_output = run_gmm(
        split_q,
        split_scale,
        w2_npu,
        s2_npu,
        b2_npu,
        group_list,
    )
    fused_output = run_gmm(
        fused_q,
        fused_scale,
        w2_npu,
        s2_npu,
        b2_npu,
        group_list,
    )
    fused_output_golden = gmm_golden(
        fused_q.cpu(),
        fused_scale.cpu(),
        down_w,
        down_s,
        b2_cpu,
    )
    torch.testing.assert_close(
        fused_output.float().cpu(),
        fused_output_golden.to(torch.bfloat16).float(),
        rtol=0.02,
        atol=0.25,
    )
    torch.testing.assert_close(
        fused_output.float().cpu(),
        split_output.float().cpu(),
        rtol=0.02,
        atol=0.25,
    )
    restored = torch_npu.npu_moe_token_unpermute(
        permuted_tokens=fused_output,
        sorted_indices=torch.abs(expanded_row_idx),
        probs=torch.ones(
            args.tokens, 1, dtype=torch.bfloat16, device=args.device
        ),
    )
    torch.testing.assert_close(restored, fused_output, rtol=0, atol=0)

    def split_activation():
        return run_split_swiglu_quant(gate_up)

    def fused_activation():
        return run_fused_swiglu_quant(gate_up, group_list)

    def split_chain():
        current_gate_up = run_gmm(
            x_q, x_scale, w13_npu, s13_npu, b13_npu, group_list
        )
        act_q, act_scale = run_split_swiglu_quant(current_gate_up)
        return run_gmm(
            act_q, act_scale, w2_npu, s2_npu, b2_npu, group_list
        )

    def fused_chain():
        current_gate_up = run_gmm(
            x_q, x_scale, w13_npu, s13_npu, b13_npu, group_list
        )
        act_q, act_scale = run_fused_swiglu_quant(
            current_gate_up, group_list
        )
        return run_gmm(
            act_q, act_scale, w2_npu, s2_npu, b2_npu, group_list
        )

    split_activation_ms = benchmark_npu(
        split_activation, args.warmup, args.iters
    )
    fused_activation_ms = benchmark_npu(
        fused_activation, args.warmup, args.iters
    )
    split_chain_ms = benchmark_npu(split_chain, args.warmup, args.iters)
    fused_chain_ms = benchmark_npu(fused_chain, args.warmup, args.iters)
    print(
        "GLM_W4A8_SWIGLU_QUANT_CHECK "
        f"tokens={args.tokens} int8_max_diff={int8_max_diff} "
        f"scale_max_abs="
        f"{(fused_scale_flat - split_scale_flat).abs().max().item():.9f} "
        "downstream_gmm_close=1 ok=1"
    )
    print(
        "GLM_W4A8_SWIGLU_QUANT_RESULT "
        f"tokens={args.tokens} split_ms={split_activation_ms:.6f} "
        f"fused_ms={fused_activation_ms:.6f} "
        f"speedup={split_activation_ms / fused_activation_ms:.4f} "
        f"warmup={args.warmup} iters={args.iters} performance_assert=0"
    )
    print(
        "GLM_W4A8_MOE_UT_OK "
        f"layer={args.layer} expert={args.expert} tokens={args.tokens} "
        f"split_chain_ms={split_chain_ms:.6f} "
        f"fused_chain_ms={fused_chain_ms:.6f} "
        f"speedup={split_chain_ms / fused_chain_ms:.4f}"
    )


if __name__ == "__main__":
    main()
