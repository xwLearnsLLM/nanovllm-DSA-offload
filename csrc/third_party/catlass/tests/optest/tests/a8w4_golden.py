# This program is free software, you can redistribute it and/or modify.
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
# BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE. See LICENSE in the root of
# the software repository for the full text of the License.

"""Golden helpers aligned with examples/59_ascend950_a8w4_mx_matmul/gen_data.py."""

from __future__ import annotations

from typing import Dict, Tuple

import torch

_BLOCK_SIZE = 32
_EPSILON = 1e-12
_MIN_SCALE_EXP = -128
_MAX_SCALE_EXP = 127

_FP8_FORMATS = {
    "E4M3": {
        "torch_dtype": torch.float8_e4m3fn,
        "emax": 8,
        "max_value": 448.0,
    },
}

_FP4_FORMATS: Dict[str, Dict[str, float]] = {
    "E2M1": {
        "exp_bits": 2,
        "mantissa_bits": 1,
        "bias": 1,
        "emax": 2,
        "max_value": 6.0,
        "min_value": -6.0,
    },
}


def _case_random_seed(*values) -> int:
    seed = 0
    for value in values:
        if isinstance(value, float):
            seed = (seed * 1_000_003 + int(round(value * 1_000))) & 0x7FFFFFFF
        else:
            seed = (seed * 1_000_003 + int(value)) & 0x7FFFFFFF
    return seed


def _set_case_random_seed(*values) -> int:
    seed = _case_random_seed(*values)
    torch.manual_seed(seed)
    return seed


def _build_e4m3_lut() -> torch.Tensor:
    bias = 7
    fp8_max = 448.0
    values = []
    for i in range(256):
        if i < 128:
            sign, val = 1, i
        else:
            sign, val = -1, i - 128
        if val == 0:
            v = 0.0
        elif val == 127:
            v = sign * fp8_max
        else:
            exp = (val >> 3) & 0x0F
            mantissa = val & 0x07
            if exp == 0:
                v = (mantissa / 8.0) * (2.0 ** (1 - bias))
            else:
                v = (1.0 + mantissa / 8.0) * (2.0 ** (exp - bias))
            v *= sign
        v = max(min(v, fp8_max), -fp8_max)
        values.append(v)
    return torch.tensor(values, dtype=torch.float32)


_FP8_LUT_CACHE: Dict[str, torch.Tensor] = {}
_FP8_LUT_POS_CACHE: Dict[str, torch.Tensor] = {}


def _get_fp8_lut_pos(format_name: str) -> torch.Tensor:
    if format_name not in _FP8_LUT_POS_CACHE:
        if format_name not in _FP8_LUT_CACHE:
            _FP8_LUT_CACHE[format_name] = _build_e4m3_lut()
        full = _FP8_LUT_CACHE[format_name]
        _FP8_LUT_POS_CACHE[format_name] = full[:128].contiguous()
    return _FP8_LUT_POS_CACHE[format_name]


def _build_fp4_lut(format_name: str) -> torch.Tensor:
    config = _FP4_FORMATS[format_name]
    exp_bits = int(config["exp_bits"])
    mantissa_bits = int(config["mantissa_bits"])
    bias = float(config["bias"])
    values = []
    for i in range(16):
        sign = (i >> 3) & 0x01
        exp = (i >> mantissa_bits) & ((1 << exp_bits) - 1)
        mantissa = i & ((1 << mantissa_bits) - 1)
        if exp == 0:
            value = 0.0 if mantissa == 0 else (mantissa / float(1 << mantissa_bits)) * (2.0 ** (1.0 - bias))
        else:
            value = (1.0 + mantissa / float(1 << mantissa_bits)) * (2.0 ** (float(exp) - bias))
        if sign == 1:
            value = -value
        values.append(value)
    return torch.tensor(values, dtype=torch.float32)


_FP4_LUT = {"E2M1": _build_fp4_lut("E2M1")}


def _e8m0_exp(max_abs: torch.Tensor, emax: int, epsilon: float = _EPSILON) -> torch.Tensor:
    zero_mask = max_abs < epsilon
    safe = torch.where(zero_mask, torch.ones_like(max_abs), max_abs)
    bits = safe.contiguous().view(torch.int32)
    exp_bits = (bits >> 23) & 0xFF
    exp = exp_bits - 127 - emax
    exp = exp.clamp(_MIN_SCALE_EXP, _MAX_SCALE_EXP)
    return torch.where(zero_mask, torch.zeros_like(exp), exp)


def _vectorized_lut_quantize_fp8(scaled: torch.Tensor, format_name: str, fp8_dtype: torch.dtype) -> torch.Tensor:
    lut_pos = _get_fp8_lut_pos(format_name)
    last_idx = lut_pos.numel() - 1
    sign = torch.sign(scaled)
    mag = scaled.abs()
    upper_idx = torch.searchsorted(lut_pos, mag).clamp(max=last_idx)
    lower_idx = (upper_idx - 1).clamp(min=0)
    upper_val = lut_pos[upper_idx]
    lower_val = lut_pos[lower_idx]
    pick_lower = (mag - lower_val) <= (upper_val - mag)
    chosen_mag = torch.where(pick_lower, lower_val, upper_val)
    snapped_fp32 = torch.where(chosen_mag == 0, torch.zeros_like(scaled), sign * chosen_mag)
    return snapped_fp32.to(fp8_dtype)


def _quantize_to_fp4_lut(values: torch.Tensor, format_name: str) -> Tuple[torch.Tensor, torch.Tensor]:
    lut = _FP4_LUT[format_name].to(values.device)
    clamped = values.clamp(_FP4_FORMATS[format_name]["min_value"], _FP4_FORMATS[format_name]["max_value"])
    distances = (clamped.unsqueeze(-1) - lut).abs()
    indices = torch.argmin(distances, dim=-1)
    return lut[indices], indices.to(torch.uint8)


def _pack_fp4_nibbles(index_matrix: torch.Tensor) -> torch.Tensor:
    rows, cols = index_matrix.shape
    if cols % 2 != 0:
        index_matrix = torch.cat(
            [index_matrix, torch.zeros((rows, 1), dtype=torch.uint8, device=index_matrix.device)],
            dim=1,
        )
    low = index_matrix[:, 0::2]
    high = index_matrix[:, 1::2] << 4
    return (low | high).to(torch.uint8)


def _quantize_fp8_axis_last(matrix: torch.Tensor, format_name: str, block_size: int = _BLOCK_SIZE):
    m, n = matrix.shape
    fp8_dtype = _FP8_FORMATS[format_name]["torch_dtype"]
    fp8_emax = _FP8_FORMATS[format_name]["emax"]
    fp8_max = _FP8_FORMATS[format_name]["max_value"]
    num_blocks = (n + block_size - 1) // block_size
    padded_n = num_blocks * block_size
    padded = matrix if padded_n == n else torch.cat(
        [matrix, torch.zeros(m, padded_n - n, dtype=matrix.dtype)], dim=1
    )
    blocks = padded.view(m, num_blocks, block_size)
    max_abs = blocks.abs().amax(dim=-1)
    exp = _e8m0_exp(max_abs, fp8_emax)
    scale = torch.exp2(exp.to(torch.float32))
    scaled = (blocks / scale.unsqueeze(-1)).clamp(-fp8_max, fp8_max)
    quant_fp8 = _vectorized_lut_quantize_fp8(scaled, format_name, fp8_dtype)
    dequant = quant_fp8.to(torch.float32) * scale.unsqueeze(-1)
    quant_fp8 = quant_fp8.reshape(m, padded_n)[:, :n].contiguous()
    dequant = dequant.reshape(m, padded_n)[:, :n].contiguous()
    padded_blocks = ((num_blocks + 1) // 2) * 2
    if padded_blocks != num_blocks:
        scale_padded = torch.ones((m, padded_blocks), dtype=torch.float32)
        scale_padded[:, :num_blocks] = scale
        scale = scale_padded
    return quant_fp8, scale, dequant


def _quantize_fp8(matrix: torch.Tensor, format_name: str, axis: int):
    if axis == 0:
        qt, st, dt = _quantize_fp8_axis_last(matrix.t().contiguous(), format_name)
        return qt.t().contiguous(), st.t().contiguous(), dt.t().contiguous()
    return _quantize_fp8_axis_last(matrix, format_name)


def _quantize_fp4_axis_last(matrix: torch.Tensor, format_name: str, block_size: int = _BLOCK_SIZE):
    m, n = matrix.shape
    padded_n = ((n + block_size - 1) // block_size) * block_size
    num_blocks = padded_n // block_size
    padded = matrix if padded_n == n else torch.zeros((m, padded_n), dtype=matrix.dtype, device=matrix.device)
    if padded_n != n:
        padded[:, :n] = matrix
    blocks = padded.view(m, num_blocks, block_size)
    max_abs = blocks.abs().amax(dim=-1)
    exp = torch.floor(torch.log2(max_abs.clamp(min=_EPSILON))) - _FP4_FORMATS[format_name]["emax"]
    exp = torch.where(max_abs < _EPSILON, torch.zeros_like(exp), exp).clamp(_MIN_SCALE_EXP, _MAX_SCALE_EXP)
    scale = torch.pow(torch.tensor(2.0, dtype=torch.float32, device=matrix.device), exp)
    scaled = blocks / scale.unsqueeze(-1)
    quantized_blocks, _ = _quantize_to_fp4_lut(scaled, format_name)
    dequant_blocks = quantized_blocks * scale.unsqueeze(-1)
    quantized = quantized_blocks.reshape(m, padded_n)[:, :n].contiguous()
    dequantized = dequant_blocks.reshape(m, padded_n)[:, :n].contiguous()
    padded_blocks = ((num_blocks + 1) // 2) * 2
    if padded_blocks != num_blocks:
        scale_padded = torch.ones((m, padded_blocks), dtype=torch.float32, device=matrix.device)
        scale_padded[:, :num_blocks] = scale
        scale = scale_padded
    return quantized, scale, dequantized


def _quantize_fp4(matrix: torch.Tensor, format_name: str, axis: int):
    if axis == 0:
        qt, st, dt = _quantize_fp4_axis_last(matrix.t().contiguous(), format_name)
        return qt.t().contiguous(), st.t().contiguous(), dt.t().contiguous()
    return _quantize_fp4_axis_last(matrix, format_name)


def _gen_data_fp8_e4m3(row: int, col: int, axis: int):
    matrix = torch.randn((row, col), dtype=torch.float32) * 10
    quant_fp8, scale_fp32, dequant_fp32 = _quantize_fp8(matrix, "E4M3", axis)
    return (
        quant_fp8.to(torch.float8_e4m3fn),
        scale_fp32.to(torch.float8_e8m0fnu),
        dequant_fp32,
    )


def _gen_data_fp4_e2m1(row: int, col: int, axis: int, trans: int):
    matrix = torch.randn((row, col), dtype=torch.float32)
    quantized_matrix, scale_matrix, dequantized_matrix = _quantize_fp4(matrix, "E2M1", axis)
    if trans == 1:
        quantized_matrix = quantized_matrix.t().contiguous()
    _, fp4_indices = _quantize_to_fp4_lut(quantized_matrix, "E2M1")
    packed = _pack_fp4_nibbles(fp4_indices)
    return packed, scale_matrix.to(torch.float8_e8m0fnu), dequantized_matrix


def _clone_int8_storage(tensor: torch.Tensor) -> torch.Tensor:
    flat = tensor.contiguous().flatten()
    return torch.empty(flat.numel(), dtype=torch.int8, device=flat.device).copy_(flat.view(torch.int8))


def prepare_a8w4_mx_inputs(
    m: int, n: int, k: int, device: str = "npu", trans_a: int = 0, trans_b: int = 1
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build inputs for example 59 (default trans_a=0, trans_b=1)."""
    _set_case_random_seed(59, m, n, k, trans_a, trans_b)
    a_fp8, a_scale, a_deq = _gen_data_fp8_e4m3(m, k, 1)
    b_packed, b_scale, b_deq = _gen_data_fp4_e2m1(k, n, 0, trans_b)

    a_scale = a_scale.reshape(a_scale.shape[0], a_scale.shape[1] // 2, 2).contiguous()
    b_scale = b_scale.reshape(b_scale.shape[0] // 2, 2, b_scale.shape[1]).contiguous()
    if trans_a == 1:
        a_scale = a_scale.permute(1, 0, 2).contiguous()
    if trans_b == 1:
        b_scale = b_scale.permute(2, 0, 1).contiguous()
    else:
        b_scale = b_scale.permute(0, 2, 1).contiguous()

    a_fp8 = a_fp8.contiguous()
    b_int8 = _clone_int8_storage(b_packed)
    expected = a_deq @ b_deq

    if device == "npu":
        a_fp8 = a_fp8.npu()
        b_int8 = b_int8.npu()
        a_scale = a_scale.npu()
        b_scale = b_scale.npu()
    return a_fp8, b_int8, a_scale, b_scale, expected


def compare_a8w4_result(result: torch.Tensor, expected: torch.Tensor, k: int) -> tuple[bool, float]:
    """Align with examples/common/golden/compare_data.hpp CompareData."""
    rtol = 1.0 / 128 if k >= 2048 else 1.0 / 256
    result_cpu = result.float().cpu()
    expected_cpu = expected.float().cpu()
    diff = (result_cpu - expected_cpu).abs()
    threshold = rtol * torch.clamp(expected_cpu.abs(), min=1.0)
    max_diff = diff.max().item()
    return bool((diff <= threshold).all()), max_diff
