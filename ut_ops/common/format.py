from __future__ import annotations

import torch


def parse_int_list(value: str) -> list[int]:
    result: list[int] = []
    for item in value.split(","):
        item = item.strip()
        if item:
            result.append(int(item))
    if not result:
        raise ValueError(f"empty int list: {value!r}")
    return result


def tensor_desc(name: str, tensor: torch.Tensor) -> str:
    return (
        f"{name}=shape={tuple(tensor.shape)} dtype={tensor.dtype} "
        f"device={tensor.device} contiguous={tensor.is_contiguous()} stride={tuple(tensor.stride())}"
    )


def diff_summary(actual: torch.Tensor, expected: torch.Tensor) -> tuple[float, float, int]:
    diff = (actual.float() - expected.float()).abs()
    max_abs = float(diff.max().item()) if diff.numel() else 0.0
    mean_abs = float(diff.mean().item()) if diff.numel() else 0.0
    return max_abs, mean_abs, int(diff.numel())
