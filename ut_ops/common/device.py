from __future__ import annotations

import random

import torch

try:
    import torch_npu  # type: ignore  # noqa: F401
except Exception:  # pragma: no cover - local non-Ascend syntax checks
    torch_npu = None


def set_device(device_name: str) -> torch.device:
    device = torch.device(device_name)
    if device.type == "npu":
        if torch_npu is None:
            raise RuntimeError("torch_npu is required for NPU runs")
        torch.npu.set_device(device)
    return device


def seed_everything(seed: int, device: torch.device | None = None) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if device is not None and device.type == "npu" and torch_npu is not None:
        torch.npu.manual_seed(seed)


def sync_device(device: torch.device) -> None:
    if device.type == "npu" and torch_npu is not None:
        torch.npu.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize(device)
