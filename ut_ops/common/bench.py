from __future__ import annotations

from time import perf_counter
from typing import Callable, TypeVar

import torch

from ut_ops.common.device import sync_device

T = TypeVar("T")


def benchmark_ms(fn: Callable[[], T], device: torch.device, warmup: int, iters: int) -> float:
    for _ in range(int(warmup)):
        fn()
    sync_device(device)
    start = perf_counter()
    for _ in range(int(iters)):
        fn()
    sync_device(device)
    return (perf_counter() - start) * 1000.0 / max(int(iters), 1)
