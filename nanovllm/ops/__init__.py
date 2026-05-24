from __future__ import annotations

import importlib
import os
from pathlib import Path


def _prepend_env_path(name: str, path: Path) -> None:
    value = str(path)
    existing = os.environ.get(name)
    if existing:
        parts = existing.split(os.pathsep)
        if value in parts:
            return
        os.environ[name] = value + os.pathsep + existing
    else:
        os.environ[name] = value


_PACKAGE_DIR = Path(__file__).resolve().parent.parent
_CUSTOM_OPP_VENDOR = (
    _PACKAGE_DIR
    / "_cann_ops_custom"
    / "vendors"
    / "nanovllm-ascend"
)
if _CUSTOM_OPP_VENDOR.exists():
    _prepend_env_path("ASCEND_CUSTOM_OPP_PATH", _CUSTOM_OPP_VENDOR)

try:
    import torch_npu  # type: ignore  # noqa: F401

    _C = importlib.import_module("nanovllm._C")
except ImportError as exc:
    raise ImportError(
        "nanovllm Ascend ops are not built. Run "
        "`bash scripts/build_nanovllm_ops.sh` on the Ascend machine first. "
        f"Original import error: {exc}"
    ) from exc


moe_gating_top_k = _C.moe_gating_top_k
npu_lightning_indexer = _C.npu_lightning_indexer
npu_sparse_flash_attention = _C.npu_sparse_flash_attention
batch_matmul_transpose = _C.batch_matmul_transpose

__all__ = [
    "batch_matmul_transpose",
    "moe_gating_top_k",
    "npu_lightning_indexer",
    "npu_sparse_flash_attention",
]
