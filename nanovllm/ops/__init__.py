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
    _PACKAGE_DIR / "_cann_ops_custom" / "vendors" / "nanovllm-ascend"
)
if _CUSTOM_OPP_VENDOR.exists():
    _prepend_env_path("ASCEND_CUSTOM_OPP_PATH", _CUSTOM_OPP_VENDOR)
    custom_opapi = _CUSTOM_OPP_VENDOR / "op_api" / "lib" / "libcust_opapi.so"
    if custom_opapi.is_file():
        os.environ["NANOVLLM_CUST_OPAPI_LIB"] = str(custom_opapi)

try:
    importlib.import_module("torch_npu")
    torch = importlib.import_module("torch")
    importlib.import_module("nanovllm._C")
except ImportError as exc:
    raise ImportError(
        "MTP offloading operators are not built. Run "
        "`bash scripts/build_nanovllm_ops.sh` on the Ascend machine first. "
        f"Original import error: {exc}"
    ) from exc


scatter_copy = torch.ops.nanovllm_dsa.scatter_copy.default
sparse_tail_attention_mtp = (
    torch.ops.nanovllm_dsa.sparse_tail_attention_mtp.default
)
fused_copy_sfa_mtp = torch.ops.nanovllm_dsa.fused_copy_sfa_mtp.default

__all__ = [
    "scatter_copy",
    "sparse_tail_attention_mtp",
    "fused_copy_sfa_mtp",
]
