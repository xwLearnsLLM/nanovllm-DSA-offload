import os
from pathlib import Path


_ROOT = Path(__file__).resolve().parents[2]
_LOCAL_OPP = _ROOT / "_custom_opp"
_OPAPI_LIBS = tuple(
    (_LOCAL_OPP / "vendors").glob("*/op_api/lib/libcust_opapi.so")
)
if len(_OPAPI_LIBS) != 1:
    raise RuntimeError(
        "nanovllm_dsa_a5 requires exactly one repository-local "
        "libcust_opapi.so. Run bash build.sh first."
    )
_OPAPI = _OPAPI_LIBS[0]
_VENDOR = _OPAPI.parents[2]

_existing = [
    value
    for value in os.getenv("ASCEND_CUSTOM_OPP_PATH", "").split(":")
    if value
]
_vendor_str = str(_VENDOR)
if _vendor_str not in _existing:
    os.environ["ASCEND_CUSTOM_OPP_PATH"] = ":".join(
        [_vendor_str, *_existing]
    )
os.environ.setdefault(
    "NANOVLLM_A5_INSTALL_OPP_PATH",
    str(_LOCAL_OPP),
)

import torch  # noqa: E402
import torch_npu  # noqa: E402,F401

from . import _C  # noqa: E402,F401


lidu_decode_update = torch.ops.nanovllm_dsa.lidu_decode_update
lidu_decode_update_out = torch.ops.nanovllm_dsa.lidu_decode_update_out
scatter_copy = torch.ops.nanovllm_dsa.scatter_copy
sparse_and_tail_attention = (
    torch.ops.nanovllm_dsa.sparse_and_tail_attention
)


def local_opapi_path() -> str:
    return str(_OPAPI)


__all__ = [
    "lidu_decode_update",
    "lidu_decode_update_out",
    "scatter_copy",
    "sparse_and_tail_attention",
    "local_opapi_path",
]
