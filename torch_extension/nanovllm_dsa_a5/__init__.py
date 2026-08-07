from __future__ import annotations

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
os.environ.setdefault("NANOVLLM_A5_INSTALL_OPP_PATH", str(_LOCAL_OPP))
os.environ["NANOVLLM_CUST_OPAPI_LIB"] = str(_OPAPI)

import torch_npu  # noqa: E402,F401

from . import _C  # noqa: E402,F401
from .ops import (  # noqa: E402
    lidu_cache_update,
    lidu_cache_update_out,
    lidu_decode_update,
    lidu_decode_update_c8,
    lidu_decode_update_c8_out,
    lidu_decode_update_out,
    scatter_copy,
    scatter_copy_c8,
    scatter_copy_c8_out,
    sparse_and_tail_attention,
    sparse_and_tail_attention_and_scatter_copy,
    sparse_and_tail_attention_and_scatter_copy_mte_pipeline,
    sparse_and_tail_attention_c8,
)


def local_opapi_path() -> str:
    return str(_OPAPI)


__all__ = [
    "lidu_decode_update",
    "lidu_decode_update_out",
    "lidu_cache_update",
    "lidu_cache_update_out",
    "lidu_decode_update_c8",
    "lidu_decode_update_c8_out",
    "scatter_copy",
    "scatter_copy_c8",
    "scatter_copy_c8_out",
    "sparse_and_tail_attention",
    "sparse_and_tail_attention_and_scatter_copy",
    "sparse_and_tail_attention_and_scatter_copy_mte_pipeline",
    "sparse_and_tail_attention_c8",
    "local_opapi_path",
]
