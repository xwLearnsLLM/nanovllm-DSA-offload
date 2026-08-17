from __future__ import annotations

import os
from pathlib import Path


_ROOT = Path(__file__).resolve().parents[2]
_LOCAL_OPPS = (_ROOT / "_custom_opp_bf16", _ROOT / "_custom_opp_c8")
_EXPLICIT_OPAPI = os.getenv("NANOVLLM_CUST_OPAPI_LIB")
if _EXPLICIT_OPAPI:
    _OPAPI = Path(_EXPLICIT_OPAPI).expanduser().resolve()
    if not _OPAPI.is_file():
        raise RuntimeError(
            f"NANOVLLM_CUST_OPAPI_LIB does not exist: {_OPAPI}"
        )
else:
    _OPAPI_LIBS = tuple(
        path
        for local_opp in _LOCAL_OPPS
        for path in (local_opp / "vendors").glob(
            "*/op_api/lib/libcust_opapi.so"
        )
    )
    if len(_OPAPI_LIBS) != 1:
        raise RuntimeError(
            "Build one operator family, or set NANOVLLM_CUST_OPAPI_LIB to "
            "the BF16/C8 libcust_opapi.so when both families are built."
        )
    _OPAPI = _OPAPI_LIBS[0].resolve()

_VENDOR = _OPAPI.parents[2]
_LOCAL_OPP = _OPAPI.parents[4]

_existing = [
    value
    for value in os.getenv("ASCEND_CUSTOM_OPP_PATH", "").split(":")
    if value
]
_ordered_vendors = []
for _vendor in [_VENDOR, *_existing]:
    _vendor_str = str(_vendor)
    if _vendor_str not in _ordered_vendors:
        _ordered_vendors.append(_vendor_str)
os.environ["ASCEND_CUSTOM_OPP_PATH"] = ":".join(_ordered_vendors)
os.environ.setdefault("NANOVLLM_A5_INSTALL_OPP_PATH", str(_LOCAL_OPP))
os.environ["NANOVLLM_CUST_OPAPI_LIB"] = str(_OPAPI)

import torch_npu  # noqa: E402,F401

from . import _C  # noqa: E402,F401
from .ops import (  # noqa: E402
    fused_li_manage,
    fused_li_manage_c8,
    fused_li_manage_c8_out,
    fused_li_manage_mtp,
    fused_li_manage_mtp_c8,
    fused_li_manage_mtp_c8_out,
    fused_li_manage_out,
    fused_copy_sparse_tail_attention,
    kvcache_scatter_copy,
    kvcache_scatter_copy_c8,
    kvcache_scatter_copy_c8_out,
    sparse_tail_attention,
    sparse_tail_attention_c8,
    sparse_tail_attention_c8_stage1,
    sparse_tail_attention_c8_stage2,
    sparse_tail_attention_c8_mtp_stage1,
    sparse_tail_attention_c8_mtp_stage2,
)


def local_opapi_path() -> str:
    return str(_OPAPI)


__all__ = [
    "fused_li_manage",
    "fused_li_manage_out",
    "fused_li_manage_mtp",
    "fused_li_manage_mtp_c8",
    "fused_li_manage_mtp_c8_out",
    "fused_li_manage_c8",
    "fused_li_manage_c8_out",
    "kvcache_scatter_copy",
    "kvcache_scatter_copy_c8",
    "kvcache_scatter_copy_c8_out",
    "sparse_tail_attention",
    "fused_copy_sparse_tail_attention",
    "sparse_tail_attention_c8",
    "sparse_tail_attention_c8_stage1",
    "sparse_tail_attention_c8_stage2",
    "sparse_tail_attention_c8_mtp_stage1",
    "sparse_tail_attention_c8_mtp_stage2",
    "local_opapi_path",
]
