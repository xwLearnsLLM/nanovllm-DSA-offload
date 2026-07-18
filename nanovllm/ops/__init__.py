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
    custom_opapi = (
        _CUSTOM_OPP_VENDOR / "op_api" / "lib" / "libcust_opapi.so"
    )
    if custom_opapi.is_file():
        # The machine may contain another vendor's libcust_opapi.so with an
        # older operator ABI.  Give the C++ adapter the exact repository-local
        # library so symbol lookup can never silently bind to that copy.
        os.environ["NANOVLLM_CUST_OPAPI_LIB"] = str(custom_opapi)

try:
    importlib.import_module("torch_npu")
    _C = importlib.import_module("nanovllm._C")
except ImportError as exc:
    raise ImportError(
        "nanovllm Ascend ops are not built. Run "
        "`bash scripts/build_nanovllm_ops.sh` on the Ascend machine first. "
        f"Original import error: {exc}"
    ) from exc


moe_gating_top_k = _C.moe_gating_top_k
npu_lightning_indexer = _C.npu_lightning_indexer
npu_gather_selection_kv_cache = _C.npu_gather_selection_kv_cache
batch_matmul_transpose = _C.batch_matmul_transpose
matmul_allreduce_add_rmsnorm = _C.matmul_allreduce_add_rmsnorm


def _missing_dsa_indexer_project(*args, **kwargs):
    raise RuntimeError(
        "dsa_indexer_project_post is not built into nanovllm._C. "
        "Run `bash scripts/build_nanovllm_ops.sh` on the Ascend machine first."
    )


dsa_indexer_project_binding_version = getattr(_C, "dsa_indexer_project_binding_version", lambda: "missing")
dsa_indexer_project_post_out = getattr(_C, "dsa_indexer_project_post_out", _missing_dsa_indexer_project)


def mla_preprocess(
    hidden_state,
    wdqkv,
    descale0,
    gamma1,
    beta1,
    wuq,
    descale1,
    gamma2,
    cos,
    sin,
    wuk,
    kv_cache,
    kv_cache_rope,
    slotmapping,
    *,
    q_out0,
    kv_cache_out0,
    q_out1,
    kv_cache_out1,
    inner_out,
    quant_scale0=None,
    quant_offset0=None,
    bias0=None,
    quant_scale1=None,
    quant_offset1=None,
    bias1=None,
    ctkv_scale=None,
    q_nope_scale=None,
    cache_mode="krope_ctkv",
    quant_mode="no_quant",
    enable_inner_out=False,
):
    return _C.mla_preprocess(
        hidden_state,
        wdqkv,
        descale0,
        gamma1,
        beta1,
        wuq,
        descale1,
        gamma2,
        cos,
        sin,
        wuk,
        kv_cache,
        kv_cache_rope,
        slotmapping,
        q_out0,
        kv_cache_out0,
        q_out1,
        kv_cache_out1,
        inner_out,
        quant_scale0,
        quant_offset0,
        bias0,
        quant_scale1,
        quant_offset1,
        bias1,
        ctkv_scale,
        q_nope_scale,
        cache_mode,
        quant_mode,
        enable_inner_out,
    )


__all__ = [
    "batch_matmul_transpose",
    "dsa_indexer_project_binding_version",
    "dsa_indexer_project_post_out",
    "matmul_allreduce_add_rmsnorm",
    "mla_preprocess",
    "moe_gating_top_k",
    "npu_gather_selection_kv_cache",
    "npu_lightning_indexer",
]
