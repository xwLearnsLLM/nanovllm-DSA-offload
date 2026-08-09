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
    torch = importlib.import_module("torch")
    # Importing the extension registers all bundled operators with
    # torch.library.  Python callers use torch.ops below; pybind is no longer
    # a second public operator surface.
    importlib.import_module("nanovllm._C")
except ImportError as exc:
    raise ImportError(
        "nanovllm Ascend ops are not built. Run "
        "`bash scripts/build_nanovllm_ops.sh` on the Ascend machine first. "
        f"Original import error: {exc}"
    ) from exc


moe_gating_top_k = torch.ops.nanovllm_dsa.moe_gating_top_k.default
batch_matmul_transpose = torch.ops.nanovllm_dsa.batch_matmul_transpose.default
matmul_allreduce_add_rmsnorm = (
    torch.ops.nanovllm_dsa.matmul_allreduce_add_rmsnorm.default
)
dsa_indexer_query_rope_inplace = (
    torch.ops.nanovllm_dsa.dsa_indexer_query_rope_inplace.default
)


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
    return torch.ops.nanovllm_dsa.mla_preprocess.default(
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
    "dsa_indexer_query_rope_inplace",
    "matmul_allreduce_add_rmsnorm",
    "mla_preprocess",
    "moe_gating_top_k",
]
