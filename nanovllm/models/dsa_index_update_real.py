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
    / "_dsa_index_update_custom"
    / "vendors"
    / "dsa_index_update_custom"
)
if _CUSTOM_OPP_VENDOR.exists():
    _prepend_env_path("ASCEND_CUSTOM_OPP_PATH", _CUSTOM_OPP_VENDOR)

_C = None
_IMPORT_ERROR: Exception | None = None
_EXPECTED_BINDING_VERSION = "manual_acl_tensor_aiv_only_v3_direct_cust_opapi"
try:
    import torch_npu  # type: ignore  # noqa: F401

    _C = importlib.import_module("nanovllm._dsa_index_update_C")
    actual_version = getattr(_C, "binding_version", lambda: "missing")()
    if actual_version != _EXPECTED_BINDING_VERSION:
        raise RuntimeError(
            "dsa_index_update binding version mismatch: "
            f"expected {_EXPECTED_BINDING_VERSION}, got {actual_version}. "
            "Rebuild with `bash scripts/build_dsa_index_update_op.sh`."
        )
except Exception as exc:  # pragma: no cover - depends on Ascend build env.
    _IMPORT_ERROR = exc
    _C = None


def is_available() -> bool:
    return _C is not None


def availability_error() -> Exception | None:
    return _IMPORT_ERROR


def binding_version() -> str | None:
    if _C is None:
        return None
    return _C.binding_version()


def extension_path() -> str | None:
    if _C is None:
        return None
    return getattr(_C, "__file__", None)


def custom_opapi_path() -> str | None:
    if _C is None:
        return None
    return _C.custom_opapi_path()


def dsa_index_update_real(
    score,
    hbm_cached_tokens_pool,
    promote_idx,
    demote_idx,
    copy_counts,
    candidate_lens,
    selected_lens,
    req_pool_entries,
    max_copy_tokens: int,
) -> None:
    if _C is None:
        raise RuntimeError(
            "dsa_index_update real op is not built. Run "
            "`bash scripts/build_dsa_index_update_op.sh` on the Ascend machine first."
        ) from _IMPORT_ERROR
    _C.dsa_index_update(
        score,
        hbm_cached_tokens_pool,
        promote_idx,
        demote_idx,
        copy_counts,
        candidate_lens,
        selected_lens,
        req_pool_entries,
        int(max_copy_tokens),
    )


__all__ = [
    "availability_error",
    "binding_version",
    "custom_opapi_path",
    "dsa_index_update_real",
    "extension_path",
    "is_available",
]
