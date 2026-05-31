from __future__ import annotations


_OPS = None
_IMPORT_ERROR: Exception | None = None
_EXPECTED_BINDING_VERSION = "dsa_indexer_project_post_csrc_v1"
try:
    import torch_npu  # type: ignore  # noqa: F401
    import nanovllm.ops as _ascend_ops

    actual_version = getattr(
        _ascend_ops,
        "dsa_indexer_project_binding_version",
        lambda: "missing",
    )()
    if actual_version != _EXPECTED_BINDING_VERSION:
        raise RuntimeError(
            "dsa_indexer_project binding version mismatch: "
            f"expected {_EXPECTED_BINDING_VERSION}, got {actual_version}. "
            "Rebuild with `bash scripts/build_nanovllm_ops.sh`."
        )
    _OPS = _ascend_ops
except Exception as exc:  # pragma: no cover - depends on Ascend build env.
    _IMPORT_ERROR = exc
    _OPS = None


def is_available() -> bool:
    return _OPS is not None


def availability_error() -> Exception | None:
    return _IMPORT_ERROR


def binding_version() -> str | None:
    if _OPS is None:
        return None
    return _OPS.dsa_indexer_project_binding_version()


def extension_path() -> str | None:
    if _OPS is None:
        return None
    return getattr(_OPS, "__file__", None)


def dsa_indexer_project_post_real(q_in, k_in, weights_in, cos, sin, score_scale: float, rope_dim: int):
    if _OPS is None:
        raise RuntimeError(
            "dsa_indexer_project real op is not built into nanovllm.ops. Run "
            "`bash scripts/build_nanovllm_ops.sh` on the Ascend machine first."
        ) from _IMPORT_ERROR
    return _OPS.dsa_indexer_project_post(q_in, k_in, weights_in, cos, sin, float(score_scale), int(rope_dim))


def dsa_indexer_project_post_real_out(q_in, k_in, weights_in, cos, sin, q_out, k_out, weights_out, score_scale: float, rope_dim: int):
    if _OPS is None:
        raise RuntimeError(
            "dsa_indexer_project real op is not built into nanovllm.ops. Run "
            "`bash scripts/build_nanovllm_ops.sh` on the Ascend machine first."
        ) from _IMPORT_ERROR
    _OPS.dsa_indexer_project_post_out(q_in, k_in, weights_in, cos, sin, q_out, k_out, weights_out, float(score_scale), int(rope_dim))
    return q_out, k_out, weights_out


__all__ = [
    "availability_error",
    "binding_version",
    "dsa_indexer_project_post_real",
    "dsa_indexer_project_post_real_out",
    "extension_path",
    "is_available",
]
