from __future__ import annotations

import torch
import torch.nn.functional as F

try:
    import torch_npu  # type: ignore
except Exception:  # pragma: no cover - local non-Ascend syntax checks
    torch_npu = None

try:
    import nanovllm.ops as ascend_ops
except Exception as exc:  # pragma: no cover - nanovllm Ascend ops are built on board.
    ascend_ops = None
    ascend_ops_import_error = exc
else:
    ascend_ops_import_error = None

_POST_OPS = ascend_ops
_POST_IMPORT_ERROR = ascend_ops_import_error


def _rotate_half_interleaved(x: torch.Tensor) -> torch.Tensor:
    even = x[..., ::2]
    odd = x[..., 1::2]
    return torch.stack((-odd, even), dim=-1).flatten(-2)


def _cos_sin_2d(cos: torch.Tensor, sin: torch.Tensor, rope_dim: int) -> tuple[torch.Tensor, torch.Tensor]:
    cos = cos.reshape(cos.shape[0], -1)[..., :rope_dim]
    sin = sin.reshape(sin.shape[0], -1)[..., :rope_dim]
    return cos, sin


def _apply_rope_reference(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    rope_dim: int,
) -> torch.Tensor:
    cos, sin = _cos_sin_2d(cos, sin, rope_dim)
    view_shape = (cos.shape[0],) + (1,) * (x.dim() - 2) + (rope_dim,)
    cos = cos.view(view_shape)
    sin = sin.view(view_shape)
    return (x * cos) + (_rotate_half_interleaved(x) * sin)


def _apply_query_rope_like_runtime(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    rope_dim: int,
) -> torch.Tensor:
    if x.device.type == "npu" and torch_npu is not None and x.dtype in (torch.float16, torch.bfloat16):
        return torch_npu.npu_rotary_mul(
            x.unsqueeze(2), cos, sin, "interleave"
        ).squeeze(2)
    return _apply_rope_reference(
        x, cos, sin, int(rope_dim)
    )


def _query_only_weights_proj(
    hidden_states: torch.Tensor,
    weights_proj_weight: torch.Tensor,
    index_weights_out: torch.Tensor,
    score_scale: float,
) -> torch.Tensor:
    index_weights_out.copy_(_query_only_weights_proj_pure(hidden_states, weights_proj_weight, float(score_scale)))
    return index_weights_out


def _query_only_weights_proj_pure(
    hidden_states: torch.Tensor,
    weights_proj_weight: torch.Tensor,
    score_scale: float,
) -> torch.Tensor:
    if weights_proj_weight.dtype == hidden_states.dtype:
        weights = F.linear(hidden_states, weights_proj_weight)
        if float(score_scale) != 1.0:
            weights = weights * float(score_scale)
        return weights
    weights = F.linear(hidden_states.float(), weights_proj_weight.float()).contiguous()
    if float(score_scale) != 1.0:
        weights = weights * float(score_scale)
    return weights.to(hidden_states.dtype)


def _dsa_indexer_project_query_only_pure(
    hidden_states: torch.Tensor,
    q_c: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    wq_b_weight: torch.Tensor,
    weights_proj_weight: torch.Tensor,
    *,
    n_head: int,
    head_dim: int,
    rope_dim: int,
    score_scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    q = F.linear(q_c, wq_b_weight).view(-1, int(n_head), int(head_dim))
    index_weights = _query_only_weights_proj_pure(hidden_states, weights_proj_weight, float(score_scale))
    q_pe, q_nope = torch.split(q, [int(rope_dim), int(head_dim) - int(rope_dim)], dim=-1)
    q_pe = _apply_query_rope_like_runtime(
        q_pe, cos, sin, int(rope_dim)
    )
    return torch.cat((q_pe, q_nope), dim=-1), index_weights


_Q_BMM_MAX_TOKENS = 64  # The measured 128-token path is slower than F.linear.


def _can_use_q_bmm(
    q_c: torch.Tensor,
    wq_b_bmm_t: torch.Tensor | None,
    enable_q_bmm: bool,
) -> bool:
    return (
        enable_q_bmm
        and wq_b_bmm_t is not None
        and ascend_ops is not None
        and hasattr(ascend_ops, "batch_matmul_transpose")
        and q_c.device.type == "npu"
        and q_c.dtype in (torch.float16, torch.bfloat16)
        and q_c.dim() == 2
        and wq_b_bmm_t.dim() == 3
        and q_c.shape[1] == wq_b_bmm_t.shape[1]
        and q_c.is_contiguous()
        and wq_b_bmm_t.is_contiguous()
        and 0 < q_c.shape[0] <= _Q_BMM_MAX_TOKENS
    )


def _check_project_outputs(
    q_index_out: torch.Tensor,
    index_k_out: torch.Tensor,
    index_weights_out: torch.Tensor,
    *,
    num_tokens: int,
    n_head: int,
    head_dim: int,
    dtype: torch.dtype,
    device: torch.device,
) -> None:
    if q_index_out.shape != (num_tokens, n_head, head_dim):
        raise ValueError(f"q_index_out shape must be {(num_tokens, n_head, head_dim)}, got {tuple(q_index_out.shape)}")
    if index_k_out.shape != (num_tokens, head_dim):
        raise ValueError(f"index_k_out shape must be {(num_tokens, head_dim)}, got {tuple(index_k_out.shape)}")
    if index_weights_out.shape != (num_tokens, n_head):
        raise ValueError(f"index_weights_out shape must be {(num_tokens, n_head)}, got {tuple(index_weights_out.shape)}")
    for name, tensor in (("q_index_out", q_index_out), ("index_k_out", index_k_out), ("index_weights_out", index_weights_out)):
        if tensor.dtype != dtype or tensor.device != device:
            raise ValueError(f"{name} must use dtype={dtype} and device={device}, got dtype={tensor.dtype} device={tensor.device}")
        if not tensor.is_contiguous():
            raise ValueError(f"{name} must be contiguous")


def _q_project(
    q_c: torch.Tensor,
    wq_b_weight: torch.Tensor,
    wq_b_bmm_t: torch.Tensor | None,
    n_head: int,
    head_dim: int,
    enable_q_bmm: bool,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    expected_shape = (q_c.shape[0], n_head, head_dim)
    if out is not None:
        if out.shape != expected_shape:
            raise ValueError(
                f"q projection output shape must be {expected_shape}, "
                f"got {tuple(out.shape)}"
            )
        if out.dtype != q_c.dtype or out.device != q_c.device:
            raise ValueError(
                "q projection output must match q_c dtype/device, got "
                f"dtype={out.dtype} device={out.device}"
            )
        if not out.is_contiguous():
            raise ValueError("q projection output must be contiguous")
    if _can_use_q_bmm(q_c, wq_b_bmm_t, enable_q_bmm):
        q = out
        if q is None:
            q = torch.empty(
                expected_shape,
                dtype=q_c.dtype,
                device=q_c.device,
            )
        # batch_matmul_transpose treats a 2-D tensor_a as shared across every
        # index head.  This avoids materializing [tokens, heads, q_lora_rank]
        # on every layer of the stable decode graph.
        ascend_ops.batch_matmul_transpose(q_c, wq_b_bmm_t, q)
        return q
    q = F.linear(q_c, wq_b_weight).view(expected_shape)
    if out is not None:
        out.copy_(q)
        return out
    return q


def _can_use_query_rope_op(
    q: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    rope_dim: int,
) -> bool:
    return (
        _POST_OPS is not None
        and hasattr(_POST_OPS, "dsa_indexer_query_rope_inplace")
        and q.device.type == "npu"
        and q.dtype in (torch.float16, torch.bfloat16)
        and cos.dtype == q.dtype
        and sin.dtype == q.dtype
        and q.is_contiguous()
        and cos.is_contiguous()
        and sin.is_contiguous()
        and 0 < int(rope_dim) <= q.shape[-1]
        and int(rope_dim) % 16 == 0
        and q.shape[-1] % 16 == 0
    )


def _run_query_rope_inplace(
    q: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    rope_dim: int,
) -> None:
    if _POST_OPS is None or not hasattr(
        _POST_OPS, "dsa_indexer_query_rope_inplace"
    ):
        raise RuntimeError(
            "dsa_indexer query RoPE op is not built into nanovllm.ops. Run "
            "`bash scripts/build_nanovllm_ops.sh` on the Ascend machine first."
        ) from _POST_IMPORT_ERROR
    _POST_OPS.dsa_indexer_query_rope_inplace(
        q,
        cos,
        sin,
        int(rope_dim),
    )


def dsa_indexer_project(
    hidden_states: torch.Tensor,
    q_c: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    wq_b_weight: torch.Tensor,
    wk_weight: torch.Tensor,
    k_norm_weight: torch.Tensor,
    k_norm_bias: torch.Tensor | None,
    weights_proj_weight: torch.Tensor,
    q_index_out: torch.Tensor,
    index_k_out: torch.Tensor,
    index_weights_out: torch.Tensor,
    *,
    n_head: int,
    head_dim: int,
    rope_dim: int,
    score_scale: float,
    wq_b_bmm_t: torch.Tensor | None = None,
    enable_q_bmm: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Project Indexer q/k/weights and write the provided output buffers."""
    _check_project_outputs(q_index_out, index_k_out, index_weights_out, num_tokens=int(hidden_states.shape[0]), n_head=int(n_head), head_dim=int(head_dim), dtype=hidden_states.dtype, device=hidden_states.device)
    q = _q_project(q_c, wq_b_weight, wq_b_bmm_t, int(n_head), int(head_dim), bool(enable_q_bmm))

    k = F.linear(hidden_states, wk_weight)

    k = F.layer_norm(k, (int(head_dim),), k_norm_weight, k_norm_bias, eps=1e-6)

    weights = F.linear(hidden_states.float(), weights_proj_weight.float()).contiguous()

    q_pe, q_nope = torch.split(q, [int(rope_dim), int(head_dim) - int(rope_dim)], dim=-1)
    k_pe, k_nope = torch.split(k, [int(rope_dim), int(head_dim) - int(rope_dim)], dim=-1)
    if q.device.type == "npu" and torch_npu is not None and q.dtype in (torch.float16, torch.bfloat16):
        q_pe = torch_npu.npu_rotary_mul(
            q_pe.unsqueeze(2), cos, sin, "interleave"
        ).squeeze(2)
        k_pe = torch_npu.npu_rotary_mul(
            k_pe.unsqueeze(1).unsqueeze(2), cos, sin, "interleave"
        ).squeeze(2).squeeze(1)
    else:
        q_pe = _apply_rope_reference(
            q_pe, cos, sin, int(rope_dim)
        )
        k_pe = _apply_rope_reference(
            k_pe.unsqueeze(1), cos, sin, int(rope_dim)
        ).squeeze(1)
    q_index_out.copy_(torch.cat((q_pe, q_nope), dim=-1))
    index_k_out.copy_(torch.cat((k_pe, k_nope), dim=-1))
    index_weights_out.copy_((weights * float(score_scale)).to(hidden_states.dtype))
    return q_index_out, index_k_out, index_weights_out


def dsa_indexer_project_query_only(
    hidden_states: torch.Tensor,
    q_c: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    wq_b_weight: torch.Tensor,
    weights_proj_weight: torch.Tensor,
    q_index_out: torch.Tensor,
    index_weights_out: torch.Tensor,
    *,
    n_head: int,
    head_dim: int,
    rope_dim: int,
    score_scale: float,
    wq_b_bmm_t: torch.Tensor | None = None,
    enable_q_bmm: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Decode-only indexer projection.
    Decode DSA update scores only need the current token's query index and per-head weights. The current decode token's index key is not part of the prefill-candidate IndexCache, so this path skips wk/k_norm/k-rope entirely.
    """

    if q_index_out.shape != (hidden_states.shape[0], n_head, head_dim):
        raise ValueError(f"q_index_out shape must be {(hidden_states.shape[0], n_head, head_dim)}, got {tuple(q_index_out.shape)}")
    if index_weights_out.shape != (hidden_states.shape[0], n_head):
        raise ValueError(f"index_weights_out shape must be {(hidden_states.shape[0], n_head)}, got {tuple(index_weights_out.shape)}")

    # Stable decode writes the shared-A BMM result directly into the
    # persistent query buffer; the non-RoPE suffix needs no slice/copy.
    q = _q_project(
        q_c,
        wq_b_weight,
        wq_b_bmm_t,
        int(n_head),
        int(head_dim),
        bool(enable_q_bmm),
        out=q_index_out,
    )

    _query_only_weights_proj(hidden_states, weights_proj_weight, index_weights_out, float(score_scale))

    if _can_use_query_rope_op(q, cos, sin, int(rope_dim)):
        _run_query_rope_inplace(
            q,
            cos,
            sin,
            int(rope_dim),
        )
        return q_index_out, index_weights_out

    q_pe = q[..., : int(rope_dim)]
    q_pe = _apply_query_rope_like_runtime(
        q_pe, cos, sin, int(rope_dim)
    )
    q_index_out[..., : int(rope_dim)].copy_(q_pe)
    if q is not q_index_out:
        q_index_out[..., int(rope_dim) :].copy_(q[..., int(rope_dim) :])
    return q_index_out, index_weights_out


__all__ = [
    "dsa_indexer_project",
    "dsa_indexer_project_query_only",
]
