from __future__ import annotations

from time import perf_counter

import torch
import torch.nn.functional as F

try:
    import torch_npu  # type: ignore
except Exception:  # pragma: no cover - local non-Ascend syntax checks
    torch_npu = None

try:
    import nanovllm.ops as ascend_ops
except Exception:  # pragma: no cover - nanovllm Ascend ops are built on board.
    ascend_ops = None

try:
    from nanovllm.models import dsa_indexer_project_real
except Exception:  # pragma: no cover - standalone AscendC op is built on board.
    dsa_indexer_project_real = None


def _profile_sync(device: torch.device) -> None:
    if device.type == "npu" and torch_npu is not None:
        torch.npu.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize(device)


def _timer_start(detail: dict[str, float] | None, sync: bool, device: torch.device) -> float | None:
    if detail is None:
        return None
    if sync:
        _profile_sync(device)
    return perf_counter()


def _timer_end(detail: dict[str, float] | None, name: str, start: float | None, sync: bool, device: torch.device) -> None:
    if detail is None or start is None:
        return
    if sync:
        _profile_sync(device)
    detail[name] = detail.get(name, 0.0) + perf_counter() - start


def _rotate_half_neox(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


def _cos_sin_2d(cos: torch.Tensor, sin: torch.Tensor, rope_dim: int) -> tuple[torch.Tensor, torch.Tensor]:
    cos = cos.reshape(cos.shape[0], -1)[..., :rope_dim]
    sin = sin.reshape(sin.shape[0], -1)[..., :rope_dim]
    return cos, sin


def _apply_rope_neox_reference(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, rope_dim: int) -> torch.Tensor:
    cos, sin = _cos_sin_2d(cos, sin, rope_dim)
    view_shape = (cos.shape[0],) + (1,) * (x.dim() - 2) + (rope_dim,)
    cos = cos.view(view_shape)
    sin = sin.view(view_shape)
    return (x * cos) + (_rotate_half_neox(x) * sin)


def _can_use_q_bmm(q_c: torch.Tensor, wq_b_bmm_t: torch.Tensor | None, enable_q_bmm: bool) -> bool:
    return (
        enable_q_bmm
        and wq_b_bmm_t is not None
        and ascend_ops is not None
        and hasattr(ascend_ops, "batch_matmul_transpose")
        and q_c.device.type == "npu"
        and q_c.dtype in (torch.float16, torch.bfloat16)
        and 0 < q_c.shape[0] <= 8
    )


def dsa_indexer_project_q_path(q_c: torch.Tensor, wq_b_bmm_t: torch.Tensor | None, enable_q_bmm: bool = False) -> str:
    return "dsa_indexer_project_bmm_transpose" if _can_use_q_bmm(q_c, wq_b_bmm_t, enable_q_bmm) else "dsa_indexer_project_linear"


def dsa_indexer_project_post_available() -> bool:
    return dsa_indexer_project_real is not None and dsa_indexer_project_real.is_available()


def _can_use_post_op(q: torch.Tensor, k: torch.Tensor, weights: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> bool:
    return (
        dsa_indexer_project_post_available()
        and q.device.type == "npu"
        and q.dtype in (torch.float16, torch.bfloat16)
        and k.dtype == q.dtype
        and cos.dtype == q.dtype
        and sin.dtype == q.dtype
        and weights.dtype == torch.float32
        and q.is_contiguous()
        and k.is_contiguous()
        and weights.is_contiguous()
        and cos.is_contiguous()
        and sin.is_contiguous()
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
) -> torch.Tensor:
    if _can_use_q_bmm(q_c, wq_b_bmm_t, enable_q_bmm):
        q_c_by_head = q_c.unsqueeze(1).expand(-1, n_head, -1).contiguous()
        q = torch.empty((q_c.shape[0], n_head, head_dim), dtype=q_c.dtype, device=q_c.device)
        ascend_ops.batch_matmul_transpose(q_c_by_head, wq_b_bmm_t, q)
        return q
    return F.linear(q_c, wq_b_weight).view(-1, n_head, head_dim)


def dsa_indexer_project_torch(
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
    detail: dict[str, float] | None = None,
    sync_detail: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """A/B implementation behind the final DSA indexer_project interface.

    The public interface already takes final output tensors. Internally this
    stage still computes q/k/weights projection with mature framework kernels,
    then uses the AscendC post op to write RoPE/scaled outputs directly into
    q_index_out/index_k_out/index_weights_out when available.
    """
    _check_project_outputs(q_index_out, index_k_out, index_weights_out, num_tokens=int(hidden_states.shape[0]), n_head=int(n_head), head_dim=int(head_dim), dtype=hidden_states.dtype, device=hidden_states.device)
    start = _timer_start(detail, sync_detail, q_c.device)
    q = _q_project(q_c, wq_b_weight, wq_b_bmm_t, int(n_head), int(head_dim), bool(enable_q_bmm))
    _timer_end(detail, "q_proj", start, sync_detail, q_c.device)

    start = _timer_start(detail, sync_detail, hidden_states.device)
    k = F.linear(hidden_states, wk_weight)
    _timer_end(detail, "k_proj", start, sync_detail, hidden_states.device)

    start = _timer_start(detail, sync_detail, k.device)
    k = F.layer_norm(k, (int(head_dim),), k_norm_weight, k_norm_bias, eps=1e-6)
    _timer_end(detail, "k_norm", start, sync_detail, k.device)

    start = _timer_start(detail, sync_detail, hidden_states.device)
    weights = F.linear(hidden_states.float(), weights_proj_weight.float()).contiguous()
    _timer_end(detail, "weights_proj", start, sync_detail, hidden_states.device)

    start = _timer_start(detail, sync_detail, q.device)
    if _can_use_post_op(q, k, weights, cos, sin):
        # B-stage true AscendC sub-op: q/k RoPE + weights scale/cast, writing final outputs in-place.
        dsa_indexer_project_real.dsa_indexer_project_post_real_out(q, k, weights, cos, sin, q_index_out, index_k_out, index_weights_out, float(score_scale), int(rope_dim))
        _timer_end(detail, "rope", start, sync_detail, q.device)
        return q_index_out, index_k_out, index_weights_out

    q_pe, q_nope = torch.split(q, [int(rope_dim), int(head_dim) - int(rope_dim)], dim=-1)
    k_pe, k_nope = torch.split(k, [int(rope_dim), int(head_dim) - int(rope_dim)], dim=-1)
    if q.device.type == "npu" and torch_npu is not None and q.dtype in (torch.float16, torch.bfloat16):
        q_pe = torch_npu.npu_rotary_mul(q_pe.unsqueeze(2), cos, sin).squeeze(2)
        k_pe = torch_npu.npu_rotary_mul(k_pe.unsqueeze(1).unsqueeze(2), cos, sin).squeeze(2).squeeze(1)
    else:
        q_pe = _apply_rope_neox_reference(q_pe, cos, sin, int(rope_dim))
        k_pe = _apply_rope_neox_reference(k_pe.unsqueeze(1), cos, sin, int(rope_dim)).squeeze(1)
    _timer_end(detail, "rope", start, sync_detail, q.device)

    q_index_out.copy_(torch.cat((q_pe, q_nope), dim=-1))
    index_k_out.copy_(torch.cat((k_pe, k_nope), dim=-1))
    index_weights_out.copy_((weights * float(score_scale)).to(hidden_states.dtype))
    return q_index_out, index_k_out, index_weights_out


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
    detail: dict[str, float] | None = None,
    sync_detail: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return dsa_indexer_project_torch(
        hidden_states,
        q_c,
        cos,
        sin,
        wq_b_weight,
        wk_weight,
        k_norm_weight,
        k_norm_bias,
        weights_proj_weight,
        q_index_out,
        index_k_out,
        index_weights_out,
        n_head=n_head,
        head_dim=head_dim,
        rope_dim=rope_dim,
        score_scale=score_scale,
        wq_b_bmm_t=wq_b_bmm_t,
        enable_q_bmm=enable_q_bmm,
        detail=detail,
        sync_detail=sync_detail,
    )


__all__ = [
    "dsa_indexer_project",
    "dsa_indexer_project_post_available",
    "dsa_indexer_project_q_path",
    "dsa_indexer_project_torch",
]
