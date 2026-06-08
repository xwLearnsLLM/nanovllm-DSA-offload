from __future__ import annotations

from time import perf_counter
from typing import Any, Callable

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

_GRAPH_LIGHTNING_INDEXER = None
_GRAPH_GATHER_SELECTION_KV_CACHE = None
_GRAPH_CUSTOM_OP_ERROR: Exception | None = None
if ascend_ops is not None:
    try:
        graph_lightning = torch.ops.nanovllm_dsa.lightning_indexer.default
        graph_schema = str(graph_lightning._schema)
        if "-> (Tensor, Tensor)" not in graph_schema:
            raise RuntimeError(f"torch.ops.nanovllm_dsa.lightning_indexer schema is stale: {graph_schema}; rebuild nanovllm ops.")
        _GRAPH_LIGHTNING_INDEXER = graph_lightning
        graph_gather = torch.ops.nanovllm_dsa.gather_selection_kv_cache.default
        gather_schema = str(graph_gather._schema)
        if "-> (Tensor, Tensor, Tensor, Tensor)" not in gather_schema:
            raise RuntimeError(f"torch.ops.nanovllm_dsa.gather_selection_kv_cache schema is stale: {gather_schema}; rebuild nanovllm ops.")
        _GRAPH_GATHER_SELECTION_KV_CACHE = graph_gather
    except Exception as exc:
        _GRAPH_CUSTOM_OP_ERROR = exc
        _GRAPH_LIGHTNING_INDEXER = None
        _GRAPH_GATHER_SELECTION_KV_CACHE = None

try:
    import torchair  # type: ignore
except Exception:  # pragma: no cover - TorchAir is optional.
    torchair = None


def _register_nanovllm_dsa_torchair_converters() -> None:
    if torchair is None or _GRAPH_LIGHTNING_INDEXER is None or _GRAPH_GATHER_SELECTION_KV_CACHE is None:
        return
    try:
        from torchair._ge_concrete_graph.fx2ge_converter import register_fx_node_ge_converter  # type: ignore
        from torchair.ge import attr  # type: ignore
    except Exception:
        return

    @register_fx_node_ge_converter(torch.ops.nanovllm_dsa.lightning_indexer.default)
    def convert_nanovllm_lightning_indexer(
        query,
        key,
        weights,
        actual_seq_lengths_query,
        actual_seq_lengths_key,
        block_table,
        layout_query: str,
        layout_key: str,
        sparse_count: int,
        sparse_mode: int,
        pre_tokens: int,
        next_tokens: int,
        return_value: bool,
        meta_outputs: Any = None,
    ):
        sparse_indices = torchair.ge.custom_op(
            "LightningIndexerVllm",
            inputs={
                "query": query,
                "key": key,
                "weights": weights,
                "actual_seq_lengths_query": actual_seq_lengths_query,
                "actual_seq_lengths_key": actual_seq_lengths_key,
                "block_table": block_table,
            },
            attrs={
                "layout_query": attr.Str(layout_query),
                "layout_key": attr.Str(layout_key),
                "sparse_count": attr.Int(sparse_count),
                "sparse_mode": attr.Int(sparse_mode),
            },
            outputs=["sparse_indices"],
        )
        return sparse_indices, sparse_indices

    @register_fx_node_ge_converter(torch.ops.nanovllm_dsa.gather_selection_kv_cache.default)
    def convert_nanovllm_gather_selection_kv_cache(
        selection_k_rope,
        selection_kv_cache,
        selection_kv_block_table,
        selection_kv_block_status,
        req_pool_entries,
        selection_topk_indices,
        full_k_rope,
        full_kv_cache,
        full_kv_block_table,
        full_kv_actual_seq,
        meta_outputs: Any = None,
    ):
        # The CANN kernel writes selection_k_rope/selection_kv_cache/status in place.
        # Keep original inputs here; TensorMove would hide that side effect.
        return torchair.ge.custom_op(
            "GatherSelectionKvCache",
            inputs={
                "selection_k_rope": selection_k_rope,
                "selection_kv_cache": selection_kv_cache,
                "selection_kv_block_table": selection_kv_block_table,
                "selection_kv_block_status": selection_kv_block_status,
                "req_pool_entries": req_pool_entries,
                "selection_topk_indices": selection_topk_indices,
                "full_k_rope": full_k_rope,
                "full_kv_cache": full_kv_cache,
                "full_kv_block_table": full_kv_block_table,
                "full_kv_actual_seq": full_kv_actual_seq,
            },
            attrs={},
            outputs=["selection_k_rope", "selection_kv_cache", "selection_kv_block_table", "selection_kv_block_status"],
        )


_register_nanovllm_dsa_torchair_converters()

_POST_OPS = None
_POST_IMPORT_ERROR: Exception | None = None
_EXPECTED_POST_BINDING_VERSION = "dsa_indexer_project_post_csrc_v1"
if ascend_ops is None:
    _POST_IMPORT_ERROR = ascend_ops_import_error
else:
    try:
        actual_version = getattr(ascend_ops, "dsa_indexer_project_binding_version", lambda: "missing")()
        if actual_version != _EXPECTED_POST_BINDING_VERSION:
            raise RuntimeError(
                "dsa_indexer_project binding version mismatch: "
                f"expected {_EXPECTED_POST_BINDING_VERSION}, got {actual_version}. "
                "Rebuild with `bash scripts/build_nanovllm_ops.sh`."
            )
        _POST_OPS = ascend_ops
    except Exception as exc:  # pragma: no cover - depends on Ascend build env.
        _POST_IMPORT_ERROR = exc
        _POST_OPS = None


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


def _apply_query_rope_like_runtime(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, rope_dim: int) -> torch.Tensor:
    # The graph path must match runtime RoPE before lightning_indexer consumes q_index.
    if x.device.type == "npu" and torch_npu is not None and x.dtype in (torch.float16, torch.bfloat16):
        return torch_npu.npu_rotary_mul(x.unsqueeze(2), cos, sin).squeeze(2)
    return _apply_rope_neox_reference(x, cos, sin, int(rope_dim))


def _rms_norm_reference(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    orig_dtype = x.dtype
    x_float = x.float()
    var = x_float.pow(2).mean(dim=-1, keepdim=True)
    return (x_float * torch.rsqrt(var + float(eps))).to(orig_dtype) * weight


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
    q_pe = _apply_query_rope_like_runtime(q_pe, cos, sin, int(rope_dim))
    return torch.cat((q_pe, q_nope), dim=-1), index_weights


def _dsa_indexer_project_query_only_with_qc_pure(
    hidden_states: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    q_a_weight: torch.Tensor,
    q_norm_weight: torch.Tensor,
    wq_b_weight: torch.Tensor,
    weights_proj_weight: torch.Tensor,
    *,
    q_norm_eps: float,
    n_head: int,
    head_dim: int,
    rope_dim: int,
    score_scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    q_c = F.linear(hidden_states, q_a_weight)
    q_c = _rms_norm_reference(q_c, q_norm_weight, float(q_norm_eps))
    return _dsa_indexer_project_query_only_pure(
        hidden_states,
        q_c,
        cos,
        sin,
        wq_b_weight,
        weights_proj_weight,
        n_head=int(n_head),
        head_dim=int(head_dim),
        rope_dim=int(rope_dim),
        score_scale=float(score_scale),
    )


def _dsa_indexer_pipeline_with_qc_functional(
    hidden_states: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    q_a_weight: torch.Tensor,
    q_norm_weight: torch.Tensor,
    wq_b_weight: torch.Tensor,
    weights_proj_weight: torch.Tensor,
    q_index_out: torch.Tensor,
    index_weights_out: torch.Tensor,
    index_cache: torch.Tensor,
    candidate_query_lens: torch.Tensor,
    candidate_lens: torch.Tensor,
    index_tables: torch.Tensor,
    selection_kpe: torch.Tensor,
    selection_ckv: torch.Tensor,
    selection_block_table: torch.Tensor,
    gather_selection_status: torch.Tensor,
    req_pool_entries: torch.Tensor,
    full_kpe: torch.Tensor,
    full_ckv: torch.Tensor,
    dram_tables: torch.Tensor,
    *,
    q_norm_eps: float,
    n_head: int,
    head_dim: int,
    rope_dim: int,
    score_scale: float,
    sparse_count: int,
    return_gather_outputs: bool = False,
) -> tuple[torch.Tensor, ...]:
    if ascend_ops is None:
        raise RuntimeError("nanovllm Ascend ops are unavailable; rebuild with `bash scripts/build_nanovllm_ops.sh`.") from ascend_ops_import_error
    q_index, index_weights = _dsa_indexer_project_query_only_with_qc_pure(
        hidden_states,
        cos,
        sin,
        q_a_weight,
        q_norm_weight,
        wq_b_weight,
        weights_proj_weight,
        q_norm_eps=float(q_norm_eps),
        n_head=int(n_head),
        head_dim=int(head_dim),
        rope_dim=int(rope_dim),
        score_scale=float(score_scale),
    )
    if _GRAPH_LIGHTNING_INDEXER is None or _GRAPH_GATHER_SELECTION_KV_CACHE is None:
        topk_indices = ascend_ops.npu_lightning_indexer(
            query=q_index,
            key=index_cache,
            weights=index_weights,
            actual_seq_lengths_query=candidate_query_lens,
            actual_seq_lengths_key=candidate_lens,
            block_table=index_tables,
            layout_query="TND",
            layout_key="PA_BSND",
            sparse_count=int(sparse_count),
            sparse_mode=3,
        )
        ascend_ops.npu_gather_selection_kv_cache(
            selection_kpe,
            selection_ckv,
            selection_block_table,
            gather_selection_status,
            req_pool_entries,
            topk_indices.view(q_index.shape[0], 1, 1, int(sparse_count)),
            full_kpe,
            full_ckv,
            dram_tables,
            candidate_lens,
        )
        if return_gather_outputs:
            return q_index, index_weights, topk_indices, selection_kpe, selection_ckv, selection_block_table, gather_selection_status
        return q_index, index_weights, topk_indices

    # candidate_query_lens is cumulative for the normal TND decode path. The
    # TorchAir mini-pipeline uses BSND query=[B, 1, N, D], so each row has S=1.
    bsnd_query_lens = torch.ones_like(candidate_query_lens)
    topk_indices = _GRAPH_LIGHTNING_INDEXER(
        q_index.unsqueeze(1),          # BSND avoids a GE Reshape after LightningIndexer.
        index_cache,
        index_weights.unsqueeze(1),
        bsnd_query_lens,
        candidate_lens,
        index_tables,
        "BSND",
        "PA_BSND",
        int(sparse_count),
        3,
        (1 << 63) - 1,
        (1 << 63) - 1,
        False,
    )[0]
    selection_kpe_out, selection_ckv_out, selection_block_table_out, gather_selection_status_out = _GRAPH_GATHER_SELECTION_KV_CACHE(
        selection_kpe,
        selection_ckv,
        selection_block_table,
        gather_selection_status,
        req_pool_entries,
        topk_indices,
        full_kpe,
        full_ckv,
        dram_tables,
        candidate_lens,
    )
    if return_gather_outputs:
        return q_index, index_weights, topk_indices, selection_kpe_out, selection_ckv_out, selection_block_table_out, gather_selection_status_out
    return q_index, index_weights, topk_indices


class DsaPipelineTorchAirCache:
    def __init__(self) -> None:
        self._compiled: dict[tuple, Callable[..., object]] = {}
        self._disabled: dict[tuple, str] = {}

    def disable(self, key: tuple, reason: str) -> None:
        self._disabled[key] = reason
        self._compiled.pop(key, None)

    def _compile_lazy_or_raise(self, key: tuple, fn, device: torch.device) -> Callable[..., object]:
        if key in self._compiled:
            return self._compiled[key]
        if key in self._disabled:
            raise RuntimeError(f"TorchAir graph is disabled for key={key}: {self._disabled[key]}")
        if torchair is None:
            self._disabled[key] = "torchair_unavailable"
            raise RuntimeError("TorchAir is unavailable; set NANOVLLM_DSA_QUERY_ONLY_BACKEND=current or install TorchAir.")
        if device.type != "npu":
            self._disabled[key] = "non_npu_device"
            raise RuntimeError(f"TorchAir DSA pipeline requires NPU tensors, got device={device}.")
        try:
            config = torchair.CompilerConfig()
            compiled = torch.compile(fn, backend=torchair.get_npu_backend(compiler_config=config), dynamic=False)
            self._compiled[key] = compiled
            return compiled
        except Exception as exc:  # pragma: no cover - depends on Ascend/TorchAir runtime.
            reason = f"compile_failed:{type(exc).__name__}"
            self._disabled[key] = reason
            raise RuntimeError(f"TorchAir DSA pipeline compile failed: {reason}") from exc

    def compile_dsa_pipeline_with_qc(
        self,
        hidden_states: torch.Tensor,
        q_a_weight: torch.Tensor,
        index_cache: torch.Tensor,
        selection_kpe: torch.Tensor,
        selection_ckv: torch.Tensor,
        selection_block_table: torch.Tensor,
        gather_selection_status: torch.Tensor,
        full_kpe: torch.Tensor,
        full_ckv: torch.Tensor,
        dram_tables: torch.Tensor,
        *,
        q_norm_eps: float,
        n_head: int,
        head_dim: int,
        rope_dim: int,
        score_scale: float,
        sparse_count: int,
    ) -> tuple[Callable[..., object], tuple]:
        key = (
            "dsa_pipeline_with_qc",
            int(hidden_states.shape[0]),
            int(hidden_states.shape[1]),
            int(q_a_weight.shape[0]),
            str(hidden_states.dtype),
            str(hidden_states.device),
            int(n_head),
            int(head_dim),
            int(rope_dim),
            float(q_norm_eps),
            float(score_scale),
            int(sparse_count),
            tuple(index_cache.shape),
            tuple(selection_kpe.shape),
            tuple(selection_ckv.shape),
            int(selection_block_table.shape[1]),
            int(gather_selection_status.shape[-1]),
            tuple(full_kpe.shape),
            tuple(full_ckv.shape),
            int(dram_tables.shape[1]),
        )

        def fn(
            hidden_states_arg,
            cos_arg,
            sin_arg,
            q_a_weight_arg,
            q_norm_weight_arg,
            wq_b_weight_arg,
            weights_proj_weight_arg,
            q_index_out_arg,
            index_weights_out_arg,
            index_cache_arg,
            candidate_query_lens_arg,
            candidate_lens_arg,
            index_tables_arg,
            selection_kpe_arg,
            selection_ckv_arg,
            selection_block_table_arg,
            gather_selection_status_arg,
            req_pool_entries_arg,
            full_kpe_arg,
            full_ckv_arg,
            dram_tables_arg,
        ):
            return _dsa_indexer_pipeline_with_qc_functional(
                hidden_states_arg,
                cos_arg,
                sin_arg,
                q_a_weight_arg,
                q_norm_weight_arg,
                wq_b_weight_arg,
                weights_proj_weight_arg,
                q_index_out_arg,
                index_weights_out_arg,
                index_cache_arg,
                candidate_query_lens_arg,
                candidate_lens_arg,
                index_tables_arg,
                selection_kpe_arg,
                selection_ckv_arg,
                selection_block_table_arg,
                gather_selection_status_arg,
                req_pool_entries_arg,
                full_kpe_arg,
                full_ckv_arg,
                dram_tables_arg,
                q_norm_eps=float(q_norm_eps),
                n_head=int(n_head),
                head_dim=int(head_dim),
                rope_dim=int(rope_dim),
                score_scale=float(score_scale),
                sparse_count=int(sparse_count),
                return_gather_outputs=True,
            )

        return self._compile_lazy_or_raise(key, fn, hidden_states.device), key


_DSA_PIPELINE_TORCHAIR_CACHE = DsaPipelineTorchAirCache()
_Q_BMM_MAX_TOKENS = 64  # Keep larger decode batches on the q BMM path; tokens=128 was slower in probe.


def _can_use_q_bmm(q_c: torch.Tensor, wq_b_bmm_t: torch.Tensor | None, enable_q_bmm: bool) -> bool:
    return (
        enable_q_bmm
        and wq_b_bmm_t is not None
        and ascend_ops is not None
        and hasattr(ascend_ops, "batch_matmul_transpose")
        and q_c.device.type == "npu"
        and q_c.dtype in (torch.float16, torch.bfloat16)
        and 0 < q_c.shape[0] <= _Q_BMM_MAX_TOKENS
    )


def dsa_indexer_project_q_path(q_c: torch.Tensor, wq_b_bmm_t: torch.Tensor | None, enable_q_bmm: bool = False) -> str:
    return "dsa_indexer_project_bmm_transpose" if _can_use_q_bmm(q_c, wq_b_bmm_t, enable_q_bmm) else "dsa_indexer_project_linear"


def dsa_indexer_project_post_available() -> bool:
    return _POST_OPS is not None


def dsa_indexer_project_post_availability_error() -> Exception | None:
    return _POST_IMPORT_ERROR


def dsa_indexer_project_post_binding_version() -> str | None:
    if _POST_OPS is None:
        return None
    return _POST_OPS.dsa_indexer_project_binding_version()


def dsa_indexer_project_post_extension_path() -> str | None:
    if _POST_OPS is None:
        return None
    return getattr(_POST_OPS, "__file__", None)


def dsa_indexer_project_post_real(q_in, k_in, weights_in, cos, sin, score_scale: float, rope_dim: int):
    if _POST_OPS is None:
        raise RuntimeError(
            "dsa_indexer_project post op is not built into nanovllm.ops. Run "
            "`bash scripts/build_nanovllm_ops.sh` on the Ascend machine first."
        ) from _POST_IMPORT_ERROR
    return _POST_OPS.dsa_indexer_project_post(q_in, k_in, weights_in, cos, sin, float(score_scale), int(rope_dim))


def dsa_indexer_project_post_real_out(q_in, k_in, weights_in, cos, sin, q_out, k_out, weights_out, score_scale: float, rope_dim: int):
    if _POST_OPS is None:
        raise RuntimeError(
            "dsa_indexer_project post op is not built into nanovllm.ops. Run "
            "`bash scripts/build_nanovllm_ops.sh` on the Ascend machine first."
        ) from _POST_IMPORT_ERROR
    _POST_OPS.dsa_indexer_project_post_out(q_in, k_in, weights_in, cos, sin, q_out, k_out, weights_out, float(score_scale), int(rope_dim))
    return q_out, k_out, weights_out


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
    then uses the AscendC post op to write RoPE/casted outputs directly into
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
        # B-stage true AscendC sub-op: q/k RoPE + weights cast, writing final outputs in-place.
        # DeepSeek-V3.2 BF16 SFA in vllm-ascend feeds raw weights_proj(x) to
        # lightning_indexer, so callers pass score_scale=1.0 here.
        dsa_indexer_project_post_real_out(q, k, weights, cos, sin, q_index_out, index_k_out, index_weights_out, float(score_scale), int(rope_dim))
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
    detail: dict[str, float] | None = None,
    sync_detail: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Decode-only indexer projection.
    Decode DSA update scores only need the current token's query index and per-head weights. The current decode token's index key is not part of the prefill-candidate IndexCache, so this path skips wk/k_norm/k-rope entirely.
    """

    if q_index_out.shape != (hidden_states.shape[0], n_head, head_dim):
        raise ValueError(f"q_index_out shape must be {(hidden_states.shape[0], n_head, head_dim)}, got {tuple(q_index_out.shape)}")
    if index_weights_out.shape != (hidden_states.shape[0], n_head):
        raise ValueError(f"index_weights_out shape must be {(hidden_states.shape[0], n_head)}, got {tuple(index_weights_out.shape)}")

    start = _timer_start(detail, sync_detail, q_c.device)
    q = _q_project(q_c, wq_b_weight, wq_b_bmm_t, int(n_head), int(head_dim), bool(enable_q_bmm))
    _timer_end(detail, "q_proj", start, sync_detail, q_c.device)

    start = _timer_start(detail, sync_detail, hidden_states.device)
    _query_only_weights_proj(hidden_states, weights_proj_weight, index_weights_out, float(score_scale))
    _timer_end(detail, "weights_proj", start, sync_detail, hidden_states.device)

    start = _timer_start(detail, sync_detail, q.device)
    q_pe, q_nope = torch.split(q, [int(rope_dim), int(head_dim) - int(rope_dim)], dim=-1)
    q_pe = _apply_query_rope_like_runtime(q_pe, cos, sin, int(rope_dim))
    q_index_out[..., : int(rope_dim)].copy_(q_pe)
    q_index_out[..., int(rope_dim) :].copy_(q_nope)
    _timer_end(detail, "rope", start, sync_detail, q.device)
    return q_index_out, index_weights_out


def dsa_indexer_pipeline_with_qc_eager(
    hidden_states: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    q_a_weight: torch.Tensor,
    q_norm_weight: torch.Tensor,
    wq_b_weight: torch.Tensor,
    weights_proj_weight: torch.Tensor,
    q_index_out: torch.Tensor,
    index_weights_out: torch.Tensor,
    index_cache: torch.Tensor,
    candidate_query_lens: torch.Tensor,
    candidate_lens: torch.Tensor,
    index_tables: torch.Tensor,
    selection_kpe: torch.Tensor,
    selection_ckv: torch.Tensor,
    selection_block_table: torch.Tensor,
    gather_selection_status: torch.Tensor,
    req_pool_entries: torch.Tensor,
    full_kpe: torch.Tensor,
    full_ckv: torch.Tensor,
    dram_tables: torch.Tensor,
    *,
    q_norm_eps: float,
    n_head: int,
    head_dim: int,
    rope_dim: int,
    score_scale: float,
    sparse_count: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return _dsa_indexer_pipeline_with_qc_functional(
        hidden_states,
        cos,
        sin,
        q_a_weight,
        q_norm_weight,
        wq_b_weight,
        weights_proj_weight,
        q_index_out,
        index_weights_out,
        index_cache,
        candidate_query_lens,
        candidate_lens,
        index_tables,
        selection_kpe,
        selection_ckv,
        selection_block_table,
        gather_selection_status,
        req_pool_entries,
        full_kpe,
        full_ckv,
        dram_tables,
        q_norm_eps=float(q_norm_eps),
        n_head=int(n_head),
        head_dim=int(head_dim),
        rope_dim=int(rope_dim),
        score_scale=float(score_scale),
        sparse_count=int(sparse_count),
    )


def dsa_indexer_pipeline_with_qc_torchair(
    hidden_states: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    q_a_weight: torch.Tensor,
    q_norm_weight: torch.Tensor,
    wq_b_weight: torch.Tensor,
    weights_proj_weight: torch.Tensor,
    q_index_out: torch.Tensor,
    index_weights_out: torch.Tensor,
    index_cache: torch.Tensor,
    candidate_query_lens: torch.Tensor,
    candidate_lens: torch.Tensor,
    index_tables: torch.Tensor,
    selection_kpe: torch.Tensor,
    selection_ckv: torch.Tensor,
    selection_block_table: torch.Tensor,
    gather_selection_status: torch.Tensor,
    req_pool_entries: torch.Tensor,
    full_kpe: torch.Tensor,
    full_ckv: torch.Tensor,
    dram_tables: torch.Tensor,
    *,
    q_norm_eps: float,
    n_head: int,
    head_dim: int,
    rope_dim: int,
    score_scale: float,
    sparse_count: int,
    detail: dict[str, float] | None = None,
    sync_detail: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if q_index_out.shape != (hidden_states.shape[0], n_head, head_dim):
        raise ValueError(f"q_index_out shape must be {(hidden_states.shape[0], n_head, head_dim)}, got {tuple(q_index_out.shape)}")
    if index_weights_out.shape != (hidden_states.shape[0], n_head):
        raise ValueError(f"index_weights_out shape must be {(hidden_states.shape[0], n_head)}, got {tuple(index_weights_out.shape)}")
    if _GRAPH_LIGHTNING_INDEXER is None or _GRAPH_GATHER_SELECTION_KV_CACHE is None:
        raise RuntimeError("TorchAir DSA pipeline requires C++ torch.ops.nanovllm_dsa lightning_indexer/gather_selection registrations. Rebuild with `bash scripts/build_nanovllm_ops.sh`.") from _GRAPH_CUSTOM_OP_ERROR

    start = _timer_start(detail, sync_detail, hidden_states.device)
    compiled, _ = _DSA_PIPELINE_TORCHAIR_CACHE.compile_dsa_pipeline_with_qc(
        hidden_states,
        q_a_weight,
        index_cache,
        selection_kpe,
        selection_ckv,
        selection_block_table,
        gather_selection_status,
        full_kpe,
        full_ckv,
        dram_tables,
        q_norm_eps=float(q_norm_eps),
        n_head=int(n_head),
        head_dim=int(head_dim),
        rope_dim=int(rope_dim),
        score_scale=float(score_scale),
        sparse_count=int(sparse_count),
    )
    try:
        with torch.inference_mode():
            (
                q_index_out,
                index_weights_out,
                topk_indices,
                _selection_kpe_out,
                _selection_ckv_out,
                _selection_block_table_out,
                _gather_selection_status_out,
            ) = compiled(
                hidden_states,
                cos,
                sin,
                q_a_weight,
                q_norm_weight,
                wq_b_weight,
                weights_proj_weight,
                q_index_out,
                index_weights_out,
                index_cache,
                candidate_query_lens,
                candidate_lens,
                index_tables,
                selection_kpe,
                selection_ckv,
                selection_block_table,
                gather_selection_status,
                req_pool_entries,
                full_kpe,
                full_ckv,
                dram_tables,
            )
        _timer_end(detail, "dsa_pipeline", start, sync_detail, hidden_states.device)
        return q_index_out, index_weights_out, topk_indices
    except Exception as exc:  # pragma: no cover - depends on Ascend/TorchAir runtime.
        key = (
            "dsa_pipeline_with_qc",
            int(hidden_states.shape[0]),
            int(hidden_states.shape[1]),
            int(q_a_weight.shape[0]),
            str(hidden_states.dtype),
            str(hidden_states.device),
            int(n_head),
            int(head_dim),
            int(rope_dim),
            float(q_norm_eps),
            float(score_scale),
            int(sparse_count),
            tuple(index_cache.shape),
            tuple(selection_kpe.shape),
            tuple(selection_ckv.shape),
            int(selection_block_table.shape[1]),
            int(gather_selection_status.shape[-1]),
            tuple(full_kpe.shape),
            tuple(full_ckv.shape),
            int(dram_tables.shape[1]),
        )
        _DSA_PIPELINE_TORCHAIR_CACHE.disable(key, f"run_failed:{type(exc).__name__}")
        _timer_end(detail, "dsa_pipeline", start, sync_detail, hidden_states.device)
        raise RuntimeError("TorchAir DSA pipeline run failed; no fallback is used for this path.") from exc


__all__ = [
    "dsa_indexer_project",
    "dsa_indexer_project_post_availability_error",
    "dsa_indexer_project_post_available",
    "dsa_indexer_project_post_binding_version",
    "dsa_indexer_project_post_extension_path",
    "dsa_indexer_project_post_real",
    "dsa_indexer_project_post_real_out",
    "dsa_indexer_project_q_path",
    "dsa_indexer_project_query_only",
    "dsa_indexer_pipeline_with_qc_eager",
    "dsa_indexer_pipeline_with_qc_torchair",
    "dsa_indexer_project_torch",
]
