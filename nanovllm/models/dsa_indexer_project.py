from __future__ import annotations

from typing import Any

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
_GRAPH_LIDU_DECODE_UPDATE = None
_GRAPH_SCATTER_COPY = None
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
        if (
            "-> (Tensor, Tensor, Tensor, Tensor)" not in gather_schema
            or gather_schema.count("!") < 3
        ):
            raise RuntimeError(f"torch.ops.nanovllm_dsa.gather_selection_kv_cache schema is stale: {gather_schema}; rebuild nanovllm ops.")
        _GRAPH_GATHER_SELECTION_KV_CACHE = graph_gather
        graph_lidu = torch.ops.nanovllm_dsa.lidu_decode_update.default
        if "Tensor(a!)" not in str(graph_lidu._schema):
            raise RuntimeError(
                "torch.ops.nanovllm_dsa.lidu_decode_update schema is stale; "
                "rebuild nanovllm ops."
            )
        _GRAPH_LIDU_DECODE_UPDATE = graph_lidu
        graph_scatter = torch.ops.nanovllm_dsa.scatter_copy.default
        if str(graph_scatter._schema).count("!") < 4:
            raise RuntimeError(
                "torch.ops.nanovllm_dsa.scatter_copy schema is stale; "
                "rebuild nanovllm ops."
            )
        _GRAPH_SCATTER_COPY = graph_scatter
    except Exception as exc:
        _GRAPH_CUSTOM_OP_ERROR = exc
        _GRAPH_LIGHTNING_INDEXER = None
        _GRAPH_GATHER_SELECTION_KV_CACHE = None
        _GRAPH_LIDU_DECODE_UPDATE = None
        _GRAPH_SCATTER_COPY = None

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

    @register_fx_node_ge_converter(torch.ops.nanovllm_dsa.lidu_decode_update.default)
    def convert_nanovllm_lidu_decode_update(
        query,
        key,
        weights,
        req_pool_entries,
        cache_slots,
        cache_tokens,
        candidate_lens,
        block_table,
        meta_outputs: Any = None,
    ):
        return torchair.ge.custom_op(
            "NanovllmLiduDecodeUpdate",
            inputs={
                "query": query,
                "key": key,
                "weights": weights,
                "req_pool_entries": req_pool_entries,
                "cache_slots": cache_slots,
                "cache_tokens": cache_tokens,
                "actual_seq_lengths_key": candidate_lens,
                "block_table": block_table,
            },
            attrs={},
            outputs=[
                "topk_index",
                "topk_slots",
                "miss_count",
                "cache_slots",
            ],
        )

    @register_fx_node_ge_converter(torch.ops.nanovllm_dsa.scatter_copy.default)
    def convert_nanovllm_scatter_copy(
        hbm_k_rope,
        hbm_kv_cache,
        dram_k_rope,
        dram_kv_cache,
        hbm_block_table,
        dram_block_table,
        source_token_ids,
        destination_slots,
        copy_counts,
        meta_outputs: Any = None,
    ):
        return torchair.ge.custom_op(
            "NanovllmKvcacheScatterCopy",
            inputs={
                "hbm_k_rope": hbm_k_rope,
                "hbm_kv_cache": hbm_kv_cache,
                "dram_k_rope": dram_k_rope,
                "dram_kv_cache": dram_kv_cache,
                "hbm_block_table": hbm_block_table,
                "dram_block_table": dram_block_table,
                "src_token_ids": source_token_ids,
                "dst_slots": destination_slots,
                "copy_counts": copy_counts,
            },
            attrs={},
            outputs=["hbm_k_rope", "hbm_kv_cache"],
        )


_register_nanovllm_dsa_torchair_converters()

_POST_OPS = None
_POST_IMPORT_ERROR: Exception | None = None
_EXPECTED_POST_BINDING_VERSION = "dsa_indexer_project_post_csrc_v2"
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


def _rotate_half_neox(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


def _rotate_half_interleaved(x: torch.Tensor) -> torch.Tensor:
    even = x[..., ::2]
    odd = x[..., 1::2]
    return torch.stack((-odd, even), dim=-1).flatten(-2)


def _normalize_rotary_mode(rotary_mode: str) -> str:
    rotary_mode = str(rotary_mode)
    if rotary_mode not in ("half", "interleave"):
        raise ValueError(
            "rotary_mode must be 'half' or 'interleave', got "
            f"{rotary_mode!r}."
        )
    return rotary_mode


def _cos_sin_2d(cos: torch.Tensor, sin: torch.Tensor, rope_dim: int) -> tuple[torch.Tensor, torch.Tensor]:
    cos = cos.reshape(cos.shape[0], -1)[..., :rope_dim]
    sin = sin.reshape(sin.shape[0], -1)[..., :rope_dim]
    return cos, sin


def _apply_rope_reference(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    rope_dim: int,
    rotary_mode: str,
) -> torch.Tensor:
    cos, sin = _cos_sin_2d(cos, sin, rope_dim)
    view_shape = (cos.shape[0],) + (1,) * (x.dim() - 2) + (rope_dim,)
    cos = cos.view(view_shape)
    sin = sin.view(view_shape)
    rotate = (
        _rotate_half_neox
        if _normalize_rotary_mode(rotary_mode) == "half"
        else _rotate_half_interleaved
    )
    return (x * cos) + (rotate(x) * sin)


def _apply_query_rope_like_runtime(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    rope_dim: int,
    rotary_mode: str,
) -> torch.Tensor:
    # The graph path must match runtime RoPE before lightning_indexer consumes q_index.
    rotary_mode = _normalize_rotary_mode(rotary_mode)
    if x.device.type == "npu" and torch_npu is not None and x.dtype in (torch.float16, torch.bfloat16):
        return torch_npu.npu_rotary_mul(
            x.unsqueeze(2), cos, sin, rotary_mode
        ).squeeze(2)
    return _apply_rope_reference(
        x, cos, sin, int(rope_dim), rotary_mode
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
    rotary_mode: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    q = F.linear(q_c, wq_b_weight).view(-1, int(n_head), int(head_dim))
    index_weights = _query_only_weights_proj_pure(hidden_states, weights_proj_weight, float(score_scale))
    q_pe, q_nope = torch.split(q, [int(rope_dim), int(head_dim) - int(rope_dim)], dim=-1)
    q_pe = _apply_query_rope_like_runtime(
        q_pe, cos, sin, int(rope_dim), rotary_mode
    )
    return torch.cat((q_pe, q_nope), dim=-1), index_weights


def _dsa_indexer_pipeline_with_qc_functional(
    hidden_states: torch.Tensor,
    q_c: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
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
    n_head: int,
    head_dim: int,
    rope_dim: int,
    score_scale: float,
    sparse_count: int,
    rotary_mode: str = "half",
    return_gather_outputs: bool = False,
) -> tuple[torch.Tensor, ...]:
    if ascend_ops is None:
        raise RuntimeError("nanovllm Ascend ops are unavailable; rebuild with `bash scripts/build_nanovllm_ops.sh`.") from ascend_ops_import_error
    q_index, index_weights = _dsa_indexer_project_query_only_pure(
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
        rotary_mode=rotary_mode,
    )
    # GatherSelection interprets full_kv_actual_seq as the full sequence seen
    # by the current query and internally excludes that newest token from the
    # reusable source range.  Nano's DRAM source contains only the completed
    # full-block candidate prefix, so pass candidate_len + the one decode
    # query.  Passing candidate_lens directly incorrectly drops candidate
    # token candidate_len - 1 on every decode step.
    gather_full_kv_lens = candidate_lens + 1
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
            gather_full_kv_lens,
        )
        if return_gather_outputs:
            return q_index, index_weights, topk_indices, selection_kpe, selection_ckv, selection_block_table, gather_selection_status
        return q_index, index_weights, topk_indices

    # candidate_query_lens is cumulative for the normal TND eager path. The
    # full-decode graph uses BSND query=[B, 1, N, D], so each row has S=1.
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
        gather_full_kv_lens,
    )
    if return_gather_outputs:
        return q_index, index_weights, topk_indices, selection_kpe_out, selection_ckv_out, selection_block_table_out, gather_selection_status_out
    return q_index, index_weights, topk_indices


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


def _run_post_op(
    q_in,
    k_in,
    weights_in,
    cos,
    sin,
    q_out,
    k_out,
    weights_out,
    score_scale: float,
    rope_dim: int,
) -> None:
    if _POST_OPS is None:
        raise RuntimeError(
            "dsa_indexer_project post op is not built into nanovllm.ops. Run "
            "`bash scripts/build_nanovllm_ops.sh` on the Ascend machine first."
        ) from _POST_IMPORT_ERROR
    _POST_OPS.dsa_indexer_project_post_out(
        q_in,
        k_in,
        weights_in,
        cos,
        sin,
        q_out,
        k_out,
        weights_out,
        float(score_scale),
        int(rope_dim),
    )


def _can_use_post_op(
    q: torch.Tensor,
    k: torch.Tensor,
    weights: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    rotary_mode: str,
) -> bool:
    return (
        _normalize_rotary_mode(rotary_mode) == "half"
        and
        _POST_OPS is not None
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
    rotary_mode: str,
) -> bool:
    # GLM uses a raw outer ACLGraph.  Keep the DeepSeek half/NeoX path on its
    # existing npugraph_ex-compatible operators for now.
    return (
        _normalize_rotary_mode(rotary_mode) == "interleave"
        and _POST_OPS is not None
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
    rotary_mode: str,
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
        _normalize_rotary_mode(rotary_mode),
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
    rotary_mode: str = "half",
    wq_b_bmm_t: torch.Tensor | None = None,
    enable_q_bmm: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Project Indexer q/k/weights and write the provided output buffers."""
    _check_project_outputs(q_index_out, index_k_out, index_weights_out, num_tokens=int(hidden_states.shape[0]), n_head=int(n_head), head_dim=int(head_dim), dtype=hidden_states.dtype, device=hidden_states.device)
    q = _q_project(q_c, wq_b_weight, wq_b_bmm_t, int(n_head), int(head_dim), bool(enable_q_bmm))

    k = F.linear(hidden_states, wk_weight)

    k = F.layer_norm(k, (int(head_dim),), k_norm_weight, k_norm_bias, eps=1e-6)

    weights = F.linear(hidden_states.float(), weights_proj_weight.float()).contiguous()

    rotary_mode = _normalize_rotary_mode(rotary_mode)
    if _can_use_post_op(q, k, weights, cos, sin, rotary_mode):
        # B-stage true AscendC sub-op: q/k RoPE + weights cast, writing final outputs in-place.
        # DeepSeek-V3.2 BF16 SFA in vllm-ascend feeds raw weights_proj(x) to
        # lightning_indexer, so callers pass score_scale=1.0 here.
        _run_post_op(q, k, weights, cos, sin, q_index_out, index_k_out, index_weights_out, float(score_scale), int(rope_dim))
        return q_index_out, index_k_out, index_weights_out

    q_pe, q_nope = torch.split(q, [int(rope_dim), int(head_dim) - int(rope_dim)], dim=-1)
    k_pe, k_nope = torch.split(k, [int(rope_dim), int(head_dim) - int(rope_dim)], dim=-1)
    if q.device.type == "npu" and torch_npu is not None and q.dtype in (torch.float16, torch.bfloat16):
        q_pe = torch_npu.npu_rotary_mul(
            q_pe.unsqueeze(2), cos, sin, rotary_mode
        ).squeeze(2)
        k_pe = torch_npu.npu_rotary_mul(
            k_pe.unsqueeze(1).unsqueeze(2), cos, sin, rotary_mode
        ).squeeze(2).squeeze(1)
    else:
        q_pe = _apply_rope_reference(
            q_pe, cos, sin, int(rope_dim), rotary_mode
        )
        k_pe = _apply_rope_reference(
            k_pe.unsqueeze(1), cos, sin, int(rope_dim), rotary_mode
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
    rotary_mode: str = "half",
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

    rotary_mode = _normalize_rotary_mode(rotary_mode)
    # GLM stable decode writes the shared-A BMM result directly into the
    # persistent query buffer.  The non-RoPE suffix is already final and no
    # longer needs a slice/copy round trip. Keep DeepSeek's npugraph_ex path
    # unchanged until this raw custom launch is registered for Dynamo.
    direct_out = q_index_out if rotary_mode == "interleave" else None
    q = _q_project(
        q_c,
        wq_b_weight,
        wq_b_bmm_t,
        int(n_head),
        int(head_dim),
        bool(enable_q_bmm),
        out=direct_out,
    )

    _query_only_weights_proj(hidden_states, weights_proj_weight, index_weights_out, float(score_scale))

    if _can_use_query_rope_op(q, cos, sin, int(rope_dim), rotary_mode):
        _run_query_rope_inplace(
            q,
            cos,
            sin,
            int(rope_dim),
            rotary_mode,
        )
        return q_index_out, index_weights_out

    q_pe = q[..., : int(rope_dim)]
    q_pe = _apply_query_rope_like_runtime(
        q_pe, cos, sin, int(rope_dim), rotary_mode
    )
    q_index_out[..., : int(rope_dim)].copy_(q_pe)
    if q is not q_index_out:
        q_index_out[..., int(rope_dim) :].copy_(q[..., int(rope_dim) :])
    return q_index_out, index_weights_out


def dsa_indexer_pipeline_with_qc_full_graph(
    hidden_states: torch.Tensor,
    q_c: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
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
    n_head: int,
    head_dim: int,
    rope_dim: int,
    score_scale: float,
    sparse_count: int,
    rotary_mode: str = "half",
) -> tuple[torch.Tensor, ...]:
    """Graph-visible DSA projection, selection, and DRAM-to-HBM gather.

    The gather inputs are declared mutable in the torch.library schema. This
    keeps the cache/status update observable even though the CANN kernel writes
    those buffers in place and the model does not consume its synthetic outputs.
    """
    if _GRAPH_LIGHTNING_INDEXER is None or _GRAPH_GATHER_SELECTION_KV_CACHE is None:
        raise RuntimeError(
            "DSA FULL_DECODE_ONLY requires the torch.library lightning_indexer "
            "and gather_selection_kv_cache registrations. Rebuild with "
            "`bash scripts/build_nanovllm_ops.sh`."
        ) from _GRAPH_CUSTOM_OP_ERROR
    return _dsa_indexer_pipeline_with_qc_functional(
        hidden_states,
        q_c,
        cos,
        sin,
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
        n_head=int(n_head),
        head_dim=int(head_dim),
        rope_dim=int(rope_dim),
        score_scale=float(score_scale),
        sparse_count=int(sparse_count),
        rotary_mode=rotary_mode,
        return_gather_outputs=True,
    )


__all__ = [
    "dsa_indexer_project",
    "dsa_indexer_project_query_only",
    "dsa_indexer_pipeline_with_qc_full_graph",
]
