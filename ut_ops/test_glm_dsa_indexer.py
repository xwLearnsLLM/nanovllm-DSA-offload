"""Single-NPU semantic test for the GLM-5.1 DSA indexer path.

The test intentionally uses GLM's real indexer dimensions with structured
synthetic weights.  Structured weights let the golden derive each projection
without calling the same GEMM implementation as the code under test.  RoPE is
checked against an independent adjacent-pair (interleaved) implementation.

After the projection check, the test runs torch-npu's native
``npu_lightning_indexer`` and validates its public TND output contract.
"""

from __future__ import annotations

import argparse
import math
import time

import torch
import torch_npu  # type: ignore

import nanovllm.ops as ascend_ops
import nanovllm.models.dsa_indexer_project as indexer_project
from nanovllm.models.dsa_indexer_project import (
    dsa_indexer_project,
    dsa_indexer_project_query_only,
)


GLM_HIDDEN_SIZE = 6144
GLM_Q_LORA_RANK = 2048
GLM_INDEX_HEADS = 32
GLM_INDEX_HEAD_DIM = 128
GLM_INDEX_ROPE_DIM = 64
LAYER_NORM_EPS = 1e-6


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate GLM-5.1 interleaved DSA indexer semantics."
    )
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--full-len", type=int, default=4096)
    parser.add_argument("--topk", type=int, default=2048)
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--bmm-warmup", type=int, default=5)
    parser.add_argument("--bmm-iters", type=int, default=20)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if torch.device(args.device).type != "npu":
        raise ValueError("--device must select one NPU, for example npu:0.")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive.")
    if args.full_len <= 0:
        raise ValueError("--full-len must be positive.")
    if args.block_size <= 0:
        raise ValueError("--block-size must be positive.")
    if args.bmm_warmup < 0 or args.bmm_iters <= 0:
        raise ValueError(
            "--bmm-warmup must be non-negative and --bmm-iters must be "
            "positive."
        )
    if not 1 <= args.topk <= min(2048, args.full_len):
        raise ValueError(
            "--topk must be in [1, min(2048, full_len)], got "
            f"topk={args.topk}, full_len={args.full_len}."
        )


def make_interleaved_cos_sin(
    batch_size: int,
    dtype: torch.dtype,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return non-trivial cos/sin in adjacent-pair GLM layout."""

    positions = torch.arange(
        1, batch_size + 1, dtype=torch.float32, device=device
    ).unsqueeze(1)
    frequencies = torch.linspace(
        0.013,
        0.377,
        GLM_INDEX_ROPE_DIM // 2,
        dtype=torch.float32,
        device=device,
    ).unsqueeze(0)
    angles = positions * frequencies
    cos = angles.cos().repeat_interleave(2, dim=-1).to(dtype)
    sin = angles.sin().repeat_interleave(2, dim=-1).to(dtype)
    return (
        cos.view(batch_size, 1, 1, GLM_INDEX_ROPE_DIM).contiguous(),
        sin.view(batch_size, 1, 1, GLM_INDEX_ROPE_DIM).contiguous(),
    )


def apply_interleaved_rope_golden(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    """Apply RoPE by rotating each adjacent pair, independently of runtime."""

    if x.shape[-1] != GLM_INDEX_ROPE_DIM:
        raise ValueError(
            f"golden expects rope dim {GLM_INDEX_ROPE_DIM}, got {x.shape[-1]}."
        )
    cos_2d = cos.reshape(cos.shape[0], GLM_INDEX_ROPE_DIM).float()
    sin_2d = sin.reshape(sin.shape[0], GLM_INDEX_ROPE_DIM).float()
    view_shape = (x.shape[0],) + (1,) * (x.dim() - 2) + (
        GLM_INDEX_ROPE_DIM,
    )
    cos_f = cos_2d.view(view_shape)
    sin_f = sin_2d.view(view_shape)

    x_f = x.float()
    even = x_f[..., 0::2]
    odd = x_f[..., 1::2]
    rotated = torch.empty_like(x_f)
    rotated[..., 0::2] = -odd
    rotated[..., 1::2] = even
    return (x_f * cos_f + rotated * sin_f).to(x.dtype)


def make_structured_projection_weights(
    dtype: torch.dtype,
    device: torch.device,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """Build one-hot projections and return their source/sign metadata."""

    q_rows = GLM_INDEX_HEADS * GLM_INDEX_HEAD_DIM
    q_source = (
        torch.arange(q_rows, dtype=torch.int64) * 17 + 11
    ) % GLM_Q_LORA_RANK
    q_sign = torch.where(
        torch.arange(q_rows) % 3 == 0,
        torch.tensor(-1.0),
        torch.tensor(1.0),
    )
    wq_b = torch.zeros(
        q_rows, GLM_Q_LORA_RANK, dtype=dtype
    )
    wq_b[torch.arange(q_rows), q_source] = q_sign.to(dtype)

    k_source = (
        torch.arange(GLM_INDEX_HEAD_DIM, dtype=torch.int64) * 29 + 7
    ) % GLM_HIDDEN_SIZE
    k_sign = torch.where(
        torch.arange(GLM_INDEX_HEAD_DIM) % 2 == 0,
        torch.tensor(1.0),
        torch.tensor(-1.0),
    )
    wk = torch.zeros(GLM_INDEX_HEAD_DIM, GLM_HIDDEN_SIZE, dtype=dtype)
    wk[torch.arange(GLM_INDEX_HEAD_DIM), k_source] = k_sign.to(dtype)

    weights_source = (
        torch.arange(GLM_INDEX_HEADS, dtype=torch.int64) * 43 + 5
    ) % GLM_HIDDEN_SIZE
    weights_sign = torch.where(
        torch.arange(GLM_INDEX_HEADS) % 2 == 0,
        torch.tensor(1.0),
        torch.tensor(-1.0),
    )
    weights_proj = torch.zeros(
        GLM_INDEX_HEADS, GLM_HIDDEN_SIZE, dtype=torch.float32
    )
    weights_proj[
        torch.arange(GLM_INDEX_HEADS), weights_source
    ] = weights_sign

    return (
        wq_b.to(device),
        wk.to(device),
        weights_proj.to(device),
        q_source.to(device),
        q_sign.to(device=device, dtype=dtype),
        torch.stack((k_source, k_sign.to(torch.int64)), dim=0).to(device),
        torch.stack(
            (weights_source, weights_sign.to(torch.int64)), dim=0
        ).to(device),
    )


def manual_layer_norm(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
) -> torch.Tensor:
    x_f = x.float()
    mean = x_f.mean(dim=-1, keepdim=True)
    variance = (x_f - mean).square().mean(dim=-1, keepdim=True)
    normalized = (x_f - mean) * torch.rsqrt(variance + LAYER_NORM_EPS)
    return (normalized * weight.float() + bias.float()).to(x.dtype)


def max_abs_diff(actual: torch.Tensor, expected: torch.Tensor) -> float:
    return float((actual.float() - expected.float()).abs().max().item())


def assert_close(
    name: str,
    actual: torch.Tensor,
    expected: torch.Tensor,
    *,
    atol: float,
    rtol: float,
) -> float:
    diff = max_abs_diff(actual, expected)
    try:
        torch.testing.assert_close(
            actual.float(), expected.float(), atol=atol, rtol=rtol
        )
    except AssertionError as error:
        raise AssertionError(
            f"{name} mismatch: shape={tuple(actual.shape)} "
            f"max_abs={diff:.6g}, atol={atol}, rtol={rtol}"
        ) from error
    return diff


def benchmark_ms(fn, warmup: int, iters: int) -> float:
    for _ in range(warmup):
        fn()
    torch.npu.synchronize()
    started = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.npu.synchronize()
    return (time.perf_counter() - started) * 1000.0 / iters


def make_paged_index_cache(
    *,
    batch_size: int,
    full_len: int,
    block_size: int,
    dtype: torch.dtype,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    blocks_per_row = math.ceil(full_len / block_size)
    # Physical block 0 is the runtime's null block. Give every request its own
    # contiguous physical blocks so range checks remain per-request logical IDs.
    num_physical_blocks = 1 + batch_size * blocks_per_row
    key = torch.randn(
        num_physical_blocks,
        block_size,
        1,
        GLM_INDEX_HEAD_DIM,
        dtype=dtype,
        device=device,
    )
    key[0].zero_()
    block_table = (
        torch.arange(
            1,
            num_physical_blocks,
            dtype=torch.int32,
            device=device,
        )
        .view(batch_size, blocks_per_row)
        .contiguous()
    )
    return key, block_table


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    validate_args(args)
    device = torch.device(args.device)
    torch.npu.set_device(device)
    torch.manual_seed(args.seed)
    torch.npu.manual_seed(args.seed)
    dtype = torch.bfloat16

    print(
        "GLM_DSA_INDEXER_CONFIG "
        f"device={device} batch={args.batch_size} full_len={args.full_len} "
        f"topk={args.topk} block_size={args.block_size} seed={args.seed} "
        f"n_head={GLM_INDEX_HEADS} head_dim={GLM_INDEX_HEAD_DIM} "
        f"q_lora_rank={GLM_Q_LORA_RANK} rotary_mode=interleave",
        flush=True,
    )

    hidden_states = torch.randn(
        args.batch_size,
        GLM_HIDDEN_SIZE,
        dtype=dtype,
        device=device,
    )
    q_c = torch.randn(
        args.batch_size,
        GLM_Q_LORA_RANK,
        dtype=dtype,
        device=device,
    )
    (
        wq_b,
        wk,
        weights_proj,
        q_source,
        q_sign,
        k_metadata,
        weights_metadata,
    ) = make_structured_projection_weights(dtype, device)
    k_source, k_sign_i64 = k_metadata[0], k_metadata[1]
    weights_source, weights_sign_i64 = (
        weights_metadata[0],
        weights_metadata[1],
    )
    k_sign = k_sign_i64.to(dtype)
    weights_sign = weights_sign_i64.to(torch.float32)

    k_norm_weight = torch.linspace(
        0.75,
        1.25,
        GLM_INDEX_HEAD_DIM,
        dtype=torch.float32,
        device=device,
    ).to(dtype)
    k_norm_bias = torch.linspace(
        -0.125,
        0.125,
        GLM_INDEX_HEAD_DIM,
        dtype=torch.float32,
        device=device,
    ).to(dtype)
    cos, sin = make_interleaved_cos_sin(args.batch_size, dtype, device)

    q_out = torch.empty(
        args.batch_size,
        GLM_INDEX_HEADS,
        GLM_INDEX_HEAD_DIM,
        dtype=dtype,
        device=device,
    )
    k_out = torch.empty(
        args.batch_size,
        GLM_INDEX_HEAD_DIM,
        dtype=dtype,
        device=device,
    )
    index_weights_out = torch.empty(
        args.batch_size,
        GLM_INDEX_HEADS,
        dtype=dtype,
        device=device,
    )
    dsa_indexer_project(
        hidden_states,
        q_c,
        cos,
        sin,
        wq_b,
        wk,
        k_norm_weight,
        k_norm_bias,
        weights_proj,
        q_out,
        k_out,
        index_weights_out,
        n_head=GLM_INDEX_HEADS,
        head_dim=GLM_INDEX_HEAD_DIM,
        rope_dim=GLM_INDEX_ROPE_DIM,
        score_scale=1.0,
        enable_q_bmm=False,
    )

    # Derive all three projections from the structured one-hot definitions.
    q_projected = (
        q_c.index_select(1, q_source) * q_sign.unsqueeze(0)
    ).view(args.batch_size, GLM_INDEX_HEADS, GLM_INDEX_HEAD_DIM)
    k_projected = hidden_states.index_select(1, k_source) * k_sign.unsqueeze(0)
    k_normalized = manual_layer_norm(
        k_projected, k_norm_weight, k_norm_bias
    )
    weights_expected = (
        hidden_states.float().index_select(1, weights_source)
        * weights_sign.unsqueeze(0)
    ).to(dtype)

    q_rope_expected = apply_interleaved_rope_golden(
        q_projected[..., :GLM_INDEX_ROPE_DIM], cos, sin
    )
    q_expected = torch.cat(
        (q_rope_expected, q_projected[..., GLM_INDEX_ROPE_DIM:]), dim=-1
    )
    k_rope_expected = apply_interleaved_rope_golden(
        k_normalized[..., :GLM_INDEX_ROPE_DIM], cos, sin
    )
    k_expected = torch.cat(
        (k_rope_expected, k_normalized[..., GLM_INDEX_ROPE_DIM:]), dim=-1
    )

    torch.npu.synchronize()
    q_diff = assert_close("q_index", q_out, q_expected, atol=0.04, rtol=0.02)
    k_diff = assert_close("index_k", k_out, k_expected, atol=0.05, rtol=0.025)
    weights_diff = assert_close(
        "index_weights",
        index_weights_out,
        weights_expected,
        atol=0.01,
        rtol=0.01,
    )
    print(
        "GLM_DSA_INDEXER_PROJECT_CHECK ok=1 "
        f"q_max_abs={q_diff:.6g} k_max_abs={k_diff:.6g} "
        f"weights_max_abs={weights_diff:.6g}",
        flush=True,
    )

    # Steady-state decode uses the query-only BMM-transpose path for batches
    # up to 64.  Its shared-A form consumes q_c=[B, K] directly instead of
    # materializing [B, H, K].  Check it against both the legacy expanded-A
    # operator input and the F.linear fallback before testing selection.
    bmm_batch = min(args.batch_size, 64)
    hidden_bmm = hidden_states[:bmm_batch]
    q_c_bmm = q_c[:bmm_batch]
    cos_bmm = cos[:bmm_batch]
    sin_bmm = sin[:bmm_batch]
    wq_b_bmm_t = (
        wq_b.view(
            GLM_INDEX_HEADS,
            GLM_INDEX_HEAD_DIM,
            GLM_Q_LORA_RANK,
        )
        .transpose(1, 2)
        .contiguous()
    )
    weights_proj_bf16 = weights_proj.to(dtype)
    q_raw_shared = torch.empty(
        bmm_batch,
        GLM_INDEX_HEADS,
        GLM_INDEX_HEAD_DIM,
        dtype=dtype,
        device=device,
    )
    q_raw_legacy = torch.empty_like(q_raw_shared)
    q_c_by_head = (
        q_c_bmm.unsqueeze(1)
        .expand(-1, GLM_INDEX_HEADS, -1)
        .contiguous()
    )
    ascend_ops.batch_matmul_transpose(q_c_bmm, wq_b_bmm_t, q_raw_shared)
    ascend_ops.batch_matmul_transpose(q_c_by_head, wq_b_bmm_t, q_raw_legacy)
    torch.npu.synchronize()
    raw_diff = assert_close(
        "query_only_shared_a_raw",
        q_raw_shared,
        q_raw_legacy,
        atol=0.04,
        rtol=0.02,
    )

    def run_shared_a() -> None:
        ascend_ops.batch_matmul_transpose(
            q_c_bmm,
            wq_b_bmm_t,
            q_raw_shared,
        )

    def run_legacy_expanded_a() -> None:
        expanded = (
            q_c_bmm.unsqueeze(1)
            .expand(-1, GLM_INDEX_HEADS, -1)
            .contiguous()
        )
        ascend_ops.batch_matmul_transpose(
            expanded,
            wq_b_bmm_t,
            q_raw_legacy,
        )

    legacy_ms = benchmark_ms(
        run_legacy_expanded_a,
        args.bmm_warmup,
        args.bmm_iters,
    )
    shared_ms = benchmark_ms(
        run_shared_a,
        args.bmm_warmup,
        args.bmm_iters,
    )
    print(
        "GLM_DSA_SHARED_A_BMM_CHECK ok=1 "
        f"batch={bmm_batch} q_max_abs={raw_diff:.6g} "
        f"legacy_expand_bmm_ms={legacy_ms:.6f} "
        f"shared_a_bmm_ms={shared_ms:.6f} "
        f"speedup={legacy_ms / shared_ms:.4f} "
        f"warmup={args.bmm_warmup} iters={args.bmm_iters}",
        flush=True,
    )

    q_linear = torch.empty(
        bmm_batch,
        GLM_INDEX_HEADS,
        GLM_INDEX_HEAD_DIM,
        dtype=dtype,
        device=device,
    )
    weights_linear = torch.empty(
        bmm_batch,
        GLM_INDEX_HEADS,
        dtype=dtype,
        device=device,
    )
    q_bmm = torch.empty_like(q_linear)
    weights_bmm = torch.empty_like(weights_linear)
    dsa_indexer_project_query_only(
        hidden_bmm,
        q_c_bmm,
        cos_bmm,
        sin_bmm,
        wq_b,
        weights_proj_bf16,
        q_linear,
        weights_linear,
        n_head=GLM_INDEX_HEADS,
        head_dim=GLM_INDEX_HEAD_DIM,
        rope_dim=GLM_INDEX_ROPE_DIM,
        score_scale=1.0,
        enable_q_bmm=False,
    )
    dsa_indexer_project_query_only(
        hidden_bmm,
        q_c_bmm,
        cos_bmm,
        sin_bmm,
        wq_b,
        weights_proj_bf16,
        q_bmm,
        weights_bmm,
        n_head=GLM_INDEX_HEADS,
        head_dim=GLM_INDEX_HEAD_DIM,
        rope_dim=GLM_INDEX_ROPE_DIM,
        score_scale=1.0,
        wq_b_bmm_t=wq_b_bmm_t,
        enable_q_bmm=True,
    )
    if not indexer_project._can_use_q_bmm(
        q_c_bmm,
        wq_b_bmm_t,
        enable_q_bmm=True,
    ):
        raise AssertionError(
            "GLM query-only test did not select the BMM-transpose hot path."
        )
    if not indexer_project._can_use_query_rope_op(
        q_bmm,
        cos_bmm,
        sin_bmm,
        GLM_INDEX_ROPE_DIM,
    ):
        raise AssertionError(
            "GLM query-only test did not select the fused in-place RoPE "
            "hot path. Rebuild nanovllm ops."
        )

    torch.npu.synchronize()
    q_bmm_diff = assert_close(
        "query_only_bmm_q_index",
        q_bmm,
        q_linear,
        atol=0.04,
        rtol=0.02,
    )
    q_bmm_golden_diff = assert_close(
        "query_only_bmm_q_index_golden",
        q_bmm,
        q_expected[:bmm_batch],
        atol=0.04,
        rtol=0.02,
    )
    weights_bmm_diff = assert_close(
        "query_only_bmm_weights",
        weights_bmm,
        weights_linear,
        atol=0.01,
        rtol=0.01,
    )
    cosine = torch.nn.functional.cosine_similarity(
        q_bmm.float().reshape(bmm_batch, -1),
        q_linear.float().reshape(bmm_batch, -1),
        dim=-1,
    )
    min_cosine = float(cosine.min().item())
    print(
        "GLM_DSA_QUERY_ONLY_BMM_CHECK ok=1 "
        f"batch={bmm_batch} path=batch_matmul_transpose_shared_a "
        f"q_max_abs={q_bmm_diff:.6g} "
        f"q_golden_max_abs={q_bmm_golden_diff:.6g} "
        f"weights_max_abs={weights_bmm_diff:.6g} "
        f"q_min_cosine={min_cosine:.9f}",
        flush=True,
    )

    q_post_legacy_raw = torch.empty_like(q_bmm)
    q_post_legacy_out = torch.empty_like(q_bmm)
    q_post_fused_out = torch.empty_like(q_bmm)

    def run_legacy_query_post() -> None:
        ascend_ops.batch_matmul_transpose(
            q_c_bmm,
            wq_b_bmm_t,
            q_post_legacy_raw,
        )
        q_pe = indexer_project._apply_query_rope_like_runtime(
            q_post_legacy_raw[..., :GLM_INDEX_ROPE_DIM],
            cos_bmm,
            sin_bmm,
            GLM_INDEX_ROPE_DIM,
            "interleave",
        )
        q_post_legacy_out[..., :GLM_INDEX_ROPE_DIM].copy_(q_pe)
        q_post_legacy_out[..., GLM_INDEX_ROPE_DIM:].copy_(
            q_post_legacy_raw[..., GLM_INDEX_ROPE_DIM:]
        )

    def run_fused_query_post() -> None:
        ascend_ops.batch_matmul_transpose(
            q_c_bmm,
            wq_b_bmm_t,
            q_post_fused_out,
        )
        ascend_ops.dsa_indexer_query_rope_inplace(
            q_post_fused_out,
            cos_bmm,
            sin_bmm,
            GLM_INDEX_ROPE_DIM,
        )

    run_legacy_query_post()
    run_fused_query_post()
    torch.npu.synchronize()
    query_post_diff = assert_close(
        "query_only_fused_post",
        q_post_fused_out,
        q_post_legacy_out,
        atol=0.04,
        rtol=0.02,
    )
    legacy_query_post_ms = benchmark_ms(
        run_legacy_query_post,
        args.bmm_warmup,
        args.bmm_iters,
    )
    fused_query_post_ms = benchmark_ms(
        run_fused_query_post,
        args.bmm_warmup,
        args.bmm_iters,
    )
    print(
        "GLM_DSA_QUERY_POST_CHECK ok=1 "
        f"batch={bmm_batch} q_max_abs={query_post_diff:.6g} "
        f"legacy_shared_bmm_rope_copy_ms={legacy_query_post_ms:.6f} "
        f"direct_bmm_fused_rope_ms={fused_query_post_ms:.6f} "
        f"speedup={legacy_query_post_ms / fused_query_post_ms:.4f} "
        f"warmup={args.bmm_warmup} iters={args.bmm_iters}",
        flush=True,
    )

    key_cache, block_table = make_paged_index_cache(
        batch_size=args.batch_size,
        full_len=args.full_len,
        block_size=args.block_size,
        dtype=dtype,
        device=device,
    )
    actual_seq_lengths_query = torch.arange(
        1,
        args.batch_size + 1,
        dtype=torch.int32,
        device=device,
    )
    actual_seq_lengths_key = torch.full(
        (args.batch_size,),
        args.full_len,
        dtype=torch.int32,
        device=device,
    )
    result = torch_npu.npu_lightning_indexer(
        query=q_out,
        key=key_cache,
        weights=index_weights_out,
        actual_seq_lengths_query=actual_seq_lengths_query,
        actual_seq_lengths_key=actual_seq_lengths_key,
        block_table=block_table,
        layout_query="TND",
        layout_key="PA_BSND",
        sparse_count=args.topk,
        sparse_mode=3,
    )
    topk_indices = result[0] if isinstance(result, (tuple, list)) else result
    if not isinstance(topk_indices, torch.Tensor):
        raise TypeError(
            "torch_npu.npu_lightning_indexer must return a Tensor or a tuple "
            f"whose first item is a Tensor, got {type(topk_indices).__name__}."
        )
    expected_shape = (args.batch_size, 1, args.topk)
    if tuple(topk_indices.shape) != expected_shape:
        raise AssertionError(
            "Unexpected LightningIndexer TND output shape: "
            f"actual={tuple(topk_indices.shape)}, expected={expected_shape}."
        )

    torch.npu.synchronize()
    topk_cpu = topk_indices.reshape(args.batch_size, args.topk).cpu()
    minimum = int(topk_cpu.min().item())
    maximum = int(topk_cpu.max().item())
    if minimum < 0 or maximum >= args.full_len:
        raise AssertionError(
            "LightningIndexer returned an out-of-range logical token ID: "
            f"min={minimum}, max={maximum}, valid=[0,{args.full_len})."
        )
    unique_per_row = [
        int(torch.unique(topk_cpu[row]).numel())
        for row in range(args.batch_size)
    ]
    if any(count != args.topk for count in unique_per_row):
        raise AssertionError(
            "LightningIndexer returned duplicate token IDs: "
            f"unique_per_row={unique_per_row}, expected={args.topk}."
        )
    print(
        "GLM_DSA_LIGHTNING_CHECK ok=1 "
        f"shape={tuple(topk_indices.shape)} dtype={topk_indices.dtype} "
        f"min={minimum} max={maximum} "
        f"unique_per_row={unique_per_row}",
        flush=True,
    )
    print(
        "GLM_DSA_INDEXER_UT_OK "
        f"batch={args.batch_size} full_len={args.full_len} topk={args.topk}",
        flush=True,
    )


if __name__ == "__main__":
    main()
