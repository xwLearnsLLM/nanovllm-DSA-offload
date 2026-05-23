import argparse
import glob
import os
from time import perf_counter

os.environ.setdefault("VLLM_ASCEND_ENABLE_NZ", "0")

import torch
import torch.nn.functional as F


def log(message: str) -> None:
    print(message, flush=True)


def _prepend_env_path(name: str, path: str) -> None:
    current = os.environ.get(name, "")
    parts = [part for part in current.split(os.pathsep) if part]
    if path not in parts:
        os.environ[name] = f"{path}{os.pathsep}{current}" if current else path


def _dedupe_env_path(name: str) -> None:
    current = os.environ.get(name, "")
    parts = []
    seen = set()
    for part in current.split(os.pathsep):
        if not part or part in seen:
            continue
        seen.add(part)
        parts.append(part)
    if parts:
        os.environ[name] = os.pathsep.join(parts)


def _ensure_vllm_ascend_custom_opp_path(vllm_ascend_module) -> str | None:
    package_dir = os.path.dirname(os.path.realpath(vllm_ascend_module.__file__))
    custom_opp_path = os.path.join(
        package_dir,
        "_cann_ops_custom",
        "vendors",
        "vllm-ascend",
    )
    if os.path.exists(custom_opp_path):
        _prepend_env_path("ASCEND_CUSTOM_OPP_PATH", custom_opp_path)
    try:
        from vllm_ascend.platform import NPUPlatform  # type: ignore

        NPUPlatform.import_kernels()
    except Exception as exc:
        log(f"COMPARE warning import_kernels failed: {exc!r}")
    _dedupe_env_path("ASCEND_CUSTOM_OPP_PATH")
    return custom_opp_path if os.path.exists(custom_opp_path) else None


def register_ascend_ops() -> None:
    log("COMPARE stage=register_import_torch_npu")
    import torch_npu  # type: ignore  # noqa: F401

    log("COMPARE stage=register_import_vllm")
    import vllm  # type: ignore  # noqa: F401

    log("COMPARE stage=register_import_vllm_ascend")
    import vllm_ascend  # type: ignore

    custom_opp_path = _ensure_vllm_ascend_custom_opp_path(vllm_ascend)
    log(
        "COMPARE stage=register_import_custom_op "
        f"custom_opp_path={custom_opp_path} "
        f"ASCEND_CUSTOM_OPP_PATH={os.environ.get('ASCEND_CUSTOM_OPP_PATH', '')}"
    )
    from vllm_ascend import vllm_ascend_C  # type: ignore  # noqa: F401

    log("COMPARE stage=register_import_layer_shard_linear")
    from vllm_ascend.ops.layer_shard_linear import (  # type: ignore  # noqa: F401
        is_hidden_layer,
        post_process_after_loading_for_shard_weight_series,
        reach_layer_for_shard_weight_series,
        register_all_layers_to_shard_weight_series,
    )
    log("COMPARE stage=register_done")


def desc(name: str, tensor: torch.Tensor) -> str:
    return (
        f"{name}=shape={tuple(tensor.shape)} dtype={tensor.dtype} "
        f"device={tensor.device} contiguous={tensor.is_contiguous()} "
        f"stride={tuple(tensor.stride())} storage_offset={tensor.storage_offset()}"
    )


def stats(name: str, tensor: torch.Tensor) -> str:
    value = tensor.float()
    finite = torch.isfinite(value)
    finite_count = int(finite.sum().item())
    if finite_count == 0:
        return f"{name}: finite=0/{value.numel()}"
    finite_value = value[finite]
    return (
        f"{name}: finite={finite_count}/{value.numel()} "
        f"min={finite_value.min().item():.6g} max={finite_value.max().item():.6g}"
    )


def pick_dump(path: str | None, pattern: str | None) -> str:
    if path:
        return path
    if not pattern:
        raise ValueError("either --dump or --dump-glob is required")
    matches = sorted(glob.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"no dump matched: {pattern}")
    return matches[-1]


def to_device(payload: dict, name: str, device: torch.device) -> torch.Tensor:
    value = payload[name]
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} is not a tensor: {type(value)!r}")
    return value.to(device=device).contiguous()


def cumulative_to_lens(ends: list[int]) -> list[int]:
    lens = []
    prev = 0
    for end in ends:
        lens.append(int(end) - prev)
        prev = int(end)
    return lens


def cumulative(values: list[int]) -> list[int]:
    total = 0
    out = []
    for value in values:
        total += int(value)
        out.append(total)
    return out


def materialize_paged_cache(
    cache: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: list[int],
) -> torch.Tensor:
    block_size = int(cache.shape[1])
    rows = []
    block_table_cpu = block_table.detach().cpu()
    for seq_idx, seq_len in enumerate(seq_lens):
        for pos in range(int(seq_len)):
            block_col = pos // block_size
            block_offset = pos % block_size
            block_id = int(block_table_cpu[seq_idx, block_col].item())
            rows.append(cache[block_id, block_offset])
    if not rows:
        return cache.new_empty((0, cache.shape[2], cache.shape[3]))
    return torch.stack(rows, dim=0).contiguous()


def make_causal_mask(mask_size: int, device: torch.device) -> torch.Tensor:
    return torch.triu(
        torch.ones(mask_size, mask_size, dtype=torch.int8, device=device),
        diagonal=1,
    ).contiguous()


def run_sfa(payload: dict, tensors: dict[str, torch.Tensor]) -> torch.Tensor:
    start = perf_counter()
    out = torch.ops._C_ascend.npu_sparse_flash_attention(
        query=tensors["query"],
        key=tensors["key"],
        value=tensors["value"],
        sparse_indices=tensors["sparse_indices"],
        scale_value=float(payload["scale_value"]),
        sparse_block_size=int(payload.get("sparse_block_size", 1)),
        block_table=tensors["block_table"],
        actual_seq_lengths_query=tensors["actual_seq_lengths_query"],
        actual_seq_lengths_kv=tensors["actual_seq_lengths_kv"],
        query_rope=tensors["query_rope"],
        key_rope=tensors["key_rope"],
        layout_query=str(payload.get("layout_query", "TND")),
        layout_kv=str(payload.get("layout_kv", "PA_BSND")),
        sparse_mode=int(payload.get("sparse_mode", 3)),
    )
    if isinstance(out, tuple):
        out = out[0]
    torch.npu.synchronize()
    log(f"COMPARE after_sfa elapsed={perf_counter() - start:.6f}s {desc('sfa_out', out)}")
    return out


def run_dense_mla_materialized(
    payload: dict,
    tensors: dict[str, torch.Tensor],
    mask_size: int,
    expand_kv_to_heads: bool,
) -> torch.Tensor:
    import torch_npu  # type: ignore

    query = tensors["query"]
    query_rope = tensors["query_rope"]
    block_table = tensors["block_table"]
    key_cache = tensors["key"]
    value_cache = tensors["value"]
    key_rope_cache = tensors["key_rope"]
    device = query.device

    q_ends = [int(x) for x in tensors["actual_seq_lengths_query"].detach().cpu().tolist()]
    q_lens = cumulative_to_lens(q_ends)
    kv_lens = [int(x) for x in tensors["actual_seq_lengths_kv"].detach().cpu().tolist()]
    kv_ends = cumulative(kv_lens)

    key = materialize_paged_cache(key_cache, block_table, kv_lens)
    value = materialize_paged_cache(value_cache, block_table, kv_lens)
    key_rope = materialize_paged_cache(key_rope_cache, block_table, kv_lens)
    num_heads = int(query.shape[1])
    num_kv_heads = int(key.shape[1])
    if expand_kv_to_heads and num_kv_heads == 1:
        key = key.expand(-1, num_heads, -1).contiguous()
        value = value.expand(-1, num_heads, -1).contiguous()
        key_rope = key_rope.expand(-1, num_heads, -1).contiguous()
        num_kv_heads = num_heads

    attn_mask = make_causal_mask(mask_size, device)
    log(
        "COMPARE dense_materialized_inputs "
        f"{desc('query', query)} {desc('key', key)} {desc('value', value)} "
        f"{desc('query_rope', query_rope)} {desc('key_rope', key_rope)} "
        f"{desc('attn_mask', attn_mask)} q_lens={q_lens} kv_lens={kv_lens}"
    )

    start = perf_counter()
    out, lse = torch_npu.npu_fused_infer_attention_score(
        query,
        key,
        value,
        query_rope=query_rope,
        key_rope=key_rope,
        num_heads=num_heads,
        num_key_value_heads=num_kv_heads,
        input_layout="TND",
        atten_mask=attn_mask,
        sparse_mode=3,
        scale=float(payload["scale_value"]),
        antiquant_mode=0,
        antiquant_scale=None,
        block_table=None,
        block_size=0,
        softmax_lse_flag=True,
        actual_seq_lengths=q_ends,
        actual_seq_lengths_kv=kv_ends,
    )
    torch.npu.synchronize()
    log(
        f"COMPARE after_dense_mla_materialized elapsed={perf_counter() - start:.6f}s "
        f"{desc('mla_out', out)} {desc('mla_lse', lse)}"
    )
    return out


def run_dense_mla_paged(
    payload: dict,
    tensors: dict[str, torch.Tensor],
    mask_size: int,
    kv_lens_mode: str,
) -> torch.Tensor:
    import torch_npu  # type: ignore

    query = tensors["query"]
    query_rope = tensors["query_rope"]
    key_cache = tensors["key"].transpose(1, 2).contiguous()
    value_cache = tensors["value"].transpose(1, 2).contiguous()
    key_rope_cache = tensors["key_rope"].transpose(1, 2).contiguous()
    block_table = tensors["block_table"]
    device = query.device

    q_ends = [int(x) for x in tensors["actual_seq_lengths_query"].detach().cpu().tolist()]
    q_lens = cumulative_to_lens(q_ends)
    kv_lens = [int(x) for x in tensors["actual_seq_lengths_kv"].detach().cpu().tolist()]
    kv_arg = cumulative(kv_lens) if kv_lens_mode == "cumulative" else kv_lens
    attn_mask = make_causal_mask(mask_size, device)
    block_size = int(key_cache.shape[2])
    num_heads = int(query.shape[1])
    num_kv_heads = int(key_cache.shape[1])

    log(
        "COMPARE dense_paged_inputs "
        f"{desc('query', query)} {desc('key_cache', key_cache)} "
        f"{desc('value_cache', value_cache)} {desc('query_rope', query_rope)} "
        f"{desc('key_rope_cache', key_rope_cache)} {desc('block_table', block_table)} "
        f"{desc('attn_mask', attn_mask)} block_size={block_size} "
        f"layout=BnNBsD q_lens={q_lens} kv_lens={kv_lens} "
        f"kv_lens_mode={kv_lens_mode} kv_arg={kv_arg}"
    )

    start = perf_counter()
    out, lse = torch_npu.npu_fused_infer_attention_score(
        query,
        key_cache,
        value_cache,
        query_rope=query_rope,
        key_rope=key_rope_cache,
        num_heads=num_heads,
        num_key_value_heads=num_kv_heads,
        input_layout="TND",
        atten_mask=attn_mask,
        sparse_mode=3,
        scale=float(payload["scale_value"]),
        antiquant_mode=0,
        antiquant_scale=None,
        block_table=block_table,
        block_size=block_size,
        softmax_lse_flag=True,
        actual_seq_lengths=q_ends,
        actual_seq_lengths_kv=kv_arg,
    )
    torch.npu.synchronize()
    log(
        f"COMPARE after_dense_mla_paged elapsed={perf_counter() - start:.6f}s "
        f"{desc('mla_out', out)} {desc('mla_lse', lse)}"
    )
    return out


def run_torch_sparse_reference(
    payload: dict,
    tensors: dict[str, torch.Tensor],
    chunk_size: int,
) -> torch.Tensor:
    query = tensors["query"]
    query_rope = tensors["query_rope"]
    sparse_indices = tensors["sparse_indices"]
    block_table = tensors["block_table"]
    key = materialize_paged_cache(
        tensors["key"],
        block_table,
        [int(x) for x in tensors["actual_seq_lengths_kv"].detach().cpu().tolist()],
    )
    value = materialize_paged_cache(
        tensors["value"],
        block_table,
        [int(x) for x in tensors["actual_seq_lengths_kv"].detach().cpu().tolist()],
    )
    key_rope = materialize_paged_cache(
        tensors["key_rope"],
        block_table,
        [int(x) for x in tensors["actual_seq_lengths_kv"].detach().cpu().tolist()],
    )
    if int(key.shape[1]) != 1:
        raise ValueError(f"sparse reference only supports absorb kv_heads=1, got {key.shape}")

    q_ends = [int(x) for x in tensors["actual_seq_lengths_query"].detach().cpu().tolist()]
    q_starts = [0] + q_ends[:-1]
    kv_lens = [int(x) for x in tensors["actual_seq_lengths_kv"].detach().cpu().tolist()]
    kv_starts = [0] + cumulative(kv_lens)[:-1]
    scale = float(payload["scale_value"])
    out = torch.empty_like(query)

    log(
        "COMPARE torch_sparse_reference_inputs "
        f"{desc('query', query)} {desc('key', key)} {desc('value', value)} "
        f"{desc('query_rope', query_rope)} {desc('key_rope', key_rope)} "
        f"{desc('sparse_indices', sparse_indices)} chunk_size={chunk_size}"
    )
    start = perf_counter()
    for seq_idx, (q_start, q_end, kv_start) in enumerate(
        zip(q_starts, q_ends, kv_starts)
    ):
        for chunk_start in range(q_start, q_end, chunk_size):
            chunk_end = min(chunk_start + chunk_size, q_end)
            selected_local = sparse_indices[chunk_start:chunk_end, 0, :]
            valid = selected_local >= 0
            selected = selected_local.clamp_min(0) + kv_start

            q = query[chunk_start:chunk_end].float()
            q_pe = query_rope[chunk_start:chunk_end].float()
            k = key[selected].squeeze(2).float()
            v = value[selected].squeeze(2).float()
            k_pe = key_rope[selected].squeeze(2).float()

            scores = torch.einsum("chd,ckd->chk", q, k)
            scores = scores + torch.einsum("chr,ckr->chk", q_pe, k_pe)
            scores = scores * scale
            scores = scores.masked_fill(~valid[:, None, :], -float("inf"))
            probs = torch.softmax(scores, dim=-1)
            probs = probs.masked_fill(~valid[:, None, :], 0)
            latent = torch.einsum("chk,ckd->chd", probs, v)
            out[chunk_start:chunk_end] = latent.to(out.dtype)
        log(
            "COMPARE torch_sparse_reference_progress "
            f"seq={seq_idx} q_range=[{q_start},{q_end}) kv_start={kv_start}"
        )
    torch.npu.synchronize()
    log(
        f"COMPARE after_torch_sparse_reference elapsed={perf_counter() - start:.6f}s "
        f"{desc('sparse_ref', out)}"
    )
    return out


def compare_outputs(
    left_name: str,
    left_out: torch.Tensor,
    right_name: str,
    right_out: torch.Tensor,
) -> None:
    left = left_out.float()
    right = right_out.float()
    diff = left - right
    abs_diff = diff.abs()
    left_min = left.min()
    left_max = left.max()
    right_min = right.min()
    right_max = right.max()
    value_range = (torch.maximum(left_max, right_max) - torch.minimum(left_min, right_min)).clamp_min(1e-6)
    max_abs = abs_diff.max()
    mean_abs = abs_diff.mean()
    left_flat = left.flatten()
    right_flat = right.flatten()
    rel_l2 = torch.linalg.vector_norm(diff.flatten()) / torch.linalg.vector_norm(right_flat).clamp_min(1e-6)
    cosine = F.cosine_similarity(left_flat, right_flat, dim=0)
    log(
        "COMPARE diff "
        f"left={left_name} right={right_name} "
        f"max_abs={max_abs.item():.6g} "
        f"mean_abs={mean_abs.item():.6g} "
        f"rms={diff.pow(2).mean().sqrt().item():.6g} "
        f"rel_l2={rel_l2.item():.6g} "
        f"cosine={cosine.item():.8f} "
        f"value_range={value_range.item():.6g} "
        f"relative_max_error={(max_abs / value_range).item():.6g} "
        f"relative_mean_abs_error={(mean_abs / value_range).item():.6g}"
    )
    log(f"COMPARE {stats(left_name, left_out)}")
    log(f"COMPARE {stats(right_name, right_out)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare SFA and dense MLA prefill from a Nano SFA dump.")
    parser.add_argument("--dump")
    parser.add_argument("--dump-glob")
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--mask-size", type=int, default=2048)
    parser.add_argument(
        "--mla-mode",
        choices=("paged", "materialized"),
        default="paged",
    )
    parser.add_argument(
        "--kv-lens-mode",
        choices=("seq", "cumulative"),
        default="seq",
    )
    parser.add_argument("--expand-materialized-kv-to-heads", action="store_true")
    parser.add_argument("--metadata-only", action="store_true")
    parser.add_argument("--skip-sfa-op", action="store_true")
    parser.add_argument("--torch-sparse-reference", action="store_true")
    parser.add_argument("--reference-chunk-size", type=int, default=64)
    args = parser.parse_args()

    log("COMPARE stage=start")
    register_ascend_ops()
    device_index = int(str(args.device).split(":")[-1])
    torch.npu.set_device(device_index)
    torch.npu.config.allow_internal_format = True
    log(f"COMPARE stage=set_device device={args.device}")

    dump_path = pick_dump(args.dump, args.dump_glob)
    log(f"COMPARE stage=load_dump path={dump_path}")
    payload = torch.load(dump_path, map_location="cpu")
    log(
        "COMPARE stage=loaded_dump "
        f"keys={sorted(str(key) for key in payload.keys())}"
    )
    if payload.get("phase") != "prefill":
        raise ValueError(f"only prefill dumps are supported, got phase={payload.get('phase')!r}")

    device = torch.device(args.device)
    log("COMPARE stage=move_tensors_to_device")
    tensors = {
        "query": to_device(payload, "query", device),
        "key": to_device(payload, "key", device),
        "value": to_device(payload, "value", device),
        "sparse_indices": to_device(payload, "sparse_indices", device).to(torch.int32),
        "block_table": to_device(payload, "block_table", device).to(torch.int32),
        "actual_seq_lengths_query": to_device(payload, "actual_seq_lengths_query", device).to(torch.int32),
        "actual_seq_lengths_kv": to_device(payload, "actual_seq_lengths_kv", device).to(torch.int32),
        "query_rope": to_device(payload, "query_rope", device),
        "key_rope": to_device(payload, "key_rope", device),
    }
    log("COMPARE stage=tensors_ready")

    log(
        "COMPARE metadata "
        f"path={dump_path} rank={payload.get('rank')} layer={payload.get('layer_id')} "
        f"phase={payload.get('phase')} scale={payload.get('scale_value')} "
        f"sparse_count={payload.get('sparse_count')} mask_size={args.mask_size} "
        f"mla_mode={args.mla_mode}"
    )
    for name, tensor in tensors.items():
        log(f"COMPARE tensor {desc(name, tensor)}")
    if args.metadata_only:
        log("COMPARE stage=metadata_only_done")
        return

    sfa_out = None
    if args.skip_sfa_op:
        log("COMPARE stage=skip_sfa_op")
    else:
        log("COMPARE stage=run_sfa")
        sfa_out = run_sfa(payload, tensors)
    if args.mla_mode == "paged":
        log("COMPARE stage=run_dense_mla_paged")
        mla_out = run_dense_mla_paged(
            payload,
            tensors,
            mask_size=args.mask_size,
            kv_lens_mode=args.kv_lens_mode,
        )
    else:
        log("COMPARE stage=run_dense_mla_materialized")
        mla_out = run_dense_mla_materialized(
            payload,
            tensors,
            mask_size=args.mask_size,
            expand_kv_to_heads=args.expand_materialized_kv_to_heads,
        )

    if args.torch_sparse_reference:
        log("COMPARE stage=run_torch_sparse_reference")
        sparse_ref = run_torch_sparse_reference(
            payload,
            tensors,
            chunk_size=args.reference_chunk_size,
        )
        log("COMPARE stage=compare_sparse_reference_to_mla")
        compare_outputs("sparse_ref", sparse_ref, "mla_out", mla_out)
    if sfa_out is not None:
        log("COMPARE stage=compare_sfa_to_mla")
        compare_outputs("sfa_out", sfa_out, "mla_out", mla_out)
    log("COMPARE stage=done")


if __name__ == "__main__":
    main()
