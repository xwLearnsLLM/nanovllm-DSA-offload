import argparse
import glob
import os
from pathlib import Path
from time import perf_counter

import torch
import torch.nn.functional as F
import torch_npu


os.environ.setdefault("VLLM_ASCEND_ENABLE_NZ", "0")


def _prepend_env_path(name: str, path: str) -> None:
    current = os.environ.get(name, "")
    parts = [part for part in current.split(os.pathsep) if part]
    if path not in parts:
        os.environ[name] = f"{path}{os.pathsep}{current}" if current else path


def register_ascend_ops() -> None:
    import vllm  # type: ignore  # noqa: F401
    import vllm_ascend  # type: ignore

    package_dir = os.path.dirname(os.path.realpath(vllm_ascend.__file__))
    custom_opp_path = os.path.join(
        package_dir,
        "_cann_ops_custom",
        "vendors",
        "vllm-ascend",
    )
    if os.path.exists(custom_opp_path):
        _prepend_env_path("ASCEND_CUSTOM_OPP_PATH", custom_opp_path)
    from vllm_ascend import vllm_ascend_C  # type: ignore  # noqa: F401
    from vllm_ascend.ops.layer_shard_linear import (  # type: ignore  # noqa: F401
        is_hidden_layer,
        post_process_after_loading_for_shard_weight_series,
        reach_layer_for_shard_weight_series,
        register_all_layers_to_shard_weight_series,
    )


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
    print(f"COMPARE after_sfa elapsed={perf_counter() - start:.6f}s {desc('sfa_out', out)}")
    return out


def run_dense_mla(
    payload: dict,
    tensors: dict[str, torch.Tensor],
    mask_size: int,
    expand_kv_to_heads: bool,
) -> torch.Tensor:
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
    print(
        "COMPARE dense_inputs "
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
    print(
        f"COMPARE after_dense_mla elapsed={perf_counter() - start:.6f}s "
        f"{desc('mla_out', out)} {desc('mla_lse', lse)}"
    )
    return out


def compare_outputs(sfa_out: torch.Tensor, mla_out: torch.Tensor) -> None:
    sfa = sfa_out.float()
    mla = mla_out.float()
    diff = sfa - mla
    abs_diff = diff.abs()
    sfa_flat = sfa.flatten()
    mla_flat = mla.flatten()
    rel_l2 = torch.linalg.vector_norm(diff.flatten()) / torch.linalg.vector_norm(mla_flat).clamp_min(1e-6)
    cosine = F.cosine_similarity(sfa_flat, mla_flat, dim=0)
    print(
        "COMPARE diff "
        f"max_abs={abs_diff.max().item():.6g} "
        f"mean_abs={abs_diff.mean().item():.6g} "
        f"rms={diff.pow(2).mean().sqrt().item():.6g} "
        f"rel_l2={rel_l2.item():.6g} "
        f"cosine={cosine.item():.8f}"
    )
    print(f"COMPARE {stats('sfa_out', sfa_out)}")
    print(f"COMPARE {stats('mla_out', mla_out)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare SFA and dense MLA prefill from a Nano SFA dump.")
    parser.add_argument("--dump")
    parser.add_argument("--dump-glob")
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--mask-size", type=int, default=2048)
    parser.add_argument("--no-expand-kv-to-heads", action="store_true")
    args = parser.parse_args()

    device_index = int(str(args.device).split(":")[-1])
    torch.npu.set_device(device_index)
    torch.npu.config.allow_internal_format = True
    register_ascend_ops()

    dump_path = pick_dump(args.dump, args.dump_glob)
    payload = torch.load(dump_path, map_location="cpu")
    if payload.get("phase") != "prefill":
        raise ValueError(f"only prefill dumps are supported, got phase={payload.get('phase')!r}")

    device = torch.device(args.device)
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

    print(
        "COMPARE metadata "
        f"path={dump_path} rank={payload.get('rank')} layer={payload.get('layer_id')} "
        f"phase={payload.get('phase')} scale={payload.get('scale_value')} "
        f"sparse_count={payload.get('sparse_count')} mask_size={args.mask_size}"
    )
    for name, tensor in tensors.items():
        print(f"COMPARE tensor {desc(name, tensor)}")

    sfa_out = run_sfa(payload, tensors)
    mla_out = run_dense_mla(
        payload,
        tensors,
        mask_size=args.mask_size,
        expand_kv_to_heads=not args.no_expand_kv_to_heads,
    )
    compare_outputs(sfa_out, mla_out)


if __name__ == "__main__":
    main()
