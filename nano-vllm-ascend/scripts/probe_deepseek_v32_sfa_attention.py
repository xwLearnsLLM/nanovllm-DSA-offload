from __future__ import annotations

"""Probe DeepSeek-V3.2 SFA ops without running the whole model.

The script mirrors the Nano attention tensors:
  ql_nope:     [T, local_heads, kv_lora_rank]
  q_pe:        [T, local_heads, qk_rope_head_dim]
  ckv_cache:   [num_blocks, block_size, 1, kv_lora_rank]
  kpe_cache:   [num_blocks, block_size, 1, qk_rope_head_dim]
  index_cache: [num_blocks, block_size, 1, index_head_dim]

It is intended to run on the Ascend machine. Use --replay-dump with files
created by NANOVLLM_DUMP_NPU_SFA_INPUTS to reproduce Nano inputs exactly.
"""

import argparse
import os
from time import perf_counter

import torch


os.environ.setdefault("VLLM_ASCEND_ENABLE_NZ", "0")


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
        print(f"PROBE warning import_kernels failed: {exc!r}", flush=True)
    _dedupe_env_path("ASCEND_CUSTOM_OPP_PATH")
    return custom_opp_path if os.path.exists(custom_opp_path) else None


def _register_ascend_ops() -> None:
    import torch_npu  # type: ignore  # noqa: F401
    import vllm  # type: ignore  # noqa: F401
    import vllm_ascend  # type: ignore  # noqa: F401

    custom_opp_path = _ensure_vllm_ascend_custom_opp_path(vllm_ascend)
    from vllm_ascend import vllm_ascend_C  # type: ignore  # noqa: F401
    from vllm_ascend.ops.layer_shard_linear import (  # type: ignore  # noqa: F401
        is_hidden_layer,
        post_process_after_loading_for_shard_weight_series,
        reach_layer_for_shard_weight_series,
        register_all_layers_to_shard_weight_series,
    )

    print(
        "PROBE env "
        f"torch_npu={getattr(torch_npu, '__file__', None)} "
        f"vllm={getattr(vllm, '__file__', None)} "
        f"vllm_ascend={getattr(vllm_ascend, '__file__', None)} "
        f"custom_opp_path={custom_opp_path} "
        f"ASCEND_CUSTOM_OPP_PATH={os.environ.get('ASCEND_CUSTOM_OPP_PATH', '')}",
        flush=True,
    )


def _desc(name: str, tensor: torch.Tensor) -> str:
    return (
        f"{name}=shape={tuple(tensor.shape)} dtype={tensor.dtype} "
        f"device={tensor.device} contiguous={tensor.is_contiguous()}"
    )


def _head(tensor: torch.Tensor, limit: int = 16) -> list[int | float]:
    if tensor.numel() == 0:
        return []
    return tensor.flatten()[:limit].detach().cpu().tolist()


def _stats(name: str, tensor: torch.Tensor) -> str:
    if not torch.is_floating_point(tensor):
        return (
            f"{name}: min={int(tensor.min().item())} "
            f"max={int(tensor.max().item())}"
        )
    value = tensor.float()
    finite = torch.isfinite(value)
    finite_count = int(finite.sum().item())
    total = int(value.numel())
    if finite_count == 0:
        return f"{name}: finite=0/{total}"
    finite_value = value[finite]
    return (
        f"{name}: finite={finite_count}/{total} "
        f"min={float(finite_value.min().item()):.6g} "
        f"max={float(finite_value.max().item()):.6g}"
    )


def _to_device(payload: dict, name: str, device: torch.device) -> torch.Tensor:
    value = payload[name]
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} is not a tensor: {type(value)!r}")
    return value.to(device=device).contiguous()


def _call_ascend_indexer(
    *,
    q_index: torch.Tensor,
    index_cache: torch.Tensor,
    weights: torch.Tensor,
    actual_seq_lengths_query: torch.Tensor,
    actual_seq_lengths_key: torch.Tensor,
    block_table: torch.Tensor,
    sparse_count: int,
) -> torch.Tensor:
    return torch.ops._C_ascend.npu_lightning_indexer(
        query=q_index,
        key=index_cache,
        weights=weights,
        actual_seq_lengths_query=actual_seq_lengths_query,
        actual_seq_lengths_key=actual_seq_lengths_key,
        block_table=block_table,
        layout_query="TND",
        layout_key="PA_BSND",
        sparse_count=sparse_count,
        sparse_mode=3,
    )


def _run_ops(
    *,
    q_index: torch.Tensor,
    weights: torch.Tensor,
    ql_nope: torch.Tensor,
    q_pe: torch.Tensor,
    ckv_cache: torch.Tensor,
    kpe_cache: torch.Tensor,
    index_cache: torch.Tensor,
    block_table: torch.Tensor,
    actual_seq_lengths_query: torch.Tensor,
    actual_seq_lengths_key: torch.Tensor,
    sparse_count: int,
    scale_value: float,
) -> torch.Tensor:
    print(
        "PROBE before_indexer "
        f"{_desc('q_index', q_index)} "
        f"{_desc('index_cache', index_cache)} "
        f"{_desc('weights', weights)} "
        f"{_desc('actual_seq_lengths_query', actual_seq_lengths_query)} "
        f"{_desc('actual_seq_lengths_key', actual_seq_lengths_key)} "
        f"{_desc('block_table', block_table)} "
        f"block_head={_head(block_table)}",
        flush=True,
    )
    start = perf_counter()
    topk = _call_ascend_indexer(
        q_index=q_index,
        index_cache=index_cache,
        weights=weights,
        actual_seq_lengths_query=actual_seq_lengths_query,
        actual_seq_lengths_key=actual_seq_lengths_key,
        block_table=block_table,
        sparse_count=sparse_count,
    )
    if isinstance(topk, tuple):
        topk = topk[0]
    torch.npu.synchronize()
    print(
        "PROBE after_indexer "
        f"elapsed={perf_counter() - start:.4f}s "
        f"{_desc('topk', topk)} topk_head={_head(topk)} "
        f"{_stats('topk', topk.detach().cpu())}",
        flush=True,
    )

    print(
        "PROBE before_sfa "
        f"{_desc('ql_nope', ql_nope)} "
        f"{_desc('ckv_cache', ckv_cache)} "
        f"{_desc('q_pe', q_pe)} "
        f"{_desc('kpe_cache', kpe_cache)} "
        f"{_desc('topk', topk)}",
        flush=True,
    )
    start = perf_counter()
    out = torch.ops._C_ascend.npu_sparse_flash_attention(
        query=ql_nope,
        key=ckv_cache,
        value=ckv_cache,
        sparse_indices=topk.to(torch.int32).contiguous(),
        scale_value=scale_value,
        sparse_block_size=1,
        actual_seq_lengths_query=actual_seq_lengths_query,
        actual_seq_lengths_kv=actual_seq_lengths_key,
        block_table=block_table,
        query_rope=q_pe,
        key_rope=kpe_cache,
        layout_query="TND",
        layout_kv="PA_BSND",
        sparse_mode=3,
    )
    if isinstance(out, tuple):
        out = out[0]
    torch.npu.synchronize()
    print(
        "PROBE after_sfa "
        f"elapsed={perf_counter() - start:.4f}s "
        f"{_desc('out', out)} {_stats('out', out.detach().cpu())}",
        flush=True,
    )
    return out


def replay_dump(path: str, device: torch.device) -> None:
    payload = torch.load(path, map_location="cpu")
    tensors = {
        "query": _to_device(payload, "query", device),
        "key": _to_device(payload, "key", device),
        "value": _to_device(payload, "value", device),
        "sparse_indices": _to_device(payload, "sparse_indices", device),
        "block_table": _to_device(payload, "block_table", device),
        "actual_seq_lengths_query": _to_device(
            payload, "actual_seq_lengths_query", device
        ),
        "actual_seq_lengths_key": _to_device(payload, "actual_seq_lengths_kv", device),
        "query_rope": _to_device(payload, "query_rope", device),
        "key_rope": _to_device(payload, "key_rope", device),
    }
    print(
        "PROBE replay_metadata "
        f"path={path} phase={payload.get('phase')} "
        f"rank={payload.get('rank')} layer={payload.get('layer_id')} "
        f"scale={payload.get('scale_value')}",
        flush=True,
    )
    for name, tensor in tensors.items():
        print(f"PROBE replay_tensor {_desc(name, tensor)}", flush=True)

    start = perf_counter()
    out = torch.ops._C_ascend.npu_sparse_flash_attention(
        query=tensors["query"],
        key=tensors["key"],
        value=tensors["value"],
        sparse_indices=tensors["sparse_indices"].to(torch.int32).contiguous(),
        scale_value=float(payload.get("scale_value", 0.1352337788608801)),
        sparse_block_size=int(payload.get("sparse_block_size", 1)),
        actual_seq_lengths_query=tensors["actual_seq_lengths_query"],
        actual_seq_lengths_kv=tensors["actual_seq_lengths_key"],
        block_table=tensors["block_table"],
        query_rope=tensors["query_rope"],
        key_rope=tensors["key_rope"],
        layout_query=str(payload.get("layout_query", "TND")),
        layout_kv=str(payload.get("layout_kv", "PA_BSND")),
        sparse_mode=int(payload.get("sparse_mode", 3)),
    )
    if isinstance(out, tuple):
        out = out[0]
    torch.npu.synchronize()
    print(
        "PROBE replay_after_sfa "
        f"elapsed={perf_counter() - start:.4f}s "
        f"{_desc('out', out)} {_stats('out', out.detach().cpu())}",
        flush=True,
    )


def build_random_case(args: argparse.Namespace, device: torch.device) -> dict:
    dtype = getattr(torch, args.dtype)
    blocks_per_seq = (args.seq_len + args.block_size - 1) // args.block_size
    block_cols = args.block_cols or blocks_per_seq
    if block_cols < blocks_per_seq:
        raise ValueError("--block-cols must cover the sequence length.")
    needed_blocks = args.block_id_base + args.batch_size * blocks_per_seq
    num_blocks = args.num_blocks or needed_blocks
    if num_blocks < needed_blocks:
        raise ValueError("--num-blocks is too small for the requested batch.")

    block_table = torch.zeros(
        args.batch_size,
        block_cols,
        dtype=torch.int32,
        device=device,
    )
    for batch_idx in range(args.batch_size):
        start = args.block_id_base + batch_idx * blocks_per_seq
        block_table[batch_idx, :blocks_per_seq] = torch.arange(
            start,
            start + blocks_per_seq,
            dtype=torch.int32,
            device=device,
        )

    if args.phase == "prefill":
        num_query_tokens = args.batch_size * args.seq_len
        actual_seq_lengths_query = torch.arange(
            args.seq_len,
            args.seq_len * args.batch_size + 1,
            args.seq_len,
            dtype=torch.int32,
            device=device,
        )
    else:
        num_query_tokens = args.batch_size
        actual_seq_lengths_query = torch.arange(
            1,
            args.batch_size + 1,
            dtype=torch.int32,
            device=device,
        )
    actual_seq_lengths_key = torch.full(
        (args.batch_size,),
        args.seq_len,
        dtype=torch.int32,
        device=device,
    )

    return {
        "q_index": torch.randn(
            num_query_tokens,
            args.index_heads,
            args.index_head_dim,
            dtype=dtype,
            device=device,
        ),
        "weights": torch.randn(
            num_query_tokens,
            args.index_heads,
            dtype=dtype,
            device=device,
        ),
        "ql_nope": torch.randn(
            num_query_tokens,
            args.local_heads,
            args.kv_lora_rank,
            dtype=dtype,
            device=device,
        ),
        "q_pe": torch.randn(
            num_query_tokens,
            args.local_heads,
            args.qk_rope_head_dim,
            dtype=dtype,
            device=device,
        ),
        "ckv_cache": torch.randn(
            num_blocks,
            args.block_size,
            1,
            args.kv_lora_rank,
            dtype=dtype,
            device=device,
        ),
        "kpe_cache": torch.randn(
            num_blocks,
            args.block_size,
            1,
            args.qk_rope_head_dim,
            dtype=dtype,
            device=device,
        ),
        "index_cache": torch.randn(
            num_blocks,
            args.block_size,
            1,
            args.index_head_dim,
            dtype=dtype,
            device=device,
        ),
        "block_table": block_table,
        "actual_seq_lengths_query": actual_seq_lengths_query,
        "actual_seq_lengths_key": actual_seq_lengths_key,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("prefill", "decode"), default="prefill")
    parser.add_argument("--seq-len", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument("--block-cols", type=int)
    parser.add_argument("--num-blocks", type=int)
    parser.add_argument("--block-id-base", type=int, default=1)
    parser.add_argument("--sparse-count", type=int, default=2048)
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--local-heads", type=int, default=32)
    parser.add_argument("--kv-lora-rank", type=int, default=512)
    parser.add_argument("--qk-rope-head-dim", type=int, default=64)
    parser.add_argument("--index-heads", type=int, default=64)
    parser.add_argument("--index-head-dim", type=int, default=128)
    parser.add_argument("--scale-value", type=float, default=0.1352337788608801)
    parser.add_argument("--replay-dump")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _register_ascend_ops()
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    torch.npu.set_device(device.index if device.index is not None else 0)
    if args.replay_dump:
        replay_dump(args.replay_dump, device)
        return

    tensors = build_random_case(args, device)
    _run_ops(
        q_index=tensors["q_index"],
        weights=tensors["weights"],
        ql_nope=tensors["ql_nope"],
        q_pe=tensors["q_pe"],
        ckv_cache=tensors["ckv_cache"],
        kpe_cache=tensors["kpe_cache"],
        index_cache=tensors["index_cache"],
        block_table=tensors["block_table"],
        actual_seq_lengths_query=tensors["actual_seq_lengths_query"],
        actual_seq_lengths_key=tensors["actual_seq_lengths_key"],
        sparse_count=args.sparse_count,
        scale_value=args.scale_value,
    )


if __name__ == "__main__":
    main()
