# SPDX-License-Identifier: Apache-2.0

"""Probe vLLM-Ascend SFA decode ops without running a full model prefill.

This mirrors the Nano DeepSeek-V3.2 decode SFA shapes:
  indexer query: [1, 64, 128]
  SFA query:     [1, 32, 512]
  PA cache:      [num_blocks, 128, 1, dim]

Run one sequence length per process. If a native op segfaults, the printed
stage marker tells which op was executing.
"""

from __future__ import annotations

import argparse
import os

os.environ.setdefault("VLLM_ASCEND_ENABLE_NZ", "0")

import torch
import torch_npu  # type: ignore  # noqa: F401
import vllm  # type: ignore  # noqa: F401
import vllm_ascend  # type: ignore  # noqa: F401
from vllm_ascend import vllm_ascend_C  # type: ignore  # noqa: F401
from vllm_ascend.ops.layer_shard_linear import (  # type: ignore  # noqa: F401
    is_hidden_layer,
    post_process_after_loading_for_shard_weight_series,
    reach_layer_for_shard_weight_series,
    register_all_layers_to_shard_weight_series,
)


def _head(tensor: torch.Tensor, limit: int = 16) -> list[int | float]:
    return tensor.flatten()[:limit].detach().cpu().tolist()


def _desc(name: str, tensor: torch.Tensor) -> str:
    return (
        f"{name}=shape={tuple(tensor.shape)} "
        f"dtype={tensor.dtype} device={tensor.device} "
        f"contiguous={tensor.is_contiguous()}"
    )


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
        raise TypeError(f"{name} in dump is not a tensor: {type(value)!r}")
    return value.to(device=device).contiguous()


def replay_dump(path: str, device: torch.device) -> None:
    payload = torch.load(path, map_location="cpu")
    query = _to_device(payload, "query", device)
    key = _to_device(payload, "key", device)
    value = _to_device(payload, "value", device)
    sparse_indices = _to_device(payload, "sparse_indices", device)
    block_table = _to_device(payload, "block_table", device)
    actual_seq_lengths_query = _to_device(
        payload, "actual_seq_lengths_query", device
    )
    actual_seq_lengths_kv = _to_device(payload, "actual_seq_lengths_kv", device)
    query_rope = _to_device(payload, "query_rope", device)
    key_rope = _to_device(payload, "key_rope", device)
    scale_value = float(payload.get("scale_value", 0.1352337788608801))
    sparse_block_size = int(payload.get("sparse_block_size", 1))
    sparse_mode = int(payload.get("sparse_mode", 3))
    layout_query = str(payload.get("layout_query", "TND"))
    layout_kv = str(payload.get("layout_kv", "PA_BSND"))

    print(
        "PROBE dump_metadata "
        f"rank={payload.get('rank')} "
        f"layer_id={payload.get('layer_id')} "
        f"seq_idx={payload.get('seq_idx')} "
        f"scale_value={scale_value}",
        flush=True,
    )
    for name, tensor in (
        ("query", query),
        ("key", key),
        ("value", value),
        ("sparse_indices", sparse_indices),
        ("block_table", block_table),
        ("actual_seq_lengths_query", actual_seq_lengths_query),
        ("actual_seq_lengths_kv", actual_seq_lengths_kv),
        ("query_rope", query_rope),
        ("key_rope", key_rope),
    ):
        print(f"PROBE dump_tensor {_desc(name, tensor)}", flush=True)
        print(f"PROBE dump_stats {_stats(name, tensor.detach().cpu())}", flush=True)

    print(
        "PROBE before_sfa_dump "
        f"{_desc('query', query)} "
        f"{_desc('key', key)} "
        f"{_desc('sparse_indices', sparse_indices)} "
        f"sparse_head={_head(sparse_indices)} "
        f"block_head={_head(block_table)}",
        flush=True,
    )
    out = torch.ops._C_ascend.npu_sparse_flash_attention(
        query=query,
        key=key,
        value=value,
        sparse_indices=sparse_indices,
        scale_value=scale_value,
        sparse_block_size=sparse_block_size,
        actual_seq_lengths_query=actual_seq_lengths_query,
        actual_seq_lengths_kv=actual_seq_lengths_kv,
        block_table=block_table,
        query_rope=query_rope,
        key_rope=key_rope,
        layout_query=layout_query,
        layout_kv=layout_kv,
        sparse_mode=sparse_mode,
    )
    if isinstance(out, tuple):
        out = out[0]
    torch.npu.synchronize()
    print(f"PROBE after_sfa_dump {_desc('out', out)}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq-len", type=int, default=2048)
    parser.add_argument("--cache-blocks", type=int, default=118)
    parser.add_argument("--block-cols", type=int, default=64)
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument("--sparse-count", type=int, default=2048)
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--dump-path",
        help="Replay a Nano NANOVLLM_DUMP_NPU_SFA_INPUTS .pt file.",
    )
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    if args.dump_path:
        replay_dump(args.dump_path, device)
        return

    blocks_needed = (args.seq_len + args.block_size - 1) // args.block_size
    if blocks_needed + 1 > args.cache_blocks:
        raise ValueError(
            "cache-blocks must include block 0 plus all sequence blocks: "
            f"need at least {blocks_needed + 1}, got {args.cache_blocks}"
        )
    if blocks_needed > args.block_cols:
        raise ValueError(
            f"block-cols must be >= {blocks_needed}, got {args.block_cols}"
        )

    q_index = torch.randn(
        1, 64, 128, dtype=torch.bfloat16, device=device
    )
    index_cache = torch.randn(
        args.cache_blocks,
        args.block_size,
        1,
        128,
        dtype=torch.bfloat16,
        device=device,
    )
    weights = torch.randn(1, 64, dtype=torch.bfloat16, device=device)
    ql_nope = torch.randn(
        1, 32, 512, dtype=torch.bfloat16, device=device
    )
    q_pe = torch.randn(1, 32, 64, dtype=torch.bfloat16, device=device)
    kv = torch.randn(
        args.cache_blocks,
        args.block_size,
        1,
        512,
        dtype=torch.bfloat16,
        device=device,
    )
    k_pe = torch.randn(
        args.cache_blocks,
        args.block_size,
        1,
        64,
        dtype=torch.bfloat16,
        device=device,
    )
    block_table = torch.zeros(
        1, args.block_cols, dtype=torch.int32, device=device
    )
    block_table[0, :blocks_needed] = torch.arange(
        1, blocks_needed + 1, dtype=torch.int32, device=device
    )
    actual_seq_lengths_query = torch.ones(
        1, dtype=torch.int32, device=device
    )
    actual_seq_lengths_key = torch.tensor(
        [args.seq_len], dtype=torch.int32, device=device
    )

    print(
        "PROBE before_indexer "
        f"seq_len={args.seq_len} "
        f"blocks_needed={blocks_needed} "
        f"{_desc('query', q_index)} "
        f"{_desc('key', index_cache)} "
        f"{_desc('block_table', block_table)} "
        f"block_head={_head(block_table)}",
        flush=True,
    )
    topk = torch.ops._C_ascend.npu_lightning_indexer(
        query=q_index,
        key=index_cache,
        weights=weights,
        actual_seq_lengths_query=actual_seq_lengths_query,
        actual_seq_lengths_key=actual_seq_lengths_key,
        block_table=block_table,
        layout_query="TND",
        layout_key="PA_BSND",
        sparse_count=args.sparse_count,
        sparse_mode=3,
    )
    if isinstance(topk, tuple):
        topk = topk[0]
    torch.npu.synchronize()
    valid_topk = int((topk >= 0).sum().item())
    print(
        "PROBE after_indexer "
        f"{_desc('topk', topk)} valid_topk={valid_topk} "
        f"topk_head={_head(topk)}",
        flush=True,
    )

    print(
        "PROBE before_sfa "
        f"{_desc('query', ql_nope)} "
        f"{_desc('key', kv)} "
        f"{_desc('sparse_indices', topk)} "
        f"{_desc('query_rope', q_pe)} "
        f"{_desc('key_rope', k_pe)}",
        flush=True,
    )
    out = torch.ops._C_ascend.npu_sparse_flash_attention(
        query=ql_nope,
        key=kv,
        value=kv,
        sparse_indices=topk,
        scale_value=0.1352337788608801,
        sparse_block_size=1,
        actual_seq_lengths_query=actual_seq_lengths_query,
        actual_seq_lengths_kv=actual_seq_lengths_key,
        block_table=block_table,
        query_rope=q_pe,
        key_rope=k_pe,
        layout_query="TND",
        layout_kv="PA_BSND",
        sparse_mode=3,
    )
    if isinstance(out, tuple):
        out = out[0]
    torch.npu.synchronize()
    print(f"PROBE after_sfa {_desc('out', out)}", flush=True)


if __name__ == "__main__":
    main()
