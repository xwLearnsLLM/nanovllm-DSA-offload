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
        f"dtype={tensor.dtype} device={tensor.device}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq-len", type=int, default=2048)
    parser.add_argument("--cache-blocks", type=int, default=118)
    parser.add_argument("--block-cols", type=int, default=64)
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument("--sparse-count", type=int, default=2048)
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device(args.device)
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
