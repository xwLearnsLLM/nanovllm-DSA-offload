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
import glob
import os

os.environ.setdefault("VLLM_ASCEND_ENABLE_NZ", "0")

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
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


def _sparse_indices_report(
    name: str,
    sparse_indices: torch.Tensor,
    actual_seq_lengths_query: torch.Tensor | None = None,
    actual_seq_lengths_kv: torch.Tensor | None = None,
    row_limit: int = 8,
) -> str:
    indices = sparse_indices.detach().cpu()
    if indices.numel() == 0:
        return f"{name}_report empty=True"
    if indices.ndim != 3:
        return (
            f"{name}_report ndim={indices.ndim} "
            f"negative_count={int((indices < 0).sum().item())}"
        )

    valid = indices >= 0
    negative_count = int((~valid).sum().item())
    valid_counts = valid.sum(dim=-1)
    seen_negative = (~valid).to(torch.int32).cumsum(dim=-1) > 0
    valid_after_negative = bool((valid & seen_negative).any().item())
    valid_values = indices[valid]
    if valid_values.numel():
        valid_min = int(valid_values.min().item())
        valid_max = int(valid_values.max().item())
    else:
        valid_min = None
        valid_max = None

    parts = [
        f"{name}_report shape={tuple(indices.shape)}",
        f"negative_count={negative_count}",
        f"valid_min={valid_min}",
        f"valid_max={valid_max}",
        f"valid_after_negative={valid_after_negative}",
        f"valid_count_head={valid_counts[:row_limit, 0].tolist()}",
    ]

    if actual_seq_lengths_query is not None and actual_seq_lengths_kv is not None:
        q_ends = actual_seq_lengths_query.detach().cpu().tolist()
        kv_lens = actual_seq_lengths_kv.detach().cpu().tolist()
        q_starts = [0] + [int(x) for x in q_ends[:-1]]
        q_ends = [int(x) for x in q_ends]
        if q_ends and q_ends[-1] == indices.shape[0] and len(q_ends) == len(kv_lens):
            q_lens = [end - start for start, end in zip(q_starts, q_ends)]
            parts.append(f"query_lens={q_lens[:row_limit]}")
            parts.append(f"kv_lens={kv_lens[:row_limit]}")
            if all(int(q_len) == int(kv_len) for q_len, kv_len in zip(q_lens, kv_lens)):
                mismatch_rows = 0
                checked_rows = 0
                for seq_idx, (start, end) in enumerate(zip(q_starts, q_ends)):
                    kv_len = int(kv_lens[seq_idx])
                    for row in range(start, end):
                        local_pos = row - start
                        expected = min(local_pos + 1, kv_len, indices.shape[-1])
                        row_counts = valid_counts[row]
                        checked_rows += 1
                        if bool((row_counts != expected).any().item()):
                            mismatch_rows += 1
                parts.append(f"prefill_causal_checked_rows={checked_rows}")
                parts.append(f"prefill_causal_mismatch_rows={mismatch_rows}")

    row_samples = []
    for row in range(min(row_limit, indices.shape[0])):
        row_samples.append(indices[row, 0, : min(16, indices.shape[-1])].tolist())
    parts.append(f"row_head_samples={row_samples}")
    return " ".join(parts)


def _to_device(payload: dict, name: str, device: torch.device) -> torch.Tensor:
    value = payload[name]
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} in dump is not a tensor: {type(value)!r}")
    return value.to(device=device).contiguous()


def _load_dump(path: str, device: torch.device) -> tuple[dict, dict[str, torch.Tensor]]:
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
        "actual_seq_lengths_kv": _to_device(
            payload, "actual_seq_lengths_kv", device
        ),
        "query_rope": _to_device(payload, "query_rope", device),
        "key_rope": _to_device(payload, "key_rope", device),
    }
    return payload, tensors


def _print_dump(payload: dict, tensors: dict[str, torch.Tensor]) -> None:
    scale_value = float(payload.get("scale_value", 0.1352337788608801))

    print(
        "PROBE dump_metadata "
        f"rank={payload.get('rank')} "
        f"layer_id={payload.get('layer_id')} "
        f"seq_idx={payload.get('seq_idx')} "
        f"scale_value={scale_value}",
        flush=True,
    )
    for name in (
        "query",
        "key",
        "value",
        "sparse_indices",
        "block_table",
        "actual_seq_lengths_query",
        "actual_seq_lengths_kv",
        "query_rope",
        "key_rope",
    ):
        tensor = tensors[name]
        print(f"PROBE dump_tensor {_desc(name, tensor)}", flush=True)
        print(f"PROBE dump_stats {_stats(name, tensor.detach().cpu())}", flush=True)
    print(
        "PROBE dump_sparse_indices "
        + _sparse_indices_report(
            "sparse_indices",
            tensors["sparse_indices"],
            tensors["actual_seq_lengths_query"],
            tensors["actual_seq_lengths_kv"],
        ),
        flush=True,
    )


def _run_sfa_dump(
    payload: dict,
    tensors: dict[str, torch.Tensor],
    prefix: str = "PROBE",
) -> torch.Tensor:
    scale_value = float(payload.get("scale_value", 0.1352337788608801))
    sparse_block_size = int(payload.get("sparse_block_size", 1))
    sparse_mode = int(payload.get("sparse_mode", 3))
    layout_query = str(payload.get("layout_query", "TND"))
    layout_kv = str(payload.get("layout_kv", "PA_BSND"))
    query = tensors["query"]
    key = tensors["key"]
    value = tensors["value"]
    sparse_indices = tensors["sparse_indices"]
    block_table = tensors["block_table"]
    actual_seq_lengths_query = tensors["actual_seq_lengths_query"]
    actual_seq_lengths_kv = tensors["actual_seq_lengths_kv"]
    query_rope = tensors["query_rope"]
    key_rope = tensors["key_rope"]

    print(
        f"{prefix} before_sfa_dump "
        f"{_desc('query', query)} "
        f"{_desc('key', key)} "
        f"{_desc('sparse_indices', sparse_indices)} "
        f"sparse_head={_head(sparse_indices)} "
        f"block_head={_head(block_table)}",
        flush=True,
    )
    print(
        f"{prefix} before_sfa_dump "
        + _sparse_indices_report(
            "sparse_indices",
            sparse_indices,
            actual_seq_lengths_query,
            actual_seq_lengths_kv,
        ),
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
    print(f"{prefix} after_sfa_dump {_desc('out', out)}", flush=True)
    return out


def replay_dump(path: str, device: torch.device) -> None:
    payload, tensors = _load_dump(path, device)
    _print_dump(payload, tensors)
    _run_sfa_dump(payload, tensors)


def _dump_path_for_rank(dump_dir: str, rank: int) -> str:
    pattern = os.path.join(dump_dir, f"sfa_rank{rank}_layer*_seq*.pt")
    matches = sorted(glob.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"no dump file matched {pattern}")
    return matches[0]


def _worker_replay_dump_dir(
    rank: int,
    world_size: int,
    dump_dir: str,
    init_method: str,
    use_dist: bool,
    repeats: int,
) -> None:
    device = torch.device(f"npu:{rank}")
    torch.npu.set_device(rank)
    if use_dist:
        dist.init_process_group(
            backend="hccl",
            init_method=init_method,
            world_size=world_size,
            rank=rank,
        )
    path = _dump_path_for_rank(dump_dir, rank)
    payload, tensors = _load_dump(path, device)
    print(
        f"PROBE_MP rank={rank} loaded path={path} "
        f"query={tuple(tensors['query'].shape)} key={tuple(tensors['key'].shape)}",
        flush=True,
    )
    for i in range(repeats):
        if use_dist:
            dist.barrier()
        _run_sfa_dump(payload, tensors, prefix=f"PROBE_MP rank={rank} iter={i}")
        if use_dist:
            dist.barrier()
    if use_dist:
        dist.destroy_process_group()
    print(f"PROBE_MP rank={rank} done", flush=True)


def replay_dump_dir(
    dump_dir: str,
    world_size: int,
    hccl_port: int,
    use_dist: bool,
    repeats: int,
) -> None:
    init_method = f"tcp://127.0.0.1:{hccl_port}"
    mp.spawn(
        _worker_replay_dump_dir,
        args=(world_size, dump_dir, init_method, use_dist, repeats),
        nprocs=world_size,
        join=True,
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
    parser.add_argument(
        "--dump-path",
        help="Replay a Nano NANOVLLM_DUMP_NPU_SFA_INPUTS .pt file.",
    )
    parser.add_argument(
        "--dump-dir",
        help="Replay one dumped file per rank concurrently.",
    )
    parser.add_argument("--world-size", type=int, default=4)
    parser.add_argument("--hccl-port", type=int, default=28089)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument(
        "--no-dist",
        action="store_true",
        help="Do not initialize HCCL for --dump-dir replay.",
    )
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    if args.dump_dir:
        replay_dump_dir(
            args.dump_dir,
            args.world_size,
            args.hccl_port,
            not args.no_dist,
            args.repeats,
        )
        return
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
