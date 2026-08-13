"""Validate GLM-5.2 MTP IndexShare with 18-bit logical SFA slots."""

from __future__ import annotations

import argparse
import math

import torch
import torch_npu  # type: ignore

import nanovllm.ops  # noqa: F401  # Load repository-local custom operators.
from ut_ops._op_utils import require_local_opapi


BLOCK_SIZE = 128
INDEX_HEADS = 32
INDEX_DIM = 128
CKV_DIM = 512
KPE_DIM = 64
TOPK = 2048


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate official LI token IDs as direct MTP SFA logical slots."
        )
    )
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--heads", type=int, default=2)
    parser.add_argument("--source-len", type=int, default=20992)
    parser.add_argument("--tail-tokens", type=int, default=64)
    parser.add_argument("--graph-replays", type=int, default=3)
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.batch_size <= 0 or args.heads <= 0:
        raise ValueError("--batch-size and --heads must be positive")
    if args.source_len <= 16384 or args.source_len > 1 << 18:
        raise ValueError("--source-len must be in (16384, 262144]")
    if args.source_len % BLOCK_SIZE:
        raise ValueError("--source-len must be divisible by 128")
    if args.tail_tokens < 0:
        raise ValueError("--tail-tokens must be nonnegative")
    if args.graph_replays < 1:
        raise ValueError("--graph-replays must be positive")


def call_li(
    query: torch.Tensor,
    key: torch.Tensor,
    weights: torch.Tensor,
    lengths: torch.Tensor,
    block_table: torch.Tensor,
) -> torch.Tensor:
    query_ends = torch.arange(
        1,
        query.shape[0] + 1,
        dtype=torch.int32,
        device=query.device,
    )
    result = torch_npu.npu_lightning_indexer(
        query=query,
        key=key,
        weights=weights,
        actual_seq_lengths_query=query_ends,
        actual_seq_lengths_key=lengths,
        block_table=block_table,
        layout_query="TND",
        layout_key="PA_BSND",
        sparse_count=TOPK,
        sparse_mode=3,
    )
    topk = result[0] if isinstance(result, (tuple, list)) else result
    if not isinstance(topk, torch.Tensor):
        raise TypeError("official LightningIndexer did not return a Tensor")
    return topk


def call_sfa(
    query_rope: torch.Tensor,
    query: torch.Tensor,
    actual_q: torch.Tensor,
    actual_kv: torch.Tensor,
    source_lens: torch.Tensor,
    topk_slots: torch.Tensor,
    block_table: torch.Tensor,
    kpe_cache: torch.Tensor,
    ckv_cache: torch.Tensor,
    output: torch.Tensor,
) -> None:
    torch.ops.nanovllm_dsa.sparse_tail_attention.default(
        query_rope,
        query,
        actual_q,
        actual_kv,
        source_lens,
        topk_slots,
        block_table,
        kpe_cache,
        ckv_cache,
        1.0 / math.sqrt(CKV_DIM + KPE_DIM),
        output,
    )


def logical_rows(
    cache: torch.Tensor,
    block_table: torch.Tensor,
    request: int,
    logical_slots: torch.Tensor,
) -> torch.Tensor:
    slots = logical_slots.to(torch.int64)
    blocks = block_table[request, slots // BLOCK_SIZE].to(torch.int64)
    return cache[blocks, slots % BLOCK_SIZE, 0]


def main() -> None:
    args = parse_args()
    validate_args(args)
    opapi = require_local_opapi()
    print(f"MTP_INDEX_SHARE_OPAPI path={opapi} local=1")
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    batch_size = args.batch_size
    total_len = args.source_len + args.tail_tokens
    blocks_per_request = math.ceil(total_len / BLOCK_SIZE)
    total_blocks = batch_size * blocks_per_request

    block_table = torch.arange(
        total_blocks, dtype=torch.int32, device=device
    ).view(batch_size, blocks_per_request)
    source_lens = torch.full(
        (batch_size,), args.source_len, dtype=torch.int32, device=device
    )
    actual_kv = torch.full(
        (batch_size,), total_len, dtype=torch.int32, device=device
    )
    actual_q = torch.arange(
        1, batch_size + 1, dtype=torch.int32, device=device
    )

    index_query = torch.randn(
        batch_size,
        INDEX_HEADS,
        INDEX_DIM,
        dtype=torch.bfloat16,
        device=device,
    )
    index_weights = torch.randn(
        batch_size, INDEX_HEADS, dtype=torch.bfloat16, device=device
    )
    index_cache = torch.randn(
        total_blocks,
        BLOCK_SIZE,
        1,
        INDEX_DIM,
        dtype=torch.bfloat16,
        device=device,
    )
    query = torch.randn(
        batch_size, args.heads, CKV_DIM,
        dtype=torch.bfloat16, device=device,
    )
    query_rope = torch.randn(
        batch_size, args.heads, KPE_DIM,
        dtype=torch.bfloat16, device=device,
    )
    ckv_cache = torch.randn(
        total_blocks, BLOCK_SIZE, 1, CKV_DIM,
        dtype=torch.bfloat16, device=device,
    )
    kpe_cache = torch.randn(
        total_blocks, BLOCK_SIZE, 1, KPE_DIM,
        dtype=torch.bfloat16, device=device,
    )
    attention_out = torch.empty_like(query)

    topk_slots = call_li(
        index_query, index_cache, index_weights, source_lens, block_table
    )
    call_sfa(
        query_rope, query, actual_q, actual_kv, source_lens, topk_slots,
        block_table, kpe_cache, ckv_cache, attention_out,
    )
    torch.npu.synchronize()

    topk_cpu = topk_slots.cpu().to(torch.int64)
    topk_max = int(topk_cpu.max())
    if topk_max < 16384:
        raise AssertionError(
            "test workload did not select an 18-bit logical slot: "
            f"topk_max={topk_max}"
        )
    if int(topk_cpu.min()) < 0 or topk_max >= args.source_len:
        raise AssertionError("LightningIndexer returned an invalid source ID")

    golden_rows = []
    scale = 1.0 / math.sqrt(CKV_DIM + KPE_DIM)
    block_table_cpu = block_table.cpu()
    ckv_cpu = ckv_cache.cpu()
    kpe_cpu = kpe_cache.cpu()
    query_cpu = query.float().cpu()
    query_rope_cpu = query_rope.float().cpu()
    for request in range(batch_size):
        logical_slots = topk_cpu[request, 0]
        if args.tail_tokens:
            logical_slots = torch.cat(
                (
                    logical_slots,
                    torch.arange(
                        args.source_len, total_len, dtype=torch.int64
                    ),
                )
            )
        key = logical_rows(
            ckv_cpu, block_table_cpu, request, logical_slots
        ).float()
        key_rope = logical_rows(
            kpe_cpu, block_table_cpu, request, logical_slots
        ).float()
        scores = (
            query_cpu[request] @ key.T
            + query_rope_cpu[request] @ key_rope.T
        ) * scale
        golden_rows.append(torch.softmax(scores, dim=-1) @ key)
    golden = torch.stack(golden_rows)
    actual = attention_out.float().cpu()
    torch.testing.assert_close(actual, golden, rtol=0.08, atol=0.08)
    max_abs = float((actual - golden).abs().max())
    print(
        "MTP_INDEX_SHARE_LOGICAL_SLOT_CHECK "
        f"batch={batch_size} source_len={args.source_len} "
        f"topk_min={int(topk_cpu.min())} topk_max={topk_max} "
        f"crosses_14bit=1 attention_max_abs={max_abs:.6f} ok=1"
    )

    graph_output = torch.empty_like(query)
    graph_topk: list[torch.Tensor] = []

    def launch() -> None:
        selected = call_li(
            index_query,
            index_cache,
            index_weights,
            source_lens,
            block_table,
        )
        graph_topk[:] = [selected]
        call_sfa(
            query_rope, query, actual_q, actual_kv, source_lens, selected,
            block_table, kpe_cache, ckv_cache, graph_output,
        )

    launch()
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph, pool=torch.npu.graph_pool_handle()):
        launch()
    for _ in range(args.graph_replays):
        graph.replay()
    torch.npu.synchronize()
    torch.testing.assert_close(graph_output, attention_out, rtol=0, atol=0)
    if int(graph_topk[0].max()) < 16384:
        raise AssertionError("graph LI output lost high logical source IDs")
    print(
        "MTP_INDEX_SHARE_GRAPH_CHECK "
        f"replays={args.graph_replays} direct_li_to_sfa=1 "
        "crosses_14bit=1 ok=1"
    )
    print("GLM_MTP_INDEX_SHARE_SLOTS_UT_OK")


if __name__ == "__main__":
    main()
