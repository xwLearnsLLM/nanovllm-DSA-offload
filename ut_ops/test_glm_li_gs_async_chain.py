"""Stress the native GLM LightningIndexer -> GatherSelection async boundary.

The tested window repeats LI immediately followed by GS without a CPU read,
event, or device synchronization between them.  It synchronizes exactly once
after every chain has been submitted, then validates the selected token sets
and every gathered KPE/CKV slot against independent CPU data.
"""

from __future__ import annotations

import argparse
import math
import os

import torch
import torch_npu  # type: ignore

import nanovllm.ops as ascend_ops
from nanovllm.models.dsa_indexer_project import (
    gather_selection_kv_cache_eager_dispatch,
)
from ut_ops.test_glm_dsa_indexer import (
    GLM_INDEX_HEAD_DIM,
    GLM_INDEX_HEADS,
    GLM_INDEX_ROPE_DIM,
    make_paged_index_cache,
    swapped_from_cpu,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stress GLM native LI -> custom GS queue ordering."
    )
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--full-len", type=int, default=8200)
    parser.add_argument("--topk", type=int, default=2048)
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument("--rounds", type=int, default=78)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--consumer",
        choices=("dispatcher", "pybind"),
        default="dispatcher",
    )
    parser.add_argument(
        "--sync-between-li-gs",
        type=int,
        choices=(0, 1),
        default=0,
        help="Positive control only; the real stress mode is 0.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if torch.device(args.device).type != "npu":
        raise ValueError("--device must select one NPU, for example npu:0.")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive.")
    if args.full_len <= 0:
        raise ValueError("--full-len must be positive.")
    if args.block_size != 128:
        raise ValueError("GatherSelection async stress currently requires --block-size 128.")
    if args.rounds <= 0:
        raise ValueError("--rounds must be positive.")
    if not 1 <= args.topk <= min(2048, args.full_len):
        raise ValueError(
            "--topk must be in [1, min(2048, full_len)], got "
            f"topk={args.topk}, full_len={args.full_len}."
        )


def run_native_li(
    query: torch.Tensor,
    weights: torch.Tensor,
    key_cache: torch.Tensor,
    query_lens: torch.Tensor,
    key_lens: torch.Tensor,
    block_table: torch.Tensor,
    topk: int,
) -> torch.Tensor:
    result = torch_npu.npu_lightning_indexer(
        query=query,
        key=key_cache,
        weights=weights,
        actual_seq_lengths_query=query_lens,
        actual_seq_lengths_key=key_lens,
        block_table=block_table,
        layout_query="TND",
        layout_key="PA_BSND",
        sparse_count=topk,
        sparse_mode=3,
    )
    output = result[0] if isinstance(result, (tuple, list)) else result
    if not isinstance(output, torch.Tensor):
        raise TypeError(
            "torch_npu.npu_lightning_indexer must return a Tensor or a tuple "
            f"whose first item is a Tensor, got {type(output).__name__}."
        )
    return output


def launch_gs(
    consumer: str,
    selection_kpe: torch.Tensor,
    selection_ckv: torch.Tensor,
    selection_table: torch.Tensor,
    status: torch.Tensor,
    req_pool_entries: torch.Tensor,
    topk_indices: torch.Tensor,
    full_kpe: torch.Tensor,
    full_ckv: torch.Tensor,
    full_table: torch.Tensor,
    full_actual_seq: torch.Tensor,
) -> None:
    args = (
        selection_kpe,
        selection_ckv,
        selection_table,
        status,
        req_pool_entries,
        topk_indices,
        full_kpe,
        full_ckv,
        full_table,
        full_actual_seq,
    )
    if consumer == "dispatcher":
        gather_selection_kv_cache_eager_dispatch(*args)
    else:
        ascend_ops.npu_gather_selection_kv_cache(*args)


def validate_topk_set(
    values: torch.Tensor,
    *,
    full_len: int,
    topk: int,
    label: str,
) -> torch.Tensor:
    values = values.reshape(-1).to(torch.int64)
    if values.numel() != topk:
        raise AssertionError(
            f"{label} size={values.numel()}, expected={topk}."
        )
    minimum = int(values.min().item())
    maximum = int(values.max().item())
    if minimum < 0 or maximum >= full_len:
        raise AssertionError(
            f"{label} contains out-of-range IDs: min={minimum}, max={maximum}, "
            f"valid=[0,{full_len})."
        )
    sorted_values = torch.sort(values).values
    if torch.unique_consecutive(sorted_values).numel() != topk:
        raise AssertionError(f"{label} contains duplicate token IDs.")
    return sorted_values


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
        "GLM_LI_GS_ASYNC_CONFIG "
        f"device={device} batch={args.batch_size} full_len={args.full_len} "
        f"topk={args.topk} block_size={args.block_size} rounds={args.rounds} "
        f"consumer={args.consumer} sync_between_li_gs={args.sync_between_li_gs} "
        f"parallel_copy={os.environ.get('NANOVLLM_GS_PARALLEL_COPY', 'auto')}",
        flush=True,
    )

    cpu_generator = torch.Generator().manual_seed(args.seed + 1)
    queries = torch.randn(
        args.rounds,
        args.batch_size,
        GLM_INDEX_HEADS,
        GLM_INDEX_HEAD_DIM,
        generator=cpu_generator,
        dtype=torch.float32,
    ).to(dtype).to(device)
    weights = torch.randn(
        args.rounds,
        args.batch_size,
        GLM_INDEX_HEADS,
        generator=cpu_generator,
        dtype=torch.float32,
    ).to(dtype).to(device)

    key_cache, full_table = make_paged_index_cache(
        batch_size=args.batch_size,
        full_len=args.full_len,
        block_size=args.block_size,
        dtype=dtype,
        device=device,
    )
    query_lens = torch.arange(
        1,
        args.batch_size + 1,
        dtype=torch.int32,
        device=device,
    )
    key_lens = torch.full(
        (args.batch_size,),
        args.full_len,
        dtype=torch.int32,
        device=device,
    )

    blocks_per_row = math.ceil(args.full_len / args.block_size)
    num_full_blocks = 1 + args.batch_size * blocks_per_row
    full_kpe_cpu = torch.randn(
        num_full_blocks,
        args.block_size,
        GLM_INDEX_ROPE_DIM,
        generator=cpu_generator,
        dtype=torch.float32,
    ).to(dtype).contiguous()
    full_ckv_cpu = torch.randn(
        num_full_blocks,
        args.block_size,
        512,
        generator=cpu_generator,
        dtype=torch.float32,
    ).to(dtype).contiguous()
    full_kpe_cpu[0].zero_()
    full_ckv_cpu[0].zero_()
    full_kpe = swapped_from_cpu(full_kpe_cpu, device)
    full_ckv = swapped_from_cpu(full_ckv_cpu, device)

    selection_blocks_per_row = math.ceil(args.topk / args.block_size)
    num_selection_blocks = (
        args.rounds * args.batch_size * selection_blocks_per_row
    )
    selection_tables_cpu = torch.arange(
        num_selection_blocks,
        dtype=torch.int32,
    ).view(
        args.rounds,
        args.batch_size,
        selection_blocks_per_row,
    )
    selection_tables = selection_tables_cpu.to(device)
    selection_kpe = torch.empty(
        num_selection_blocks,
        args.block_size,
        GLM_INDEX_ROPE_DIM,
        dtype=dtype,
        device=device,
    )
    selection_ckv = torch.empty(
        num_selection_blocks,
        args.block_size,
        512,
        dtype=dtype,
        device=device,
    )
    statuses = torch.full(
        (args.rounds, args.batch_size, 1, 1, args.topk + 1),
        -1,
        dtype=torch.int32,
        device=device,
    )
    captured_topk = torch.empty(
        args.rounds,
        args.batch_size,
        1,
        args.topk,
        dtype=torch.int32,
        device=device,
    )
    req_pool_entries = torch.arange(
        args.batch_size,
        dtype=torch.int32,
        device=device,
    )
    full_actual_seq = torch.full(
        (args.batch_size,),
        args.full_len + 1,
        dtype=torch.int32,
        device=device,
    )

    token_ids = torch.arange(args.topk, dtype=torch.int64).view(1, 1, 1, -1)
    row_offsets = (
        torch.arange(args.batch_size, dtype=torch.int64).view(1, -1, 1, 1)
        * 997
    )
    round_offsets = (
        torch.arange(args.rounds, dtype=torch.int64).view(-1, 1, 1, 1)
        * 211
    )
    poison_templates = (
        (token_ids + row_offsets + round_offsets) % args.full_len
    ).to(torch.int32).to(device)

    # All setup work is complete before the strict async window starts.
    torch.npu.synchronize()
    full_table_cpu = full_table.cpu()
    torch.npu.synchronize()

    poison_keepalive: list[torch.Tensor] = []
    topk_storage_reuse = 0
    expected_shape = (args.batch_size, 1, args.topk)
    for round_idx in range(args.rounds):
        topk_indices = run_native_li(
            queries[round_idx],
            weights[round_idx],
            key_cache,
            query_lens,
            key_lens,
            full_table,
            args.topk,
        )
        if tuple(topk_indices.shape) != expected_shape:
            raise AssertionError(
                "Unexpected LightningIndexer output shape: "
                f"actual={tuple(topk_indices.shape)}, expected={expected_shape}."
            )
        topk_ptr = topk_indices.data_ptr()
        if args.sync_between_li_gs:
            torch.npu.synchronize()
        launch_gs(
            args.consumer,
            selection_kpe,
            selection_ckv,
            selection_tables[round_idx],
            statuses[round_idx],
            req_pool_entries,
            topk_indices.view(args.batch_size, 1, 1, args.topk),
            full_kpe,
            full_ckv,
            full_table,
            full_actual_seq,
        )
        # This copy is deliberately after GS submission, so the tested
        # producer -> consumer edge remains an immediate LI -> GS chain.
        captured_topk[round_idx].copy_(topk_indices)
        del topk_indices

        poison = torch.empty(
            expected_shape,
            dtype=torch.int32,
            device=device,
        )
        if poison.data_ptr() == topk_ptr:
            topk_storage_reuse += 1
        poison.copy_(poison_templates[round_idx])
        poison_keepalive.append(poison)

    # This is the only synchronization inside/after the tested chain series.
    torch.npu.synchronize()
    statuses_cpu = statuses.cpu()
    selection_kpe_cpu = selection_kpe.cpu()
    selection_ckv_cpu = selection_ckv.cpu()
    captured_topk_cpu = captured_topk.cpu()

    positions = torch.arange(args.topk, dtype=torch.int64)
    for round_idx in range(args.rounds):
        for row in range(args.batch_size):
            label = f"round={round_idx} row={row}"
            actual = int(statuses_cpu[round_idx, row, 0, 0, args.topk].item())
            if actual != args.topk:
                raise AssertionError(
                    f"{label} GS status length={actual}, expected={args.topk}."
                )
            selected = statuses_cpu[
                round_idx, row, 0, 0, : args.topk
            ].to(torch.int64)
            selected_sorted = validate_topk_set(
                selected,
                full_len=args.full_len,
                topk=args.topk,
                label=f"{label} GS",
            )
            reference_sorted = validate_topk_set(
                captured_topk_cpu[round_idx, row],
                full_len=args.full_len,
                topk=args.topk,
                label=f"{label} LI reference",
            )
            if not torch.equal(selected_sorted, reference_sorted):
                missing = reference_sorted[~torch.isin(reference_sorted, selected_sorted)]
                extra = selected_sorted[~torch.isin(selected_sorted, reference_sorted)]
                raise AssertionError(
                    f"{label} LI -> GS set mismatch: "
                    f"missing={missing[:8].tolist()}, extra={extra[:8].tolist()}."
                )

            dst_blocks = selection_tables_cpu[round_idx, row].to(torch.int64)[
                positions // args.block_size
            ]
            dst_offsets = positions % args.block_size
            src_blocks = full_table_cpu[row].to(torch.int64)[
                selected // args.block_size
            ]
            src_offsets = selected % args.block_size
            if not torch.equal(
                selection_kpe_cpu[dst_blocks, dst_offsets],
                full_kpe_cpu[src_blocks, src_offsets],
            ):
                raise AssertionError(f"{label} gathered KPE mismatch.")
            if not torch.equal(
                selection_ckv_cpu[dst_blocks, dst_offsets],
                full_ckv_cpu[src_blocks, src_offsets],
            ):
                raise AssertionError(f"{label} gathered CKV mismatch.")

    print(
        "GLM_DSA_LI_GS_ASYNC_UT_OK "
        f"rounds={args.rounds} batch={args.batch_size} full_len={args.full_len} "
        f"topk={args.topk} consumer={args.consumer} "
        f"sync_between_li_gs={args.sync_between_li_gs} "
        f"topk_storage_reuse={topk_storage_reuse}/{args.rounds}",
        flush=True,
    )


if __name__ == "__main__":
    main()
