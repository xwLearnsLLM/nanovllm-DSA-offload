"""Stress the GLM query BMM -> native LightningIndexer async boundary.

The production decode path launches the custom batch-matmul-transpose query
projection, native RoPE/copies, and torch-npu LightningIndexer without a CPU
read or device synchronization between them.  This test repeats that exact
producer/consumer chain for every GLM layer, synchronizes once after all
chains have been submitted, then compares each asynchronous LI result with a
reference LI invocation over the completed projection tensors.
"""

from __future__ import annotations

import argparse

import torch
import torch_npu  # type: ignore

from nanovllm.models.dsa_indexer_project import (
    dsa_indexer_project_q_path,
    dsa_indexer_project_query_only,
)
from ut_ops.test_glm_dsa_indexer import (
    GLM_HIDDEN_SIZE,
    GLM_INDEX_HEAD_DIM,
    GLM_INDEX_HEADS,
    GLM_INDEX_ROPE_DIM,
    GLM_Q_LORA_RANK,
    make_interleaved_cos_sin,
    make_paged_index_cache,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stress GLM custom query BMM -> native LI queue ordering."
    )
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--full-len", type=int, default=8200)
    parser.add_argument("--topk", type=int, default=2048)
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument("--rounds", type=int, default=78)
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if torch.device(args.device).type != "npu":
        raise ValueError("--device must select one NPU, for example npu:0.")
    if not 1 <= args.batch_size <= 64:
        raise ValueError("--batch-size must be in [1, 64] for the query BMM path.")
    if args.full_len <= 0:
        raise ValueError("--full-len must be positive.")
    if args.block_size <= 0:
        raise ValueError("--block-size must be positive.")
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


def validate_topk(
    actual: torch.Tensor,
    expected: torch.Tensor,
    *,
    full_len: int,
    topk: int,
    round_idx: int,
) -> None:
    if actual.shape != expected.shape:
        raise AssertionError(
            f"round={round_idx} LI shape mismatch: "
            f"actual={tuple(actual.shape)} expected={tuple(expected.shape)}."
        )
    actual_rows = actual.reshape(-1, topk).to(torch.int64)
    expected_rows = expected.reshape(-1, topk).to(torch.int64)
    for row in range(actual_rows.shape[0]):
        values = actual_rows[row]
        minimum = int(values.min().item())
        maximum = int(values.max().item())
        if minimum < 0 or maximum >= full_len:
            raise AssertionError(
                f"round={round_idx} row={row} contains out-of-range IDs: "
                f"min={minimum} max={maximum} valid=[0,{full_len})."
            )
        actual_sorted = torch.sort(values).values
        if torch.unique_consecutive(actual_sorted).numel() != topk:
            raise AssertionError(
                f"round={round_idx} row={row} contains duplicate token IDs."
            )
        expected_sorted = torch.sort(expected_rows[row]).values
        if not torch.equal(actual_sorted, expected_sorted):
            mismatch = int((actual_sorted != expected_sorted).sum().item())
            raise AssertionError(
                f"round={round_idx} row={row} async LI differs from the "
                f"completed-projection reference in {mismatch}/{topk} slots."
            )


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
        "GLM_Q_BMM_LI_ASYNC_CONFIG "
        f"device={device} batch={args.batch_size} full_len={args.full_len} "
        f"topk={args.topk} block_size={args.block_size} rounds={args.rounds} "
        f"seed={args.seed}",
        flush=True,
    )

    wq_b = (
        torch.randn(
            GLM_INDEX_HEADS * GLM_INDEX_HEAD_DIM,
            GLM_Q_LORA_RANK,
            dtype=dtype,
            device=device,
        )
        * 0.02
    ).contiguous()
    wq_b_bmm_t = (
        wq_b.view(GLM_INDEX_HEADS, GLM_INDEX_HEAD_DIM, GLM_Q_LORA_RANK)
        .transpose(1, 2)
        .contiguous()
    )
    weights_proj = (
        torch.randn(
            GLM_INDEX_HEADS,
            GLM_HIDDEN_SIZE,
            dtype=dtype,
            device=device,
        )
        * 0.02
    ).contiguous()
    q_c_inputs = torch.randn(
        args.rounds,
        args.batch_size,
        GLM_Q_LORA_RANK,
        dtype=dtype,
        device=device,
    )
    hidden_inputs = torch.randn(
        args.rounds,
        args.batch_size,
        GLM_HIDDEN_SIZE,
        dtype=dtype,
        device=device,
    )
    cos, sin = make_interleaved_cos_sin(args.batch_size, dtype, device)
    key_cache, block_table = make_paged_index_cache(
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
    torch.npu.synchronize()

    selected_path = dsa_indexer_project_q_path(
        q_c_inputs[0],
        wq_b_bmm_t,
        enable_q_bmm=True,
    )
    if selected_path != "dsa_indexer_project_bmm_transpose":
        raise AssertionError(
            "The async UT did not select the custom BMM-transpose path: "
            f"selected={selected_path}."
        )

    q_outputs: list[torch.Tensor] = []
    weight_outputs: list[torch.Tensor] = []
    async_topk: list[torch.Tensor] = []
    for round_idx in range(args.rounds):
        q_out = torch.empty(
            args.batch_size,
            GLM_INDEX_HEADS,
            GLM_INDEX_HEAD_DIM,
            dtype=dtype,
            device=device,
        )
        weights_out = torch.empty(
            args.batch_size,
            GLM_INDEX_HEADS,
            dtype=dtype,
            device=device,
        )
        dsa_indexer_project_query_only(
            hidden_inputs[round_idx],
            q_c_inputs[round_idx],
            cos,
            sin,
            wq_b,
            weights_proj,
            q_out,
            weights_out,
            n_head=GLM_INDEX_HEADS,
            head_dim=GLM_INDEX_HEAD_DIM,
            rope_dim=GLM_INDEX_ROPE_DIM,
            score_scale=1.0,
            rotary_mode="interleave",
            wq_b_bmm_t=wq_b_bmm_t,
            enable_q_bmm=True,
        )
        topk_indices = run_native_li(
            q_out,
            weights_out,
            key_cache,
            query_lens,
            key_lens,
            block_table,
            args.topk,
        )
        q_outputs.append(q_out)
        weight_outputs.append(weights_out)
        async_topk.append(topk_indices)

    # The stress window above intentionally contains no CPU read, event, or
    # device synchronization between the custom producer and native consumer.
    torch.npu.synchronize()
    async_topk_cpu = [tensor.cpu() for tensor in async_topk]

    reference_topk = [
        run_native_li(
            q_outputs[round_idx],
            weight_outputs[round_idx],
            key_cache,
            query_lens,
            key_lens,
            block_table,
            args.topk,
        )
        for round_idx in range(args.rounds)
    ]
    torch.npu.synchronize()
    reference_topk_cpu = [tensor.cpu() for tensor in reference_topk]

    for round_idx in range(args.rounds):
        validate_topk(
            async_topk_cpu[round_idx],
            reference_topk_cpu[round_idx],
            full_len=args.full_len,
            topk=args.topk,
            round_idx=round_idx,
        )

    print(
        "GLM_Q_BMM_LI_ASYNC_UT_OK "
        f"rounds={args.rounds} batch={args.batch_size} "
        f"full_len={args.full_len} topk={args.topk} path={selected_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
