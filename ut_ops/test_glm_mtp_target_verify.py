"""Validate GLM MTP target verification MLA against token-by-token decode."""

from __future__ import annotations

import argparse
import math

import torch
import torch_npu  # type: ignore


BLOCK_SIZE = 128
CKV_DIM = 512
KPE_DIM = 64
QK_HEAD_DIM = 192
MASK_SIZE = 2048


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare FIA-v2 TND_NTD causal MTP verification with the "
            "ordinary FIA-v2 one-token decode path."
        )
    )
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--heads", type=int, default=2)
    parser.add_argument("--prefix-len", type=int, default=4096)
    parser.add_argument("--query-lens", default="2,4")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--atol", type=float, default=0.04)
    parser.add_argument("--rtol", type=float, default=0.02)
    parser.add_argument("--min-cosine", type=float, default=0.999)
    parser.add_argument(
        "--graph-replays",
        type=int,
        default=0,
        help=(
            "Capture the largest query length with FIA-v2 external-task "
            "refresh and replay it this many times."
        ),
    )
    return parser.parse_args()


def parse_query_lens(raw: str) -> list[int]:
    values = [int(value.strip()) for value in raw.split(",") if value.strip()]
    if not values or any(value <= 1 or value > MASK_SIZE for value in values):
        raise ValueError(
            f"--query-lens must contain integers in [2, {MASK_SIZE}]."
        )
    return values


def random_bf16(
    generator: torch.Generator,
    *shape: int,
) -> torch.Tensor:
    return (
        torch.randn(*shape, generator=generator, dtype=torch.float32)
        .mul_(0.02)
        .to(torch.bfloat16)
    )


def make_inputs(
    batch_size: int,
    heads: int,
    prefix_len: int,
    query_len: int,
    seed: int,
    device: torch.device,
) -> tuple[torch.Tensor, ...]:
    generator = torch.Generator().manual_seed(seed + query_len)
    blocks_per_request = math.ceil(
        (prefix_len + query_len) / BLOCK_SIZE
    )
    total_blocks = batch_size * blocks_per_request
    block_table = torch.randperm(
        total_blocks, generator=generator, dtype=torch.int64
    ).to(torch.int32).view(batch_size, blocks_per_request)
    key_cache = random_bf16(
        generator, total_blocks, 1, BLOCK_SIZE, CKV_DIM
    )
    key_rope_cache = random_bf16(
        generator, total_blocks, 1, BLOCK_SIZE, KPE_DIM
    )

    # Make speculative positions strongly distinguishable. If TND_NTD's
    # bottom-right causal mask leaks even one future token, its larger key
    # dominates softmax and this UT fails instead of hiding the bug in a
    # long random prefix.
    pattern = torch.where(
        torch.arange(CKV_DIM) % 2 == 0,
        torch.tensor(0.5),
        torch.tensor(-0.5),
    ).to(torch.bfloat16)
    query = pattern.view(1, 1, 1, CKV_DIM).expand(
        batch_size, query_len, heads, CKV_DIM
    ).clone()
    query_rope = torch.zeros(
        batch_size, query_len, heads, KPE_DIM, dtype=torch.bfloat16
    )
    for row in range(batch_size):
        for step in range(query_len):
            logical_position = prefix_len + step
            physical_block = int(
                block_table[row, logical_position // BLOCK_SIZE]
            )
            offset = logical_position % BLOCK_SIZE
            key_cache[physical_block, 0, offset] = pattern * (step + 1)
            key_rope_cache[physical_block, 0, offset].zero_()
    return (
        query.to(device),
        query_rope.to(device),
        key_cache.to(device),
        key_rope_cache.to(device),
        block_table.to(device),
    )


def run_batched_verify(
    query: torch.Tensor,
    query_rope: torch.Tensor,
    key_cache: torch.Tensor,
    key_rope_cache: torch.Tensor,
    block_table: torch.Tensor,
    prefix_len: int,
    mask: torch.Tensor,
    actual_seq_kvlen: list[int] | None = None,
) -> torch.Tensor:
    batch_size, query_len, heads, _ = query.shape
    total_tokens = batch_size * query_len
    if actual_seq_kvlen is None:
        actual_seq_kvlen = [prefix_len + query_len] * batch_size
    output, _ = torch_npu.npu_fused_infer_attention_score_v2(
        query.view(total_tokens, heads, CKV_DIM).contiguous(),
        key_cache,
        key_cache,
        query_rope=query_rope.view(
            total_tokens, heads, KPE_DIM
        ).contiguous(),
        key_rope=key_rope_cache,
        num_query_heads=heads,
        num_key_value_heads=1,
        input_layout="TND_NTD",
        atten_mask=mask,
        sparse_mode=3,
        softmax_scale=1.0 / math.sqrt(QK_HEAD_DIM),
        block_table=block_table,
        block_size=BLOCK_SIZE,
        actual_seq_qlen=[
            (row + 1) * query_len for row in range(batch_size)
        ],
        actual_seq_kvlen=actual_seq_kvlen,
    )
    return output.view(
        heads, batch_size, query_len, CKV_DIM
    ).permute(1, 2, 0, 3).contiguous()


def run_graph_replay_check(
    query: torch.Tensor,
    query_rope: torch.Tensor,
    key_cache: torch.Tensor,
    key_rope_cache: torch.Tensor,
    block_table: torch.Tensor,
    prefix_len: int,
    mask: torch.Tensor,
    replays: int,
    atol: float,
    rtol: float,
) -> None:
    """Exercise the same FIA-v2 external-task pattern used by MTP graphs."""

    batch_size, query_len, heads, _ = query.shape
    total_tokens = batch_size * query_len
    flat_query = query.view(total_tokens, heads, CKV_DIM).contiguous()
    flat_query_rope = query_rope.view(
        total_tokens, heads, KPE_DIM
    ).contiguous()
    actual_seq_qlen = [
        (row + 1) * query_len for row in range(batch_size)
    ]
    capture_kvlen = [prefix_len + query_len] * batch_size
    kwargs = {
        "query_rope": flat_query_rope,
        "key_rope": key_rope_cache,
        "num_query_heads": heads,
        "num_key_value_heads": 1,
        "input_layout": "TND_NTD",
        "atten_mask": mask,
        "sparse_mode": 3,
        "softmax_scale": 1.0 / math.sqrt(QK_HEAD_DIM),
        "block_table": block_table,
        "block_size": BLOCK_SIZE,
        "actual_seq_qlen": actual_seq_qlen,
        "actual_seq_kvlen": capture_kvlen,
    }
    eager_output, _ = (
        torch_npu.npu_fused_infer_attention_score_v2(
            flat_query,
            key_cache,
            key_cache,
            **kwargs,
        )
    )
    graph_output = torch.empty(
        (heads, total_tokens, CKV_DIM),
        dtype=flat_query.dtype,
        device=flat_query.device,
    )
    graph_lse = torch.empty(
        total_tokens,
        dtype=flat_query.dtype,
        device=flat_query.device,
    )
    workspace = (
        torch_npu._npu_fused_infer_attention_score_v2_get_max_workspace(
            flat_query,
            key_cache,
            key_cache,
            **kwargs,
        )
    )
    attention_op = torch_npu.npu_fused_infer_attention_score_v2.out

    # Separate the out-variant contract from graph capture. This mirrors the
    # output/LSE allocation used by vLLM-Ascend's MLA graph implementation.
    attention_op(
        flat_query,
        key_cache,
        key_cache,
        **kwargs,
        workspace=workspace,
        out=[graph_output, graph_lse],
    )
    torch.npu.synchronize()
    torch.testing.assert_close(
        graph_output.float().cpu(),
        eager_output.float().cpu(),
        atol=atol,
        rtol=rtol,
    )
    print(
        "GLM_MTP_TARGET_VERIFY_OUT_CHECK "
        f"query_len={query_len} ok=1"
    )

    graph = torch.npu.NPUGraph()
    pool = torch.npu.graph_pool_handle()
    with torch.npu.graph(graph, pool=pool):
        # torch.npu.graph may switch to its capture stream. Resolve the
        # stream inside the context, exactly as the model attention path does.
        capture_stream = torch.npu.current_stream()
        event = torch.npu.ExternalEvent()
        event.wait(capture_stream)
        event.reset(capture_stream)
        torch.npu.graph_task_group_begin(capture_stream)
        attention_op(
            flat_query,
            key_cache,
            key_cache,
            **kwargs,
            workspace=workspace,
            out=[graph_output, graph_lse],
        )
        task_handle = torch.npu.graph_task_group_end(capture_stream)
    torch.npu.synchronize()

    update_stream = torch.npu.Stream()

    def replay_graph(kv_lengths: list[int]) -> None:
        torch.npu.current_stream().synchronize()
        graph.replay()
        with torch.npu.stream(update_stream):
            torch.npu.graph_task_update_begin(update_stream, task_handle)
            try:
                attention_op(
                    flat_query,
                    key_cache,
                    key_cache,
                    **(kwargs | {"actual_seq_kvlen": kv_lengths}),
                    workspace=workspace,
                    out=[graph_output, graph_lse],
                )
            finally:
                torch.npu.graph_task_update_end(update_stream)
            event.record(update_stream)
        torch.npu.synchronize()

    # Capturing an external task does not make its capture-time output a valid
    # inference result. Validate only after a real replay plus task refresh.
    replay_graph(capture_kvlen)
    torch.testing.assert_close(
        graph_output.float().cpu(),
        eager_output.float().cpu(),
        atol=atol,
        rtol=rtol,
    )

    max_abs = 0.0
    for replay in range(replays):
        # Change both device input data and host-side KV-length attributes.
        flat_query.mul_(-1)
        kv_lengths = [
            prefix_len + query_len - ((replay + row) % 3)
            for row in range(batch_size)
        ]
        replay_graph(kv_lengths)
        golden = run_batched_verify(
            query,
            query_rope,
            key_cache,
            key_rope_cache,
            block_table,
            prefix_len,
            mask,
            kv_lengths,
        )
        actual = graph_output.view(
            heads, batch_size, query_len, CKV_DIM
        ).permute(1, 2, 0, 3).contiguous()
        torch.npu.synchronize()
        actual_cpu = actual.float().cpu()
        golden_cpu = golden.float().cpu()
        replay_max_abs = float((actual_cpu - golden_cpu).abs().max())
        max_abs = max(max_abs, replay_max_abs)
        torch.testing.assert_close(
            actual_cpu, golden_cpu, atol=atol, rtol=rtol
        )
    print(
        "GLM_MTP_TARGET_VERIFY_GRAPH_CHECK "
        f"query_len={query_len} replays={replays} "
        f"max_abs={max_abs:.9f} dynamic_kvlen=1 ok=1"
    )


def run_sequential_decode(
    query: torch.Tensor,
    query_rope: torch.Tensor,
    key_cache: torch.Tensor,
    key_rope_cache: torch.Tensor,
    block_table: torch.Tensor,
    prefix_len: int,
) -> torch.Tensor:
    batch_size, query_len, heads, _ = query.shape
    outputs = []
    for step in range(query_len):
        output, _ = torch_npu.npu_fused_infer_attention_score_v2(
            query[:, step].view(
                batch_size, heads, 1, CKV_DIM
            ).contiguous(),
            key_cache,
            key_cache,
            query_rope=query_rope[:, step].view(
                batch_size, heads, 1, KPE_DIM
            ).contiguous(),
            key_rope=key_rope_cache,
            num_query_heads=heads,
            num_key_value_heads=1,
            input_layout="BNSD_NBSD",
            atten_mask=None,
            sparse_mode=0,
            softmax_scale=1.0 / math.sqrt(QK_HEAD_DIM),
            block_table=block_table,
            block_size=BLOCK_SIZE,
            actual_seq_qlen=None,
            actual_seq_kvlen=[
                prefix_len + step + 1
            ] * batch_size,
        )
        outputs.append(
            output.view(heads, batch_size, 1, CKV_DIM)
            .permute(1, 2, 0, 3)
        )
    return torch.cat(outputs, dim=1).contiguous()


def check_close(
    actual: torch.Tensor,
    expected: torch.Tensor,
    query_len: int,
    atol: float,
    rtol: float,
    min_cosine: float,
) -> None:
    actual_f = actual.float()
    expected_f = expected.float()
    max_abs = float((actual_f - expected_f).abs().max().cpu())
    cosine = torch.nn.functional.cosine_similarity(
        actual_f.reshape(-1, CKV_DIM),
        expected_f.reshape(-1, CKV_DIM),
        dim=-1,
    )
    min_cos = float(cosine.min().cpu())
    torch.testing.assert_close(
        actual_f.cpu(), expected_f.cpu(), atol=atol, rtol=rtol
    )
    if min_cos < min_cosine:
        raise AssertionError(
            "MTP target verification cosine similarity is too low: "
            f"query_len={query_len}, actual={min_cos:.9f}, "
            f"required={min_cosine:.9f}."
        )
    print(
        "GLM_MTP_TARGET_VERIFY_CHECK "
        f"query_len={query_len} max_abs={max_abs:.9f} "
        f"min_cosine={min_cos:.9f} ok=1"
    )


def main() -> None:
    args = parse_args()
    query_lens = parse_query_lens(args.query_lens)
    if (
        args.batch_size <= 0
        or args.heads <= 0
        or args.prefix_len <= 0
        or args.atol < 0
        or args.rtol < 0
        or args.graph_replays < 0
        or not 0 <= args.min_cosine <= 1
    ):
        raise ValueError(
            "batch-size, heads and prefix-len must be positive; tolerances "
            "must be non-negative and min-cosine must be in [0,1]."
        )
    device = torch.device(args.device)
    if device.type != "npu":
        raise ValueError("--device must select one NPU, for example npu:0.")
    torch.npu.set_device(device)
    torch.npu.config.allow_internal_format = False
    mask = torch.triu(
        torch.ones(
            MASK_SIZE,
            MASK_SIZE,
            dtype=torch.int8,
            device=device,
        ),
        diagonal=1,
    ).contiguous()
    print(
        "GLM_MTP_TARGET_VERIFY_CONFIG "
        f"device={device} batch={args.batch_size} heads={args.heads} "
        f"prefix_len={args.prefix_len} query_lens={query_lens} "
        f"seed={args.seed}"
    )
    for query_len in query_lens:
        inputs = make_inputs(
            args.batch_size,
            args.heads,
            args.prefix_len,
            query_len,
            args.seed,
            device,
        )
        batched = run_batched_verify(*inputs, args.prefix_len, mask)
        sequential = run_sequential_decode(*inputs, args.prefix_len)
        torch.npu.synchronize()
        check_close(
            batched,
            sequential,
            query_len,
            args.atol,
            args.rtol,
            args.min_cosine,
        )
        del inputs, batched, sequential
    if args.graph_replays:
        query_len = max(query_lens)
        inputs = make_inputs(
            args.batch_size,
            args.heads,
            args.prefix_len,
            query_len,
            args.seed + 1000,
            device,
        )
        run_graph_replay_check(
            *inputs,
            args.prefix_len,
            mask,
            args.graph_replays,
            args.atol,
            args.rtol,
        )
    print("GLM_MTP_TARGET_VERIFY_UT_OK")


if __name__ == "__main__":
    main()
