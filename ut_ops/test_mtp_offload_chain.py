"""End-to-end MTP3 LIDU -> SCATTER -> sparse-and-tail Attention UT."""

from __future__ import annotations

import argparse
import math
from dataclasses import replace

import torch
import torch_npu  # type: ignore

import nanovllm.ops  # noqa: F401  # Load repository-local custom operators.
from ut_ops._op_utils import require_local_opapi
import test_fused_li_manage_mtp as lidu_ut


QUERY_COUNT = lidu_ut.QUERY_COUNT
BLOCK_SIZE = lidu_ut.BLOCK_SIZE
TOPK = lidu_ut.TOPK
UNION_CAPACITY = lidu_ut.UNION_CAPACITY
CKV_DIM = lidu_ut.CKV_DIM
KPE_DIM = lidu_ut.KPE_DIM
KNOWN_FUSED_ATTENTION_ATOL = 0.1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate the real MTP-LIDU outputs through SCATTER and MTP3 "
            "sparse-and-tail Attention on one NPU."
        )
    )
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--heads", type=int, default=2)
    parser.add_argument("--source-len", type=int, default=20992)
    parser.add_argument("--cache-tokens", type=int, default=8192)
    parser.add_argument("--tail-tokens", type=int, default=64)
    parser.add_argument("--graph-replays", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument(
        "--perf-miss-count",
        type=int,
        default=300,
        help=(
            "Exact miss count per request in the performance-only case. "
            "Semantic and graph checks retain broad miss coverage."
        ),
    )
    parser.add_argument(
        "--allow-fused-attention-diff",
        action="store_true",
        help=(
            "Allow the known COPYSFA-MTP numerical difference from the split "
            "Attention path while retaining the CPU golden and cache checks."
        ),
    )
    parser.add_argument("--skip-performance", action="store_true")
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if args.heads <= 0 or args.heads > 128 or args.heads & (args.heads - 1):
        raise ValueError("--heads must be a power of two in [1,128]")
    if (
        args.source_len < UNION_CAPACITY
        or args.source_len > lidu_ut.MAX_SOURCE_CAPACITY
        or args.source_len % BLOCK_SIZE
    ):
        raise ValueError("--source-len must be block aligned and in [8192,2^18]")
    if (
        args.cache_tokens < UNION_CAPACITY
        or args.cache_tokens > args.source_len
        or args.cache_tokens % BLOCK_SIZE
    ):
        raise ValueError(
            "--cache-tokens must be block aligned and in [8192,source-len]"
        )
    if args.tail_tokens < 0 or args.graph_replays < 2:
        raise ValueError("--tail-tokens must be >=0 and --graph-replays must be >=2")
    if args.warmup < 0 or args.iters <= 0:
        raise ValueError("--warmup must be >=0 and --iters must be positive")
    if not 0 <= args.perf_miss_count <= UNION_CAPACITY:
        raise ValueError("--perf-miss-count must be in [0,8192]")


def logical_rows(
    cache: torch.Tensor,
    block_table: torch.Tensor,
    request: int,
    logical_slots: torch.Tensor,
) -> torch.Tensor:
    slots = logical_slots.to(torch.int64)
    blocks = block_table[request, slots // BLOCK_SIZE].to(torch.int64)
    return cache[blocks, slots % BLOCK_SIZE]


def make_miss_fractions(batch_size: int) -> tuple[float, ...]:
    if batch_size == 1:
        return (1.0,)
    return tuple(request / (batch_size - 1) for request in range(batch_size))


def initialize_hbm(
    *,
    case: lidu_ut.MtpCase,
    cache_tokens: int,
    final_kv_len: int,
    dram_kpe: torch.Tensor,
    dram_ckv: torch.Tensor,
    dram_table: torch.Tensor,
    hbm_table: torch.Tensor,
    hbm_blocks: int,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Materialize the pre-update LIDU token->slot state in HBM."""

    initial_kpe = torch.zeros(
        hbm_blocks, BLOCK_SIZE, KPE_DIM, dtype=torch.bfloat16
    )
    initial_ckv = torch.zeros(
        hbm_blocks, BLOCK_SIZE, CKV_DIM, dtype=torch.bfloat16
    )
    source_ids = torch.empty(
        case.batch_size, cache_tokens, dtype=torch.int32
    )
    destination_slots = torch.empty_like(source_ids)
    for request in range(case.batch_size):
        pool_row = int(case.req_pool_entries_cpu[request])
        state = case.initial_cache_cpu[pool_row, : case.source_capacity]
        sources = torch.nonzero(state >= 0).flatten()
        if sources.numel() != cache_tokens:
            raise AssertionError(
                f"request={request}: initial LIDU state does not contain C tokens"
            )
        source_ids[request] = sources.to(torch.int32)
        destination_slots[request] = state[sources]
    lidu_ut._apply_scatter_reference(
        initial_kpe,
        initial_ckv,
        dram_kpe,
        dram_ckv,
        hbm_table,
        dram_table,
        source_ids,
        destination_slots,
        torch.full((case.batch_size,), cache_tokens, dtype=torch.int32),
    )

    dense_count = final_kv_len - cache_tokens
    dense_kpe = torch.randn(
        case.batch_size,
        dense_count,
        KPE_DIM,
        generator=generator,
        dtype=torch.float32,
    ).mul_(0.25).to(torch.bfloat16)
    dense_ckv = torch.randn(
        case.batch_size,
        dense_count,
        CKV_DIM,
        generator=generator,
        dtype=torch.float32,
    ).mul_(0.25).to(torch.bfloat16)
    dense_slots = torch.arange(cache_tokens, final_kv_len, dtype=torch.int64)
    for request in range(case.batch_size):
        blocks = hbm_table[
            request, dense_slots // BLOCK_SIZE
        ].to(torch.int64)
        offsets = dense_slots % BLOCK_SIZE
        initial_kpe[blocks, offsets] = dense_kpe[request]
        initial_ckv[blocks, offsets] = dense_ckv[request]
    return initial_kpe.contiguous(), initial_ckv.contiguous()


def expected_after_scatter(
    *,
    initial_kpe: torch.Tensor,
    initial_ckv: torch.Tensor,
    dram_kpe: torch.Tensor,
    dram_ckv: torch.Tensor,
    hbm_table: torch.Tensor,
    dram_table: torch.Tensor,
    lidu_outputs: tuple[torch.Tensor, ...],
) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
    expected_kpe = initial_kpe.clone()
    expected_ckv = initial_ckv.clone()
    counts = lidu_outputs[4].cpu()
    lidu_ut._apply_scatter_reference(
        expected_kpe,
        expected_ckv,
        dram_kpe,
        dram_ckv,
        hbm_table,
        dram_table,
        lidu_outputs[2].cpu(),
        lidu_outputs[3].cpu(),
        counts,
    )
    return (
        expected_kpe,
        expected_ckv,
        [int(value) for value in counts.tolist()],
    )


def cache_mismatch_diagnostic(
    *,
    name: str,
    actual: torch.Tensor,
    expected: torch.Tensor,
    initial: torch.Tensor,
    hbm_table: torch.Tensor,
    dram_cache: torch.Tensor,
    dram_table: torch.Tensor,
    lidu_outputs: tuple[torch.Tensor, ...],
) -> str:
    mismatch = actual != expected
    mismatch_elements = int(mismatch.sum())
    mismatch_rows = mismatch.any(dim=-1)
    mismatch_token_rows = int(mismatch_rows.sum())
    first = torch.nonzero(mismatch, as_tuple=False)[0]
    physical_block = int(first[0])
    block_offset = int(first[1])
    feature = int(first[2])

    table_match = torch.nonzero(
        hbm_table == physical_block, as_tuple=False
    )
    request = -1
    logical_slot = -1
    expected_source = -1
    actual_source = -1
    expected_occurrences = "[]"
    actual_occurrences = "[]"
    if table_match.numel():
        request = int(table_match[0, 0])
        block_column = int(table_match[0, 1])
        logical_slot = block_column * BLOCK_SIZE + block_offset
        counts = lidu_outputs[4].cpu().to(torch.int64)
        destinations = lidu_outputs[3].cpu().to(torch.int64)
        sources = lidu_outputs[2].cpu().to(torch.int64)
        count = int(counts[request])
        destination_match = torch.nonzero(
            destinations[request, :count] == logical_slot,
            as_tuple=False,
        )
        if destination_match.numel():
            expected_source = int(
                sources[request, int(destination_match[0, 0])]
            )

    actual_row = actual[physical_block, block_offset]
    expected_row = expected[physical_block, block_offset]
    initial_row = initial[physical_block, block_offset]
    if request >= 0:
        request_blocks = dram_table[request].to(torch.int64)
        request_rows = dram_cache.index_select(0, request_blocks).reshape(
            -1, actual_row.numel()
        )
        source_match = torch.nonzero(
            (request_rows == actual_row).all(dim=-1), as_tuple=False
        )
        if source_match.numel():
            actual_source = int(source_match[0, 0])
        aligned_sources = lidu_outputs[1].reshape(
            -1, QUERY_COUNT, TOPK
        )[request].cpu().to(torch.int64)
        aligned_slots = lidu_outputs[0].reshape(
            -1, QUERY_COUNT, TOPK
        )[request].cpu().to(torch.int64)

        def _format_occurrences(source: int) -> str:
            if source < 0:
                return "[]"
            positions = torch.nonzero(
                aligned_sources == source, as_tuple=False
            )
            values: list[str] = []
            for query_idx, topk_idx in positions[:8].tolist():
                slot = int(aligned_slots[query_idx, topk_idx])
                values.append(f"q{query_idx}:k{topk_idx}->s{slot}")
            return "[" + ",".join(values) + "]"

        expected_occurrences = _format_occurrences(expected_source)
        actual_occurrences = _format_occurrences(actual_source)
    return (
        f"{name} payload mismatch: mismatch_elements={mismatch_elements} "
        f"mismatch_token_rows={mismatch_token_rows} "
        f"first_physical_block={physical_block} "
        f"first_block_offset={block_offset} first_feature={feature} "
        f"request={request} logical_slot={logical_slot} "
        f"expected_source={expected_source} actual_source={actual_source} "
        f"expected_occurrences={expected_occurrences} "
        f"actual_occurrences={actual_occurrences} "
        f"actual_still_initial={int(torch.equal(actual_row, initial_row))} "
        f"actual_first={float(actual_row[feature]):.7f} "
        f"expected_first={float(expected_row[feature]):.7f} "
        f"initial_first={float(initial_row[feature]):.7f}"
    )


def attention_golden(
    *,
    query: torch.Tensor,
    query_rope: torch.Tensor,
    kpe: torch.Tensor,
    ckv: torch.Tensor,
    hbm_table: torch.Tensor,
    sparse_slots: torch.Tensor,
    cache_tokens: int,
    tail_tokens: int,
    scale: float,
) -> torch.Tensor:
    rows: list[torch.Tensor] = []
    sparse = sparse_slots.reshape(-1, TOPK).to(torch.int64)
    for request in range(hbm_table.shape[0]):
        for query_idx in range(QUERY_COUNT):
            row = request * QUERY_COUNT + query_idx
            dense_end = cache_tokens + tail_tokens + query_idx + 1
            logical_slots = torch.cat(
                (
                    sparse[row],
                    torch.arange(cache_tokens, dense_end, dtype=torch.int64),
                )
            )
            selected_ckv = logical_rows(
                ckv, hbm_table, request, logical_slots
            ).float()
            selected_kpe = logical_rows(
                kpe, hbm_table, request, logical_slots
            ).float()
            scores = (
                query[row].float() @ selected_ckv.T
                + query_rope[row].float() @ selected_kpe.T
            ) * scale
            rows.append(torch.softmax(scores, dim=-1) @ selected_ckv)
    return torch.stack(rows)


def call_attention_out(
    *,
    query: torch.Tensor,
    query_rope: torch.Tensor,
    kpe: torch.Tensor,
    ckv: torch.Tensor,
    sparse_slots: torch.Tensor,
    cache_tokens: torch.Tensor,
    hbm_table: torch.Tensor,
    actual_q: torch.Tensor,
    actual_kv: torch.Tensor,
    scale: float,
    output: torch.Tensor,
) -> torch.Tensor:
    latent_cache = ckv.view(-1, BLOCK_SIZE, 1, CKV_DIM)
    rope_cache = kpe.view(-1, BLOCK_SIZE, 1, KPE_DIM)
    return torch.ops.nanovllm_dsa.sparse_tail_attention_mtp_out.default(
        query,
        latent_cache,
        latent_cache,
        sparse_slots,
        cache_tokens,
        hbm_table,
        actual_q,
        actual_kv,
        query_rope,
        rope_cache,
        scale,
        output,
    )


def run_chain(args: argparse.Namespace, device: torch.device) -> None:
    batch_size = args.batch_size
    cache_tokens = args.cache_tokens
    tail_tokens = args.tail_tokens
    final_kv_len = cache_tokens + tail_tokens + QUERY_COUNT
    hbm_capacity = math.ceil(final_kv_len / BLOCK_SIZE) * BLOCK_SIZE
    case = lidu_ut.make_case(
        name="mtp_offload_chain",
        device=device,
        dtype=torch.bfloat16,
        candidate_lens=(args.source_len,) * batch_size,
        cache_tokens=(cache_tokens,) * batch_size,
        miss_fractions=make_miss_fractions(batch_size),
        source_capacity=args.source_len,
        seed=args.seed + 5000,
    )
    generator = torch.Generator().manual_seed(args.seed + 5017)
    dram_table_cpu, dram_blocks = lidu_ut._random_block_table(
        batch_size, args.source_len // BLOCK_SIZE, generator
    )
    hbm_table_cpu, hbm_blocks = lidu_ut._random_block_table(
        batch_size, hbm_capacity // BLOCK_SIZE, generator
    )
    dram_kpe_cpu = torch.randn(
        dram_blocks,
        BLOCK_SIZE,
        KPE_DIM,
        generator=generator,
        dtype=torch.float32,
    ).mul_(0.25).to(torch.bfloat16)
    dram_ckv_cpu = torch.randn(
        dram_blocks,
        BLOCK_SIZE,
        CKV_DIM,
        generator=generator,
        dtype=torch.float32,
    ).mul_(0.25).to(torch.bfloat16)
    initial_kpe_cpu, initial_ckv_cpu = initialize_hbm(
        case=case,
        cache_tokens=cache_tokens,
        final_kv_len=final_kv_len,
        dram_kpe=dram_kpe_cpu,
        dram_ckv=dram_ckv_cpu,
        dram_table=dram_table_cpu,
        hbm_table=hbm_table_cpu,
        hbm_blocks=hbm_blocks,
        generator=generator,
    )

    query_cpu = torch.randn(
        batch_size * QUERY_COUNT,
        args.heads,
        CKV_DIM,
        generator=generator,
        dtype=torch.float32,
    ).mul_(0.25).to(torch.bfloat16)
    query_rope_cpu = torch.randn(
        batch_size * QUERY_COUNT,
        args.heads,
        KPE_DIM,
        generator=generator,
        dtype=torch.float32,
    ).mul_(0.25).to(torch.bfloat16)
    query = query_cpu.to(device)
    query_rope = query_rope_cpu.to(device)
    dram_kpe = lidu_ut._swapped_from_cpu(dram_kpe_cpu, device)
    dram_ckv = lidu_ut._swapped_from_cpu(dram_ckv_cpu, device)
    dram_table = dram_table_cpu.to(device)
    hbm_table = hbm_table_cpu.to(device)
    actual_q = torch.arange(
        QUERY_COUNT,
        batch_size * QUERY_COUNT + 1,
        QUERY_COUNT,
        dtype=torch.int32,
        device=device,
    )
    actual_kv = torch.full(
        (batch_size,), final_kv_len, dtype=torch.int32, device=device
    )
    scale = 1.0 / math.sqrt(CKV_DIM + KPE_DIM)

    def launch(
        cache_slots: torch.Tensor,
        hbm_kpe: torch.Tensor,
        hbm_ckv: torch.Tensor,
        lidu_buffers: tuple[torch.Tensor, ...],
        attention_output: torch.Tensor,
    ) -> tuple[tuple[torch.Tensor, ...], torch.Tensor]:
        lidu_outputs = lidu_ut.call_mtp_out(
            case, cache_slots, *lidu_buffers
        )
        kpe_alias, ckv_alias = torch.ops.nanovllm_dsa.scatter_copy.default(
            hbm_kpe,
            hbm_ckv,
            dram_kpe,
            dram_ckv,
            hbm_table,
            dram_table,
            lidu_outputs[2],
            lidu_outputs[3],
            lidu_outputs[4],
        )
        attention = call_attention_out(
            query=query,
            query_rope=query_rope,
            kpe=kpe_alias,
            ckv=ckv_alias,
            sparse_slots=lidu_outputs[0],
            cache_tokens=case.cache_tokens,
            hbm_table=hbm_table,
            actual_q=actual_q,
            actual_kv=actual_kv,
            scale=scale,
            output=attention_output,
        )
        return lidu_outputs, attention

    def launch_fused(
        cache_slots: torch.Tensor,
        hbm_kpe: torch.Tensor,
        hbm_ckv: torch.Tensor,
        lidu_buffers: tuple[torch.Tensor, ...],
        attention_output: torch.Tensor,
    ) -> tuple[tuple[torch.Tensor, ...], torch.Tensor]:
        lidu_outputs = lidu_ut.call_mtp_out(
            case, cache_slots, *lidu_buffers
        )
        attention = (
            torch.ops.nanovllm_dsa.fused_copy_sfa_mtp_out.default(
                query_rope,
                query,
                actual_q,
                actual_kv,
                case.cache_tokens,
                lidu_outputs[0],
                lidu_outputs[1],
                lidu_outputs[4],
                hbm_table,
                dram_table,
                hbm_kpe.view(-1, BLOCK_SIZE, 1, KPE_DIM),
                hbm_ckv.view(-1, BLOCK_SIZE, 1, CKV_DIM),
                dram_kpe,
                dram_ckv,
                scale,
                attention_output,
            )
        )
        if attention.data_ptr() != attention_output.data_ptr():
            raise AssertionError(
                "fused_copy_sfa_mtp_out did not return its caller-owned output"
            )
        return lidu_outputs, attention

    def validate_chain(
        *,
        label: str,
        before_cache: torch.Tensor,
        cache_slots: torch.Tensor,
        hbm_kpe: torch.Tensor,
        hbm_ckv: torch.Tensor,
        lidu_outputs: tuple[torch.Tensor, ...],
        attention: torch.Tensor,
    ) -> tuple[list[int], float]:
        counts = lidu_ut.validate_result(
            case,
            before_cache,
            cache_slots,
            lidu_outputs,
            label=label,
        )
        expected_kpe, expected_ckv, payload_counts = expected_after_scatter(
            initial_kpe=initial_kpe_cpu,
            initial_ckv=initial_ckv_cpu,
            dram_kpe=dram_kpe_cpu,
            dram_ckv=dram_ckv_cpu,
            hbm_table=hbm_table_cpu,
            dram_table=dram_table_cpu,
            lidu_outputs=lidu_outputs,
        )
        if counts != payload_counts:
            raise AssertionError(f"{label}: LIDU and SCATTER counts differ")
        actual_kpe = hbm_kpe.cpu()
        actual_ckv = hbm_ckv.cpu()
        if not torch.equal(actual_kpe, expected_kpe):
            raise AssertionError(
                f"{label}: "
                + cache_mismatch_diagnostic(
                    name="COPYSFA KPE",
                    actual=actual_kpe,
                    expected=expected_kpe,
                    initial=initial_kpe_cpu,
                    hbm_table=hbm_table_cpu,
                    dram_cache=dram_kpe_cpu,
                    dram_table=dram_table_cpu,
                    lidu_outputs=lidu_outputs,
                )
            )
        if not torch.equal(actual_ckv, expected_ckv):
            raise AssertionError(
                f"{label}: "
                + cache_mismatch_diagnostic(
                    name="COPYSFA CKV",
                    actual=actual_ckv,
                    expected=expected_ckv,
                    initial=initial_ckv_cpu,
                    hbm_table=hbm_table_cpu,
                    dram_cache=dram_ckv_cpu,
                    dram_table=dram_table_cpu,
                    lidu_outputs=lidu_outputs,
                )
            )
        golden = attention_golden(
            query=query_cpu,
            query_rope=query_rope_cpu,
            kpe=expected_kpe,
            ckv=expected_ckv,
            hbm_table=hbm_table_cpu,
            sparse_slots=lidu_outputs[0].cpu(),
            cache_tokens=cache_tokens,
            tail_tokens=tail_tokens,
            scale=scale,
        )
        actual = attention.float().cpu()
        golden_atol = (
            KNOWN_FUSED_ATTENTION_ATOL
            if args.allow_fused_attention_diff
            else 0.08
        )
        torch.testing.assert_close(actual, golden, rtol=0.08, atol=golden_atol)
        return counts, float((actual - golden).abs().max())

    eager_cache = case.initial_cache_cpu.to(device)
    eager_kpe = initial_kpe_cpu.to(device)
    eager_ckv = initial_ckv_cpu.to(device)
    eager_attention_buffer = torch.empty(
        batch_size * QUERY_COUNT,
        args.heads,
        CKV_DIM,
        dtype=torch.bfloat16,
        device=device,
    )
    eager_outputs, eager_attention = launch(
        eager_cache,
        eager_kpe,
        eager_ckv,
        lidu_ut.make_outputs(case),
        eager_attention_buffer,
    )
    torch.npu.synchronize()
    eager_counts, eager_max_abs = validate_chain(
        label="mtp_offload_chain/eager",
        before_cache=case.initial_cache_cpu,
        cache_slots=eager_cache,
        hbm_kpe=eager_kpe,
        hbm_ckv=eager_ckv,
        lidu_outputs=eager_outputs,
        attention=eager_attention,
    )
    if max(eager_counts) <= TOPK or max(eager_counts) > UNION_CAPACITY:
        raise AssertionError(
            "chain coverage must include a union miss_count in (2048,8192]"
        )
    print(
        "MTP_OFFLOAD_CHAIN_CHECK "
        f"batch={batch_size} heads={args.heads} source_len={args.source_len} "
        f"cache_tokens={cache_tokens} tail_tokens={tail_tokens} "
        f"miss_counts={eager_counts} total_misses={sum(eager_counts)} "
        f"attention_max_abs={eager_max_abs:.6f} over_2048=1 ok=1",
        flush=True,
    )

    fused_cache = case.initial_cache_cpu.to(device)
    fused_kpe = initial_kpe_cpu.to(device)
    fused_ckv = initial_ckv_cpu.to(device)
    fused_attention_buffer = torch.empty_like(eager_attention_buffer)
    fused_outputs, fused_attention = launch_fused(
        fused_cache,
        fused_kpe,
        fused_ckv,
        lidu_ut.make_outputs(case),
        fused_attention_buffer,
    )
    torch.npu.synchronize()
    fused_counts, fused_max_abs = validate_chain(
        label="mtp_offload_chain/fused_eager",
        before_cache=case.initial_cache_cpu,
        cache_slots=fused_cache,
        hbm_kpe=fused_kpe,
        hbm_ckv=fused_ckv,
        lidu_outputs=fused_outputs,
        attention=fused_attention,
    )
    if fused_counts != eager_counts:
        raise AssertionError("split and fused eager miss counts differ")
    lidu_ut._compare_valid_outputs(
        case,
        fused_outputs,
        eager_outputs,
        label="mtp_offload_chain/split_fused",
    )
    if not torch.equal(fused_cache.cpu(), eager_cache.cpu()):
        raise AssertionError("split and fused LIDU states differ")
    if not torch.equal(fused_kpe.cpu(), eager_kpe.cpu()) or not torch.equal(
        fused_ckv.cpu(), eager_ckv.cpu()
    ):
        raise AssertionError("split and fused HBM cache payloads differ")
    split_fused_max_abs = float(
        (fused_attention.float() - eager_attention.float()).abs().max().cpu()
    )
    if not args.allow_fused_attention_diff:
        torch.testing.assert_close(fused_attention, eager_attention, rtol=0, atol=0)
    print(
        "FUSED_COPY_SFA_MTP_CHECK "
        f"batch={batch_size} misses={fused_counts} "
        "caller_owned_output=1 cache_alias_outputs=0 "
        f"attention_max_abs={fused_max_abs:.6f} "
        f"split_fused_max_abs={split_fused_max_abs:.6f} "
        f"split_exact={int(split_fused_max_abs == 0.0)} "
        f"known_diff_allowed={int(args.allow_fused_attention_diff)} ok=1",
        flush=True,
    )

    # Warm up LIDU plus the caller-owned fused interface on disposable state.
    warm_cache = case.initial_cache_cpu.to(device)
    warm_kpe = initial_kpe_cpu.to(device)
    warm_ckv = initial_ckv_cpu.to(device)
    warm_attention = torch.empty_like(eager_attention_buffer)
    launch_fused(
        warm_cache,
        warm_kpe,
        warm_ckv,
        lidu_ut.make_outputs(case),
        warm_attention,
    )
    torch.npu.synchronize()

    graph_cache = case.initial_cache_cpu.to(device)
    graph_kpe = initial_kpe_cpu.to(device)
    graph_ckv = initial_ckv_cpu.to(device)
    graph_buffers = lidu_ut.make_outputs(case)
    graph_attention_buffer = torch.empty_like(eager_attention_buffer)
    graph = torch.npu.NPUGraph()
    pool = torch.npu.graph_pool_handle()
    with torch.npu.graph(graph, pool=pool):
        graph_outputs, graph_attention = launch_fused(
            graph_cache,
            graph_kpe,
            graph_ckv,
            graph_buffers,
            graph_attention_buffer,
        )
    torch.npu.synchronize()

    # Capture may execute.  Replay from the exact pre-update cache state.
    graph_cache.copy_(case.initial_cache_cpu.to(device))
    graph_kpe.copy_(initial_kpe_cpu.to(device))
    graph_ckv.copy_(initial_ckv_cpu.to(device))
    torch.npu.synchronize()
    graph.replay()
    torch.npu.synchronize()
    graph_counts, graph_max_abs = validate_chain(
        label="mtp_offload_chain/graph",
        before_cache=case.initial_cache_cpu,
        cache_slots=graph_cache,
        hbm_kpe=graph_kpe,
        hbm_ckv=graph_ckv,
        lidu_outputs=graph_outputs,
        attention=graph_attention,
    )
    if graph_counts != fused_counts:
        raise AssertionError("eager and graph chain miss counts differ")
    lidu_ut._compare_valid_outputs(
        case, graph_outputs, fused_outputs, label="mtp_offload_chain/eager_graph"
    )
    if not torch.equal(graph_cache.cpu(), fused_cache.cpu()):
        raise AssertionError("eager and graph LIDU states differ")
    if not torch.equal(graph_kpe.cpu(), fused_kpe.cpu()) or not torch.equal(
        graph_ckv.cpu(), fused_ckv.cpu()
    ):
        raise AssertionError("eager and graph HBM cache payloads differ")
    torch.testing.assert_close(graph_attention, fused_attention, rtol=0, atol=0)

    # Identical replay must be zero-miss; SCATTER must leave HBM unchanged and
    # Attention must remain deterministic while consuming the same top-k slots.
    repeat_cache_before = graph_cache.cpu()
    repeat_kpe_before = graph_kpe.cpu()
    repeat_ckv_before = graph_ckv.cpu()
    repeat_attention_before = graph_attention.cpu()
    graph.replay()
    torch.npu.synchronize()
    repeat_counts = lidu_ut.validate_result(
        case,
        repeat_cache_before,
        graph_cache,
        graph_outputs,
        label="mtp_offload_chain/graph_repeat",
    )
    if any(repeat_counts):
        raise AssertionError("identical full-chain replay must be zero miss")
    if not torch.equal(graph_kpe.cpu(), repeat_kpe_before) or not torch.equal(
        graph_ckv.cpu(), repeat_ckv_before
    ):
        raise AssertionError("zero-miss full-chain replay modified HBM cache")
    repeat_attention = graph_attention.cpu()
    repeat_attention_max_abs = float(
        (repeat_attention.float() - repeat_attention_before.float()).abs().max()
    )
    if args.allow_fused_attention_diff:
        if repeat_attention_max_abs > KNOWN_FUSED_ATTENTION_ATOL:
            raise AssertionError(
                "zero-miss full-chain replay Attention difference exceeds "
                f"{KNOWN_FUSED_ATTENTION_ATOL}: max_abs={repeat_attention_max_abs}"
            )
    elif not torch.equal(repeat_attention, repeat_attention_before):
        raise AssertionError("zero-miss full-chain replay changed Attention output")
    print(
        "MTP_OFFLOAD_CHAIN_GRAPH_CHECK "
        f"replays={args.graph_replays} first_nonzero_miss=1 "
        "repeat_zero_miss=1 lidu_to_scatter_dependency=1 "
        "scatter_to_attention_dependency=1 out_buffer=1 "
        f"attention_max_abs={graph_max_abs:.6f} "
        f"repeat_attention_max_abs={repeat_attention_max_abs:.6f} "
        f"known_diff_allowed={int(args.allow_fused_attention_diff)} ok=1",
        flush=True,
    )

    if not args.skip_performance:
        perf_initial_cache_cpu, _ = lidu_ut._make_cache_state(
            topk_rows=case.topk_cpu,
            candidate_lens=(args.source_len,) * batch_size,
            cache_tokens=(cache_tokens,) * batch_size,
            req_pool_entries=case.req_pool_entries_cpu,
            source_capacity=case.source_capacity,
            miss_fractions=(0.0,) * batch_size,
            exact_miss_counts=(args.perf_miss_count,) * batch_size,
            generator=torch.Generator().manual_seed(args.seed + 8000),
            pool_size=batch_size + 3,
        )
        perf_case = replace(
            case,
            name="mtp_offload_chain_typical_perf",
            initial_cache_cpu=perf_initial_cache_cpu,
        )
        perf_initial_kpe_cpu, perf_initial_ckv_cpu = initialize_hbm(
            case=perf_case,
            cache_tokens=cache_tokens,
            final_kv_len=final_kv_len,
            dram_kpe=dram_kpe_cpu,
            dram_ckv=dram_ckv_cpu,
            dram_table=dram_table_cpu,
            hbm_table=hbm_table_cpu,
            hbm_blocks=hbm_blocks,
            generator=torch.Generator().manual_seed(args.seed + 8017),
        )
        perf_cache = perf_initial_cache_cpu.to(device)
        perf_outputs = lidu_ut.call_mtp_out(
            perf_case, perf_cache, *lidu_ut.make_outputs(perf_case)
        )
        torch.npu.synchronize()
        perf_counts = [int(value) for value in perf_outputs[4].cpu().tolist()]
        aligned_dram_reads = int((perf_outputs[1] >= 0).sum().cpu())
        expected_perf_counts = [args.perf_miss_count] * batch_size
        if perf_counts != expected_perf_counts:
            raise AssertionError(
                f"typical performance miss counts={perf_counts}, "
                f"expected={expected_perf_counts}"
            )

        split_perf_kpe = perf_initial_kpe_cpu.to(device)
        split_perf_ckv = perf_initial_ckv_cpu.to(device)
        split_perf_out = torch.empty_like(eager_attention_buffer)
        fused_perf_kpe = perf_initial_kpe_cpu.to(device)
        fused_perf_ckv = perf_initial_ckv_cpu.to(device)
        fused_perf_out = torch.empty_like(eager_attention_buffer)

        def split_copy_attention() -> torch.Tensor:
            kpe_alias, ckv_alias = torch.ops.nanovllm_dsa.scatter_copy.default(
                split_perf_kpe,
                split_perf_ckv,
                dram_kpe,
                dram_ckv,
                hbm_table,
                dram_table,
                perf_outputs[2],
                perf_outputs[3],
                perf_outputs[4],
            )
            return call_attention_out(
                query=query,
                query_rope=query_rope,
                kpe=kpe_alias,
                ckv=ckv_alias,
                sparse_slots=perf_outputs[0],
                cache_tokens=perf_case.cache_tokens,
                hbm_table=hbm_table,
                actual_q=actual_q,
                actual_kv=actual_kv,
                scale=scale,
                output=split_perf_out,
            )

        def fused_copy_attention() -> torch.Tensor:
            return torch.ops.nanovllm_dsa.fused_copy_sfa_mtp_out.default(
                query_rope,
                query,
                actual_q,
                actual_kv,
                perf_case.cache_tokens,
                perf_outputs[0],
                perf_outputs[1],
                perf_outputs[4],
                hbm_table,
                dram_table,
                fused_perf_kpe.view(-1, BLOCK_SIZE, 1, KPE_DIM),
                fused_perf_ckv.view(-1, BLOCK_SIZE, 1, CKV_DIM),
                dram_kpe,
                dram_ckv,
                scale,
                fused_perf_out,
            )

        def elapsed_ms(fn) -> float:
            for _ in range(args.warmup):
                fn()
            torch.npu.synchronize()
            start = torch.npu.Event(enable_timing=True)
            end = torch.npu.Event(enable_timing=True)
            start.record()
            for _ in range(args.iters):
                fn()
            end.record()
            end.synchronize()
            return float(start.elapsed_time(end)) / args.iters

        split_ms = elapsed_ms(split_copy_attention)
        fused_ms = elapsed_ms(fused_copy_attention)
        print(
            "FUSED_COPY_SFA_MTP_PERF_RESULT "
            f"batch={batch_size} miss_per_request={args.perf_miss_count} "
            f"total_misses={sum(perf_counts)} "
            f"aligned_dram_reads={aligned_dram_reads} "
            f"split_ms={split_ms:.6f} fused_ms={fused_ms:.6f} "
            f"speedup={split_ms / fused_ms:.4f} "
            "performance_assert=0 implementation=source_aware_v2 "
            f"warmup={args.warmup} iters={args.iters}",
            flush=True,
        )

    # Additional identical replays exercise persistent state without repeating
    # expensive CPU golden construction.
    for _ in range(max(args.graph_replays - 2, 0)):
        graph.replay()
    torch.npu.synchronize()
    print(
        "MTP_OFFLOAD_CHAIN_UT_OK",
        flush=True,
    )


def main() -> None:
    args = parse_args()
    validate_args(args)
    device = torch.device(args.device)
    if device.type != "npu":
        raise ValueError("--device must select one NPU, for example npu:0")
    torch.npu.set_device(device)
    torch.npu.config.allow_internal_format = False
    opapi_path = require_local_opapi()
    print(f"MTP_OFFLOAD_CHAIN_OPAPI path={opapi_path} local=1", flush=True)
    print(
        "MTP_OFFLOAD_CHAIN_CONFIG "
        f"device={device} query_len={QUERY_COUNT} batch={args.batch_size} "
        f"heads={args.heads} source_len={args.source_len} "
        f"cache_tokens={args.cache_tokens} tail_tokens={args.tail_tokens} "
        f"graph_replays={args.graph_replays} seed={args.seed}",
        flush=True,
    )
    run_chain(args, device)


if __name__ == "__main__":
    main()
