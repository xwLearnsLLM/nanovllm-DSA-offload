"""COPYSFA-MTP semantic, graph, and split-baseline checks."""

from __future__ import annotations

import argparse
import math

import torch
import torch_npu  # type: ignore

import nanovllm.ops  # noqa: F401  # Load repository-local custom operators.
from ut_ops._op_utils import require_local_opapi
import _mtp_fixture as fixture


QUERY_COUNT = fixture.QUERY_COUNT
BLOCK_SIZE = fixture.BLOCK_SIZE
TOPK = fixture.TOPK
UNION_CAPACITY = fixture.UNION_CAPACITY
CKV_DIM = fixture.CKV_DIM
KPE_DIM = fixture.KPE_DIM
KNOWN_FUSED_ATTENTION_ATOL = 0.1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate COPYSFA-MTP against SCATTER plus MTP3 sparse-and-tail "
            "Attention on one NPU."
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
        "--perf-miss-overlap-rate",
        type=float,
        default=1.0 / 3.0,
        help=(
            "Normalized overlap of unique misses across four queries: "
            "(query-level miss occurrences - unique misses) / "
            "(3 * unique misses). 0 means no repeated source read; 1 means "
            "every miss appears in all four queries. Default 1/3 makes each "
            "miss appear in two queries on average."
        ),
    )
    parser.add_argument(
        "--perf-hit-overlap-rate",
        type=float,
        default=0.0,
        help=(
            "Fraction of the per-query hit suffix shared by all four MTP "
            "queries in the performance-only fixture. 0 retains query-local "
            "hits; 1 makes the common hit prefix fully shared."
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
    parser.add_argument(
        "--diagnose-attention",
        action="store_true",
        help=(
            "Compare zero-miss HBM, nonzero-miss HBM-only, and nonzero-miss "
            "mixed-source COPYSFA-MTP paths for tail=0 and the configured tail."
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
        or args.source_len > fixture.MAX_SOURCE_CAPACITY
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
    if not 0.0 <= args.perf_miss_overlap_rate <= 1.0:
        raise ValueError("--perf-miss-overlap-rate must be in [0,1]")
    if not 0.0 <= args.perf_hit_overlap_rate <= 1.0:
        raise ValueError("--perf-hit-overlap-rate must be in [0,1]")
    query_miss_occurrences = args.perf_miss_count + round(
        3 * args.perf_miss_count * args.perf_miss_overlap_rate
    )
    if query_miss_occurrences > UNION_CAPACITY:
        raise ValueError(
            "the requested miss count/overlap needs more than four TopK2048 "
            "rows; reduce --perf-miss-count or --perf-miss-overlap-rate"
        )
    if args.perf_miss_count > args.source_len - args.cache_tokens:
        raise ValueError(
            "the performance fixture needs at least --perf-miss-count source "
            "tokens outside the initial cache"
        )


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
    case: fixture.MtpCase,
    cache_tokens: int,
    final_kv_len: int,
    dram_kpe: torch.Tensor,
    dram_ckv: torch.Tensor,
    dram_table: torch.Tensor,
    hbm_table: torch.Tensor,
    hbm_blocks: int,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Materialize the pre-update token-to-slot state in HBM."""

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
                f"request={request}: initial cache state does not contain C tokens"
            )
        source_ids[request] = sources.to(torch.int32)
        destination_slots[request] = state[sources]
    fixture.apply_scatter_reference(
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
    metadata: tuple[torch.Tensor, ...],
) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
    expected_kpe = initial_kpe.clone()
    expected_ckv = initial_ckv.clone()
    counts = metadata[4].cpu()
    fixture.apply_scatter_reference(
        expected_kpe,
        expected_ckv,
        dram_kpe,
        dram_ckv,
        hbm_table,
        dram_table,
        metadata[2].cpu(),
        metadata[3].cpu(),
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
    metadata: tuple[torch.Tensor, ...],
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
        counts = metadata[4].cpu().to(torch.int64)
        destinations = metadata[3].cpu().to(torch.int64)
        sources = metadata[2].cpu().to(torch.int64)
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
        aligned_sources = metadata[1].reshape(
            -1, QUERY_COUNT, TOPK
        )[request].cpu().to(torch.int64)
        aligned_slots = metadata[0].reshape(
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
    torch.ops.nanovllm_dsa.sparse_tail_attention_mtp.default(
        query_rope,
        query,
        actual_q,
        actual_kv,
        cache_tokens,
        sparse_slots,
        hbm_table,
        rope_cache,
        latent_cache,
        scale,
        output,
    )
    return output


def call_fused_attention_out(
    *,
    query: torch.Tensor,
    query_rope: torch.Tensor,
    actual_q: torch.Tensor,
    actual_kv: torch.Tensor,
    cache_tokens: torch.Tensor,
    topk_dst_slots: torch.Tensor,
    topk_src_ids: torch.Tensor,
    topk_miss_counts: torch.Tensor,
    miss_src_ids: torch.Tensor,
    miss_dst_slots: torch.Tensor,
    miss_counts: torch.Tensor,
    hbm_table: torch.Tensor,
    dram_table: torch.Tensor,
    hbm_kpe: torch.Tensor,
    hbm_ckv: torch.Tensor,
    dram_kpe: torch.Tensor,
    dram_ckv: torch.Tensor,
    scale: float,
    output: torch.Tensor,
) -> torch.Tensor:
    torch.ops.nanovllm_dsa.fused_copy_sfa_mtp.default(
        query_rope,
        query,
        actual_q,
        actual_kv,
        cache_tokens,
        topk_dst_slots,
        topk_src_ids,
        topk_miss_counts,
        miss_src_ids,
        miss_dst_slots,
        miss_counts,
        hbm_table,
        dram_table,
        hbm_kpe.view(-1, BLOCK_SIZE, 1, KPE_DIM),
        hbm_ckv.view(-1, BLOCK_SIZE, 1, CKV_DIM),
        dram_kpe,
        dram_ckv,
        scale,
        output,
    )
    return output


def attention_diff_by_query(
    lhs: torch.Tensor, rhs: torch.Tensor
) -> tuple[float, list[float]]:
    difference = (lhs.float() - rhs.float()).abs().reshape(
        -1, QUERY_COUNT, lhs.shape[-2], lhs.shape[-1]
    )
    per_query = difference.amax(dim=(0, 2, 3)).cpu().tolist()
    return float(difference.max().cpu()), [float(value) for value in per_query]


def validate_topk_miss_prefix(metadata: tuple[torch.Tensor, ...]) -> None:
    topk_src_ids = metadata[1].reshape(-1, TOPK).cpu()
    topk_miss_counts = metadata[5].reshape(-1).cpu()
    if topk_miss_counts.numel() != topk_src_ids.shape[0]:
        raise AssertionError(
            "topk_miss_counts must contain one entry for every MTP query"
        )
    for query_idx, count in enumerate(topk_miss_counts.tolist()):
        if count < 0 or count > TOPK:
            raise AssertionError(
                f"topk_miss_counts[{query_idx}]={count} is outside [0, {TOPK}]"
            )
        if count and not bool((topk_src_ids[query_idx, :count] >= 0).all()):
            raise AssertionError(
                f"query {query_idx} has a hit in its miss prefix"
            )
        if count < TOPK and not bool((topk_src_ids[query_idx, count:] < 0).all()):
            raise AssertionError(
                f"query {query_idx} has a miss after its hit suffix starts"
            )


def topk_reuse_stats(case: fixture.MtpCase) -> tuple[int, int, int, float, float]:
    total_positions = case.batch_size * QUERY_COUNT * TOPK
    unique_tokens = sum(int(union.numel()) for union in case.union_cpu)
    reuse_occurrences = total_positions - unique_tokens
    reuse_ratio = reuse_occurrences / total_positions
    normalized_overlap = reuse_occurrences / (
        case.batch_size * (QUERY_COUNT - 1) * TOPK
    )
    return (
        total_positions,
        unique_tokens,
        reuse_occurrences,
        reuse_ratio,
        normalized_overlap,
    )


def run_chain(args: argparse.Namespace, device: torch.device) -> None:
    batch_size = args.batch_size
    cache_tokens = args.cache_tokens
    tail_tokens = args.tail_tokens
    allow_known_fused_diff = (
        args.allow_fused_attention_diff or args.diagnose_attention
    )
    final_kv_len = cache_tokens + tail_tokens + QUERY_COUNT
    hbm_capacity = math.ceil(final_kv_len / BLOCK_SIZE) * BLOCK_SIZE
    case = fixture.make_case(
        name="mtp_offload_chain",
        device=device,
        batch_size=batch_size,
        source_capacity=args.source_len,
        cache_tokens=cache_tokens,
        miss_fractions=make_miss_fractions(batch_size),
        seed=args.seed + 5000,
        topk_profile="broad",
    )
    generator = torch.Generator().manual_seed(args.seed + 5017)
    dram_table_cpu, dram_blocks = fixture.random_block_table(
        batch_size, args.source_len // BLOCK_SIZE, generator
    )
    hbm_table_cpu, hbm_blocks = fixture.random_block_table(
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
    dram_kpe = fixture.swapped_from_cpu(dram_kpe_cpu, device)
    dram_ckv = fixture.swapped_from_cpu(dram_ckv_cpu, device)
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
        hbm_kpe: torch.Tensor,
        hbm_ckv: torch.Tensor,
        metadata: tuple[torch.Tensor, ...],
        attention_output: torch.Tensor,
    ) -> torch.Tensor:
        torch.ops.nanovllm_dsa.scatter_copy.default(
            metadata[2],
            metadata[3],
            metadata[4],
            hbm_table,
            dram_table,
            hbm_kpe,
            hbm_ckv,
            dram_kpe,
            dram_ckv,
        )
        attention = call_attention_out(
            query=query,
            query_rope=query_rope,
            kpe=hbm_kpe,
            ckv=hbm_ckv,
            sparse_slots=metadata[0],
            cache_tokens=case.cache_tokens,
            hbm_table=hbm_table,
            actual_q=actual_q,
            actual_kv=actual_kv,
            scale=scale,
            output=attention_output,
        )
        return attention

    def launch_fused(
        hbm_kpe: torch.Tensor,
        hbm_ckv: torch.Tensor,
        metadata: tuple[torch.Tensor, ...],
        attention_output: torch.Tensor,
    ) -> torch.Tensor:
        torch.ops.nanovllm_dsa.fused_copy_sfa_mtp.default(
            query_rope,
            query,
            actual_q,
            actual_kv,
            case.cache_tokens,
            metadata[0],
            metadata[1],
            metadata[5],
            metadata[2],
            metadata[3],
            metadata[4],
            hbm_table,
            dram_table,
            hbm_kpe.view(-1, BLOCK_SIZE, 1, KPE_DIM),
            hbm_ckv.view(-1, BLOCK_SIZE, 1, CKV_DIM),
            dram_kpe,
            dram_ckv,
            scale,
            attention_output,
        )
        return attention_output

    def validate_chain(
        *,
        label: str,
        hbm_kpe: torch.Tensor,
        hbm_ckv: torch.Tensor,
        metadata: tuple[torch.Tensor, ...],
        attention: torch.Tensor,
    ) -> tuple[list[int], float]:
        counts = [int(value) for value in metadata[4].cpu().tolist()]
        expected_kpe, expected_ckv, payload_counts = expected_after_scatter(
            initial_kpe=initial_kpe_cpu,
            initial_ckv=initial_ckv_cpu,
            dram_kpe=dram_kpe_cpu,
            dram_ckv=dram_ckv_cpu,
            hbm_table=hbm_table_cpu,
            dram_table=dram_table_cpu,
            metadata=metadata,
        )
        if counts != payload_counts:
            raise AssertionError(f"{label}: metadata and SCATTER counts differ")
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
                    metadata=metadata,
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
                    metadata=metadata,
                )
            )
        golden = attention_golden(
            query=query_cpu,
            query_rope=query_rope_cpu,
            kpe=expected_kpe,
            ckv=expected_ckv,
            hbm_table=hbm_table_cpu,
            sparse_slots=metadata[0].cpu(),
            cache_tokens=cache_tokens,
            tail_tokens=tail_tokens,
            scale=scale,
        )
        actual = attention.float().cpu()
        golden_atol = KNOWN_FUSED_ATTENTION_ATOL if allow_known_fused_diff else 0.08
        torch.testing.assert_close(actual, golden, rtol=0.08, atol=golden_atol)
        return counts, float((actual - golden).abs().max())

    eager_outputs = fixture.materialize_metadata(case)
    # This check copies metadata to host, so it must remain outside NPU Graph
    # capture.  launch_fused() itself is capture-safe.
    validate_topk_miss_prefix(eager_outputs)
    eager_kpe = initial_kpe_cpu.to(device)
    eager_ckv = initial_ckv_cpu.to(device)
    eager_attention_buffer = torch.empty(
        batch_size * QUERY_COUNT,
        args.heads,
        CKV_DIM,
        dtype=torch.bfloat16,
        device=device,
    )
    eager_attention = launch(
        eager_kpe,
        eager_ckv,
        eager_outputs,
        eager_attention_buffer,
    )
    torch.npu.synchronize()
    eager_counts, eager_max_abs = validate_chain(
        label="mtp_offload_chain/eager",
        hbm_kpe=eager_kpe,
        hbm_ckv=eager_ckv,
        metadata=eager_outputs,
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

    fused_kpe = initial_kpe_cpu.to(device)
    fused_ckv = initial_ckv_cpu.to(device)
    fused_attention_buffer = torch.empty_like(eager_attention_buffer)
    fused_attention = launch_fused(
        fused_kpe,
        fused_ckv,
        eager_outputs,
        fused_attention_buffer,
    )
    torch.npu.synchronize()
    fused_counts, fused_max_abs = validate_chain(
        label="mtp_offload_chain/fused_eager",
        hbm_kpe=fused_kpe,
        hbm_ckv=fused_ckv,
        metadata=eager_outputs,
        attention=fused_attention,
    )
    if fused_counts != eager_counts:
        raise AssertionError("split and fused eager miss counts differ")
    if not torch.equal(fused_kpe.cpu(), eager_kpe.cpu()) or not torch.equal(
        fused_ckv.cpu(), eager_ckv.cpu()
    ):
        raise AssertionError("split and fused HBM cache payloads differ")
    split_fused_max_abs = float(
        (fused_attention.float() - eager_attention.float()).abs().max().cpu()
    )
    if not allow_known_fused_diff:
        torch.testing.assert_close(fused_attention, eager_attention, rtol=0, atol=0)
    print(
        "FUSED_COPY_SFA_MTP_CHECK "
        f"batch={batch_size} misses={fused_counts} "
        "caller_owned_output=1 cache_alias_outputs=0 "
        f"attention_max_abs={fused_max_abs:.6f} "
        f"split_fused_max_abs={split_fused_max_abs:.6f} "
        f"split_exact={int(split_fused_max_abs == 0.0)} "
        f"known_diff_allowed={int(allow_known_fused_diff)} ok=1",
        flush=True,
    )

    if args.diagnose_attention:
        hbm_only_src_ids = torch.full_like(eager_outputs[1], -1)
        canonical_counts = torch.zeros_like(eager_outputs[4])
        nonzero_counts = torch.ones_like(eager_outputs[4])
        tail_cases = sorted({0, tail_tokens})
        hit_sentinel_diffs: list[float] = []

        def diagnostic_fused(
            *,
            kpe_seed: torch.Tensor,
            ckv_seed: torch.Tensor,
            topk_src_ids: torch.Tensor,
            topk_miss_counts: torch.Tensor,
            miss_counts: torch.Tensor,
            actual_kv: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            diagnostic_kpe = kpe_seed.clone()
            diagnostic_ckv = ckv_seed.clone()
            diagnostic_output = torch.empty_like(eager_attention_buffer)
            call_fused_attention_out(
                query=query,
                query_rope=query_rope,
                actual_q=actual_q,
                actual_kv=actual_kv,
                cache_tokens=case.cache_tokens,
                topk_dst_slots=eager_outputs[0],
                topk_src_ids=topk_src_ids,
                topk_miss_counts=topk_miss_counts,
                miss_src_ids=eager_outputs[2],
                miss_dst_slots=eager_outputs[3],
                miss_counts=miss_counts,
                hbm_table=hbm_table,
                dram_table=dram_table,
                hbm_kpe=diagnostic_kpe,
                hbm_ckv=diagnostic_ckv,
                dram_kpe=dram_kpe,
                dram_ckv=dram_ckv,
                scale=scale,
                output=diagnostic_output,
            )
            return diagnostic_output, diagnostic_kpe, diagnostic_ckv

        for diagnostic_tail in tail_cases:
            diagnostic_actual_kv = torch.full(
                (batch_size,),
                cache_tokens + diagnostic_tail + QUERY_COUNT,
                dtype=torch.int32,
                device=device,
            )
            split_output = torch.empty_like(eager_attention_buffer)
            call_attention_out(
                query=query,
                query_rope=query_rope,
                kpe=eager_kpe,
                ckv=eager_ckv,
                sparse_slots=eager_outputs[0],
                cache_tokens=case.cache_tokens,
                hbm_table=hbm_table,
                actual_q=actual_q,
                actual_kv=diagnostic_actual_kv,
                scale=scale,
                output=split_output,
            )

            canonical_output, canonical_kpe, canonical_ckv = diagnostic_fused(
                kpe_seed=eager_kpe,
                ckv_seed=eager_ckv,
                topk_src_ids=hbm_only_src_ids,
                topk_miss_counts=torch.zeros_like(eager_outputs[5]),
                miss_counts=canonical_counts,
                actual_kv=diagnostic_actual_kv,
            )
            nonzero_hbm_output, nonzero_hbm_kpe, nonzero_hbm_ckv = (
                diagnostic_fused(
                    kpe_seed=eager_kpe,
                    ckv_seed=eager_ckv,
                    topk_src_ids=hbm_only_src_ids,
                    topk_miss_counts=torch.zeros_like(eager_outputs[5]),
                    miss_counts=nonzero_counts,
                    actual_kv=diagnostic_actual_kv,
                )
            )
            mixed_output, mixed_kpe, mixed_ckv = diagnostic_fused(
                kpe_seed=initial_kpe_cpu.to(device),
                ckv_seed=initial_ckv_cpu.to(device),
                topk_src_ids=eager_outputs[1],
                topk_miss_counts=eager_outputs[5],
                miss_counts=eager_outputs[4],
                actual_kv=diagnostic_actual_kv,
            )
            torch.npu.synchronize()

            if not torch.equal(canonical_kpe, eager_kpe) or not torch.equal(
                canonical_ckv, eager_ckv
            ):
                raise AssertionError("canonical HBM-only diagnostic modified cache")
            if not torch.equal(nonzero_hbm_kpe, eager_kpe) or not torch.equal(
                nonzero_hbm_ckv, eager_ckv
            ):
                raise AssertionError("nonzero HBM-only diagnostic modified cache")
            if not torch.equal(mixed_kpe, eager_kpe) or not torch.equal(
                mixed_ckv, eager_ckv
            ):
                raise AssertionError("mixed-source diagnostic cache copy differs")

            comparisons = (
                ("split_vs_canonical_hbm", split_output, canonical_output),
                (
                    "canonical_vs_nonzero_hbm",
                    canonical_output,
                    nonzero_hbm_output,
                ),
                (
                    "nonzero_hbm_vs_nonzero_mixed",
                    nonzero_hbm_output,
                    mixed_output,
                ),
                ("split_vs_nonzero_mixed", split_output, mixed_output),
            )
            for pair, lhs, rhs in comparisons:
                max_abs, query_max_abs = attention_diff_by_query(lhs, rhs)
                if pair == "canonical_vs_nonzero_hbm":
                    hit_sentinel_diffs.append(max_abs)
                formatted_query_max = ",".join(
                    f"{value:.6f}" for value in query_max_abs
                )
                print(
                    "FUSED_COPY_SFA_MTP_ATTENTION_DIAGNOSTIC "
                    f"tail_tokens={diagnostic_tail} pair={pair} "
                    f"max_abs={max_abs:.6f} "
                    f"query_max_abs=[{formatted_query_max}] "
                    "cache_exact=1",
                    flush=True,
                )

        hit_sentinel_max_abs = max(hit_sentinel_diffs, default=0.0)
        if hit_sentinel_max_abs != 0.0:
            raise AssertionError(
                "nonzero request-level miss_count changed an all-HBM "
                "Attention gather: "
                f"max_abs={hit_sentinel_max_abs:.9f}"
            )
        print(
            "FUSED_COPY_SFA_MTP_HIT_SENTINEL_CHECK "
            f"tail_cases={tail_cases} max_abs={hit_sentinel_max_abs:.9f} "
            "exact=1 ok=1",
            flush=True,
        )

    # Warm up the caller-owned fused interface on disposable state.
    warm_kpe = initial_kpe_cpu.to(device)
    warm_ckv = initial_ckv_cpu.to(device)
    warm_attention = torch.empty_like(eager_attention_buffer)
    launch_fused(
        warm_kpe,
        warm_ckv,
        eager_outputs,
        warm_attention,
    )
    torch.npu.synchronize()

    graph_kpe = initial_kpe_cpu.to(device)
    graph_ckv = initial_ckv_cpu.to(device)
    graph_metadata = tuple(tensor.clone() for tensor in eager_outputs)
    graph_attention_buffer = torch.empty_like(eager_attention_buffer)
    graph = torch.npu.NPUGraph()
    pool = torch.npu.graph_pool_handle()
    with torch.npu.graph(graph, pool=pool):
        graph_attention = launch_fused(
            graph_kpe,
            graph_ckv,
            graph_metadata,
            graph_attention_buffer,
        )
    torch.npu.synchronize()

    # Capture may execute.  Replay from the exact pre-update cache state.
    graph_kpe.copy_(initial_kpe_cpu.to(device))
    graph_ckv.copy_(initial_ckv_cpu.to(device))
    torch.npu.synchronize()
    graph.replay()
    torch.npu.synchronize()
    graph_counts, graph_max_abs = validate_chain(
        label="mtp_offload_chain/graph",
        hbm_kpe=graph_kpe,
        hbm_ckv=graph_ckv,
        metadata=graph_metadata,
        attention=graph_attention,
    )
    if graph_counts != fused_counts:
        raise AssertionError("eager and graph chain miss counts differ")
    if not torch.equal(graph_kpe.cpu(), fused_kpe.cpu()) or not torch.equal(
        graph_ckv.cpu(), fused_ckv.cpu()
    ):
        raise AssertionError("eager and graph HBM cache payloads differ")
    torch.testing.assert_close(graph_attention, fused_attention, rtol=0, atol=0)

    # Reapplying identical copy metadata must be idempotent and Attention must
    # remain deterministic while consuming the same TopK slots.
    repeat_kpe_before = graph_kpe.cpu()
    repeat_ckv_before = graph_ckv.cpu()
    repeat_attention_before = graph_attention.cpu()
    graph.replay()
    torch.npu.synchronize()
    if not torch.equal(graph_kpe.cpu(), repeat_kpe_before) or not torch.equal(
        graph_ckv.cpu(), repeat_ckv_before
    ):
        raise AssertionError("identical metadata replay changed HBM cache")
    repeat_attention = graph_attention.cpu()
    repeat_attention_max_abs = float(
        (repeat_attention.float() - repeat_attention_before.float()).abs().max()
    )
    if allow_known_fused_diff:
        if repeat_attention_max_abs > KNOWN_FUSED_ATTENTION_ATOL:
            raise AssertionError(
                "identical metadata replay Attention difference exceeds "
                f"{KNOWN_FUSED_ATTENTION_ATOL}: max_abs={repeat_attention_max_abs}"
            )
    elif not torch.equal(repeat_attention, repeat_attention_before):
        raise AssertionError("identical metadata replay changed Attention output")
    print(
        "MTP_OFFLOAD_CHAIN_GRAPH_CHECK "
        f"replays={args.graph_replays} first_nonzero_miss=1 "
        "repeat_copy_idempotent=1 metadata_to_copy_dependency=1 "
        "scatter_to_attention_dependency=1 out_buffer=1 "
        f"attention_max_abs={graph_max_abs:.6f} "
        f"repeat_attention_max_abs={repeat_attention_max_abs:.6f} "
        f"known_diff_allowed={int(allow_known_fused_diff)} ok=1",
        flush=True,
    )

    if not args.skip_performance:
        perf_case = fixture.make_case(
            name="mtp_offload_chain_typical_perf",
            device=device,
            batch_size=batch_size,
            source_capacity=case.source_capacity,
            cache_tokens=cache_tokens,
            miss_fractions=(0.0,) * batch_size,
            exact_miss_count=args.perf_miss_count,
            miss_overlap_rate=args.perf_miss_overlap_rate,
            hit_overlap_rate=args.perf_hit_overlap_rate,
            seed=args.seed + 8000,
            topk_profile="miss_overlap",
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
        perf_outputs = fixture.materialize_metadata(perf_case)
        validate_topk_miss_prefix(perf_outputs)
        (
            topk_positions_total,
            topk_unique_tokens_total,
            topk_reuse_occurrences_total,
            topk_reuse_ratio,
            topk_overlap_rate_actual,
        ) = topk_reuse_stats(perf_case)
        perf_counts = [int(value) for value in perf_outputs[4].cpu().tolist()]
        query_miss_occurrences = int((perf_outputs[1] >= 0).sum().cpu())
        (
            query_miss_occurrences_mean,
            actual_overlap_rate,
        ) = fixture.metadata_stats(perf_case, perf_outputs)
        query_miss_means = (
            (perf_outputs[1].view(batch_size, QUERY_COUNT, TOPK) >= 0)
            .sum(dim=2)
            .float()
            .mean(dim=0)
            .cpu()
            .tolist()
        )
        expected_perf_counts = [args.perf_miss_count] * batch_size
        if perf_counts != expected_perf_counts:
            raise AssertionError(
                f"typical performance miss counts={perf_counts}, "
                f"expected={expected_perf_counts}"
            )
        expected_query_misses = args.perf_miss_count + round(
            3 * args.perf_miss_count * args.perf_miss_overlap_rate
        )
        if query_miss_occurrences != batch_size * expected_query_misses:
            raise AssertionError(
                "constructed query-level miss occurrences="
                f"{query_miss_occurrences}, expected="
                f"{batch_size * expected_query_misses}"
            )
        formatted_query_miss_means = ",".join(
            f"{value:.2f}" for value in query_miss_means
        )

        split_perf_kpe = perf_initial_kpe_cpu.to(device)
        split_perf_ckv = perf_initial_ckv_cpu.to(device)
        split_perf_out = torch.empty_like(eager_attention_buffer)
        fused_perf_kpe = perf_initial_kpe_cpu.to(device)
        fused_perf_ckv = perf_initial_ckv_cpu.to(device)
        fused_perf_out = torch.empty_like(eager_attention_buffer)

        def scatter_copy_only() -> None:
            torch.ops.nanovllm_dsa.scatter_copy.default(
                perf_outputs[2],
                perf_outputs[3],
                perf_outputs[4],
                hbm_table,
                dram_table,
                split_perf_kpe,
                split_perf_ckv,
                dram_kpe,
                dram_ckv,
            )

        def sparse_attention_only() -> torch.Tensor:
            return call_attention_out(
                query=query,
                query_rope=query_rope,
                kpe=split_perf_kpe,
                ckv=split_perf_ckv,
                sparse_slots=perf_outputs[0],
                cache_tokens=perf_case.cache_tokens,
                hbm_table=hbm_table,
                actual_q=actual_q,
                actual_kv=actual_kv,
                scale=scale,
                output=split_perf_out,
            )

        def split_copy_attention() -> torch.Tensor:
            scatter_copy_only()
            return sparse_attention_only()

        def fused_copy_attention() -> torch.Tensor:
            torch.ops.nanovllm_dsa.fused_copy_sfa_mtp.default(
                query_rope,
                query,
                actual_q,
                actual_kv,
                perf_case.cache_tokens,
                perf_outputs[0],
                perf_outputs[1],
                perf_outputs[5],
                perf_outputs[2],
                perf_outputs[3],
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
            return fused_perf_out

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
        scatter_ms = elapsed_ms(scatter_copy_only)
        sfa_ms = elapsed_ms(sparse_attention_only)
        split_component_sum_ms = scatter_ms + sfa_ms
        print(
            "FUSED_COPY_SFA_MTP_PERF_RESULT "
            f"batch={batch_size} "
            f"unique_misses_per_request={args.perf_miss_count} "
            f"total_unique_misses={sum(perf_counts)} "
            f"query_miss_occurrences_total={query_miss_occurrences} "
            "query_miss_occurrences_per_request="
            f"{query_miss_occurrences_mean:.2f} "
            f"query_miss_occurrences_by_query=[{formatted_query_miss_means}] "
            f"miss_overlap_rate_requested={args.perf_miss_overlap_rate:.6f} "
            f"miss_overlap_rate_actual={actual_overlap_rate:.6f} "
            f"hit_overlap_rate_requested={args.perf_hit_overlap_rate:.6f} "
            f"topk_positions_total={topk_positions_total} "
            f"topk_unique_tokens_total={topk_unique_tokens_total} "
            f"topk_reuse_occurrences_total={topk_reuse_occurrences_total} "
            f"topk_reuse_ratio={topk_reuse_ratio:.6f} "
            f"topk_overlap_rate_actual={topk_overlap_rate_actual:.6f} "
            f"split_ms={split_ms:.6f} "
            f"kvcache_scatter_copy_ms={scatter_ms:.6f} "
            f"sparse_tail_attention_mtp_ms={sfa_ms:.6f} "
            f"split_component_sum_ms={split_component_sum_ms:.6f} "
            f"fused_ms={fused_ms:.6f} "
            f"speedup={split_ms / fused_ms:.4f} "
            "query_gather_writeback=1 performance_assert=0 "
            "implementation=source_aware_all_hit_pair_v6 "
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
        f"perf_unique_misses={args.perf_miss_count} "
        f"perf_miss_overlap_rate={args.perf_miss_overlap_rate:.6f} "
        f"graph_replays={args.graph_replays} seed={args.seed}",
        flush=True,
    )
    run_chain(args, device)


if __name__ == "__main__":
    main()
