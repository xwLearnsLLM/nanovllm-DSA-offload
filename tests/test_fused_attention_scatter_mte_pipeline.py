#!/usr/bin/env python3
"""Check and tune the promoted A5 fused Attention/SCATTER MTE pipeline."""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import torch

import nanovllm_dsa_a5
import torch_npu  # type: ignore  # noqa: E402,F401

from test_fused_attention_scatter import (
    BLOCK_SIZE,
    CKV_DIM,
    KPE_DIM,
    SPARSE_COUNT,
    Case,
    active_physical_token_indices,
    build_cpu_attention_golden,
    event_benchmark,
    guard_physical_token_indices,
    make_case,
)


FUTURE_WORKSPACE_MAX_MISS = 400


@dataclass
class PipelineCase:
    base: Case
    baseline_kpe: torch.Tensor
    baseline_ckv: torch.Tensor
    pipeline_kpe: torch.Tensor
    pipeline_ckv: torch.Tensor
    zero_counts: torch.Tensor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="npu:0")
    parser.add_argument(
        "--mode", choices=("all", "check", "bench", "profile"),
        default="all",
    )
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--source-len", type=int, default=65536)
    parser.add_argument("--cache-tokens", type=int, default=8192)
    parser.add_argument("--tail-tokens", type=int, default=64)
    parser.add_argument("--miss-min", type=int, default=0)
    parser.add_argument("--miss-max", type=int, default=300)
    parser.add_argument("--prefetch-rows", default="5")
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--profile-replays", type=int, default=4)
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> tuple[int, ...]:
    if not 0 <= args.miss_min <= args.miss_max <= SPARSE_COUNT:
        raise ValueError(
            "--miss-min/--miss-max must satisfy "
            f"0 <= min <= max <= {SPARSE_COUNT}."
        )
    if args.warmup < 0 or args.iters <= 0:
        raise ValueError("--warmup must be non-negative and --iters positive.")
    if args.profile_replays <= 0:
        raise ValueError("--profile-replays must be positive.")
    prefetch_rows = tuple(
        int(item.strip())
        for item in args.prefetch_rows.split(",")
        if item.strip()
    )
    if not prefetch_rows or any(
        rows < 0 or rows > 16 for rows in prefetch_rows
    ):
        raise ValueError("--prefetch-rows must contain values in [0,16].")
    if args.mode == "profile" and len(prefetch_rows) != 1:
        raise ValueError("--mode profile requires one --prefetch-rows value.")
    return prefetch_rows


def make_pipeline_case(args: argparse.Namespace) -> PipelineCase:
    base_args = argparse.Namespace(
        device=args.device,
        batch_size=args.batch_size,
        heads=args.heads,
        source_len=args.source_len,
        cache_tokens=args.cache_tokens,
        tail_tokens=args.tail_tokens,
        miss_min=args.miss_min,
        miss_max=args.miss_max,
        seed=args.seed,
    )
    base = make_case(base_args)
    initial_kpe = base.fused_kpe
    initial_ckv = base.fused_ckv
    return PipelineCase(
        base=base,
        baseline_kpe=initial_kpe.clone(),
        baseline_ckv=initial_ckv.clone(),
        pipeline_kpe=initial_kpe.clone(),
        pipeline_ckv=initial_ckv.clone(),
        zero_counts=torch.zeros_like(base.copy_counts),
    )


def launch_baseline(case: PipelineCase) -> torch.Tensor:
    base = case.base
    output, _, _ = (
        nanovllm_dsa_a5
        .sparse_and_tail_attention_and_scatter_copy(
            base.query,
            case.baseline_ckv,
            base.sparse_slots,
            base.cache_tokens,
            base.hbm_block_table,
            base.actual_q,
            base.actual_kv,
            base.query_rope,
            case.baseline_kpe,
            base.dram_kpe,
            base.dram_ckv,
            base.dram_block_table,
            base.source_token_ids,
            base.copy_counts,
            base.scale,
        )
    )
    return output


def launch_pipeline(
    case: PipelineCase,
    prefetch_rows: int,
    copy_counts: torch.Tensor | None = None,
) -> torch.Tensor:
    base = case.base
    output, _, _ = (
        nanovllm_dsa_a5
        .sparse_and_tail_attention_and_scatter_copy(
            base.query,
            case.pipeline_ckv,
            base.sparse_slots,
            base.cache_tokens,
            base.hbm_block_table,
            base.actual_q,
            base.actual_kv,
            base.query_rope,
            case.pipeline_kpe,
            base.dram_kpe,
            base.dram_ckv,
            base.dram_block_table,
            base.source_token_ids,
            base.copy_counts if copy_counts is None else copy_counts,
            base.scale,
            prefetch_rows,
        )
    )
    return output


def launch_scatter(case: PipelineCase) -> tuple[torch.Tensor, torch.Tensor]:
    base = case.base
    return nanovllm_dsa_a5.scatter_copy(
        base.serial_kpe.view(-1, BLOCK_SIZE, KPE_DIM),
        base.serial_ckv.view(-1, BLOCK_SIZE, CKV_DIM),
        base.dram_kpe,
        base.dram_ckv,
        base.hbm_block_table,
        base.dram_block_table,
        base.source_token_ids,
        base.sparse_slots.view(base.sparse_slots.size(0), SPARSE_COUNT),
        base.copy_counts,
    )


def launch_sfa(case: PipelineCase) -> torch.Tensor:
    base = case.base
    return nanovllm_dsa_a5.sparse_and_tail_attention(
        base.query,
        base.serial_ckv,
        base.serial_ckv,
        base.sparse_slots,
        base.cache_tokens,
        base.hbm_block_table,
        base.actual_q,
        base.actual_kv,
        base.query_rope,
        base.serial_kpe,
        base.scale,
    )


def launch_serial(case: PipelineCase) -> torch.Tensor:
    launch_scatter(case)
    return launch_sfa(case)


def assert_pipeline_cache_exact(
    case: PipelineCase,
    destination_batches: torch.Tensor,
    destination_rows: torch.Tensor,
    source_rows: torch.Tensor,
    actual_ckv: torch.Tensor,
    actual_kpe: torch.Tensor,
    expected_ckv: torch.Tensor,
    expected_kpe: torch.Tensor,
    prefetch_rows: int,
) -> None:
    bad_ckv = (actual_ckv != expected_ckv).any(dim=1)
    bad_kpe = (actual_kpe != expected_kpe).any(dim=1)
    bad = bad_ckv | bad_kpe
    if bool(bad.any()):
        bad_indices = bad.nonzero().flatten()
        counts = case.base.copy_counts.cpu().tolist()
        positions = torch.cat(
            [torch.arange(int(count)) for count in counts]
        )
        bad_by_batch = []
        for batch in destination_batches[bad_indices].unique().tolist():
            batch_mask = destination_batches[bad_indices] == batch
            bad_by_batch.append(
                f"{batch}:{int(batch_mask.sum())}/{int(counts[batch])}"
            )
        print(
            "A5_MTE_PIPELINE_CACHE_MISMATCH "
            f"prefetch_rows={prefetch_rows} "
            f"bad_ckv_rows={int(bad_ckv.sum())} "
            f"bad_kpe_rows={int(bad_kpe.sum())} "
            f"bad_by_batch={','.join(bad_by_batch)}",
            flush=True,
        )
        cache_tokens = case.base.cache_tokens.cpu()
        actual_kv = case.base.actual_kv.cpu()
        for active_index in bad_indices[:32].tolist():
            batch = int(destination_batches[active_index])
            miss_pos = int(positions[active_index])
            tail_tokens = int(actual_kv[batch] - cache_tokens[batch])
            active_tail_end = SPARSE_COUNT - int(counts[batch]) + tail_tokens
            virtual_row = active_tail_end + miss_pos
            tile_row = virtual_row % BLOCK_SIZE
            total_virtual_rows = SPARSE_COUNT + tail_tokens
            tile_start = virtual_row // BLOCK_SIZE * BLOCK_SIZE
            tile_size = min(
                BLOCK_SIZE, total_virtual_rows - tile_start
            )
            pair_count = (tile_size + 1) // 2
            first_owner_rows = min(
                ((pair_count + 1) // 2) * 2, tile_size
            )
            scheduled_owner = int(tile_row >= first_owner_rows)
            ckv_max_abs = float(
                (actual_ckv[active_index].float() -
                 expected_ckv[active_index].float()).abs().max()
            )
            kpe_max_abs = float(
                (actual_kpe[active_index].float() -
                 expected_kpe[active_index].float()).abs().max()
            )
            print(
                "A5_MTE_PIPELINE_BAD_ROW "
                f"active_index={active_index} batch={batch} "
                f"miss_pos={miss_pos} count={counts[batch]} "
                f"virtual_row_mod128={tile_row} "
                f"owner={scheduled_owner} "
                f"source_physical={int(source_rows[active_index])} "
                f"destination_physical="
                f"{int(destination_rows[active_index])} "
                f"ckv_max_abs={ckv_max_abs:.9f} "
                f"kpe_max_abs={kpe_max_abs:.9f}",
                flush=True,
            )
        raise AssertionError(
            "MTE pipeline did not persist every active CKV/KPE row."
        )

    torch.testing.assert_close(actual_ckv, expected_ckv, rtol=0, atol=0)
    torch.testing.assert_close(actual_kpe, expected_kpe, rtol=0, atol=0)


def check_semantics(
    case: PipelineCase, prefetch_rows_values: tuple[int, ...]
) -> None:
    base = case.base
    destination_batches, destination_cpu = active_physical_token_indices(
        base.hbm_block_table,
        base.sparse_slots[:, 0, :],
        base.copy_counts,
    )
    _, source_cpu = active_physical_token_indices(
        base.dram_block_table,
        base.source_token_ids,
        base.copy_counts,
    )
    destination = destination_cpu.to(base.query.device)
    guard_cpu = guard_physical_token_indices(base)
    guard = guard_cpu.to(base.query.device)
    initial_pipeline_ckv_guard = (
        case.pipeline_ckv.view(-1, CKV_DIM)[guard].cpu().clone()
    )
    initial_pipeline_kpe_guard = (
        case.pipeline_kpe.view(-1, KPE_DIM)[guard].cpu().clone()
    )
    cpu_golden = build_cpu_attention_golden(base, source_cpu)

    baseline_output = launch_baseline(case)
    torch.npu.synchronize()
    baseline_output_cpu = baseline_output.cpu().float()
    torch.testing.assert_close(
        baseline_output_cpu, cpu_golden, rtol=8e-2, atol=8e-2
    )
    expected_ckv = base.dram_ckv_cpu.view(-1, CKV_DIM)[source_cpu]
    expected_kpe = base.dram_kpe_cpu.view(-1, KPE_DIM)[source_cpu]

    for rows in prefetch_rows_values:
        case.pipeline_kpe.copy_(base.fused_kpe)
        case.pipeline_ckv.copy_(base.fused_ckv)
        pipeline_output = launch_pipeline(case, rows)
        torch.npu.synchronize()
        pipeline_output_cpu = pipeline_output.cpu().float()
        torch.testing.assert_close(
            pipeline_output_cpu, cpu_golden, rtol=8e-2, atol=8e-2
        )
        pipeline_ckv_rows = (
            case.pipeline_ckv.view(-1, CKV_DIM)[destination].cpu()
        )
        pipeline_kpe_rows = (
            case.pipeline_kpe.view(-1, KPE_DIM)[destination].cpu()
        )
        assert_pipeline_cache_exact(
            case,
            destination_batches,
            destination_cpu,
            source_cpu,
            pipeline_ckv_rows,
            pipeline_kpe_rows,
            expected_ckv,
            expected_kpe,
            rows,
        )
        torch.testing.assert_close(
            case.pipeline_ckv.view(-1, CKV_DIM)[guard].cpu(),
            initial_pipeline_ckv_guard,
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            case.pipeline_kpe.view(-1, KPE_DIM)[guard].cpu(),
            initial_pipeline_kpe_guard,
            rtol=0,
            atol=0,
        )
        output_diff = (pipeline_output_cpu - baseline_output_cpu).abs()
        old_tolerance_close = torch.isclose(
            pipeline_output_cpu,
            baseline_output_cpu,
            rtol=2e-2,
            atol=2e-2,
        )
        cosine = torch.nn.functional.cosine_similarity(
            pipeline_output_cpu.flatten(),
            baseline_output_cpu.flatten(),
            dim=0,
        )
        print(
            "A5_MTE_PIPELINE_CHECK output=cpu_golden_close cache=exact "
            f"guards=unchanged prefetch_rows={rows} "
            f"baseline_pipeline_max_abs={float(output_diff.max()):.9f} "
            f"baseline_pipeline_mean_abs={float(output_diff.mean()):.9f} "
            f"old_tolerance_close_fraction="
            f"{float(old_tolerance_close.float().mean()):.9f} "
            f"baseline_pipeline_cosine={float(cosine):.9f}",
            flush=True,
        )


def benchmark(
    case: PipelineCase,
    prefetch_rows_values: tuple[int, ...],
    warmup: int,
    iters: int,
) -> None:
    base = case.base
    payload = int(base.copy_counts.sum()) * (CKV_DIM + KPE_DIM) * 2
    for rows in prefetch_rows_values:
        baseline_ms = event_benchmark(
            lambda: launch_baseline(case), warmup, iters
        )
        scatter_ms = event_benchmark(
            lambda: launch_scatter(case), warmup, iters
        )
        sfa_ms = event_benchmark(
            lambda: launch_sfa(case), warmup, iters
        )
        serial_ms = event_benchmark(
            lambda: launch_serial(case), warmup, iters
        )
        pipeline_sfa_ms = event_benchmark(
            lambda: launch_pipeline(case, rows, case.zero_counts),
            warmup,
            iters,
        )
        pipeline_ms = event_benchmark(
            lambda: launch_pipeline(case, rows), warmup, iters
        )
        component_serial_ms = scatter_ms + sfa_ms
        hidden_copy_ms = component_serial_ms - pipeline_ms
        overlap_ratio = (
            hidden_copy_ms / scatter_ms if scatter_ms > 0 else 0.0
        )
        payload_gbs = payload / pipeline_ms / 1.0e6
        print(
            "A5_MTE_PIPELINE_RESULT "
            f"batch={base.query.size(0)} heads={base.query.size(1)} "
            f"miss_min={int(base.copy_counts.min())} "
            f"miss_max={int(base.copy_counts.max())} "
            f"miss_mean={float(base.copy_counts.float().mean()):.3f} "
            f"prefetch_rows={rows} baseline_fused_ms={baseline_ms:.6f} "
            f"scatter_ms={scatter_ms:.6f} sfa_ms={sfa_ms:.6f} "
            f"serial_ms={serial_ms:.6f} "
            f"pipeline_sfa_ms={pipeline_sfa_ms:.6f} "
            f"pipeline_fused_ms={pipeline_ms:.6f} "
            f"component_serial_ms={component_serial_ms:.6f} "
            f"hidden_copy_ms={hidden_copy_ms:.6f} "
            f"overlap_ratio={overlap_ratio:.6f} "
            f"payload_GBs={payload_gbs:.6f} "
            f"speedup_vs_baseline={baseline_ms / pipeline_ms:.6f} "
            f"warmup={warmup} iters={iters}",
            flush=True,
        )


def profile(case: PipelineCase, rows: int, replays: int) -> None:
    launch_scatter(case)
    launch_sfa(case)
    launch_baseline(case)
    launch_pipeline(case, rows)
    torch.npu.synchronize()
    print(f"A5_MTE_PROFILE_SCATTER_BEGIN replays={replays}", flush=True)
    for _ in range(replays):
        launch_scatter(case)
    torch.npu.synchronize()
    print("A5_MTE_PROFILE_SCATTER_END", flush=True)
    print(f"A5_MTE_PROFILE_SFA_BEGIN replays={replays}", flush=True)
    for _ in range(replays):
        launch_sfa(case)
    torch.npu.synchronize()
    print("A5_MTE_PROFILE_SFA_END", flush=True)
    print(f"A5_MTE_PROFILE_BASELINE_BEGIN replays={replays}", flush=True)
    for _ in range(replays):
        launch_baseline(case)
    torch.npu.synchronize()
    print("A5_MTE_PROFILE_BASELINE_END", flush=True)
    print(f"A5_MTE_PROFILE_PIPELINE_BEGIN replays={replays}", flush=True)
    for _ in range(replays):
        launch_pipeline(case, rows)
    torch.npu.synchronize()
    print("A5_MTE_PROFILE_PIPELINE_END", flush=True)


def main() -> None:
    args = parse_args()
    prefetch_rows_values = validate_args(args)
    torch.npu.set_device(torch.device(args.device))
    case = make_pipeline_case(args)
    if args.miss_max == 0:
        workspace_path = "zero_miss_sfa"
    elif args.miss_max <= FUTURE_WORKSPACE_MAX_MISS:
        workspace_path = "future_workspace"
    elif args.miss_min > FUTURE_WORKSPACE_MAX_MISS:
        workspace_path = "three_workspace_fallback"
    else:
        workspace_path = "mixed_future_and_fallback"
    print(
        "A5_MTE_PIPELINE_CONFIG "
        f"mode={args.mode} batch={args.batch_size} heads={args.heads} "
        f"miss_range=[{args.miss_min},{args.miss_max}] "
        f"prefetch_rows={prefetch_rows_values} "
        f"workspace_path={workspace_path} "
        f"opapi={nanovllm_dsa_a5.local_opapi_path()}",
        flush=True,
    )
    if args.mode in ("all", "check"):
        check_semantics(case, prefetch_rows_values)
    if args.mode in ("all", "bench"):
        benchmark(case, prefetch_rows_values, args.warmup, args.iters)
    if args.mode == "profile":
        profile(case, prefetch_rows_values[0], args.profile_replays)
    print("A5_MTE_PIPELINE_TEST_OK", flush=True)


if __name__ == "__main__":
    main()
