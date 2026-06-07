from __future__ import annotations

import argparse
from time import perf_counter

import torch

from nanovllm.models.dsa_indexer_project import dsa_indexer_pipeline_with_qc_eager, dsa_indexer_pipeline_with_qc_torchair
from ut_ops.common.device import set_device, sync_device
from ut_ops.common.format import tensor_desc


def rand_bf16(shape: tuple[int, ...], device: torch.device) -> torch.Tensor:
    return torch.randn(shape, dtype=torch.float32, device=device).to(torch.bfloat16).contiguous()


def make_cos_sin(tokens: int, rope_dim: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    base = torch.arange(tokens * rope_dim, dtype=torch.float32, device=device).view(tokens, rope_dim) / max(rope_dim, 1)
    return base.cos().to(torch.bfloat16).view(tokens, 1, 1, rope_dim).contiguous(), base.sin().to(torch.bfloat16).view(tokens, 1, 1, rope_dim).contiguous()


def make_inputs(args, device: torch.device, seed: int) -> dict[str, torch.Tensor]:
    torch.manual_seed(seed)
    sparse_blocks = (args.topk + args.block_size - 1) // args.block_size
    full_blocks_per_req = (args.full_len + args.block_size - 1) // args.block_size
    hidden_states = rand_bf16((args.batch_size, args.hidden_size), device)
    cos, sin = make_cos_sin(args.batch_size, args.rope_dim, device)
    selection_blocks = args.batch_size * sparse_blocks
    full_blocks = args.batch_size * full_blocks_per_req
    return {
        "hidden_states": hidden_states,
        "cos": cos,
        "sin": sin,
        "q_a_weight": rand_bf16((args.q_lora_rank, args.hidden_size), device),
        "q_norm_weight": rand_bf16((args.q_lora_rank,), device),
        "wq_b_weight": rand_bf16((args.n_head * args.head_dim, args.q_lora_rank), device),
        "weights_proj_weight": rand_bf16((args.n_head, args.hidden_size), device),
        "index_cache": rand_bf16((full_blocks, args.block_size, 1, args.head_dim), device),
        "candidate_query_lens": torch.arange(1, args.batch_size + 1, dtype=torch.int32, device=device),
        "candidate_lens": torch.full((args.batch_size,), args.full_len, dtype=torch.int32, device=device),
        "index_tables": torch.arange(full_blocks, dtype=torch.int32, device=device).view(args.batch_size, full_blocks_per_req),
        "selection_kpe": torch.zeros((selection_blocks, args.block_size, args.rope_dim), dtype=torch.bfloat16, device=device),
        "selection_ckv": torch.zeros((selection_blocks, args.block_size, args.kv_dim), dtype=torch.bfloat16, device=device),
        "selection_block_table": torch.arange(selection_blocks, dtype=torch.int32, device=device).view(args.batch_size, sparse_blocks),
        "gather_selection_status": torch.full((args.pool_capacity, 1, 1, args.topk + 1), -1, dtype=torch.int32, device=device),
        "req_pool_entries": torch.arange(args.batch_size, dtype=torch.int32, device=device) % args.pool_capacity,
        "full_kpe": rand_bf16((full_blocks, args.block_size, args.rope_dim), device),
        "full_ckv": rand_bf16((full_blocks, args.block_size, args.kv_dim), device),
        "dram_tables": torch.arange(full_blocks, dtype=torch.int32, device=device).view(args.batch_size, full_blocks_per_req),
    }


def clone_mutable(tensors: dict[str, torch.Tensor], args) -> dict[str, torch.Tensor]:
    cloned = dict(tensors)
    for name in ("selection_kpe", "selection_ckv", "gather_selection_status"):
        cloned[name] = tensors[name].clone()
    cloned["q_index_out"] = torch.empty((tensors["hidden_states"].shape[0], args.n_head, args.head_dim), dtype=torch.bfloat16, device=tensors["hidden_states"].device)
    cloned["index_weights_out"] = torch.empty((tensors["hidden_states"].shape[0], tensors["weights_proj_weight"].shape[0]), dtype=torch.bfloat16, device=tensors["hidden_states"].device)
    return cloned


def call_eager(t, args):
    return dsa_indexer_pipeline_with_qc_eager(
        t["hidden_states"], t["cos"], t["sin"], t["q_a_weight"], t["q_norm_weight"], t["wq_b_weight"], t["weights_proj_weight"],
        t["q_index_out"], t["index_weights_out"], t["index_cache"], t["candidate_query_lens"], t["candidate_lens"], t["index_tables"],
        t["selection_kpe"], t["selection_ckv"], t["selection_block_table"], t["gather_selection_status"], t["req_pool_entries"],
        t["full_kpe"], t["full_ckv"], t["dram_tables"], q_norm_eps=args.q_norm_eps, n_head=args.n_head, head_dim=args.head_dim,
        rope_dim=args.rope_dim, score_scale=1.0, sparse_count=args.topk,
    )


def call_torchair(t, args):
    return dsa_indexer_pipeline_with_qc_torchair(
        t["hidden_states"], t["cos"], t["sin"], t["q_a_weight"], t["q_norm_weight"], t["wq_b_weight"], t["weights_proj_weight"],
        t["q_index_out"], t["index_weights_out"], t["index_cache"], t["candidate_query_lens"], t["candidate_lens"], t["index_tables"],
        t["selection_kpe"], t["selection_ckv"], t["selection_block_table"], t["gather_selection_status"], t["req_pool_entries"],
        t["full_kpe"], t["full_ckv"], t["dram_tables"], q_norm_eps=args.q_norm_eps, n_head=args.n_head, head_dim=args.head_dim,
        rope_dim=args.rope_dim, score_scale=1.0, sparse_count=args.topk,
    )


def diff(name: str, actual: torch.Tensor, expected: torch.Tensor) -> tuple[float, float]:
    d = (actual.float() - expected.float()).abs()
    max_abs = float(d.max().item()) if d.numel() else 0.0
    mean_abs = float(d.mean().item()) if d.numel() else 0.0
    print(f"DSA_PIPELINE_DIFF {name}: max_abs={max_abs:.6g} mean_abs={mean_abs:.6g}")
    return max_abs, mean_abs


def bench(fn, device: torch.device, warmup: int, iters: int) -> float:
    for _ in range(max(warmup, 0)):
        fn()
    sync_device(device)
    start = perf_counter()
    for _ in range(max(iters, 1)):
        fn()
    sync_device(device)
    return (perf_counter() - start) * 1000.0 / max(iters, 1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--pool-capacity", type=int, default=256)
    parser.add_argument("--full-len", type=int, default=16384)
    parser.add_argument("--topk", type=int, default=2048)
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument("--hidden-size", type=int, default=7168)
    parser.add_argument("--q-lora-rank", type=int, default=1536)
    parser.add_argument("--n-head", type=int, default=64)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--rope-dim", type=int, default=64)
    parser.add_argument("--kv-dim", type=int, default=512)
    parser.add_argument("--q-norm-eps", type=float, default=1e-6)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if args.pool_capacity < args.batch_size:
        raise ValueError("--pool-capacity must be >= --batch-size")
    if args.topk % args.block_size != 0:
        raise ValueError("--topk must be divisible by --block-size")
    device = set_device(args.device)
    base = make_inputs(args, device, args.seed)
    for name in ("hidden_states", "index_cache", "selection_kpe", "selection_ckv", "full_kpe", "full_ckv", "index_tables"):
        print("DSA_PIPELINE_TENSOR " + tensor_desc(name, base[name]))

    eager = clone_mutable(base, args)
    graph = clone_mutable(base, args)
    q_eager, w_eager, topk_eager = call_eager(eager, args)
    q_graph, w_graph, topk_graph = call_torchair(graph, args)
    sync_device(device)

    q_diff, _ = diff("q_index", q_graph, q_eager)
    w_diff, _ = diff("index_weights", w_graph, w_eager)
    topk_bad = int((topk_graph != topk_eager).sum().item())
    print(f"DSA_PIPELINE_DIFF topk_bad_count={topk_bad}")
    kpe_diff, _ = diff("selection_kpe", graph["selection_kpe"], eager["selection_kpe"])
    ckv_diff, _ = diff("selection_ckv", graph["selection_ckv"], eager["selection_ckv"])
    status_bad = int((graph["gather_selection_status"] != eager["gather_selection_status"]).sum().item())
    print(f"DSA_PIPELINE_DIFF status_bad_count={status_bad}")
    if q_diff != 0.0 or w_diff != 0.0 or topk_bad or kpe_diff != 0.0 or ckv_diff != 0.0 or status_bad:
        raise AssertionError("TorchAir DSA pipeline differs from eager pipeline")

    eager_ms = bench(lambda: call_eager(clone_mutable(base, args), args), device, args.warmup, args.iters)
    torchair_ms = bench(lambda: call_torchair(clone_mutable(base, args), args), device, args.warmup, args.iters)
    print(f"DSA_PIPELINE_BENCH eager_avg_ms={eager_ms:.6f} torchair_avg_ms={torchair_ms:.6f} warmup={args.warmup} iters={args.iters}")
    print("DSA_PIPELINE_OK")


if __name__ == "__main__":
    main()
