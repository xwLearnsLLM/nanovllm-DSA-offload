import argparse
import json
import os
from time import perf_counter

import torch
import torch_npu  # type: ignore  # noqa: F401
import nanovllm.ops as ascend_ops


def log(message: str) -> None:
    print(message, flush=True)


def register_ascend_ops() -> None:
    log("MOE_GATE stage=register_import_torch_npu")
    log("MOE_GATE stage=register_import_nanovllm_ops")
    log(f"MOE_GATE ASCEND_CUSTOM_OPP_PATH={os.environ.get('ASCEND_CUSTOM_OPP_PATH', '')}")
    log("MOE_GATE stage=register_done")


def load_config(path: str | None) -> dict:
    if path is None:
        return {}
    config_path = path
    if os.path.isdir(config_path):
        config_path = os.path.join(config_path, "config.json")
    if not os.path.exists(config_path):
        raise FileNotFoundError(config_path)
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def torch_grouped_topk(
    router_logits: torch.Tensor,
    *,
    top_k: int,
    topk_group: int,
    num_expert_group: int,
    scoring_func: str,
    renormalize: bool,
    routed_scaling_factor: float,
    bias: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    router_logits = router_logits.float()
    if scoring_func == "softmax":
        scores = torch.softmax(router_logits, dim=-1)
    elif scoring_func == "sigmoid":
        scores = router_logits.sigmoid()
    else:
        raise ValueError(f"Unsupported scoring function: {scoring_func}")
    if bias is not None and bias.dtype != scores.dtype:
        bias = bias.to(scores.dtype)

    if num_expert_group > 1:
        num_tokens = scores.shape[0]
        experts_per_group = scores.shape[-1] // num_expert_group
        if experts_per_group * num_expert_group != scores.shape[-1]:
            raise ValueError("num_expert_group must divide num_experts")
        if bias is not None:
            original_scores = scores
            biased_scores = scores + bias.unsqueeze(0)
            group_take = min(2, experts_per_group)
            group_scores = (
                biased_scores.view(num_tokens, num_expert_group, experts_per_group)
                .topk(group_take, dim=-1)[0]
                .sum(dim=-1)
            )
            scores_for_select = biased_scores
        else:
            original_scores = scores
            group_scores = scores.view(
                num_tokens,
                num_expert_group,
                experts_per_group,
            ).max(dim=-1).values
            scores_for_select = scores

        topk_group = min(topk_group, num_expert_group)
        top_k = min(top_k, topk_group * experts_per_group)
        group_idx = torch.topk(group_scores, k=topk_group, dim=-1, sorted=False).indices
        group_mask = torch.zeros_like(group_scores)
        group_mask.scatter_(1, group_idx, 1)
        score_mask = (
            group_mask.unsqueeze(-1)
            .expand(num_tokens, num_expert_group, experts_per_group)
            .reshape(num_tokens, -1)
        )
        masked_scores = scores_for_select.masked_fill(~score_mask.bool(), float("-inf"))
        topk_ids = torch.topk(masked_scores, k=top_k, dim=-1, sorted=False).indices
        topk_weights = original_scores.gather(1, topk_ids)
    else:
        if bias is not None:
            original_scores = scores
            topk_ids = torch.topk(
                scores + bias.unsqueeze(0),
                k=top_k,
                dim=-1,
                sorted=False,
            ).indices
            topk_weights = original_scores.gather(1, topk_ids)
        else:
            topk_weights, topk_ids = torch.topk(
                scores,
                k=top_k,
                dim=-1,
                sorted=False,
            )

    if renormalize:
        topk_weights = topk_weights / topk_weights.sum(
            dim=-1,
            keepdim=True,
        ).clamp_min(1e-20)
    if routed_scaling_factor != 1.0:
        topk_weights = topk_weights * routed_scaling_factor
    return topk_weights.float(), topk_ids.long()


def npu_grouped_topk(
    router_logits: torch.Tensor,
    *,
    top_k: int,
    topk_group: int,
    num_expert_group: int,
    scoring_func: str,
    renormalize: bool,
    routed_scaling_factor: float,
    bias: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    norm_type = 1 if scoring_func == "sigmoid" else 0
    router_logits = router_logits.float()
    if bias is not None and bias.dtype != router_logits.dtype:
        bias = bias.to(router_logits.dtype)
    topk_weights, topk_ids, _ = ascend_ops.moe_gating_top_k(
        router_logits,
        k=top_k,
        k_group=topk_group,
        group_count=num_expert_group,
        group_select_mode=1,
        renorm=0,
        norm_type=norm_type,
        out_flag=False,
        routed_scaling_factor=1.0,
        eps=1e-20,
        bias_opt=bias,
    )
    topk_weights = topk_weights.float()
    if renormalize:
        topk_weights = topk_weights / topk_weights.sum(
            dim=-1,
            keepdim=True,
        ).clamp_min(1e-20)
    if routed_scaling_factor != 1.0:
        topk_weights = topk_weights * routed_scaling_factor
    return topk_weights.float(), topk_ids.long()


def sort_by_id(
    weights: torch.Tensor,
    ids: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    sorted_ids, order = torch.sort(ids, dim=-1)
    sorted_weights = weights.gather(1, order)
    return sorted_weights, sorted_ids


def tensor_desc(name: str, tensor: torch.Tensor) -> str:
    return (
        f"{name}=shape={tuple(tensor.shape)} dtype={tensor.dtype} "
        f"device={tensor.device} contiguous={tensor.is_contiguous()}"
    )


def parse_dtype(name: str) -> torch.dtype:
    normalized = name.lower()
    if normalized in ("float32", "fp32"):
        return torch.float32
    if normalized in ("bfloat16", "bf16"):
        return torch.bfloat16
    if normalized in ("float16", "fp16"):
        return torch.float16
    raise ValueError(f"Unsupported dtype: {name}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--model-config", default=None)
    parser.add_argument("--tokens", type=int, default=None)
    parser.add_argument("--experts", type=int, default=None)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--n-group", type=int, default=None)
    parser.add_argument("--topk-group", type=int, default=None)
    parser.add_argument("--scoring-func", default=None)
    parser.add_argument("--routed-scaling-factor", type=float, default=None)
    parser.add_argument("--renormalize", type=int, default=None)
    parser.add_argument("--bias-mode", choices=("auto", "none", "random"), default="auto")
    parser.add_argument("--logits-dtype", default="float32")
    parser.add_argument("--bias-dtype", default="float32")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--iters", type=int, default=5)
    parser.add_argument("--try-renorm-one", action="store_true")
    args = parser.parse_args()

    log("MOE_GATE stage=start")
    register_ascend_ops()
    torch.npu.set_device(args.device)
    device = torch.device(args.device)

    config = load_config(args.model_config)
    tokens = args.tokens or 3072
    experts = args.experts or int(config.get("n_routed_experts", 32))
    top_k = args.top_k or int(config.get("num_experts_per_tok", 8))
    n_group = args.n_group or int(config.get("n_group", 8))
    topk_group = args.topk_group or int(config.get("topk_group", 4))
    scoring_func = args.scoring_func or str(config.get("scoring_func", "sigmoid"))
    routed_scaling_factor = (
        args.routed_scaling_factor
        if args.routed_scaling_factor is not None
        else float(config.get("routed_scaling_factor", 1.0))
    )
    renormalize = (
        bool(args.renormalize)
        if args.renormalize is not None
        else bool(config.get("norm_topk_prob", True))
    )
    topk_method = config.get("topk_method")
    use_bias = args.bias_mode == "random" or (
        args.bias_mode == "auto" and topk_method == "noaux_tc"
    )

    torch.manual_seed(args.seed)
    logits_dtype = parse_dtype(args.logits_dtype)
    bias_dtype = parse_dtype(args.bias_dtype)
    router_logits = torch.randn(
        (tokens, experts),
        dtype=logits_dtype,
        device=device,
    )
    bias = (
        torch.randn((experts,), dtype=bias_dtype, device=device) * 0.01
        if use_bias
        else None
    )

    log(
        "MOE_GATE config "
        f"tokens={tokens} experts={experts} top_k={top_k} "
        f"n_group={n_group} topk_group={topk_group} "
        f"scoring_func={scoring_func} renormalize={renormalize} "
        f"routed_scaling_factor={routed_scaling_factor} "
        f"topk_method={topk_method} use_bias={use_bias}"
    )
    log("MOE_GATE " + tensor_desc("router_logits", router_logits))
    if bias is not None:
        log("MOE_GATE " + tensor_desc("bias", bias))

    ref_weights, ref_ids = torch_grouped_topk(
        router_logits,
        top_k=top_k,
        topk_group=topk_group,
        num_expert_group=n_group,
        scoring_func=scoring_func,
        renormalize=renormalize,
        routed_scaling_factor=routed_scaling_factor,
        bias=bias,
    )
    torch.npu.synchronize()
    log("MOE_GATE stage=after_torch_reference")

    if args.try_renorm_one:
        try:
            renorm_one_logits = router_logits.float()
            renorm_one_bias = (
                bias.to(renorm_one_logits.dtype)
                if bias is not None and bias.dtype != renorm_one_logits.dtype
                else bias
            )
            ascend_ops.moe_gating_top_k(
                renorm_one_logits,
                k=top_k,
                k_group=topk_group,
                group_count=n_group,
                group_select_mode=1,
                renorm=1,
                norm_type=1 if scoring_func == "sigmoid" else 0,
                out_flag=False,
                routed_scaling_factor=1.0,
                eps=1e-20,
                bias_opt=renorm_one_bias,
            )
            torch.npu.synchronize()
            log("MOE_GATE renorm_one=ok")
        except Exception as exc:
            log(f"MOE_GATE renorm_one=failed error={exc!r}")

    for _ in range(args.warmup):
        npu_grouped_topk(
            router_logits,
            top_k=top_k,
            topk_group=topk_group,
            num_expert_group=n_group,
            scoring_func=scoring_func,
            renormalize=renormalize,
            routed_scaling_factor=routed_scaling_factor,
            bias=bias,
        )
    torch.npu.synchronize()

    elapsed = []
    npu_weights = None
    npu_ids = None
    for _ in range(args.iters):
        torch.npu.synchronize()
        start = perf_counter()
        npu_weights, npu_ids = npu_grouped_topk(
            router_logits,
            top_k=top_k,
            topk_group=topk_group,
            num_expert_group=n_group,
            scoring_func=scoring_func,
            renormalize=renormalize,
            routed_scaling_factor=routed_scaling_factor,
            bias=bias,
        )
        torch.npu.synchronize()
        elapsed.append((perf_counter() - start) * 1000)

    assert npu_weights is not None
    assert npu_ids is not None
    log("MOE_GATE stage=after_npu_op")
    log("MOE_GATE " + tensor_desc("ref_weights", ref_weights))
    log("MOE_GATE " + tensor_desc("ref_ids", ref_ids))
    log("MOE_GATE " + tensor_desc("npu_weights", npu_weights))
    log("MOE_GATE " + tensor_desc("npu_ids", npu_ids))

    ref_weights_s, ref_ids_s = sort_by_id(ref_weights, ref_ids)
    npu_weights_s, npu_ids_s = sort_by_id(npu_weights, npu_ids)
    id_mismatch = int((ref_ids_s != npu_ids_s).sum().item())
    diff = (ref_weights_s - npu_weights_s).abs()
    max_abs = float(diff.max().item()) if diff.numel() else 0.0
    mean_abs = float(diff.mean().item()) if diff.numel() else 0.0
    denom = float(ref_weights_s.abs().max().clamp_min(1e-6).item())
    max_rel = max_abs / denom
    mean_rel = mean_abs / denom
    log(
        "MOE_GATE diff "
        f"id_mismatch={id_mismatch} max_abs={max_abs:.6g} "
        f"mean_abs={mean_abs:.6g} max_rel={max_rel:.6g} "
        f"mean_rel={mean_rel:.6g} ref_sum_head={ref_weights[:4].sum(-1).detach().cpu().tolist()} "
        f"npu_sum_head={npu_weights[:4].sum(-1).detach().cpu().tolist()}"
    )
    log(
        "MOE_GATE bench "
        f"avg_ms={sum(elapsed) / len(elapsed):.6f} "
        f"min_ms={min(elapsed):.6f} max_ms={max(elapsed):.6f}"
    )
    log("MOE_GATE stage=done")


if __name__ == "__main__":
    main()
