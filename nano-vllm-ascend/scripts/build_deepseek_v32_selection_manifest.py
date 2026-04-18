from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from export_deepseek_v32_pruned_impl import (
    SOURCE_FORMAT_HF_BF16,
    ShardedTensorReader,
    detect_source_format,
    resolve_model_dir,
)


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def _load_trace_hidden_states(
    trace_dir: Path,
    layer_idx: int,
    *,
    max_trace_files: int = 0,
    max_tokens: int = 0,
) -> torch.Tensor:
    pattern = f"layer_{layer_idx:03d}_call_*.pt"
    files = sorted(trace_dir.glob(pattern))
    if max_trace_files > 0:
        files = files[:max_trace_files]
    chunks: list[torch.Tensor] = []
    remaining_tokens = max_tokens if max_tokens > 0 else None
    for file_path in files:
        payload = torch.load(file_path, map_location="cpu")
        hidden_states = payload["hidden_states"].to(torch.float32)
        if remaining_tokens is not None:
            if remaining_tokens <= 0:
                break
            hidden_states = hidden_states[:remaining_tokens]
            remaining_tokens -= hidden_states.shape[0]
        if hidden_states.numel() > 0:
            chunks.append(hidden_states)
    if not chunks:
        raise FileNotFoundError(
            f"No trace tensors found for layer {layer_idx} under '{trace_dir}'."
        )
    return torch.cat(chunks, dim=0)


def _original_grouped_topk(
    hidden_states: torch.Tensor,
    gate_weight: torch.Tensor,
    *,
    scoring_func: str,
    top_k: int,
    num_groups: int,
    topk_group: int,
    renormalize: bool,
    bias: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    router_logits = torch.nn.functional.linear(
        hidden_states.to(torch.float32),
        gate_weight.to(torch.float32),
    )
    if scoring_func == "softmax":
        scores = torch.softmax(router_logits, dim=-1)
    elif scoring_func == "sigmoid":
        scores = router_logits.sigmoid()
    else:
        raise ValueError(f"Unsupported scoring function: {scoring_func}")

    num_experts = scores.shape[-1]
    num_groups = max(1, min(num_groups, num_experts))
    top_k = max(1, min(top_k, num_experts))
    if num_groups > 1:
        experts_per_group = num_experts // num_groups
        if experts_per_group * num_groups != num_experts:
            raise ValueError("n_group must divide the number of routed experts.")
        if bias is not None:
            original_scores = scores
            biased_scores = scores + bias.to(torch.float32).unsqueeze(0)
            group_take = min(2, experts_per_group)
            group_scores = (
                biased_scores.view(-1, num_groups, experts_per_group)
                .topk(group_take, dim=-1)[0]
                .sum(dim=-1)
            )
        else:
            biased_scores = scores
            group_scores = scores.view(-1, num_groups, experts_per_group).max(dim=-1).values
        topk_group = max(1, min(topk_group, num_groups))
        top_k = min(top_k, topk_group * experts_per_group)
        group_idx = torch.topk(
            group_scores,
            k=topk_group,
            dim=-1,
            sorted=False,
        ).indices
        group_mask = torch.zeros_like(group_scores)
        group_mask.scatter_(1, group_idx, 1)
        score_mask = (
            group_mask.unsqueeze(-1)
            .expand(-1, num_groups, experts_per_group)
            .reshape(-1, num_experts)
        )
        tmp_scores = biased_scores.masked_fill(~score_mask.bool(), float("-inf"))
        if bias is not None:
            topk_ids = torch.topk(tmp_scores, k=top_k, dim=-1, sorted=False).indices
            topk_weights = original_scores.gather(1, topk_ids)
        else:
            topk_weights, topk_ids = torch.topk(
                tmp_scores,
                k=top_k,
                dim=-1,
                sorted=False,
            )
    else:
        if bias is not None:
            original_scores = scores
            topk_ids = torch.topk(
                scores + bias.to(torch.float32).unsqueeze(0),
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
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
    return topk_weights, topk_ids


def _accumulate_expert_scores(
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    num_experts: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    hit_count = torch.bincount(topk_ids.reshape(-1), minlength=num_experts).to(
        torch.float32
    )
    weight_sum = torch.zeros(num_experts, dtype=torch.float32)
    weight_sum.scatter_add_(0, topk_ids.reshape(-1), topk_weights.reshape(-1))
    return hit_count, weight_sum


def _select_group_balanced_from_scores(
    scores: torch.Tensor,
    *,
    keep_k: int,
    num_groups: int,
) -> list[int]:
    num_experts = scores.shape[0]
    experts_per_group = num_experts // num_groups
    if experts_per_group * num_groups != num_experts:
        raise ValueError("n_group must divide n_routed_experts for balanced selection.")

    base_keep = keep_k // num_groups
    remainder = keep_k % num_groups
    selected: list[int] = []
    leftover_candidates: list[tuple[float, int]] = []

    for group_idx in range(num_groups):
        start = group_idx * experts_per_group
        end = start + experts_per_group
        group_scores = scores[start:end]
        group_keep = min(base_keep, experts_per_group)
        if group_keep > 0:
            chosen = torch.topk(
                group_scores,
                k=group_keep,
                dim=0,
                largest=True,
                sorted=False,
            ).indices + start
            selected.extend(int(index) for index in chosen.tolist())
        if group_keep < experts_per_group:
            ranked = torch.topk(
                group_scores,
                k=experts_per_group,
                dim=0,
                largest=True,
                sorted=True,
            ).indices.tolist()
            for local_index in ranked[group_keep:]:
                expert_idx = start + int(local_index)
                leftover_candidates.append((float(scores[expert_idx].item()), expert_idx))

    if remainder > 0:
        leftover_candidates.sort(reverse=True)
        selected.extend(expert_idx for _, expert_idx in leftover_candidates[:remainder])

    selected = sorted(set(selected))
    if len(selected) != keep_k:
        raise ValueError(
            f"Balanced selection expected {keep_k} experts, got {len(selected)}."
        )
    return selected


def _select_global_topk_from_scores(scores: torch.Tensor, keep_k: int) -> list[int]:
    selected = torch.topk(scores, k=keep_k, dim=0, largest=True, sorted=False).indices
    return sorted(int(index) for index in selected.tolist())


def build_manifest(
    source_model: Path,
    trace_dir: Path,
    output_manifest: Path,
    *,
    keep_k: int,
    selection_strategy: str,
    score_metric: str,
    max_trace_files_per_layer: int,
    max_tokens_per_layer: int,
) -> None:
    source_model = resolve_model_dir(source_model)
    source_format = detect_source_format(source_model)
    if source_format != SOURCE_FORMAT_HF_BF16:
        raise ValueError("Selection manifest builder supports HF BF16 sources only.")

    reader = ShardedTensorReader(source_model, source_format)
    config = _load_json(source_model / "config.json")
    num_layers = int(config["num_hidden_layers"])
    first_k_dense = int(config["first_k_dense_replace"])
    num_experts = int(config["n_routed_experts"])
    num_groups = max(1, int(config.get("n_group", 1) or 1))
    topk_group = max(1, int(config.get("topk_group", 1) or 1))
    top_k = int(config["num_experts_per_tok"])
    scoring_func = str(config.get("scoring_func", "sigmoid"))
    renormalize = bool(config.get("norm_topk_prob", True))

    manifest: dict[str, list[int]] = {}
    stats: dict[str, dict] = {}

    for layer_idx in range(first_k_dense, num_layers):
        hidden_states = _load_trace_hidden_states(
            trace_dir,
            layer_idx,
            max_trace_files=max_trace_files_per_layer,
            max_tokens=max_tokens_per_layer,
        )
        gate_weight = reader.get_tensor(f"model.layers.{layer_idx}.mlp.gate.weight")
        bias = reader.get_optional_tensor(
            f"model.layers.{layer_idx}.mlp.gate.e_score_correction_bias"
        )
        topk_weights, topk_ids = _original_grouped_topk(
            hidden_states,
            gate_weight,
            scoring_func=scoring_func,
            top_k=top_k,
            num_groups=num_groups,
            topk_group=topk_group,
            renormalize=renormalize,
            bias=bias,
        )
        hit_count, weight_sum = _accumulate_expert_scores(
            topk_weights,
            topk_ids,
            num_experts=num_experts,
        )
        score_source = weight_sum if score_metric == "weight_sum" else hit_count
        if selection_strategy == "group_balanced":
            selected = _select_group_balanced_from_scores(
                score_source,
                keep_k=keep_k,
                num_groups=num_groups,
            )
        else:
            selected = _select_global_topk_from_scores(score_source, keep_k)
        manifest[str(layer_idx)] = selected
        stats[str(layer_idx)] = {
            "num_trace_tokens": int(hidden_states.shape[0]),
            "selection_strategy": selection_strategy,
            "score_metric": score_metric,
            "selected_experts": selected,
            "top_hit_experts": _select_global_topk_from_scores(hit_count, min(16, num_experts)),
            "top_weight_experts": _select_global_topk_from_scores(weight_sum, min(16, num_experts)),
        }

    with output_manifest.open("w", encoding="utf-8") as file:
        json.dump(manifest, file, ensure_ascii=False, indent=2)
        file.write("\n")

    stats_path = output_manifest.with_name(
        f"{output_manifest.stem}.stats{output_manifest.suffix}"
    )
    with stats_path.open("w", encoding="utf-8") as file:
        json.dump(
            {
                "source_model": str(source_model),
                "trace_dir": str(trace_dir),
                "keep_routed_experts": keep_k,
                "selection_strategy": selection_strategy,
                "score_metric": score_metric,
                "layers": stats,
            },
            file,
            ensure_ascii=False,
            indent=2,
        )
        file.write("\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a DeepSeek-V3.2 routed-expert selection manifest from traced "
            "MoE inputs and the original BF16 gate weights."
        )
    )
    parser.add_argument("source_model", type=Path)
    parser.add_argument("trace_dir", type=Path)
    parser.add_argument("output_manifest", type=Path)
    parser.add_argument(
        "--keep-routed-experts",
        type=int,
        default=8,
        help="Number of routed experts to keep per MoE layer.",
    )
    parser.add_argument(
        "--selection-strategy",
        choices=("group_balanced", "global_topk"),
        default="group_balanced",
        help="How to choose the retained experts from traced routing statistics.",
    )
    parser.add_argument(
        "--score-metric",
        choices=("weight_sum", "hit_count"),
        default="weight_sum",
        help="Use summed routing weights or raw hit counts to rank experts.",
    )
    parser.add_argument(
        "--max-trace-files-per-layer",
        type=int,
        default=0,
        help="Optional cap on the number of trace files loaded per layer.",
    )
    parser.add_argument(
        "--max-tokens-per-layer",
        type=int,
        default=0,
        help="Optional cap on total traced tokens consumed per layer.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    build_manifest(
        args.source_model,
        args.trace_dir,
        args.output_manifest,
        keep_k=args.keep_routed_experts,
        selection_strategy=args.selection_strategy,
        score_metric=args.score_metric,
        max_trace_files_per_layer=args.max_trace_files_per_layer,
        max_tokens_per_layer=args.max_tokens_per_layer,
    )


if __name__ == "__main__":
    main()
