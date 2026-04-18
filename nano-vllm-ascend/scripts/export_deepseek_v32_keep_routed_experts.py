from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from export_deepseek_v32_pruned_impl import (
    SOURCE_FORMAT_HF_BF16,
    ShardedTensorReader,
    copy_metadata_files,
    detect_source_format,
    resolve_model_dir,
    export_attention_block,
    export_mlp_block,
    save_shard,
    to_dtype,
    write_tokenizer_files,
)


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def _normalize_bias_score(bias: torch.Tensor | None) -> torch.Tensor | None:
    if bias is None:
        return None
    score = bias.float().abs()
    max_score = float(score.max().item()) if score.numel() > 0 else 0.0
    if max_score > 0:
        score = score / max_score
    return score


def _compute_expert_scores(
    reader: ShardedTensorReader,
    layer_idx: int,
) -> torch.Tensor:
    layer_prefix = f"model.layers.{layer_idx}.mlp"
    gate_weight = reader.get_tensor(f"{layer_prefix}.gate.weight").float()
    scores = gate_weight.norm(dim=1)
    bias = reader.get_optional_tensor(f"{layer_prefix}.gate.e_score_correction_bias")
    bias_score = _normalize_bias_score(bias)
    if bias_score is not None:
        scores = scores * (1.0 + bias_score)
    return scores


def select_experts_by_gate_weight(
    reader: ShardedTensorReader,
    layer_idx: int,
    keep_k: int,
) -> list[int]:
    scores = _compute_expert_scores(reader, layer_idx)
    selected = torch.topk(scores, k=keep_k, dim=0, largest=True, sorted=False).indices
    return sorted(int(index) for index in selected.tolist())


def select_experts_group_balanced(
    reader: ShardedTensorReader,
    source_config: dict,
    layer_idx: int,
    keep_k: int,
) -> list[int]:
    scores = _compute_expert_scores(reader, layer_idx)
    num_experts = int(source_config["n_routed_experts"])
    num_groups = max(1, int(source_config.get("n_group", 1) or 1))
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
            remaining_scores = torch.topk(
                group_scores,
                k=experts_per_group,
                dim=0,
                largest=True,
                sorted=True,
            ).indices.tolist()
            for local_index in remaining_scores[group_keep:]:
                expert_idx = start + int(local_index)
                leftover_candidates.append((float(scores[expert_idx].item()), expert_idx))

    if remainder > 0:
        leftover_candidates.sort(reverse=True)
        selected.extend(expert_idx for _, expert_idx in leftover_candidates[:remainder])

    selected = sorted(set(selected))
    if len(selected) != keep_k:
        raise ValueError(
            f"Balanced expert selection expected {keep_k} experts, got {len(selected)}."
        )
    return selected


def load_selection_manifest(path: Path) -> dict[int, list[int]]:
    raw = _load_json(path)
    manifest: dict[int, list[int]] = {}
    for key, value in raw.items():
        manifest[int(key)] = [int(item) for item in value]
    return manifest


def export_routed_moe_block(
    reader: ShardedTensorReader,
    layer_idx: int,
    dtype: torch.dtype,
    *,
    keep_k: int,
    selected_experts: list[int],
) -> dict[str, torch.Tensor]:
    layer_prefix = f"model.layers.{layer_idx}"
    source_prefix = f"{layer_prefix}.mlp"
    tensors: dict[str, torch.Tensor] = {}

    for proj_name in ("gate_proj", "up_proj", "down_proj"):
        source_name = f"{source_prefix}.shared_experts.{proj_name}.weight"
        target_name = f"{source_prefix}.shared_experts.{proj_name}.weight"
        tensors[target_name] = to_dtype(reader.get_tensor(source_name), dtype)

    gate_weight = reader.get_tensor(f"{source_prefix}.gate.weight")
    tensors[f"{source_prefix}.gate.weight"] = to_dtype(
        gate_weight.index_select(
            0, torch.tensor(selected_experts, dtype=torch.long)
        ),
        dtype,
    )

    bias = reader.get_optional_tensor(f"{source_prefix}.gate.e_score_correction_bias")
    if bias is not None:
        tensors[f"{source_prefix}.gate.e_score_correction_bias"] = bias.index_select(
            0, torch.tensor(selected_experts, dtype=torch.long)
        ).to(torch.float32).contiguous()

    proj_names = ("gate_proj", "up_proj", "down_proj")
    for new_idx, old_idx in enumerate(selected_experts):
        for proj_name in proj_names:
            source_name = (
                f"{source_prefix}.experts.{old_idx}.{proj_name}.weight"
            )
            target_name = (
                f"{source_prefix}.experts.{new_idx}.{proj_name}.weight"
            )
            tensors[target_name] = to_dtype(reader.get_tensor(source_name), dtype)

    if len(selected_experts) != keep_k:
        raise ValueError(
            f"Layer {layer_idx} expected {keep_k} experts, got {len(selected_experts)}."
        )
    return tensors


def export_config(
    source_config: dict,
    *,
    keep_k: int,
    num_experts_per_tok: int,
    n_group: int,
    topk_group: int,
) -> dict:
    config = dict(source_config)
    config["model_type"] = "deepseek_v32"
    config["architectures"] = ["DeepseekV32ForCausalLM"]
    config["torch_dtype"] = "bfloat16"
    config["dtype"] = "bfloat16"
    config["num_nextn_predict_layers"] = 0
    config["n_routed_experts"] = keep_k
    config["num_experts_per_tok"] = num_experts_per_tok
    config["n_group"] = n_group
    config["topk_group"] = topk_group
    config["nanovllm_pruned_shared_only"] = False
    config["nanovllm_pruned_keep_routed_experts"] = True
    config["nanovllm_export_format"] = "bf16_keep_routed_experts"
    config["nanovllm_retained_routed_experts_per_layer"] = keep_k
    config["tokenizer_class"] = "LlamaTokenizerFast"
    return config


def export_model(
    source_dir: Path,
    output_dir: Path,
    *,
    keep_k: int,
    num_experts_per_tok: int,
    selection_manifest: dict[int, list[int]] | None,
    selection_strategy: str,
) -> None:
    source_dir = resolve_model_dir(source_dir)
    if output_dir.exists():
        if any(output_dir.iterdir()):
            raise FileExistsError(
                f"Output directory '{output_dir}' already exists and is not empty."
            )
    else:
        output_dir.mkdir(parents=True)

    source_format = detect_source_format(source_dir)
    if source_format != SOURCE_FORMAT_HF_BF16:
        raise ValueError(
            "This exporter currently supports Hugging Face BF16 safetensors "
            "source models only."
        )

    reader = ShardedTensorReader(source_dir, source_format)
    source_config = _load_json(source_dir / "config.json")
    original_num_experts = int(source_config["n_routed_experts"])
    original_n_group = max(1, int(source_config.get("n_group", 1) or 1))
    original_topk_group = max(1, int(source_config.get("topk_group", 1) or 1))
    if keep_k < 1:
        raise ValueError("--keep-routed-experts must be >= 1.")
    if keep_k > original_num_experts:
        raise ValueError(
            f"--keep-routed-experts={keep_k} exceeds source n_routed_experts="
            f"{original_num_experts}."
        )
    if num_experts_per_tok < 1:
        raise ValueError("--num-experts-per-tok must be >= 1.")

    if (
        selection_strategy == "group_balanced"
        and keep_k % original_n_group == 0
    ):
        export_n_group = original_n_group
        experts_per_group_after_prune = keep_k // original_n_group
        export_topk_group = min(original_topk_group, export_n_group)
        max_num_experts_per_tok = export_topk_group * experts_per_group_after_prune
    else:
        export_n_group = 1
        export_topk_group = 1
        max_num_experts_per_tok = keep_k

    effective_num_experts_per_tok = min(num_experts_per_tok, max_num_experts_per_tok)
    dtype = torch.bfloat16
    total_size = 0
    weight_map: dict[str, str] = {}

    copy_metadata_files(source_dir, output_dir)
    write_tokenizer_files(
        source_dir,
        output_dir,
        max_model_len=int(source_config["max_position_embeddings"]),
    )

    embeddings = {
        "model.embed_tokens.weight": to_dtype(
            reader.get_tensor("model.embed_tokens.weight"),
            dtype,
        )
    }
    total_size += save_shard(
        output_dir,
        "model-embeddings.safetensors",
        embeddings,
        weight_map,
    )

    first_k_dense = int(source_config["first_k_dense_replace"])
    num_layers = int(source_config["num_hidden_layers"])
    selected_by_layer: dict[int, list[int]] = {}

    for layer_idx in range(num_layers):
        layer_prefix = f"model.layers.{layer_idx}"
        layer_tensors = export_attention_block(
            reader,
            layer_prefix,
            dtype,
            source_format,
        )
        if layer_idx < first_k_dense:
            layer_tensors.update(
                export_mlp_block(
                    reader,
                    layer_prefix,
                    source_prefix=f"{layer_prefix}.mlp",
                    dtype=dtype,
                    source_format=source_format,
                )
            )
        else:
            selected = (
                selection_manifest[layer_idx]
                if selection_manifest is not None and layer_idx in selection_manifest
                else (
                    select_experts_group_balanced(
                        reader,
                        source_config,
                        layer_idx,
                        keep_k,
                    )
                    if selection_strategy == "group_balanced"
                    else select_experts_by_gate_weight(reader, layer_idx, keep_k)
                )
            )
            if len(selected) != keep_k:
                raise ValueError(
                    f"Layer {layer_idx} selection size {len(selected)} != keep_k {keep_k}."
                )
            selected_by_layer[layer_idx] = list(selected)
            layer_tensors.update(
                export_routed_moe_block(
                    reader,
                    layer_idx,
                    dtype,
                    keep_k=keep_k,
                    selected_experts=selected,
                )
            )

        shard_name = f"model-layer-{layer_idx:03d}.safetensors"
        total_size += save_shard(output_dir, shard_name, layer_tensors, weight_map)

    lm_head = reader.get_optional_tensor("lm_head.weight")
    if lm_head is None:
        if bool(source_config.get("tie_word_embeddings", False)):
            lm_head = embeddings["model.embed_tokens.weight"]
        else:
            raise KeyError("Tensor 'lm_head.weight' is missing from the source model.")

    final_tensors = {
        "model.norm.weight": to_dtype(reader.get_tensor("model.norm.weight"), dtype),
        "lm_head.weight": to_dtype(lm_head, dtype),
    }
    total_size += save_shard(
        output_dir,
        "model-final.safetensors",
        final_tensors,
        weight_map,
    )

    config = export_config(
        source_config,
        keep_k=keep_k,
        num_experts_per_tok=effective_num_experts_per_tok,
        n_group=export_n_group,
        topk_group=export_topk_group,
    )
    with (output_dir / "config.json").open("w", encoding="utf-8") as file:
        json.dump(config, file, ensure_ascii=False, indent=2)
        file.write("\n")

    index = {
        "metadata": {"total_size": total_size},
        "weight_map": dict(sorted(weight_map.items())),
    }
    with (output_dir / "model.safetensors.index.json").open("w", encoding="utf-8") as file:
        json.dump(index, file, ensure_ascii=False, indent=2)
        file.write("\n")

    notes = {
        "source_model": str(source_dir),
        "source_format": source_format,
        "export_format": "bf16_keep_routed_experts",
        "kept_shared_experts": int(source_config.get("n_shared_experts", 1) or 1),
        "kept_routed_experts_per_layer": keep_k,
        "num_experts_per_tok": int(config["num_experts_per_tok"]),
        "n_group": int(config["n_group"]),
        "topk_group": int(config["topk_group"]),
        "selection_method": (
            "manifest"
            if selection_manifest is not None
            else selection_strategy
        ),
        "selected_routed_experts": {
            str(layer_idx): selected
            for layer_idx, selected in sorted(selected_by_layer.items())
        },
        "dropped_components": [
            "unselected routed experts",
            "nextn predict layers",
        ],
        "routing_adjustments": {
            "n_group": int(config["n_group"]),
            "topk_group": int(config["topk_group"]),
            "max_num_experts_per_tok": int(max_num_experts_per_tok),
        },
    }
    with (output_dir / "nanovllm_keep_routed_experts_notes.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(notes, file, ensure_ascii=False, indent=2)
        file.write("\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export DeepSeek-V3.2 into a BF16 model directory that keeps the "
            "shared expert and a small number of routed experts per MoE layer "
            "for nano-vllm-ascend."
        )
    )
    parser.add_argument("source_model", type=Path)
    parser.add_argument("output_model", type=Path)
    parser.add_argument(
        "--keep-routed-experts",
        type=int,
        default=8,
        help="Number of routed experts to keep in each MoE layer.",
    )
    parser.add_argument(
        "--num-experts-per-tok",
        type=int,
        default=4,
        help=(
            "Number of routed experts activated per token in the exported config. "
            "This will be clipped to --keep-routed-experts."
        ),
    )
    parser.add_argument(
        "--selection-strategy",
        choices=("group_balanced", "global_topk"),
        default="group_balanced",
        help=(
            "How to choose the retained routed experts. 'group_balanced' keeps "
            "the best experts evenly across the original DeepSeek expert groups."
        ),
    )
    parser.add_argument(
        "--selection-manifest",
        type=Path,
        default=None,
        help=(
            "Optional JSON file mapping layer indices to retained routed expert "
            "ids. If omitted, a gate-weight norm heuristic is used."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    selection_manifest = (
        load_selection_manifest(args.selection_manifest)
        if args.selection_manifest is not None
        else None
    )
    export_model(
        args.source_model,
        args.output_model,
        keep_k=args.keep_routed_experts,
        num_experts_per_tok=args.num_experts_per_tok,
        selection_manifest=selection_manifest,
        selection_strategy=args.selection_strategy,
    )


if __name__ == "__main__":
    main()
