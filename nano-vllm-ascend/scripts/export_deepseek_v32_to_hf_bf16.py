from __future__ import annotations

import argparse
import json
import shutil
from collections import OrderedDict
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

from export_deepseek_v32_pruned_impl import ensure_output_dir, resolve_model_dir


FLOAT8_DTYPES = {
    torch.float8_e4m3fn,
    torch.float8_e5m2,
}
if hasattr(torch, "float8_e4m3fnuz"):
    FLOAT8_DTYPES.add(torch.float8_e4m3fnuz)
if hasattr(torch, "float8_e5m2fnuz"):
    FLOAT8_DTYPES.add(torch.float8_e5m2fnuz)

STANDARD_METADATA_EXCLUDE = {
    "model.safetensors.index.json",
    "quant_model_weights.safetensors.index.json",
    "nanovllm_hf_bf16_export_notes.json",
}

SCALE_SUFFIX = "_scale_inv"


def _is_fp8_dtype(dtype: torch.dtype) -> bool:
    return dtype in FLOAT8_DTYPES


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def _copy_metadata_files(source_dir: Path, output_dir: Path) -> list[str]:
    copied: list[str] = []
    for path in sorted(source_dir.iterdir()):
        if not path.is_file():
            continue
        if path.name in STANDARD_METADATA_EXCLUDE:
            continue
        if path.suffix == ".safetensors":
            continue
        if path.name.endswith(".safetensors.index.json"):
            continue
        shutil.copy2(path, output_dir / path.name)
        copied.append(path.name)
    return copied


def _sanitize_config(config: dict) -> dict:
    sanitized = dict(config)
    sanitized["torch_dtype"] = "bfloat16"
    sanitized["dtype"] = "bfloat16"
    for key in (
        "quantization_config",
        "compression_config",
        "quant_method",
        "activation_scheme",
        "weight_block_size",
    ):
        sanitized.pop(key, None)
    return sanitized


def _rewrite_config_files(source_dir: Path, output_dir: Path) -> list[str]:
    rewritten: list[str] = []
    for filename in ("config.json", "configuration.json"):
        source_path = source_dir / filename
        if not source_path.is_file():
            continue
        config = _sanitize_config(_load_json(source_path))
        with (output_dir / filename).open("w", encoding="utf-8") as file:
            json.dump(config, file, ensure_ascii=False, indent=2)
            file.write("\n")
        rewritten.append(filename)
    return rewritten


def _dequant_fp8_weight(
    weight: torch.Tensor,
    scale_inv: torch.Tensor,
    *,
    block_size: int,
) -> torch.Tensor:
    if weight.dim() != 2 or scale_inv.dim() != 2:
        raise ValueError(
            "Official DeepSeek fp8_cast_bf16.py expects 2D FP8 weights and 2D "
            f"scale_inv tensors, got weight={tuple(weight.shape)} and "
            f"scale_inv={tuple(scale_inv.shape)}."
        )

    rows, cols = weight.shape
    expected_shape = (
        (rows + block_size - 1) // block_size,
        (cols + block_size - 1) // block_size,
    )
    if scale_inv.shape != expected_shape:
        raise ValueError(
            f"scale_inv shape mismatch for weight {tuple(weight.shape)}. "
            f"Expected {expected_shape}, got {tuple(scale_inv.shape)}."
        )

    weight_fp32 = weight.to(torch.float32)
    output = torch.empty_like(weight_fp32)
    scale_fp32 = scale_inv.to(torch.float32)

    for row_block in range(expected_shape[0]):
        row_start = row_block * block_size
        row_end = min(row_start + block_size, rows)
        for col_block in range(expected_shape[1]):
            col_start = col_block * block_size
            col_end = min(col_start + block_size, cols)
            output[row_start:row_end, col_start:col_end] = (
                weight_fp32[row_start:row_end, col_start:col_end]
                * scale_fp32[row_block, col_block]
            )

    return output.to(torch.bfloat16).contiguous()


class _ShardCache:
    def __init__(self, source_dir: Path, weight_map: dict[str, str], load_device: str) -> None:
        self.source_dir = source_dir
        self.weight_map = weight_map
        self.load_device = load_device
        self.loaded_files: OrderedDict[str, dict[str, torch.Tensor]] = OrderedDict()

    def _load_shard(self, shard_name: str) -> dict[str, torch.Tensor]:
        shard_path = self.source_dir / shard_name
        if not shard_path.is_file():
            raise FileNotFoundError(f"Missing source shard: '{shard_path}'.")

        state_dict = load_file(str(shard_path), device=self.load_device)
        self.loaded_files[shard_name] = state_dict
        self.loaded_files.move_to_end(shard_name)

        while len(self.loaded_files) > 2:
            old_shard_name, _ = self.loaded_files.popitem(last=False)
            if self.load_device == "cuda":
                torch.cuda.empty_cache()
            del old_shard_name

        return state_dict

    def get_shard(self, shard_name: str) -> dict[str, torch.Tensor]:
        state_dict = self.loaded_files.get(shard_name)
        if state_dict is not None:
            self.loaded_files.move_to_end(shard_name)
            return state_dict
        return self._load_shard(shard_name)

    def get_tensor(self, tensor_name: str) -> torch.Tensor:
        shard_name = self.weight_map.get(tensor_name)
        if shard_name is None:
            raise KeyError(f"Tensor '{tensor_name}' is missing from model.safetensors.index.json.")
        state_dict = self.get_shard(shard_name)
        return state_dict[tensor_name]


def export_model(
    source_model: Path,
    output_model: Path,
    *,
    block_size: int,
    load_device: str,
) -> None:
    source_dir = resolve_model_dir(source_model)
    ensure_output_dir(output_model)

    index_path = source_dir / "model.safetensors.index.json"
    if not index_path.is_file():
        raise FileNotFoundError(
            f"'{index_path}' is missing. This minimal exporter expects a Hugging Face "
            "FP8/BF16 checkpoint with model.safetensors.index.json."
        )

    source_index = _load_json(index_path)
    source_weight_map = dict(source_index["weight_map"])
    cache = _ShardCache(source_dir, source_weight_map, load_device)

    copied_files = _copy_metadata_files(source_dir, output_model)
    rewritten_config_files = _rewrite_config_files(source_dir, output_model)
    output_weight_map: dict[str, str] = {}
    converted_weights: list[str] = []
    total_size = 0

    shard_names = sorted(set(source_weight_map.values()))
    for shard_name in shard_names:
        current_state_dict = cache.get_shard(shard_name)
        new_state_dict: dict[str, torch.Tensor] = {}

        for weight_name, weight in current_state_dict.items():
            if weight_name.endswith(SCALE_SUFFIX):
                continue

            if _is_fp8_dtype(weight.dtype):
                scale_inv_name = f"{weight_name}{SCALE_SUFFIX}"
                if scale_inv_name not in source_weight_map:
                    raise KeyError(
                        f"Missing paired scale tensor '{scale_inv_name}' for FP8 tensor "
                        f"'{weight_name}'."
                    )
                scale_inv = cache.get_tensor(scale_inv_name)
                new_state_dict[weight_name] = _dequant_fp8_weight(
                    weight,
                    scale_inv,
                    block_size=block_size,
                )
                converted_weights.append(weight_name)
            else:
                new_state_dict[weight_name] = weight.contiguous()

        if not new_state_dict:
            continue

        save_file(new_state_dict, str(output_model / shard_name))
        for tensor_name, tensor in new_state_dict.items():
            output_weight_map[tensor_name] = shard_name
            total_size += tensor.numel() * tensor.element_size()

    output_index = {
        "metadata": {"total_size": total_size},
        "weight_map": dict(sorted(output_weight_map.items())),
    }
    with (output_model / "model.safetensors.index.json").open("w", encoding="utf-8") as file:
        json.dump(output_index, file, ensure_ascii=False, indent=2)
        file.write("\n")

    notes = {
        "source_model": str(source_dir),
        "copied_metadata_files": copied_files,
        "rewritten_config_files": rewritten_config_files,
        "load_device": load_device,
        "block_size": block_size,
        "converted_fp8_weight_count": len(converted_weights),
        "converted_fp8_weights": converted_weights,
    }
    with (output_model / "nanovllm_hf_bf16_export_notes.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(notes, file, ensure_ascii=False, indent=2)
        file.write("\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Cast a DeepSeek HF FP8 checkpoint to BF16 in the same shard layout. "
            "This intentionally stays close to DeepSeek's official fp8_cast_bf16.py: "
            "it only applies weight + weight_scale_inv dequantization and copies the "
            "rest of the files unchanged."
        )
    )
    parser.add_argument("source_model", type=Path)
    parser.add_argument("output_model", type=Path)
    parser.add_argument(
        "--block-size",
        type=int,
        default=128,
        help="Block size for weight dequantization. Matches the official DeepSeek default.",
    )
    parser.add_argument(
        "--load-device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="Device used to load shards during conversion. Defaults to cuda when available.",
    )
    parser.add_argument(
        "--num-experts-per-tok",
        type=int,
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--n-group",
        type=int,
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--topk-group",
        type=int,
        default=None,
        help=argparse.SUPPRESS,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.load_device == "auto":
        load_device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        load_device = args.load_device

    export_model(
        args.source_model,
        args.output_model,
        block_size=args.block_size,
        load_device=load_device,
    )


if __name__ == "__main__":
    main()
