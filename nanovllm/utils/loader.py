from __future__ import annotations

import json
import os
from dataclasses import dataclass
from glob import glob
from typing import Any

import torch
from safetensors import safe_open
from torch import nn


_QUANT_TYPES = {"FLOAT", "W8A8_DYNAMIC", "W4A8_DYNAMIC"}


@dataclass(frozen=True)
class WeightTarget:
    """A checkpoint weight mapped to a parameter-specific loader call."""

    name: str
    loader_args: tuple[Any, ...] = ()


def default_weight_loader(param: nn.Parameter, loaded_weight: torch.Tensor):
    param.data.copy_(loaded_weight)


def quant_tensor_types(description: dict[str, Any] | None) -> dict[str, str]:
    """Return only tensor entries from ModelSlim's flat description object."""

    if not description:
        return {}
    return {
        name: quant_type
        for name, quant_type in description.items()
        if "." in name and quant_type in _QUANT_TYPES
    }


def load_quant_description(path: str) -> dict[str, Any] | None:
    description_path = os.path.join(path, "quant_model_description.json")
    if not os.path.isfile(description_path):
        return None
    with open(description_path, "r", encoding="utf-8") as file:
        return json.load(file)


def dequantize_w8a8_weight(
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    weight_offset: torch.Tensor | None = None,
) -> torch.Tensor:
    """Dequantize a symmetric ModelSlim W8A8 weight on the host."""

    if weight.dtype != torch.int8:
        raise TypeError(
            f"W8A8 weight must be int8, got {weight.dtype}."
        )
    if weight_scale.numel() != weight.shape[0]:
        raise ValueError(
            "W8A8 weight_scale must contain one value per output channel: "
            f"weight={tuple(weight.shape)}, scale={tuple(weight_scale.shape)}."
        )
    scale = weight_scale.reshape(weight.shape[0], *([1] * (weight.dim() - 1)))
    dequantized = weight.to(torch.float32)
    if weight_offset is not None:
        if weight_offset.numel() != weight.shape[0]:
            raise ValueError(
                "W8A8 weight_offset must contain one value per output channel: "
                f"weight={tuple(weight.shape)}, "
                f"offset={tuple(weight_offset.shape)}."
            )
        offset = weight_offset.reshape(
            weight.shape[0], *([1] * (weight.dim() - 1))
        )
        dequantized.sub_(offset.to(torch.float32))
    return dequantized.mul_(scale.to(torch.float32))


def _load_weight_map(path: str) -> dict[str, str]:
    index_files = sorted(glob(os.path.join(path, "*.safetensors.index.json")))
    if not index_files:
        return {}
    with open(index_files[0], "r", encoding="utf-8") as file:
        index = json.load(file)
    return dict(index.get("weight_map", {}))


def _get_companion_tensor(
    *,
    tensor_name: str,
    current_file: str,
    current_reader,
    current_keys: set[str],
    root: str,
    weight_map: dict[str, str],
) -> torch.Tensor:
    if tensor_name in current_keys:
        return current_reader.get_tensor(tensor_name)
    shard = weight_map.get(tensor_name)
    if shard is None:
        raise KeyError(
            f"Quantized weight companion {tensor_name!r} was not found "
            f"while reading {current_file!r}."
        )
    shard_path = os.path.join(root, shard)
    with safe_open(shard_path, "pt", "cpu") as reader:
        return reader.get_tensor(tensor_name)


def load_model(
    model: nn.Module,
    path: str,
    name_mapping=None,
    *,
    quant_description: dict[str, Any] | None = None,
) -> set[str]:
    """Load safetensors weights, including the GLM ModelSlim export format.

    W8A8_DYNAMIC dense weights are dequantized to the model parameter dtype so
    the existing BF16 MLA/MLP implementation can be reused in the first GLM
    bring-up stage. W4A8_DYNAMIC routed-expert tensors stay packed and are sent
    to parameter-specific loaders through :class:`WeightTarget`.
    """

    if quant_description is None:
        quant_description = load_quant_description(path)
    packed_modules_mapping = getattr(model, "packed_modules_mapping", {})
    tensor_types = quant_tensor_types(quant_description)
    weight_map = _load_weight_map(path) if tensor_types else {}

    loaded_parameters: set[str] = set()
    for file in sorted(glob(os.path.join(path, "*.safetensors"))):
        with safe_open(file, "pt", "cpu") as reader:
            current_keys = set(reader.keys())
            for weight_name in reader.keys():
                quant_type = tensor_types.get(weight_name)

                # W8 auxiliary tensors are consumed with their base weight.
                if quant_type == "W8A8_DYNAMIC" and not weight_name.endswith(
                    ".weight"
                ):
                    continue

                target: str | WeightTarget | None = weight_name
                if name_mapping is not None:
                    target = name_mapping(weight_name)
                    if target is None:
                        continue

                tensor = reader.get_tensor(weight_name)
                if (
                    quant_type == "W4A8_DYNAMIC"
                    and weight_name.endswith(".weight_offset")
                ):
                    if torch.count_nonzero(tensor).item() != 0:
                        raise ValueError(
                            "GLM W4A8 supports symmetric weights only, but "
                            f"{weight_name!r} contains a non-zero offset."
                        )
                    continue
                if quant_type == "W8A8_DYNAMIC":
                    scale_name = weight_name.removesuffix(
                        ".weight"
                    ) + ".weight_scale"
                    if tensor_types.get(scale_name) != quant_type:
                        raise ValueError(
                            f"Missing W8A8_DYNAMIC metadata for {scale_name!r}."
                        )
                    scale = _get_companion_tensor(
                        tensor_name=scale_name,
                        current_file=file,
                        current_reader=reader,
                        current_keys=current_keys,
                        root=path,
                        weight_map=weight_map,
                    )
                    offset_name = weight_name.removesuffix(
                        ".weight"
                    ) + ".weight_offset"
                    if tensor_types.get(offset_name) != quant_type:
                        raise ValueError(
                            f"Missing W8A8_DYNAMIC metadata for "
                            f"{offset_name!r}."
                        )
                    offset = _get_companion_tensor(
                        tensor_name=offset_name,
                        current_file=file,
                        current_reader=reader,
                        current_keys=current_keys,
                        root=path,
                        weight_map=weight_map,
                    )
                    tensor = dequantize_w8a8_weight(tensor, scale, offset)

                if isinstance(target, WeightTarget):
                    try:
                        param = model.get_parameter(target.name)
                    except AttributeError as error:
                        raise AttributeError(
                            f"Failed to locate parameter {target.name!r} "
                            f"mapped from {weight_name!r}."
                        ) from error
                    weight_loader = getattr(
                        param, "weight_loader", default_weight_loader
                    )
                    weight_loader(param, tensor, *target.loader_args)
                    loaded_parameters.add(target.name)
                    continue

                target_name = target
                for checkpoint_name, (packed_name, shard_id) in (
                    packed_modules_mapping.items()
                ):
                    if checkpoint_name not in weight_name:
                        continue
                    param_name = target_name
                    if checkpoint_name in param_name:
                        param_name = param_name.replace(
                            checkpoint_name, packed_name
                        )
                    elif (
                        checkpoint_name in ("gate_proj", "up_proj")
                        and "gate_up_proj" in param_name
                    ):
                        param_name = param_name.replace(
                            "gate_up_proj", packed_name
                        )
                    param = model.get_parameter(param_name)
                    weight_loader = getattr(param, "weight_loader")
                    weight_loader(param, tensor, shard_id)
                    loaded_parameters.add(param_name)
                    break
                else:
                    try:
                        param = model.get_parameter(target_name)
                    except AttributeError as error:
                        raise AttributeError(
                            f"Failed to locate parameter {target_name!r} "
                            f"mapped from {weight_name!r}."
                        ) from error
                    weight_loader = getattr(
                        param, "weight_loader", default_weight_loader
                    )
                    weight_loader(param, tensor)
                    loaded_parameters.add(target_name)

    return loaded_parameters
