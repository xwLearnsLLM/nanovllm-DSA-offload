from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file


COPY_FILES = [
    "generation_config.json",
    "tokenizer.json",
    "tokenizer.model",
    "merges.txt",
    "vocab.json",
    "README.md",
    "configuration.json",
]

BOS_TOKEN = "<\uFF5Cbegin\u2581of\u2581sentence\uFF5C>"
EOS_TOKEN = "<\uFF5Cend\u2581of\u2581sentence\uFF5C>"
DEEPSEEK_USER_TOKEN = "<\uFF5CUser\uFF5C>"
DEEPSEEK_ASSISTANT_TOKEN = "<\uFF5CAssistant\uFF5C>"
DEFAULT_CHAT_TEMPLATE = """{% if not add_generation_prompt is defined %}{% set add_generation_prompt = false %}{% endif %}
{% set ns = namespace(system_prompt='') %}
{%- for message in messages %}
    {%- if message['role'] == 'system' %}
        {%- if ns.system_prompt %}
            {% set ns.system_prompt = ns.system_prompt + '\\n\\n' + message['content'] %}
        {%- else %}
            {% set ns.system_prompt = message['content'] %}
        {%- endif %}
    {%- endif %}
{%- endfor %}
{{ bos_token }}{{ ns.system_prompt }}
{%- for message in messages %}
    {%- if message['role'] == 'user' %}
        {{ '<\uFF5CUser\uFF5C>' + message['content'] }}
    {%- elif message['role'] == 'assistant' %}
        {{ '<\uFF5CAssistant\uFF5C>' + message['content'] + eos_token }}
    {%- endif %}
{%- endfor %}
{%- if add_generation_prompt %}
    {{ '<\uFF5CAssistant\uFF5C>' }}
{%- endif %}"""

SOURCE_FORMAT_W8A8 = "w8a8"
SOURCE_FORMAT_HF_BF16 = "hf_bf16"

W8A8_STATIC_ATTN_LINEARS = (
    "q_a_proj",
    "q_b_proj",
    "kv_a_proj_with_mqa",
    "o_proj",
)

BF16_ATTN_LINEAR_BIAS = {
    "q_a_proj": True,
    "q_b_proj": True,
    "kv_a_proj_with_mqa": True,
    "kv_b_proj": False,
    "o_proj": True,
}

ATTN_COMMON_SUFFIXES = (
    "self_attn.q_a_layernorm.weight",
    "self_attn.kv_a_layernorm.weight",
    "self_attn.indexer.wq_b.weight",
    "self_attn.indexer.wk.weight",
    "self_attn.indexer.k_norm.weight",
    "self_attn.indexer.k_norm.bias",
    "self_attn.indexer.weights_proj.weight",
    "input_layernorm.weight",
    "post_attention_layernorm.weight",
)

MLP_PROJ_NAMES = ("gate_proj", "up_proj", "down_proj")


def _detect_source_format_in_dir(source_dir: Path) -> str | None:
    if (source_dir / "quant_model_weights.safetensors.index.json").is_file():
        return SOURCE_FORMAT_W8A8
    if (source_dir / "model.safetensors.index.json").is_file():
        return SOURCE_FORMAT_HF_BF16
    if list(source_dir.glob("*.safetensors")):
        return SOURCE_FORMAT_HF_BF16

    return None


def resolve_model_dir(source_dir: Path) -> Path:
    direct_match = _detect_source_format_in_dir(source_dir)
    if direct_match is not None:
        return source_dir

    index_candidates = list(
        source_dir.rglob("quant_model_weights.safetensors.index.json")
    )
    index_candidates += list(source_dir.rglob("model.safetensors.index.json"))
    if index_candidates:
        index_candidates.sort(
            key=lambda path: (len(path.parts), str(path))
        )
        return index_candidates[0].parent

    snapshot_root = source_dir / "snapshots"
    if snapshot_root.is_dir():
        snapshot_dirs = sorted(
            (path for path in snapshot_root.iterdir() if path.is_dir()),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        for snapshot_dir in snapshot_dirs:
            if _detect_source_format_in_dir(snapshot_dir) is not None:
                return snapshot_dir

    return source_dir


def detect_source_format(source_dir: Path) -> str:
    resolved_dir = resolve_model_dir(source_dir)
    source_format = _detect_source_format_in_dir(resolved_dir)
    if source_format is not None:
        return source_format

    raise FileNotFoundError(
        "Unable to detect model format. Expected either "
        "'quant_model_weights.safetensors.index.json' for Ascend W8A8 "
        "or standard Hugging Face safetensors shards. "
        "If you downloaded from Hugging Face cache, try pointing to the "
        "snapshot directory that contains the actual shard files. "
        "If you intended to use the flat model directory, the safetensors "
        "download may not have finished yet."
    )


class ShardedTensorReader:
    def __init__(self, model_dir: Path, source_format: str) -> None:
        self.model_dir = model_dir
        self.source_format = source_format
        self.weight_map = self._build_weight_map()

    def _build_weight_map(self) -> dict[str, str]:
        if self.source_format == SOURCE_FORMAT_W8A8:
            index_path = self.model_dir / "quant_model_weights.safetensors.index.json"
        else:
            index_path = self.model_dir / "model.safetensors.index.json"

        if index_path.is_file():
            with index_path.open("r", encoding="utf-8") as file:
                index = json.load(file)
            return dict(index["weight_map"])

        weight_map: dict[str, str] = {}
        for shard_path in sorted(self.model_dir.glob("*.safetensors")):
            with safe_open(str(shard_path), framework="pt", device="cpu") as file:
                for name in file.keys():
                    weight_map[name] = shard_path.name
        if not weight_map:
            raise FileNotFoundError(
                f"No safetensors files were found under '{self.model_dir}'."
            )
        return weight_map

    def has_tensor(self, name: str) -> bool:
        return name in self.weight_map

    def get_tensor(self, name: str) -> torch.Tensor:
        shard_name = self.weight_map.get(name)
        if shard_name is None:
            raise KeyError(f"Tensor '{name}' is missing from the source index.")
        shard_path = self.model_dir / shard_name
        if not shard_path.is_file():
            raise FileNotFoundError(
                f"Missing source shard '{shard_path}'. "
                "Please download the original safetensors weights first."
            )
        with safe_open(str(shard_path), framework="pt", device="cpu") as file:
            return file.get_tensor(name)

    def get_optional_tensor(self, name: str) -> torch.Tensor | None:
        if not self.has_tensor(name):
            return None
        return self.get_tensor(name)


def to_dtype(tensor: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    return tensor.to(dtype).contiguous()


def as_channel_vector(tensor: torch.Tensor, channels: int) -> torch.Tensor:
    tensor = tensor.reshape(-1).to(torch.float32)
    if tensor.numel() == 1:
        return tensor.expand(channels)
    if tensor.numel() != channels:
        raise ValueError(
            f"Expected {channels} scale/bias values, got {tensor.numel()}."
        )
    return tensor


def as_broadcast_scale(scale: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    scale = scale.to(torch.float32)
    if scale.numel() == weight.shape[0]:
        view_shape = (weight.shape[0],) + (1,) * (weight.ndim - 1)
        return scale.reshape(view_shape)
    while scale.ndim < weight.ndim:
        scale = scale.unsqueeze(-1)
    return scale


def dequant_dynamic_weight(
    reader: ShardedTensorReader,
    prefix: str,
    dtype: torch.dtype,
) -> torch.Tensor:
    weight = reader.get_tensor(f"{prefix}.weight").to(torch.float32)
    scale = reader.get_tensor(f"{prefix}.weight_scale")
    scale = as_broadcast_scale(scale, weight)
    return to_dtype(weight * scale, dtype)


def dequant_static_linear(
    reader: ShardedTensorReader,
    prefix: str,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    weight = reader.get_tensor(f"{prefix}.weight").to(torch.float32)
    out_features = weight.shape[0]
    input_scale = reader.get_tensor(f"{prefix}.input_scale").reshape(-1)
    if input_scale.numel() != 1:
        raise ValueError(
            f"Expected scalar input_scale for '{prefix}', got {tuple(input_scale.shape)}."
        )
    input_scale_value = float(input_scale[0].item())
    if input_scale_value == 0.0:
        raise ValueError(f"input_scale is zero for '{prefix}'.")
    deq_scale = as_channel_vector(
        reader.get_tensor(f"{prefix}.deq_scale"),
        out_features,
    )
    quant_bias = as_channel_vector(
        reader.get_tensor(f"{prefix}.quant_bias"),
        out_features,
    )

    weight_scale = (deq_scale / input_scale_value).unsqueeze(-1)
    bias = quant_bias * deq_scale
    return to_dtype(weight * weight_scale, dtype), to_dtype(bias, dtype)


def copy_linear_weight(
    reader: ShardedTensorReader,
    prefix: str,
    dtype: torch.dtype,
    *,
    bias_required: bool,
) -> dict[str, torch.Tensor]:
    weight = to_dtype(reader.get_tensor(f"{prefix}.weight"), dtype)
    tensors = {f"{prefix}.weight": weight}

    bias = reader.get_optional_tensor(f"{prefix}.bias")
    if bias is not None:
        tensors[f"{prefix}.bias"] = to_dtype(bias, dtype)
    elif bias_required:
        tensors[f"{prefix}.bias"] = torch.zeros(weight.shape[0], dtype=dtype)
    return tensors


def export_attention_block(
    reader: ShardedTensorReader,
    layer_prefix: str,
    dtype: torch.dtype,
    source_format: str,
) -> dict[str, torch.Tensor]:
    tensors: dict[str, torch.Tensor] = {}
    attn_prefix = f"{layer_prefix}.self_attn"

    if source_format == SOURCE_FORMAT_W8A8:
        for linear_name in W8A8_STATIC_ATTN_LINEARS:
            weight, bias = dequant_static_linear(
                reader,
                f"{attn_prefix}.{linear_name}",
                dtype,
            )
            tensors[f"{attn_prefix}.{linear_name}.weight"] = weight
            tensors[f"{attn_prefix}.{linear_name}.bias"] = bias
        tensors[f"{attn_prefix}.kv_b_proj.weight"] = to_dtype(
            reader.get_tensor(f"{attn_prefix}.kv_b_proj.weight"),
            dtype,
        )
    else:
        for linear_name, bias_required in BF16_ATTN_LINEAR_BIAS.items():
            tensors.update(
                copy_linear_weight(
                    reader,
                    f"{attn_prefix}.{linear_name}",
                    dtype,
                    bias_required=bias_required,
                )
            )

    for suffix in ATTN_COMMON_SUFFIXES:
        tensors[f"{layer_prefix}.{suffix}"] = to_dtype(
            reader.get_tensor(f"{layer_prefix}.{suffix}"),
            dtype,
        )
    return tensors


def export_mlp_block(
    reader: ShardedTensorReader,
    layer_prefix: str,
    *,
    source_prefix: str,
    dtype: torch.dtype,
    source_format: str,
) -> dict[str, torch.Tensor]:
    tensors: dict[str, torch.Tensor] = {}
    for proj_name in MLP_PROJ_NAMES:
        target_name = f"{layer_prefix}.mlp.{proj_name}.weight"
        source_name = f"{source_prefix}.{proj_name}"
        if source_format == SOURCE_FORMAT_W8A8:
            tensors[target_name] = dequant_dynamic_weight(
                reader,
                source_name,
                dtype,
            )
        else:
            tensors[target_name] = to_dtype(
                reader.get_tensor(f"{source_name}.weight"),
                dtype,
            )
    return tensors


def export_layer(
    reader: ShardedTensorReader,
    config: dict,
    layer_idx: int,
    dtype: torch.dtype,
    source_format: str,
) -> dict[str, torch.Tensor]:
    layer_prefix = f"model.layers.{layer_idx}"
    tensors = export_attention_block(reader, layer_prefix, dtype, source_format)

    first_k_dense = int(config["first_k_dense_replace"])
    if layer_idx < first_k_dense:
        source_prefix = f"{layer_prefix}.mlp"
    else:
        source_prefix = f"{layer_prefix}.mlp.shared_experts"
    tensors.update(
        export_mlp_block(
            reader,
            layer_prefix,
            source_prefix=source_prefix,
            dtype=dtype,
            source_format=source_format,
        )
    )
    return tensors


def export_config(source_config: dict, source_format: str) -> dict:
    config = dict(source_config)
    config["model_type"] = "deepseek_v32"
    config["architectures"] = ["DeepseekV32ForCausalLM"]
    config["torch_dtype"] = "bfloat16"
    config["dtype"] = "bfloat16"
    config["num_nextn_predict_layers"] = 0
    config["n_routed_experts"] = 0
    config["num_experts_per_tok"] = 0
    config["nanovllm_pruned_shared_only"] = True
    config["nanovllm_export_format"] = (
        "bf16_from_w8a8"
        if source_format == SOURCE_FORMAT_W8A8
        else "bf16_from_hf"
    )
    config["tokenizer_class"] = "LlamaTokenizerFast"
    return config


def save_shard(
    output_dir: Path,
    filename: str,
    tensors: dict[str, torch.Tensor],
    weight_map: dict[str, str],
) -> int:
    path = output_dir / filename
    save_file(tensors, str(path))
    for tensor_name in tensors:
        weight_map[tensor_name] = filename
    return sum(tensor.numel() * tensor.element_size() for tensor in tensors.values())


def copy_metadata_files(source_dir: Path, output_dir: Path) -> None:
    for filename in COPY_FILES:
        src = source_dir / filename
        if src.is_file():
            shutil.copy2(src, output_dir / filename)


def _load_json_if_exists(path: Path) -> dict:
    if not path.is_file():
        return {}
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def _added_token(content: str) -> dict:
    return {
        "__type": "AddedToken",
        "content": content,
        "lstrip": False,
        "normalized": True,
        "rstrip": False,
        "single_word": False,
    }


def _normalize_token_entry(value: str | dict | None, fallback: str) -> dict:
    if isinstance(value, dict):
        return value
    if isinstance(value, str) and value:
        return _added_token(value)
    return _added_token(fallback)


def write_tokenizer_files(
    source_dir: Path,
    output_dir: Path,
    *,
    max_model_len: int,
) -> None:
    tokenizer_config = _load_json_if_exists(source_dir / "tokenizer_config.json")
    tokenizer_config["tokenizer_class"] = "LlamaTokenizerFast"
    tokenizer_config["tokenizer_file"] = "tokenizer.json"
    tokenizer_config["model_max_length"] = max_model_len
    tokenizer_config["legacy"] = True
    tokenizer_config["add_bos_token"] = False
    tokenizer_config["add_eos_token"] = False
    tokenizer_config["bos_token"] = _normalize_token_entry(
        tokenizer_config.get("bos_token"),
        BOS_TOKEN,
    )
    tokenizer_config["eos_token"] = _normalize_token_entry(
        tokenizer_config.get("eos_token"),
        EOS_TOKEN,
    )
    tokenizer_config["pad_token"] = _normalize_token_entry(
        tokenizer_config.get("pad_token"),
        tokenizer_config["eos_token"]["content"],
    )
    if tokenizer_config.get("unk_token") is not None:
        tokenizer_config["unk_token"] = _normalize_token_entry(
            tokenizer_config.get("unk_token"),
            str(tokenizer_config["unk_token"]),
        )
    tokenizer_config["chat_template"] = tokenizer_config.get(
        "chat_template", DEFAULT_CHAT_TEMPLATE
    )
    with (output_dir / "tokenizer_config.json").open("w", encoding="utf-8") as file:
        json.dump(tokenizer_config, file, ensure_ascii=False, indent=2)
        file.write("\n")

    special_tokens_map = _load_json_if_exists(source_dir / "special_tokens_map.json")
    special_tokens_map["bos_token"] = tokenizer_config["bos_token"]
    special_tokens_map["eos_token"] = tokenizer_config["eos_token"]
    special_tokens_map["pad_token"] = tokenizer_config["pad_token"]
    if tokenizer_config.get("unk_token") is not None:
        special_tokens_map["unk_token"] = tokenizer_config["unk_token"]
    with (output_dir / "special_tokens_map.json").open("w", encoding="utf-8") as file:
        json.dump(special_tokens_map, file, ensure_ascii=False, indent=2)
        file.write("\n")


def ensure_output_dir(output_dir: Path) -> None:
    if output_dir.exists():
        if any(output_dir.iterdir()):
            raise FileExistsError(
                f"Output directory '{output_dir}' already exists and is not empty."
            )
    else:
        output_dir.mkdir(parents=True)


def export_model(
    source_dir: Path,
    output_dir: Path,
    *,
    source_format: str,
) -> None:
    source_dir = resolve_model_dir(source_dir)
    ensure_output_dir(output_dir)
    reader = ShardedTensorReader(source_dir, source_format)

    with (source_dir / "config.json").open("r", encoding="utf-8") as file:
        source_config = json.load(file)

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

    num_layers = int(source_config["num_hidden_layers"])
    for layer_idx in range(num_layers):
        layer_tensors = export_layer(
            reader,
            source_config,
            layer_idx,
            dtype,
            source_format,
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

    config = export_config(source_config, source_format)
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
        "export_format": "shared-only-bf16",
        "kept_moe_path": "shared_experts only",
        "dropped_components": [
            "all routed experts",
            "all router/gate tensors",
            "nextn predict layers",
        ],
        "tokenizer_tokens": {
            "bos": BOS_TOKEN,
            "eos": EOS_TOKEN,
            "user": DEEPSEEK_USER_TOKEN,
            "assistant": DEEPSEEK_ASSISTANT_TOKEN,
        },
    }
    if source_format == SOURCE_FORMAT_W8A8:
        notes["static_w8a8_reconstruction"] = {
            "weight_scale": "deq_scale / input_scale",
            "bias": "quant_bias * deq_scale",
        }
        notes["dynamic_w8a8_reconstruction"] = {
            "weight_scale": "weight_scale",
        }
    else:
        notes["bf16_copy_mode"] = {
            "attention": "copy weights directly and synthesize zero biases where nano expects them",
            "mlp": "copy dense/shared-expert weights directly",
        }
    with (output_dir / "nanovllm_export_notes.json").open("w", encoding="utf-8") as file:
        json.dump(notes, file, ensure_ascii=False, indent=2)
        file.write("\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export DeepSeek-V3.2 into a shared-expert-only BF16 model "
            "directory that nano-vllm-ascend can load."
        )
    )
    parser.add_argument(
        "source_model",
        type=Path,
        help=(
            "Path to the original DeepSeek-V3.2 model directory. Supports "
            "either Ascend W8A8 or standard Hugging Face BF16 safetensors."
        ),
    )
    parser.add_argument(
        "output_model",
        type=Path,
        help="Path to write the pruned BF16 model directory.",
    )
    parser.add_argument(
        "--source-format",
        choices=("auto", SOURCE_FORMAT_W8A8, SOURCE_FORMAT_HF_BF16),
        default="auto",
        help="Override source model format detection.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_format = (
        detect_source_format(args.source_model)
        if args.source_format == "auto"
        else args.source_format
    )
    print(f"[export_deepseek_v32_pruned] detected source format: {source_format}")
    export_model(
        args.source_model,
        args.output_model,
        source_format=source_format,
    )


if __name__ == "__main__":
    main()
