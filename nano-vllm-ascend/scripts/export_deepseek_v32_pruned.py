from export_deepseek_v32_pruned_impl import main as _impl_main

if __name__ == "__main__":
    _impl_main()
    raise SystemExit(0)

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
    "README.md",
    "configuration.json",
]

BOS_TOKEN = "<\uff5cbegin\u2581of\u2581sentence\uff5c>"
EOS_TOKEN = "<\uff5cend\u2581of\u2581sentence\uff5c>"
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
        {{ '<｜User｜>' + message['content'] }}
    {%- elif message['role'] == 'assistant' %}
        {{ '<｜Assistant｜>' + message['content'] + eos_token }}
    {%- endif %}
{%- endfor %}
{%- if add_generation_prompt %}
    {{ '<｜Assistant｜>' }}
{%- endif %}"""

ATTN_STATIC_LINEARS = (
    "q_a_proj",
    "q_b_proj",
    "kv_a_proj_with_mqa",
    "o_proj",
)

FLOAT_LAYER_SUFFIXES = (
    "self_attn.q_a_layernorm.weight",
    "self_attn.kv_a_layernorm.weight",
    "self_attn.kv_b_proj.weight",
    "self_attn.indexer.wq_b.weight",
    "self_attn.indexer.wk.weight",
    "self_attn.indexer.k_norm.weight",
    "self_attn.indexer.k_norm.bias",
    "self_attn.indexer.weights_proj.weight",
    "input_layernorm.weight",
    "post_attention_layernorm.weight",
)

MLP_PROJ_NAMES = ("gate_proj", "up_proj", "down_proj")


class ShardedTensorReader:
    def __init__(self, model_dir: Path) -> None:
        self.model_dir = model_dir
        index_path = model_dir / "quant_model_weights.safetensors.index.json"
        with index_path.open("r", encoding="utf-8") as file:
            index = json.load(file)
        self.weight_map = index["weight_map"]

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

    # Upstream W8A8 static path uses:
    #   x_q = quantize(x, input_scale, input_offset)
    #   y = matmul(x_q, w_int8) * deq_scale + quant_bias * deq_scale
    # so the BF16 equivalent uses per-output weight scale deq_scale/input_scale
    # and keeps the quant_bias contribution as an explicit float bias term.
    weight_scale = (deq_scale / input_scale_value).unsqueeze(-1)
    bias = quant_bias * deq_scale
    return to_dtype(weight * weight_scale, dtype), to_dtype(bias, dtype)


def export_attention_block(
    reader: ShardedTensorReader,
    layer_prefix: str,
    dtype: torch.dtype,
) -> dict[str, torch.Tensor]:
    tensors: dict[str, torch.Tensor] = {}
    for linear_name in ATTN_STATIC_LINEARS:
        weight, bias = dequant_static_linear(
            reader,
            f"{layer_prefix}.self_attn.{linear_name}",
            dtype,
        )
        tensors[f"{layer_prefix}.self_attn.{linear_name}.weight"] = weight
        tensors[f"{layer_prefix}.self_attn.{linear_name}.bias"] = bias

    for suffix in FLOAT_LAYER_SUFFIXES:
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
) -> dict[str, torch.Tensor]:
    tensors: dict[str, torch.Tensor] = {}
    for proj_name in MLP_PROJ_NAMES:
        tensors[f"{layer_prefix}.mlp.{proj_name}.weight"] = dequant_dynamic_weight(
            reader,
            f"{source_prefix}.{proj_name}",
            dtype,
        )
    return tensors


def export_layer(
    reader: ShardedTensorReader,
    config: dict,
    layer_idx: int,
    dtype: torch.dtype,
) -> dict[str, torch.Tensor]:
    layer_prefix = f"model.layers.{layer_idx}"
    tensors = export_attention_block(reader, layer_prefix, dtype)

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
        )
    )
    return tensors


def export_config(source_config: dict) -> dict:
    config = dict(source_config)
    config["architectures"] = ["DeepseekV32ForCausalLM"]
    config["torch_dtype"] = "bfloat16"
    config["num_nextn_predict_layers"] = 0
    config["n_routed_experts"] = 0
    config["num_experts_per_tok"] = 0
    config["nanovllm_pruned_shared_only"] = True
    config["nanovllm_export_format"] = "bf16_from_w8a8"
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
    tokenizer_config["bos_token"] = tokenizer_config.get(
        "bos_token", _added_token(BOS_TOKEN)
    )
    tokenizer_config["eos_token"] = tokenizer_config.get(
        "eos_token", _added_token(EOS_TOKEN)
    )
    tokenizer_config["pad_token"] = tokenizer_config.get(
        "pad_token", tokenizer_config["eos_token"]
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


def export_model(source_dir: Path, output_dir: Path) -> None:
    ensure_output_dir(output_dir)
    reader = ShardedTensorReader(source_dir)

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
    total_size += save_shard(output_dir, "model-embeddings.safetensors", embeddings, weight_map)

    num_layers = int(source_config["num_hidden_layers"])
    for layer_idx in range(num_layers):
        layer_tensors = export_layer(reader, source_config, layer_idx, dtype)
        shard_name = f"model-layer-{layer_idx:03d}.safetensors"
        total_size += save_shard(output_dir, shard_name, layer_tensors, weight_map)

    final_tensors = {
        "model.norm.weight": to_dtype(reader.get_tensor("model.norm.weight"), dtype),
        "lm_head.weight": to_dtype(reader.get_tensor("lm_head.weight"), dtype),
    }
    total_size += save_shard(output_dir, "model-final.safetensors", final_tensors, weight_map)

    config = export_config(source_config)
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
        "export_format": "shared-only-bf16",
        "kept_moe_path": "shared_experts only",
        "dropped_components": [
            "all routed experts",
            "all router/gate tensors",
            "nextn predict layers",
        ],
        "static_w8a8_reconstruction": {
            "weight_scale": "deq_scale / input_scale",
            "bias": "quant_bias * deq_scale",
        },
        "dynamic_w8a8_reconstruction": {
            "weight_scale": "weight_scale",
        },
    }
    with (output_dir / "nanovllm_export_notes.json").open("w", encoding="utf-8") as file:
        json.dump(notes, file, ensure_ascii=False, indent=2)
        file.write("\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export DeepSeek-V3.2-W8A8 into a shared-expert-only BF16 model "
            "directory that nano-vllm-ascend can load."
        )
    )
    parser.add_argument("source_model", type=Path, help="Path to the original DeepSeek-V3.2-W8A8 model directory.")
    parser.add_argument("output_model", type=Path, help="Path to write the pruned BF16 model directory.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    export_model(args.source_model, args.output_model)


if __name__ == "__main__":
    main()
