import argparse
import json
from pathlib import Path


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

BOS_TOKEN = "<\uFF5Cbegin\u2581of\u2581sentence\uFF5C>"
EOS_TOKEN = "<\uFF5Cend\u2581of\u2581sentence\uFF5C>"
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


def _read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def _write_json(path: Path, data: dict) -> None:
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)
        file.write("\n")


def _added_token(content: str) -> dict:
    return {
        "__type": "AddedToken",
        "content": content,
        "lstrip": False,
        "normalized": True,
        "rstrip": False,
        "single_word": False,
    }


def repair_export(model_dir: Path) -> None:
    config_path = model_dir / "config.json"
    tokenizer_config_path = model_dir / "tokenizer_config.json"
    special_tokens_map_path = model_dir / "special_tokens_map.json"

    config = _read_json(config_path)
    config["tokenizer_class"] = "LlamaTokenizerFast"
    _write_json(config_path, config)

    tokenizer_config = (
        _read_json(tokenizer_config_path) if tokenizer_config_path.is_file() else {}
    )
    tokenizer_config["tokenizer_class"] = "LlamaTokenizerFast"
    tokenizer_config["tokenizer_file"] = "tokenizer.json"
    tokenizer_config["model_max_length"] = int(
        config.get("max_position_embeddings", tokenizer_config.get("model_max_length", 131072))
    )
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
    _write_json(tokenizer_config_path, tokenizer_config)

    special_tokens_map = (
        _read_json(special_tokens_map_path)
        if special_tokens_map_path.is_file()
        else {}
    )
    special_tokens_map["bos_token"] = tokenizer_config["bos_token"]
    special_tokens_map["eos_token"] = tokenizer_config["eos_token"]
    special_tokens_map["pad_token"] = tokenizer_config["pad_token"]
    if tokenizer_config.get("unk_token") is not None:
        special_tokens_map["unk_token"] = tokenizer_config["unk_token"]
    _write_json(special_tokens_map_path, special_tokens_map)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Repair tokenizer/config metadata for a DeepSeek-V3.2 shared-only "
            "export without re-running the full prune."
        )
    )
    parser.add_argument("model_dir", type=Path, help="Path to the exported pruned model directory.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repair_export(args.model_dir)


if __name__ == "__main__":
    main()
