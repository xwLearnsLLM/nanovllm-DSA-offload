import os
from pathlib import Path
import json

from nanovllm import LLM, SamplingParams
from nanovllm.utils.logger import init_logger
from transformers import LlamaTokenizerFast

logger = init_logger(__name__)

DEEPSEEK_USER_TOKEN = "<\uFF5CUser\uFF5C>"
DEEPSEEK_ASSISTANT_TOKEN = "<\uFF5CAssistant\uFF5C>"
DEFAULT_MODEL_CANDIDATES = (
    r"E:\LLM\models\Deepseek-V3.2-Pruned-95B-BF16",
    "/data/model/Deepseek-V3.2-Pruned-95B-BF16",
)


def get_env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    return default if value is None else int(value)


def get_env_float(name: str, default: float) -> float:
    value = os.environ.get(name)
    return default if value is None else float(value)


def get_env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.lower() in ("1", "true", "yes", "on")


def resolve_model_path() -> str:
    configured = os.environ.get("NANOVLLM_MODEL")
    if configured:
        return os.path.expanduser(configured)
    for candidate in DEFAULT_MODEL_CANDIDATES:
        expanded = os.path.expanduser(candidate)
        if os.path.isdir(expanded):
            return expanded
    raise FileNotFoundError(
        "Set NANOVLLM_MODEL to the BF16-exported DeepSeek-V3.2-Pruned-95B "
        "directory. If you only have the original FP8 checkpoint, convert it "
        "first with `python scripts/export_deepseek_v32_to_hf_bf16.py "
        "<source_model> <output_model>`."
    )


def load_model_config(model_path: str) -> dict:
    config_path = Path(model_path) / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"Missing config.json under {model_path!r}.")
    with config_path.open("r", encoding="utf-8") as file:
        return json.load(file)


def ensure_bf16_export(model_path: str) -> None:
    config = load_model_config(model_path)
    quantization_config = config.get("quantization_config") or {}
    quant_method = str(quantization_config.get("quant_method", "")).lower()
    if quant_method != "fp8":
        return
    source_hint = model_path.rstrip("\\/")
    output_hint = f"{source_hint}-BF16"
    raise ValueError(
        "The example must point to the BF16-exported DeepSeek-V3.2-Pruned-95B "
        "directory, not the original FP8 checkpoint. Run "
        f"`python scripts/export_deepseek_v32_to_hf_bf16.py {source_hint} "
        f"{output_hint}` and then set NANOVLLM_MODEL={output_hint!r}."
    )


def encode_plain_prompts(
    tokenizer,
    prompts: list[str],
    *,
    add_bos: bool,
) -> list[list[int]]:
    bos_token_id = tokenizer.bos_token_id
    prompt_token_ids = []
    for prompt in prompts:
        token_ids = tokenizer.encode(prompt, add_special_tokens=False)
        if add_bos and bos_token_id is not None:
            token_ids = [bos_token_id] + token_ids
        prompt_token_ids.append(token_ids)
    return prompt_token_ids


def format_prompts(prompts: list[str], use_deepseek_chat: bool) -> list[str]:
    if not use_deepseek_chat:
        return prompts
    return [
        f"{DEEPSEEK_USER_TOKEN}{prompt}{DEEPSEEK_ASSISTANT_TOKEN}"
        for prompt in prompts
    ]


def main():
    model_path = resolve_model_path()
    ensure_bf16_export(model_path)
    tokenizer_path = os.path.expanduser(
        os.environ.get(
            "NANOVLLM_TOKENIZER",
            model_path,
        )
    )
    tensor_parallel_size = get_env_int("NANOVLLM_TP_SIZE", 4)
    enable_expert_parallel = get_env_bool(
        "NANOVLLM_ENABLE_EXPERT_PARALLEL",
        True,
    )
    max_model_len = get_env_int("NANOVLLM_MAX_MODEL_LEN", 256)
    max_num_batched_tokens = get_env_int(
        "NANOVLLM_MAX_BATCHED_TOKENS",
        512,
    )
    max_num_seqs = get_env_int("NANOVLLM_MAX_NUM_SEQS", 4)
    kvcache_block_size = get_env_int("NANOVLLM_KVCACHE_BLOCK_SIZE", 128)
    max_gen_tokens = get_env_int("NANOVLLM_MAX_GEN_TOKENS", 64)
    temperature = get_env_float("NANOVLLM_TEMPERATURE", 0.0)
    gpu_memory_utilization = get_env_float(
        "NANOVLLM_GPU_MEMORY_UTILIZATION",
        0.95,
    )
    skip_warmup = get_env_bool("NANOVLLM_SKIP_WARMUP", True)
    use_deepseek_chat = get_env_bool("NANOVLLM_USE_DEEPSEEK_CHAT", True)
    add_bos_env = os.environ.get("NANOVLLM_ADD_BOS")
    if add_bos_env is None:
        add_bos = use_deepseek_chat
    else:
        add_bos = add_bos_env.lower() in ("1", "true", "yes", "on")

    llm = LLM(
        model_path,
        enforce_eager=True,
        tensor_parallel_size=tensor_parallel_size,
        enable_expert_parallel=enable_expert_parallel,
        max_model_len=max_model_len,
        max_num_batched_tokens=max_num_batched_tokens,
        max_num_seqs=max_num_seqs,
        kvcache_block_size=kvcache_block_size,
        skip_warmup=skip_warmup,
        trust_remote_code=True,
        gpu_memory_utilization=gpu_memory_utilization,
    )
    prompt_tokenizer = (
        llm.tokenizer
        if tokenizer_path == model_path
        else LlamaTokenizerFast.from_pretrained(tokenizer_path, legacy=True)
    )
    sampling_params = SamplingParams(
        temperature=temperature,
        max_tokens=max_gen_tokens,
    )

    logger.info(
        "example config: tp=%s, max_model_len=%s, max_num_batched_tokens=%s, "
        "max_num_seqs=%s, kvcache_block_size=%s, skip_warmup=%s, "
        "max_gen_tokens=%s, temperature=%s, use_deepseek_chat=%s, add_bos=%s, "
        "enable_expert_parallel=%s, gpu_memory_utilization=%s, tokenizer_path=%s",
        tensor_parallel_size,
        max_model_len,
        max_num_batched_tokens,
        max_num_seqs,
        kvcache_block_size,
        skip_warmup,
        max_gen_tokens,
        temperature,
        use_deepseek_chat,
        add_bos,
        enable_expert_parallel,
        gpu_memory_utilization,
        tokenizer_path,
    )

    prompts = [
        "Answer briefly: what is 2 + 3 * 4?",
        "Explain sparse attention in one short paragraph.",
        "Translate 'hello world' into Chinese.",
    ]
    formatted_prompts = format_prompts(prompts, use_deepseek_chat)
    prompt_token_ids = encode_plain_prompts(
        prompt_tokenizer,
        formatted_prompts,
        add_bos=add_bos,
    )

    outputs = llm.generate(prompt_token_ids, sampling_params, use_tqdm=False)
    for prompt, prompt_ids, output in zip(prompts, prompt_token_ids, outputs):
        external_text = prompt_tokenizer.decode(
            output["token_ids"],
            skip_special_tokens=False,
        )
        logger.info("")
        logger.info(f"Prompt: {prompt!r}")
        logger.info(
            "prompt len: %s, completion len: %s, text: %r, external_text: %r, "
            "first_prompt_ids: %s, first_output_ids: %s",
            output["prompt_len"],
            len(output["token_ids"]),
            output["text"],
            external_text,
            prompt_ids[:16],
            output["token_ids"][:16],
        )


if __name__ == "__main__":
    main()
