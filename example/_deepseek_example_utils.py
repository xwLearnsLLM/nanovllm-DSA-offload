import os

from nanovllm import LLM, SamplingParams


MODEL_PATH = "/home/models/Deepseek-V3.2-Pruned-95B-BF/"
TP_SIZE = 4
DEEPSEEK_USER_TOKEN = "<\uFF5CUser\uFF5C>"
DEEPSEEK_ASSISTANT_TOKEN = "<\uFF5CAssistant\uFF5C>"


def env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.lower() in ("1", "true", "yes", "on")


def env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    return default if value is None else int(value)


def env_float(name: str, default: float) -> float:
    value = os.environ.get(name)
    return default if value is None else float(value)


def model_path() -> str:
    return os.environ.get("NANOVLLM_MODEL", MODEL_PATH)


def make_llm(
    *,
    max_model_len: int,
    max_num_prefill_seqs_per_step: int,
    max_num_decode_seqs_per_step: int,
) -> LLM:
    return LLM(
        model_path(),
        enforce_eager=True,
        tensor_parallel_size=env_int("NANOVLLM_TP_SIZE", TP_SIZE),
        enable_expert_parallel=env_bool("NANOVLLM_ENABLE_EXPERT_PARALLEL", True),
        max_model_len=max_model_len,
        max_num_prefill_seqs_per_step=max_num_prefill_seqs_per_step,
        max_num_decode_seqs_per_step=max_num_decode_seqs_per_step,
        kvcache_block_size=env_int("NANOVLLM_KVCACHE_BLOCK_SIZE", 128),
        trust_remote_code=True,
    )


def prompt_tokenizer(llm: LLM):
    return llm.tokenizer


def encode_prompts(tokenizer, prompts: list[str]) -> list[list[int]]:
    use_chat = env_bool("NANOVLLM_USE_DEEPSEEK_CHAT", False)
    add_bos = env_bool("NANOVLLM_ADD_BOS", use_chat)
    encoded = []
    for prompt in prompts:
        if use_chat:
            prompt = f"{DEEPSEEK_USER_TOKEN}{prompt}{DEEPSEEK_ASSISTANT_TOKEN}"
        token_ids = tokenizer.encode(prompt, add_special_tokens=False)
        if add_bos and tokenizer.bos_token_id is not None:
            token_ids = [tokenizer.bos_token_id] + token_ids
        encoded.append(token_ids)
    return encoded


def sampling_params(max_tokens: int) -> SamplingParams:
    return SamplingParams(
        temperature=env_float("NANOVLLM_TEMPERATURE", 0.02),
        max_tokens=env_int("NANOVLLM_MAX_GEN_TOKENS", max_tokens),
        ignore_eos=env_bool("NANOVLLM_IGNORE_EOS", False),
    )


def print_outputs(prompts: list[str], prompt_token_ids: list[list[int]], outputs) -> None:
    for prompt, ids, output in zip(prompts, prompt_token_ids, outputs):
        one_line_prompt = " ".join(prompt.split())
        print("prompt_len:", len(ids))
        print("prompt    :", one_line_prompt[:300])
        print("response  :", repr(output["text"]))
        print("token_ids :", output["token_ids"])
        print()
