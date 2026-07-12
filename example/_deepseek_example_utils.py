import os

from nanovllm import LLM, SamplingParams


MODEL_PATH = "/home/models/DeepSeek-V3.2-REAP-345B-A37B-BF16/"
TP_SIZE = 16
DEEPSEEK_USER_TOKEN = "<\uFF5CUser\uFF5C>"
DEEPSEEK_ASSISTANT_TOKEN = "<\uFF5CAssistant\uFF5C>"


def env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    value = value.strip().lower()
    if value in ("1", "true", "yes", "on"):
        return True
    if value in ("0", "false", "no", "off"):
        return False
    raise ValueError(f"{name} must be a boolean, got {value!r}.")


def env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {value!r}.") from exc


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
    enforce_eager = env_bool("NANOVLLM_ENFORCE_EAGER", False)
    return LLM(
        model_path(),
        enforce_eager=enforce_eager,
        decode_graph_capture_sizes=(max_num_decode_seqs_per_step,),
        tensor_parallel_size=env_int("NANOVLLM_TP_SIZE", TP_SIZE),
        enable_expert_parallel=env_bool("NANOVLLM_ENABLE_EXPERT_PARALLEL", True),
        max_model_len=max_model_len,
        max_num_prefill_seqs_per_step=max_num_prefill_seqs_per_step,
        max_num_decode_seqs_per_step=max_num_decode_seqs_per_step,
        kvcache_block_size=env_int("NANOVLLM_KVCACHE_BLOCK_SIZE", 128),
        num_hbm_kvcache_blocks=env_int("NANOVLLM_HBM_NUM_BLOCKS", -1),
        num_dram_kvcache_blocks=env_int("NANOVLLM_DRAM_NUM_BLOCKS", -1),
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
