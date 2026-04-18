import os

from nanovllm import LLM, SamplingParams

MODEL_PATH = "/home/models/Deepseek-V3.2-Pruned-95B-BF/"
TP_SIZE    = 4
DEEPSEEK_USER_TOKEN = "<\uFF5CUser\uFF5C>"
DEEPSEEK_ASSISTANT_TOKEN = "<\uFF5CAssistant\uFF5C>"


def get_env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.lower() in ("1", "true", "yes", "on")


def encode_prompts(tokenizer, prompts: list[str]) -> list[list[int]]:
    use_deepseek_chat = get_env_bool("NANOVLLM_USE_DEEPSEEK_CHAT", False)
    add_bos = get_env_bool("NANOVLLM_ADD_BOS", use_deepseek_chat)
    bos_token_id = tokenizer.bos_token_id
    prompt_token_ids = []
    for prompt in prompts:
        if use_deepseek_chat:
            prompt = f"{DEEPSEEK_USER_TOKEN}{prompt}{DEEPSEEK_ASSISTANT_TOKEN}"
        token_ids = tokenizer.encode(prompt, add_special_tokens=False)
        if add_bos and bos_token_id is not None:
            token_ids = [bos_token_id] + token_ids
        prompt_token_ids.append(token_ids)
    return prompt_token_ids


def maybe_print_debug_prompts(tokenizer, prompts: list[str], prompt_token_ids: list[list[int]]) -> None:
    if not get_env_bool("NANOVLLM_DEBUG_PROMPTS", False):
        return
    for prompt, token_ids in zip(prompts, prompt_token_ids):
        print("debug prompt:", repr(prompt))
        print("debug prompt_token_ids:", token_ids[:32])
        print(
            "debug prompt_tokens:",
            tokenizer.convert_ids_to_tokens(token_ids[:32]),
        )


def maybe_print_debug_outputs(tokenizer, outputs) -> None:
    if not get_env_bool("NANOVLLM_DEBUG_OUTPUT_TOKENS", False):
        return
    for output in outputs:
        token_ids = output["token_ids"][:16]
        print("debug output_token_ids:", token_ids)
        print(
            "debug output_tokens:",
            tokenizer.convert_ids_to_tokens(token_ids),
        )

if __name__ == '__main__' :
    prompts = [
        "The capital city of China is",
        "calculate 2 + 3 * 4 = ",
        "List all prime numbers < 100: 2, 3, ",
        "1, 3, 7, 15, 31, ",
        "Large Language Model is a type of AI model that",
        "If Tom is shorter than Jerry, Spike is taller than Tom, then the shortest person is",
        "If a train leaves at 2 PM and takes 3 hours to arrive, it will arrive at",
    ]

    tp_size = int(os.environ.get("NANOVLLM_TP_SIZE", str(TP_SIZE)))
    enable_expert_parallel = get_env_bool(
        "NANOVLLM_ENABLE_EXPERT_PARALLEL",
        True,
    )

    llm = LLM(
        model=MODEL_PATH,
        enforce_eager=True,
        tensor_parallel_size=tp_size,
        enable_expert_parallel=enable_expert_parallel,
        max_model_len=8192,
        max_num_batched_tokens=8192,
        max_num_seqs=128,
        kvcache_block_size=128,
        skip_warmup=1,
        trust_remote_code=True,
        gpu_memory_utilization=0.95,
    )

    prompt_token_ids = encode_prompts(llm.tokenizer, prompts)
    maybe_print_debug_prompts(llm.tokenizer, prompts, prompt_token_ids)
    outputs = llm.generate(
        prompt_token_ids,
        SamplingParams(temperature=0.02, max_tokens=32),
    )
    maybe_print_debug_outputs(llm.tokenizer, outputs)

    for (prompt, output) in zip(prompts, outputs):
        print("prompt  :", prompt)
        print("response:", repr(output["text"]))
