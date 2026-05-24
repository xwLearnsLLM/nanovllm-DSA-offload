from _deepseek_example_utils import (
    DEEPSEEK_ASSISTANT_TOKEN,
    DEEPSEEK_USER_TOKEN,
    encode_prompts,
    env_bool,
    env_float,
    env_int,
    make_llm,
    print_outputs,
    prompt_tokenizer,
)
from nanovllm import SamplingParams


def build_exact_token_prompt(tokenizer, target_len: int) -> list[int]:
    if target_len <= 0:
        raise ValueError("target_len must be positive")

    use_chat = env_bool("NANOVLLM_USE_DEEPSEEK_CHAT", True)
    add_bos = env_bool("NANOVLLM_ADD_BOS", use_chat)
    prefix = DEEPSEEK_USER_TOKEN if use_chat else ""
    suffix = DEEPSEEK_ASSISTANT_TOKEN if use_chat else ""
    prefix_ids = tokenizer.encode(prefix, add_special_tokens=False)
    suffix_ids = tokenizer.encode(suffix, add_special_tokens=False)
    if add_bos and tokenizer.bos_token_id is not None:
        prefix_ids = [tokenizer.bos_token_id] + prefix_ids

    body_len = target_len - len(prefix_ids) - len(suffix_ids)
    if body_len <= 0:
        raise ValueError("target_len is too small for the prompt wrapper")

    seed = (
        "DeepSeek sparse attention validation. "
        "This deterministic sentence builds a controlled prefill prompt. "
    )
    seed_ids = tokenizer.encode(seed, add_special_tokens=False)
    repeats = (body_len + len(seed_ids) - 1) // len(seed_ids)
    return prefix_ids + (seed_ids * repeats)[:body_len] + suffix_ids


def main() -> None:
    max_model_len = env_int("NANOVLLM_MAX_MODEL_LEN", 256)
    max_num_batched_tokens = env_int("NANOVLLM_MAX_BATCHED_TOKENS", max_model_len)
    max_num_seqs = env_int("NANOVLLM_MAX_NUM_SEQS", 1)
    long_prompt_tokens = env_int("NANOVLLM_LONG_PROMPT_TOKENS", 0)

    llm = make_llm(
        max_model_len=max_model_len,
        max_num_batched_tokens=max_num_batched_tokens,
        max_num_seqs=max_num_seqs,
    )
    tokenizer = prompt_tokenizer(llm)

    if long_prompt_tokens > 0:
        prompts = [f"<exact token prompt: {long_prompt_tokens}>"]
        prompt_token_ids = [build_exact_token_prompt(tokenizer, long_prompt_tokens)]
    else:
        prompts = [
            "Answer briefly: what is 2 + 3 * 4?",
            "Translate 'hello world' into Chinese.",
            "Name one practical use of sparse attention.",
        ]
        prompt_token_ids = encode_prompts(tokenizer, prompts)

    for i, ids in enumerate(prompt_token_ids, 1):
        print(f"prompt {i} token_len={len(ids)} first_ids={ids[:16]}")

    outputs = llm.generate(
        prompt_token_ids,
        SamplingParams(
            temperature=env_float("NANOVLLM_TEMPERATURE", 0.0),
            max_tokens=env_int("NANOVLLM_MAX_GEN_TOKENS", 1),
            ignore_eos=env_bool("NANOVLLM_IGNORE_EOS", True),
        ),
    )
    print_outputs(prompts, prompt_token_ids, outputs)


if __name__ == "__main__":
    main()
