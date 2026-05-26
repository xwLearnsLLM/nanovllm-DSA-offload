import os
import random

from _deepseek_example_utils import (
    DEEPSEEK_ASSISTANT_TOKEN,
    DEEPSEEK_USER_TOKEN,
    env_bool,
    env_float,
    env_int,
    make_llm,
    print_outputs,
    prompt_tokenizer,
)
from nanovllm.engine.dsa_offload import compute_sparse_blocks
from nanovllm import SamplingParams


def _env_int_any(names: tuple[str, ...], default: int) -> int:
    for name in names:
        value = os.environ.get(name)
        if value is not None:
            return int(value)
    return default


def parse_prompt_lengths() -> list[int]:
    exact_lengths = os.environ.get("NANOVLLM_PROMPT_LENGTHS", "").strip()
    if exact_lengths:
        lengths = [
            int(item.strip())
            for item in exact_lengths.split(",")
            if item.strip()
        ]
        if not lengths:
            raise ValueError("NANOVLLM_PROMPT_LENGTHS is set but empty")
        return lengths

    legacy_long = env_int("NANOVLLM_LONG_PROMPT_TOKENS", 0)
    default_min = legacy_long if legacy_long > 0 else 128
    default_max = legacy_long if legacy_long > 0 else default_min
    num_prompts = _env_int_any(
        ("NANOVLLM_TEST_NUM_PROMPTS", "NANOVLLM_NUM_PROMPTS"),
        1,
    )
    min_tokens = _env_int_any(
        ("NANOVLLM_PROMPT_MIN_TOKENS", "NANOVLLM_MIN_PROMPT_TOKENS"),
        default_min,
    )
    max_tokens = _env_int_any(
        ("NANOVLLM_PROMPT_MAX_TOKENS", "NANOVLLM_MAX_PROMPT_TOKENS"),
        default_max,
    )
    if num_prompts <= 0:
        raise ValueError("num prompts must be positive")
    if min_tokens <= 0 or max_tokens <= 0:
        raise ValueError("prompt token lengths must be positive")
    if min_tokens > max_tokens:
        raise ValueError("prompt min tokens must be <= prompt max tokens")

    seed = env_int("NANOVLLM_PROMPT_SEED", 0)
    rng = random.Random(seed)
    return [rng.randint(min_tokens, max_tokens) for _ in range(num_prompts)]


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


def print_prompt_plan(lengths: list[int], block_size: int) -> None:
    print("prompt plan:")
    for i, length in enumerate(lengths, 1):
        full_blocks = length // block_size
        sparse_blocks = compute_sparse_blocks(full_blocks)
        release_blocks = max(0, full_blocks - sparse_blocks)
        print(
            f"  prompt {i}: target_len={length}, "
            f"full_blocks={full_blocks}, sparse_blocks={sparse_blocks}, "
            f"release_blocks={release_blocks}"
        )


def main() -> None:
    prompt_lengths = parse_prompt_lengths()
    max_prompt_len = max(prompt_lengths)
    max_gen_tokens = env_int("NANOVLLM_MAX_GEN_TOKENS", 1)
    max_model_len = env_int(
        "NANOVLLM_MAX_MODEL_LEN",
        max_prompt_len + max_gen_tokens,
    )
    if max_model_len < max_prompt_len + max_gen_tokens:
        raise ValueError(
            "NANOVLLM_MAX_MODEL_LEN must cover prompt length plus generation "
            f"tokens: got {max_model_len}, need at least "
            f"{max_prompt_len + max_gen_tokens}."
        )

    max_num_batched_tokens = env_int(
        "NANOVLLM_MAX_BATCHED_TOKENS",
        max(sum(prompt_lengths), max_model_len),
    )
    max_num_seqs = env_int("NANOVLLM_MAX_NUM_SEQS", len(prompt_lengths))
    if max_num_seqs < len(prompt_lengths):
        raise ValueError(
            "NANOVLLM_MAX_NUM_SEQS must be >= number of prompts: "
            f"got {max_num_seqs}, need {len(prompt_lengths)}."
        )
    if max_num_batched_tokens < max_model_len:
        raise ValueError(
            "NANOVLLM_MAX_BATCHED_TOKENS must be >= NANOVLLM_MAX_MODEL_LEN: "
            f"got {max_num_batched_tokens}, need at least {max_model_len}."
        )
    block_size = env_int("NANOVLLM_KVCACHE_BLOCK_SIZE", 128)

    llm = make_llm(
        max_model_len=max_model_len,
        max_num_batched_tokens=max_num_batched_tokens,
        max_num_seqs=max_num_seqs,
    )
    tokenizer = prompt_tokenizer(llm)

    prompts = [
        f"<random exact-token prompt {i}: target_len={length}>"
        for i, length in enumerate(prompt_lengths, 1)
    ]
    prompt_token_ids = [
        build_exact_token_prompt(tokenizer, length)
        for length in prompt_lengths
    ]

    print(
        "test config: "
        f"num_prompts={len(prompt_lengths)}, "
        f"prompt_min={min(prompt_lengths)}, "
        f"prompt_max={max(prompt_lengths)}, "
        f"max_model_len={max_model_len}, "
        f"max_num_batched_tokens={max_num_batched_tokens}, "
        f"max_num_seqs={max_num_seqs}, "
        f"max_gen_tokens={max_gen_tokens}"
    )
    print_prompt_plan([len(ids) for ids in prompt_token_ids], block_size)
    for i, ids in enumerate(prompt_token_ids, 1):
        print(f"prompt {i} token_len={len(ids)} first_ids={ids[:16]}")

    outputs = llm.generate(
        prompt_token_ids,
        SamplingParams(
            temperature=env_float("NANOVLLM_TEMPERATURE", 0.0),
            max_tokens=max_gen_tokens,
            ignore_eos=env_bool("NANOVLLM_IGNORE_EOS", True),
        ),
    )
    print_outputs(prompts, prompt_token_ids, outputs)


if __name__ == "__main__":
    main()
