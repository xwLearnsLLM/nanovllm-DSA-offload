"""Short-sequence eager smoke test for GLM-5.1-w4a8."""

from _example_utils import (
    env_bool,
    env_int,
    make_llm,
    print_outputs,
)
from nanovllm import SamplingParams


PROMPTS = [
    "中国的首都是哪里？只回答城市名。",
    "Calculate 2 + 3 * 4. Return only the number.",
    "Write one short sentence explaining what a large language model is.",
]


def main() -> None:
    max_model_len = env_int("NANOVLLM_MAX_MODEL_LEN", 512)
    max_tokens = env_int("NANOVLLM_MAX_GEN_TOKENS", 8)
    llm = make_llm(
        max_model_len=max_model_len,
        max_num_prefill_seqs_per_step=1,
        max_num_decode_seqs_per_step=len(PROMPTS),
    )
    tokenizer = llm.tokenizer
    prompt_token_ids = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=False,
            return_dict=False,
        )
        for prompt in PROMPTS
    ]
    longest = max(len(ids) for ids in prompt_token_ids)
    if longest + max_tokens > max_model_len:
        raise ValueError(
            "GLM chat prompt plus generated tokens exceeds "
            f"NANOVLLM_MAX_MODEL_LEN: {longest}+{max_tokens}>"
            f"{max_model_len}."
        )
    print(
        "GLM short-sequence smoke: "
        f"batch={len(PROMPTS)}, prompt_max={longest}, "
        f"max_tokens={max_tokens}, max_model_len={max_model_len}"
    )
    outputs = llm.generate(
        prompt_token_ids,
        SamplingParams(
            temperature=0.0,
            max_tokens=max_tokens,
            ignore_eos=env_bool("NANOVLLM_IGNORE_EOS", False),
        ),
    )
    print_outputs(PROMPTS, prompt_token_ids, outputs)


if __name__ == "__main__":
    main()
