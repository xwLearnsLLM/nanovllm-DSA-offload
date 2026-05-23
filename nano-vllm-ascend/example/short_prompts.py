from _deepseek_example_utils import (
    encode_prompts,
    make_llm,
    print_outputs,
    prompt_tokenizer,
    sampling_params,
)


prompts = [
    "The capital city of China is",
    "calculate 2 + 3 * 4 = ",
    "List all prime numbers < 100: 2, 3, ",
    "1, 3, 7, 15, 31, ",
    "Large Language Model is a type of AI model that",
    "If Tom is shorter than Jerry, Spike is taller than Tom, then the shortest person is",
    "If a train leaves at 2 PM and takes 3 hours to arrive, it will arrive at",
]


if __name__ == "__main__":
    llm = make_llm(
        max_model_len=512,
        max_num_batched_tokens=4096,
        max_num_seqs=len(prompts),
    )
    tokenizer = prompt_tokenizer(llm)
    prompt_token_ids = encode_prompts(tokenizer, prompts)
    outputs = llm.generate(
        prompt_token_ids,
        sampling_params(max_tokens=16),
        use_tqdm=True,
    )
    print_outputs(prompts, prompt_token_ids, outputs)
