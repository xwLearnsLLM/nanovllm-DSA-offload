from _example_utils import (
    encode_prompts,
    make_llm,
    print_outputs,
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
        # DSA graph capture has a fixed 2048-token sparse budget even though
        # these short requests themselves remain on the eager decode path.
        max_model_len=2048,
        max_num_prefill_seqs_per_step=1,
        max_num_decode_seqs_per_step=len(prompts),
    )
    tokenizer = llm.tokenizer
    prompt_token_ids = encode_prompts(tokenizer, prompts)
    outputs = llm.generate(
        prompt_token_ids,
        sampling_params(max_tokens=16),
    )
    print_outputs(prompts, prompt_token_ids, outputs)
