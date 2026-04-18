from nanovllm import LLM, SamplingParams

MODEL_PATH = "/home/models/Deepseek-V3.2-Pruned-95B-BF/"
TP_SIZE    = 4

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

    llm = LLM(
        model=MODEL_PATH,
        enforce_eager=True,
        tensor_parallel_size=TP_SIZE,
        enable_expert_parallel=True,
        max_model_len=8192,
        max_num_batched_tokens=8192,
        max_num_seqs=128,
        kvcache_block_size=128,
        skip_warmup=1,
        trust_remote_code=True,
        gpu_memory_utilization=0.95,
    )

    outputs = llm.generate(prompts, SamplingParams(temperature=0.02, max_tokens=32))

    for (prompt, output) in zip(prompts, outputs):
        print("prompt  :", prompt)
        print("response:", repr(output["text"]))
