import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from nanovllm import LLM, SamplingParams


DEFAULT_PROMPT = "Introduce yourself in one short sentence."


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Minimal DeepSeek-V3.2 chat example for nano-vllm-ascend-ep."
    )
    parser.add_argument(
        "prompt",
        nargs="?",
        default=DEFAULT_PROMPT,
        help="Single user prompt.",
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("NANOVLLM_MODEL"),
        help="Path to the BF16 DeepSeek-V3.2 model directory.",
    )
    parser.add_argument(
        "--tp",
        type=int,
        default=get_env_int("NANOVLLM_TP_SIZE", 4),
        help="Tensor parallel size.",
    )
    parser.add_argument(
        "--ep",
        dest="ep",
        action="store_true",
        help="Enable expert parallel.",
    )
    parser.add_argument(
        "--no-ep",
        dest="ep",
        action="store_false",
        help="Disable expert parallel.",
    )
    parser.set_defaults(
        ep=get_env_bool("NANOVLLM_ENABLE_EXPERT_PARALLEL", True)
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=get_env_int("NANOVLLM_MAX_GEN_TOKENS", 64),
        help="Maximum generated tokens.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=get_env_float("NANOVLLM_TEMPERATURE", 0.0),
        help="Sampling temperature.",
    )
    return parser.parse_args()


def build_prompt_token_ids(tokenizer, prompt: str) -> list[int]:
    messages = [{"role": "user", "content": prompt}]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
    )


def main() -> None:
    args = parse_args()
    if not args.model:
        raise ValueError(
            "Set --model or NANOVLLM_MODEL to the BF16 DeepSeek-V3.2 model directory."
        )

    llm = LLM(
        args.model,
        enforce_eager=True,
        tensor_parallel_size=args.tp,
        enable_expert_parallel=args.ep,
        max_model_len=get_env_int("NANOVLLM_MAX_MODEL_LEN", 256),
        max_num_batched_tokens=get_env_int("NANOVLLM_MAX_BATCHED_TOKENS", 512),
        max_num_seqs=1,
        kvcache_block_size=get_env_int("NANOVLLM_KVCACHE_BLOCK_SIZE", 128),
        skip_warmup=get_env_bool("NANOVLLM_SKIP_WARMUP", True),
        trust_remote_code=True,
        gpu_memory_utilization=get_env_float(
            "NANOVLLM_GPU_MEMORY_UTILIZATION",
            0.95,
        ),
    )
    sampling_params = SamplingParams(
        temperature=args.temperature,
        max_tokens=args.max_new_tokens,
    )
    prompt_token_ids = [build_prompt_token_ids(llm.tokenizer, args.prompt)]
    output = llm.generate(
        prompt_token_ids,
        sampling_params,
        use_tqdm=False,
    )[0]
    print(output["text"])


if __name__ == "__main__":
    main()
