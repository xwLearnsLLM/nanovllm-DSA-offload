from __future__ import annotations

"""Run vLLM-Ascend and print the real SFA/indexer op inputs.

This script is for comparing Nano's DeepSeek-V3.2 SFA call with the
working vllm-ascend 0.19 path on the Ascend machine. It installs a
sitecustomize hook through PYTHONPATH so TP worker processes also trace
torch.ops._C_ascend calls.
"""

import argparse
import importlib.util
import os
import sys
from pathlib import Path


DEEPSEEK_USER_TOKEN = "<\uFF5CUser\uFF5C>"
DEEPSEEK_ASSISTANT_TOKEN = "<\uFF5CAssistant\uFF5C>"


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        return default


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.lower() in ("1", "true", "yes", "on")


def _install_trace_hook_for_this_process() -> None:
    trace_dir = Path(__file__).resolve().parent / "sfa_trace_sitecustomize"
    existing = os.environ.get("PYTHONPATH", "")
    parts = [part for part in existing.split(os.pathsep) if part]
    if str(trace_dir) not in parts:
        os.environ["PYTHONPATH"] = (
            f"{trace_dir}{os.pathsep}{existing}" if existing else str(trace_dir)
        )
    if str(trace_dir) not in sys.path:
        sys.path.insert(0, str(trace_dir))

    sitecustomize_path = trace_dir / "sitecustomize.py"
    spec = importlib.util.spec_from_file_location(
        "_sfa_trace_sitecustomize_runtime",
        sitecustomize_path,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {sitecustomize_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)


def _load_tokenizer(model_path: str):
    from transformers import PreTrainedTokenizerFast

    return PreTrainedTokenizerFast.from_pretrained(
        model_path,
        trust_remote_code=True,
    )


def _build_prompt_token_ids(
    tokenizer,
    target_len: int,
    *,
    use_chat_wrapper: bool,
    add_bos: bool,
) -> list[int]:
    prefix_text = DEEPSEEK_USER_TOKEN if use_chat_wrapper else ""
    suffix_text = DEEPSEEK_ASSISTANT_TOKEN if use_chat_wrapper else ""
    prefix_ids = tokenizer.encode(prefix_text, add_special_tokens=False)
    suffix_ids = tokenizer.encode(suffix_text, add_special_tokens=False)
    if add_bos and tokenizer.bos_token_id is not None:
        prefix_ids = [tokenizer.bos_token_id] + prefix_ids

    body_len = target_len - len(prefix_ids) - len(suffix_ids)
    if body_len <= 0:
        raise ValueError("prompt token target is too small for the wrapper.")

    seed = (
        "DeepSeek sparse attention validation. "
        "This repeated sentence builds a deterministic long prefill prompt. "
    )
    body = seed
    body_ids = tokenizer.encode(body, add_special_tokens=False)
    while len(body_ids) < body_len:
        body += seed
        body_ids = tokenizer.encode(body, add_special_tokens=False)
    return prefix_ids + body_ids[:body_len] + suffix_ids


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default=os.environ.get(
            "VLLM_TRACE_MODEL",
            os.environ.get(
                "MODEL",
                "/home/models/Deepseek-V3.2-Pruned-95B-BF/",
            ),
        ),
    )
    parser.add_argument(
        "--tp-size",
        type=int,
        default=_env_int("VLLM_TRACE_TP_SIZE", _env_int("TP_SIZE", 4)),
    )
    parser.add_argument(
        "--prompt-tokens",
        type=int,
        default=_env_int("VLLM_TRACE_PROMPT_TOKENS", 2048),
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=_env_int("VLLM_TRACE_MAX_MODEL_LEN", 2176),
    )
    parser.add_argument(
        "--max-num-batched-tokens",
        type=int,
        default=_env_int("VLLM_TRACE_MAX_BATCHED_TOKENS", 3072),
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=_env_int("VLLM_TRACE_MAX_TOKENS", 1),
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=_env_float("VLLM_TRACE_TEMPERATURE", 0.0),
    )
    parser.add_argument(
        "--trace-max-calls",
        type=int,
        default=_env_int("SFA_TRACE_MAX_CALLS", 8),
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=_env_float("VLLM_TRACE_GPU_MEMORY_UTILIZATION", 0.95),
    )
    parser.add_argument(
        "--no-chat-wrapper",
        action="store_true",
    )
    parser.add_argument(
        "--no-bos",
        action="store_true",
    )
    parser.add_argument(
        "--no-expert-parallel",
        action="store_true",
    )
    parser.add_argument(
        "--dump-dir",
        default=os.environ.get("SFA_TRACE_DUMP_DIR", ""),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.environ["SFA_TRACE_ENABLE"] = "1"
    os.environ["SFA_TRACE_MAX_CALLS"] = str(args.trace_max_calls)
    os.environ.setdefault("SFA_TRACE_SYNC", "1")
    if args.dump_dir:
        os.environ["SFA_TRACE_DUMP_DIR"] = args.dump_dir
    _install_trace_hook_for_this_process()

    from vllm import LLM, SamplingParams

    tokenizer = _load_tokenizer(args.model)
    prompt_token_ids = _build_prompt_token_ids(
        tokenizer,
        args.prompt_tokens,
        use_chat_wrapper=not args.no_chat_wrapper,
        add_bos=not args.no_bos,
    )
    print(
        "SFA_TRACE prompt "
        f"requested={args.prompt_tokens} actual={len(prompt_token_ids)} "
        f"first_ids={prompt_token_ids[:16]}",
        flush=True,
    )
    print(
        "SFA_TRACE config "
        f"model={args.model} tp={args.tp_size} "
        f"max_model_len={args.max_model_len} "
        f"max_num_batched_tokens={args.max_num_batched_tokens} "
        f"max_tokens={args.max_tokens} "
        f"expert_parallel={not args.no_expert_parallel}",
        flush=True,
    )

    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tp_size,
        enable_expert_parallel=not args.no_expert_parallel,
        max_model_len=args.max_model_len,
        max_num_batched_tokens=args.max_num_batched_tokens,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=True,
        trust_remote_code=True,
    )
    outputs = llm.generate(
        [{"prompt_token_ids": prompt_token_ids}],
        SamplingParams(
            temperature=args.temperature,
            max_tokens=args.max_tokens,
        ),
        use_tqdm=False,
    )
    for output in outputs:
        print(
            "SFA_TRACE output "
            f"prompt_len={len(output.prompt_token_ids or [])} "
            f"text={output.outputs[0].text!r}",
            flush=True,
        )


if __name__ == "__main__":
    main()
