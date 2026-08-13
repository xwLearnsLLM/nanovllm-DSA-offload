"""Run the longest sorted DuReader/LongBench requests through nano-vLLM."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from _example_utils import (
    GLM_ASSISTANT_TOKEN,
    GLM_USER_TOKEN,
    decode_step_limits,
    make_llm,
)
from nanovllm import SamplingParams


DATASET_PATH = Path(__file__).with_name("dureader.jsonl")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--prompt_count",
        type=int,
        default=1,
        help="Number of longest sorted DuReader requests to run.",
    )
    parser.add_argument(
        "--prompt_len",
        type=int,
        default=None,
        help=(
            "Repeat every tokenized DuReader prompt and truncate it from "
            "the front to exactly this many tokens. By default, preserve "
            "the original prompt lengths."
        ),
    )
    return parser.parse_args()


def load_requests(prompt_count: int) -> list[dict]:
    if prompt_count <= 0:
        raise ValueError("--prompt_count must be positive.")
    if not DATASET_PATH.is_file():
        raise FileNotFoundError(f"DuReader dataset is missing: {DATASET_PATH}")

    with DATASET_PATH.open(encoding="utf-8") as file:
        records = [json.loads(line) for line in file if line.strip()]
    if prompt_count > len(records):
        raise ValueError(
            f"--prompt_count={prompt_count} exceeds dataset size {len(records)}."
        )
    lengths = [int(record["length"]) for record in records]
    if any(left < right for left, right in zip(lengths, lengths[1:])):
        raise RuntimeError("dureader.jsonl must be sorted by length descending.")
    for record in records[:prompt_count]:
        if not isinstance(record.get("context"), str) or not isinstance(
            record.get("input"), str
        ):
            raise ValueError("Every DuReader record requires string context and input.")
    return records[:prompt_count]


def prompt_wrapper_ids(tokenizer) -> tuple[list[int], list[int]]:
    prefix_ids = tokenizer.encode(GLM_USER_TOKEN, add_special_tokens=False)
    suffix_ids = tokenizer.encode(GLM_ASSISTANT_TOKEN, add_special_tokens=False)
    if tokenizer.bos_token_id is not None:
        prefix_ids = [tokenizer.bos_token_id] + prefix_ids
    return prefix_ids, suffix_ids


def build_prompt_token_ids(tokenizer, record: dict) -> list[int]:
    body = (
        "请基于以下文章回答问题，只输出答案，不要解释。\n\n"
        f"文章：\n{record['context']}\n\n"
        f"问题：{record['input']}\n"
        "答案："
    )
    prefix_ids, suffix_ids = prompt_wrapper_ids(tokenizer)
    body_ids = tokenizer.encode(body, add_special_tokens=False)
    return prefix_ids + body_ids + suffix_ids


def resize_prompt_token_ids(
    token_ids: list[int],
    target_len: int,
) -> list[int]:
    if target_len <= 0:
        raise ValueError("--prompt_len must be positive.")
    if not token_ids:
        raise ValueError("Cannot resize an empty DuReader prompt.")
    repeats = (target_len + len(token_ids) - 1) // len(token_ids)
    return (token_ids * repeats)[-target_len:]


def main() -> None:
    args = parse_args()
    records = load_requests(args.prompt_count)
    max_steps, max_tokens = decode_step_limits(16)

    if args.prompt_len is not None and args.prompt_len <= 0:
        raise ValueError("--prompt_len must be positive.")

    # The input IDs are built after LLM initialization, so use the dataset
    # length as a safe initial model-length bound. The script verifies the
    # final tokenizer lengths immediately afterward.
    if args.prompt_len is None:
        max_model_len = (
            max(int(record["length"]) for record in records)
            + max_tokens
            + 128
        )
    else:
        max_model_len = args.prompt_len + max_tokens
    llm = make_llm(
        max_model_len=max_model_len,
        max_num_prefill_seqs_per_step=1,
        max_num_decode_seqs_per_step=len(records),
    )
    prompt_token_ids = [
        build_prompt_token_ids(llm.tokenizer, record) for record in records
    ]
    if args.prompt_len is not None:
        prompt_token_ids = [
            resize_prompt_token_ids(ids, args.prompt_len)
            for ids in prompt_token_ids
        ]
    actual_max_len = max(len(ids) for ids in prompt_token_ids) + max_tokens
    if actual_max_len > llm.config.max_model_len:
        raise RuntimeError(
            "Tokenized DuReader prompt exceeds max_model_len: "
            f"need={actual_max_len}, configured={llm.config.max_model_len}."
        )

    print(
        "dureader config: "
        f"prompt_count={len(records)}, "
        f"dataset_length_range={int(records[-1]['length'])}-"
        f"{int(records[0]['length'])}, "
        f"prompt_token_range={min(map(len, prompt_token_ids))}-"
        f"{max(map(len, prompt_token_ids))}, "
        f"prompt_len_override={args.prompt_len}, "
        f"max_model_len={llm.config.max_model_len}, "
        f"max_steps={max_steps}, "
        f"max_completion_tokens={max_tokens}"
    )

    outputs = llm.generate(
        prompt_token_ids,
        SamplingParams(
            temperature=0.0,
            max_tokens=max_tokens,
            max_steps=max_steps,
            ignore_eos=True,
        ),
    )
    for index, (record, ids, output) in enumerate(
        zip(records, prompt_token_ids, outputs),
        1,
    ):
        print(
            f"request={index} id={record['_id']} "
            f"dataset_length={int(record['length'])} prompt_len={len(ids)}"
        )
        print("response :", repr(output["text"]))
        print("token_ids:", output["token_ids"])


if __name__ == "__main__":
    main()
