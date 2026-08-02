"""Run the longest sorted DuReader/LongBench requests through nano-vLLM."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from _example_utils import (
    GLM_ASSISTANT_TOKEN,
    GLM_USER_TOKEN,
    env_int,
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


def main() -> None:
    args = parse_args()
    records = load_requests(args.prompt_count)
    max_gen_tokens = env_int("NANOVLLM_MAX_GEN_TOKENS", 16)

    # The input IDs are built after LLM initialization, so use the dataset
    # length as a safe initial model-length bound. The script verifies the
    # final tokenizer lengths immediately afterward.
    max_model_len = max(int(record["length"]) for record in records) + max_gen_tokens + 128
    llm = make_llm(
        max_model_len=max_model_len,
        max_num_prefill_seqs_per_step=1,
        max_num_decode_seqs_per_step=len(records),
    )
    prompt_token_ids = [
        build_prompt_token_ids(llm.tokenizer, record) for record in records
    ]
    actual_max_len = max(len(ids) for ids in prompt_token_ids) + max_gen_tokens
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
        f"max_model_len={llm.config.max_model_len}, "
        f"max_gen_tokens={max_gen_tokens}"
    )

    outputs = llm.generate(
        prompt_token_ids,
        SamplingParams(
            temperature=0.0,
            max_tokens=max_gen_tokens,
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
