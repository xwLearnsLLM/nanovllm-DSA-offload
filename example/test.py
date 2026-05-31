import os
import random
import ctypes
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
custom_so_path = (
PROJECT_ROOT
/ "nanovllm"
/ "_cann_ops_custom"
/ "vendors"
/ "nanovllm-ascend"
/ "op_api"
/ "lib"
/ "libopapi.so"
)
so_str = str(custom_so_path)
try:
    ctypes.CDLL(so_str, mode=ctypes.RTLD_GLOBAL)
except Exception as e:
    print("load custom op failed.")

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


def _prompt_wrapper_ids(tokenizer) -> tuple[list[int], list[int]]:
    use_chat = env_bool("NANOVLLM_USE_DEEPSEEK_CHAT", True)
    add_bos = env_bool("NANOVLLM_ADD_BOS", use_chat)
    prefix = DEEPSEEK_USER_TOKEN if use_chat else ""
    suffix = DEEPSEEK_ASSISTANT_TOKEN if use_chat else ""
    prefix_ids = tokenizer.encode(prefix, add_special_tokens=False)
    suffix_ids = tokenizer.encode(suffix, add_special_tokens=False)
    if add_bos and tokenizer.bos_token_id is not None:
        prefix_ids = [tokenizer.bos_token_id] + prefix_ids
    return prefix_ids, suffix_ids


def _meaningful_long_qa_text() -> tuple[str, str, str]:
    intro = """
Read the following long archive packet and answer the final question with one
short phrase.

Archive packet title: The Hawthorn Bridge Water Project.

The packet describes a valley repair project that rebuilt a temporary supply
crossing after a flood. The important distinction is that Willow Bridge was an
early footbridge that washed out, while Hawthorn Bridge was the later temporary
bridge used by supply carts during the signed final inspection.
"""
    evidence_block = """
Ledger section: The Silver Orchard water project began as a repair plan for
three villages. Mira Patel kept the repair ledger. Elena Ruiz supervised the
engineering notes. Early in the project, workers built Willow Bridge from pine
boards and rope. Willow Bridge was intended only for foot traffic. After three
days of heavy rain, Willow Bridge washed out and was marked with a red circle
on the hazard map.

Replacement section: The committee then built a second temporary bridge with
iron pins, ash beams, and a gravel approach. The second bridge was named
Hawthorn Bridge because hawthorn trees marked the crossing. After the flood,
every delivery receipt used the name Hawthorn Bridge. The bridge carried lime,
pump parts, sacks of oats, and the spare intake screen. Loaded carts crossed
one at a time. The archive index says Hawthorn Bridge was the active temporary
supply crossing during the final inspection.

Inspection section: The final inspection was held in the schoolhouse. The
committee asked whether the channel leaked, whether the pump could run for two
hours, and whether the temporary bridge was safe for supply carts. The signed
inspection notes say the temporary bridge was safe for carts only if carts
crossed one at a time. The bridge name in those signed inspection notes was
Hawthorn Bridge, not Willow Bridge. Mira Patel signed the notes, and Elena Ruiz
countersigned them.

Correction section: A later clerk wrote a confusing margin note mentioning
Willow Bridge beside the final inspection. The staff correction says the margin
note was copied from the first week of repairs and should not override the
signed inspection notes. The typed archive copy preserves the signed inspection
wording and removes the mistaken margin note. The search term Willow Bridge
points to the flood damage file. The search term Hawthorn Bridge points to the
final inspection file.

Audit section: HAWTHORN-ACTIVE-CROSSING, HAWTHORN-SUPPLY-MAP,
HAWTHORN-CART-ROUTE, and HAWTHORN-FINAL-NOTE all refer to the same temporary
bridge. WILLOW-DAMAGE-FILE, WILLOW-FLOOD-NOTE, and WILLOW-OLD-DRAFT refer to
the washed-out early bridge. The answer should use the bridge name from the
signed final inspection notes.
"""
    final_section = """
Final evidence summary:
1. Willow Bridge was the early footbridge and washed out before the final
inspection.
2. Hawthorn Bridge was the later temporary supply crossing.
3. The signed final inspection notes identify the temporary bridge as
Hawthorn Bridge.
4. The later Willow Bridge margin note is explicitly marked as a mistake.

Question: What was the name of the temporary bridge used during the final
inspection?

Answer with exactly the bridge name.
"""
    return intro, evidence_block, final_section


def build_base_meaningful_prompt(tokenizer, base_target_len: int = 10000) -> list[int]:
    if base_target_len <= 0:
        raise ValueError("base_target_len must be positive")

    prefix_ids, suffix_ids = _prompt_wrapper_ids(tokenizer)
    intro, evidence_block, final_section = _meaningful_long_qa_text()
    intro_ids = tokenizer.encode(intro, add_special_tokens=False)
    evidence_ids = tokenizer.encode(evidence_block, add_special_tokens=False)
    final_ids = tokenizer.encode(final_section, add_special_tokens=False)

    fixed_len = len(prefix_ids) + len(intro_ids) + len(final_ids) + len(suffix_ids)
    if base_target_len <= fixed_len:
        body_ids = intro_ids + final_ids
        return (prefix_ids + body_ids + suffix_ids)[-base_target_len:]

    evidence_budget = base_target_len - fixed_len
    repeats = (evidence_budget + len(evidence_ids) - 1) // len(evidence_ids)
    evidence = (evidence_ids * repeats)[:evidence_budget]
    return prefix_ids + intro_ids + evidence + final_ids + suffix_ids


def build_meaningful_prompt_from_base(base_ids: list[int], target_len: int) -> list[int]:
    if target_len <= 0:
        raise ValueError("target_len must be positive")
    if not base_ids:
        raise ValueError("base prompt must not be empty")
    repeats = (target_len + len(base_ids) - 1) // len(base_ids)
    return (base_ids * repeats)[-target_len:]


def build_exact_token_prompt(tokenizer, target_len: int, base_ids: list[int] | None = None) -> list[int]:
    if base_ids is None:
        base_ids = build_base_meaningful_prompt(
            tokenizer,
            env_int("NANOVLLM_MEANINGFUL_BASE_TOKENS", 10000),
        )
    return build_meaningful_prompt_from_base(base_ids, target_len)


def _decode_prompt_tail(tokenizer, token_ids: list[int], max_chars: int = 300) -> str:
    try:
        text = tokenizer.decode(token_ids[-512:], skip_special_tokens=False)
    except TypeError:
        text = tokenizer.decode(token_ids[-512:])
    text = " ".join(text.split())
    return text[-max_chars:]


def build_random_exact_token_prompt(tokenizer, target_len: int) -> list[int]:
    if target_len <= 0:
        raise ValueError("target_len must be positive")

    prefix_ids, suffix_ids = _prompt_wrapper_ids(tokenizer)
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

    max_num_prefill_seqs_per_step = env_int("NANOVLLM_MAX_PREFILL_SEQS_PER_STEP", 1)
    max_num_decode_seqs_per_step = env_int("NANOVLLM_MAX_DECODE_SEQS_PER_STEP", len(prompt_lengths))
    if max_num_prefill_seqs_per_step <= 0:
        raise ValueError("NANOVLLM_MAX_PREFILL_SEQS_PER_STEP must be > 0.")
    if max_num_decode_seqs_per_step <= 0:
        raise ValueError("NANOVLLM_MAX_DECODE_SEQS_PER_STEP must be > 0.")
    block_size = env_int("NANOVLLM_KVCACHE_BLOCK_SIZE", 128)

    llm = make_llm(
        max_model_len=max_model_len,
        max_num_prefill_seqs_per_step=max_num_prefill_seqs_per_step,
        max_num_decode_seqs_per_step=max_num_decode_seqs_per_step,
    )
    tokenizer = prompt_tokenizer(llm)

    prompt_style = os.environ.get("NANOVLLM_TEST_PROMPT_STYLE", "meaningful").strip().lower()
    if prompt_style == "random":
        base_ids = None
        prompt_token_ids = [build_random_exact_token_prompt(tokenizer, length) for length in prompt_lengths]
        prompts = [f"<random exact-token prompt {i}: target_len={length}>" for i, length in enumerate(prompt_lengths, 1)]
    elif prompt_style == "meaningful":
        base_ids = build_base_meaningful_prompt(tokenizer, env_int("NANOVLLM_MEANINGFUL_BASE_TOKENS", 10000))
        prompt_token_ids = [build_exact_token_prompt(tokenizer, length, base_ids) for length in prompt_lengths]
        prompts = [
            f"<meaningful long-QA suffix prompt {i}: target_len={length}, tail='{_decode_prompt_tail(tokenizer, ids)}'>"
            for i, (length, ids) in enumerate(zip(prompt_lengths, prompt_token_ids), 1)
        ]
    else:
        raise ValueError("NANOVLLM_TEST_PROMPT_STYLE must be 'meaningful' or 'random'.")

    print(
        "test config: "
        f"num_prompts={len(prompt_lengths)}, "
        f"prompt_min={min(prompt_lengths)}, "
        f"prompt_max={max(prompt_lengths)}, "
        f"max_model_len={max_model_len}, "
        f"max_num_prefill_seqs_per_step={max_num_prefill_seqs_per_step}, "
        f"max_num_decode_seqs_per_step={max_num_decode_seqs_per_step}, "
        f"max_gen_tokens={max_gen_tokens}, "
        f"prompt_style={prompt_style}, "
        f"meaningful_base_tokens={len(base_ids) if base_ids is not None else 0}"
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
