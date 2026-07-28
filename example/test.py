import os

from _example_utils import (
    GLM_ASSISTANT_TOKEN,
    GLM_USER_TOKEN,
    env_int,
    make_llm,
    print_outputs,
)
from nanovllm.engine.dsa_offload import (
    LIDU_OFFLOAD_MODES,
    lidu_cache_tokens,
)
from nanovllm import SamplingParams


def parse_prompt_lengths() -> list[int]:
    value = os.environ.get("NANOVLLM_PROMPT_LENGTHS", "8200,8201")
    lengths = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not lengths or any(length <= 0 for length in lengths):
        raise ValueError("NANOVLLM_PROMPT_LENGTHS must contain positive integers.")
    return lengths


def _prompt_wrapper_ids(tokenizer) -> tuple[list[int], list[int]]:
    prefix_ids = tokenizer.encode(GLM_USER_TOKEN, add_special_tokens=False)
    suffix_ids = tokenizer.encode(GLM_ASSISTANT_TOKEN, add_special_tokens=False)
    if tokenizer.bos_token_id is not None:
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
        base_ids = build_base_meaningful_prompt(tokenizer)
    return build_meaningful_prompt_from_base(base_ids, target_len)


def _decode_prompt_tail(tokenizer, token_ids: list[int], max_chars: int = 300) -> str:
    try:
        text = tokenizer.decode(token_ids[-512:], skip_special_tokens=False)
    except TypeError:
        text = tokenizer.decode(token_ids[-512:])
    text = " ".join(text.split())
    return text[-max_chars:]


def print_prompt_plan(
    lengths: list[int],
    block_size: int,
    offload_mode: str,
) -> None:
    print("prompt plan:")
    for i, length in enumerate(lengths, 1):
        full_blocks = length // block_size
        if offload_mode in LIDU_OFFLOAD_MODES:
            cache_tokens = lidu_cache_tokens(length)
            cache_blocks = (
                cache_tokens // block_size
                if cache_tokens
                else full_blocks
            )
            delays_cache_arena = 0 < cache_blocks < full_blocks
            detail = (
                f"lidu_cache_tokens={cache_tokens}, "
                f"final_prefill_release_blocks="
                f"{full_blocks if delays_cache_arena else 0}, "
                f"first_decode_allocate_blocks="
                f"{cache_blocks if delays_cache_arena else 0}"
            )
        else:
            detail = f"tail_tokens={length % block_size}"
        print(
            f"  prompt {i}: target_len={length}, "
            f"full_blocks={full_blocks}, {detail}"
        )


def main() -> None:
    prompt_lengths = parse_prompt_lengths()
    max_prompt_len = max(prompt_lengths)
    max_gen_tokens = env_int("NANOVLLM_MAX_GEN_TOKENS", 16)
    max_model_len = max_prompt_len + max_gen_tokens
    max_num_prefill_seqs_per_step = 1
    max_num_decode_seqs_per_step = len(prompt_lengths)
    block_size = env_int("NANOVLLM_KVCACHE_BLOCK_SIZE", 128)

    llm = make_llm(
        max_model_len=max_model_len,
        max_num_prefill_seqs_per_step=max_num_prefill_seqs_per_step,
        max_num_decode_seqs_per_step=max_num_decode_seqs_per_step,
    )
    tokenizer = llm.tokenizer

    base_ids = build_base_meaningful_prompt(tokenizer)
    prompt_token_ids = [
        build_exact_token_prompt(tokenizer, length, base_ids)
        for length in prompt_lengths
    ]
    prompts = [
        f"<meaningful long-QA suffix prompt {i}: target_len={length}, "
        f"tail='{_decode_prompt_tail(tokenizer, ids)}'>"
        for i, (length, ids) in enumerate(zip(prompt_lengths, prompt_token_ids), 1)
    ]

    print(
        "test config: "
        f"num_prompts={len(prompt_lengths)}, "
        f"prompt_min={min(prompt_lengths)}, "
        f"prompt_max={max(prompt_lengths)}, "
        f"max_model_len={max_model_len}, "
        f"max_num_prefill_seqs_per_step={max_num_prefill_seqs_per_step}, "
        f"prefill_chunk_size={env_int('NANOVLLM_PREFILL_CHUNK_SIZE', 0)}, "
        f"max_num_decode_seqs_per_step={max_num_decode_seqs_per_step}, "
        f"offload_mode={llm.config.offload_mode}, "
        f"max_gen_tokens={max_gen_tokens}, "
        f"meaningful_base_tokens={len(base_ids)}"
    )
    print_prompt_plan(
        [len(ids) for ids in prompt_token_ids],
        block_size,
        llm.config.offload_mode,
    )
    for i, ids in enumerate(prompt_token_ids, 1):
        print(f"prompt {i} token_len={len(ids)} first_ids={ids[:16]}")

    outputs = llm.generate(
        prompt_token_ids,
        SamplingParams(
            temperature=0.0,
            max_tokens=max_gen_tokens,
            ignore_eos=True,
        ),
    )
    print_outputs(prompts, prompt_token_ids, outputs)


if __name__ == "__main__":
    main()
