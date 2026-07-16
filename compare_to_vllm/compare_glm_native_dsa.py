"""Compare vLLM-Ascend GLM native DSA on nano-vLLM's exact long prompt."""

from __future__ import annotations

import hashlib
import os
import struct


DEFAULT_MODEL = "/mnt/models/GLM-5.1-w4a8/"
DEEPSEEK_USER_TOKEN = "<\uFF5CUser\uFF5C>"
DEEPSEEK_ASSISTANT_TOKEN = "<\uFF5CAssistant\uFF5C>"


def env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    value = value.strip().lower()
    if value in ("1", "true", "yes", "on"):
        return True
    if value in ("0", "false", "no", "off"):
        return False
    raise ValueError(f"{name} must be a boolean, got {value!r}")


def env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {value!r}") from exc


def env_float(name: str, default: float) -> float:
    value = os.environ.get(name)
    return default if value is None else float(value)


def _prompt_wrapper_ids(tokenizer) -> tuple[list[int], list[int]]:
    # Keep this byte-for-byte equivalent to example/test.py.  In particular,
    # this intentionally uses the same explicit wrapper instead of GLM's chat
    # template so the two runtimes receive identical token IDs.
    prefix_ids = tokenizer.encode(DEEPSEEK_USER_TOKEN, add_special_tokens=False)
    suffix_ids = tokenizer.encode(
        DEEPSEEK_ASSISTANT_TOKEN,
        add_special_tokens=False,
    )
    if tokenizer.bos_token_id is not None:
        prefix_ids = [tokenizer.bos_token_id] + prefix_ids
    return prefix_ids, suffix_ids


def _meaningful_long_qa_text() -> tuple[str, str, str]:
    # This text and its whitespace deliberately match example/test.py.
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


def build_base_meaningful_prompt(
    tokenizer,
    base_target_len: int = 10000,
) -> list[int]:
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


def build_exact_token_prompt(base_ids: list[int], target_len: int) -> list[int]:
    if target_len <= 0:
        raise ValueError("target_len must be positive")
    if not base_ids:
        raise ValueError("base prompt must not be empty")
    repeats = (target_len + len(base_ids) - 1) // len(base_ids)
    return (base_ids * repeats)[-target_len:]


def token_fingerprint(token_ids: list[int]) -> str:
    payload = b"".join(struct.pack("<q", token_id) for token_id in token_ids)
    return hashlib.sha256(payload).hexdigest()


def _read_hf_config(llm):
    vllm_config = llm.llm_engine.vllm_config
    model_config = vllm_config.model_config
    hf_config = getattr(model_config, "hf_text_config", None)
    return hf_config if hf_config is not None else model_config.hf_config


def main() -> None:
    from vllm import LLM, SamplingParams, TokensPrompt

    model = os.environ.get("VLLM_MODEL", DEFAULT_MODEL)
    tp_size = env_int("VLLM_TP_SIZE", 16)
    prompt_len = env_int("VLLM_PROMPT_LENGTH", 8200)
    max_tokens = env_int("VLLM_MAX_GEN_TOKENS", 2)
    max_num_batched_tokens = env_int("VLLM_MAX_NUM_BATCHED_TOKENS", 1024)
    enforce_eager = env_bool("VLLM_ENFORCE_EAGER", True)

    if prompt_len <= 2048:
        raise ValueError(
            "VLLM_PROMPT_LENGTH must be greater than 2048 to exercise GLM DSA"
        )
    if max_tokens < 2:
        raise ValueError(
            "VLLM_MAX_GEN_TOKENS must be at least 2 to observe the first "
            "decode result"
        )
    if max_num_batched_tokens <= 0:
        raise ValueError("VLLM_MAX_NUM_BATCHED_TOKENS must be positive")

    print(
        "vLLM GLM native-DSA comparison: "
        f"model={model}, tp={tp_size}, prompt_len={prompt_len}, "
        f"max_tokens={max_tokens}, max_model_len={prompt_len + max_tokens}, "
        f"max_num_batched_tokens={max_num_batched_tokens}, "
        f"enforce_eager={enforce_eager}"
    )

    llm = LLM(
        model=model,
        tensor_parallel_size=tp_size,
        enable_expert_parallel=env_bool("VLLM_ENABLE_EXPERT_PARALLEL", True),
        max_model_len=prompt_len + max_tokens,
        max_num_batched_tokens=max_num_batched_tokens,
        max_num_seqs=1,
        gpu_memory_utilization=env_float("VLLM_GPU_MEMORY_UTILIZATION", 0.95),
        block_size=env_int("VLLM_KVCACHE_BLOCK_SIZE", 128),
        enable_prefix_caching=False,
        enable_chunked_prefill=True,
        trust_remote_code=True,
        enforce_eager=enforce_eager,
    )

    hf_config = _read_hf_config(llm)
    model_type = getattr(hf_config, "model_type", None)
    index_topk = getattr(hf_config, "index_topk", None)
    print(
        "vLLM GLM DSA config: "
        f"model_type={model_type}, index_topk={index_topk}, "
        f"index_n_heads={getattr(hf_config, 'index_n_heads', None)}, "
        f"index_head_dim={getattr(hf_config, 'index_head_dim', None)}, "
        "indexer_rope_interleave="
        f"{getattr(hf_config, 'indexer_rope_interleave', None)}"
    )
    if model_type != "glm_moe_dsa" or index_topk != 2048:
        raise RuntimeError(
            "This is not the expected GLM native-DSA configuration: "
            f"model_type={model_type!r}, index_topk={index_topk!r}"
        )

    tokenizer = llm.get_tokenizer()
    base_ids = build_base_meaningful_prompt(tokenizer)
    prompt_token_ids = build_exact_token_prompt(base_ids, prompt_len)
    print(
        "exact prompt: "
        f"base_len={len(base_ids)}, token_len={len(prompt_token_ids)}, "
        f"first_ids={prompt_token_ids[:16]}, "
        f"last_ids={prompt_token_ids[-16:]}, "
        f"sha256_i64le={token_fingerprint(prompt_token_ids)}"
    )

    outputs = llm.generate(
        [TokensPrompt(prompt_token_ids=prompt_token_ids)],
        SamplingParams(
            temperature=0.0,
            max_tokens=max_tokens,
            min_tokens=max_tokens,
            ignore_eos=True,
        ),
        use_tqdm=False,
    )
    completion = outputs[0].outputs[0]
    output_ids = list(completion.token_ids)
    print("response  :", repr(completion.text))
    print("token_ids :", output_ids)

    if output_ids[:2] == [39, 672]:
        print("VLLM_GLM_DSA_RESULT=matches_dense_reference_[39,672]")
    elif output_ids[:2] == [39, 0]:
        print("VLLM_GLM_DSA_RESULT=matches_nanovllm_failure_[39,0]")
    else:
        print(f"VLLM_GLM_DSA_RESULT=other_{output_ids[:2]}")


if __name__ == "__main__":
    main()
