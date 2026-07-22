import os

from nanovllm import LLM


DEFAULT_MODEL_PATH = "/mnt/models/GLM-5.1-w4a8/"
DEFAULT_TP_SIZE = 16
GLM_USER_TOKEN = "<\uFF5CUser\uFF5C>"
GLM_ASSISTANT_TOKEN = "<\uFF5CAssistant\uFF5C>"


def env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    value = value.strip().lower()
    if value in ("1", "true", "yes", "on"):
        return True
    if value in ("0", "false", "no", "off"):
        return False
    raise ValueError(f"{name} must be a boolean, got {value!r}.")


def env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {value!r}.") from exc


def model_path() -> str:
    return os.environ.get("NANOVLLM_MODEL", DEFAULT_MODEL_PATH)


def make_llm(
    *,
    max_model_len: int,
    max_num_prefill_seqs_per_step: int,
    max_num_decode_seqs_per_step: int,
    enforce_eager: bool | None = None,
) -> LLM:
    if enforce_eager is None:
        enforce_eager = env_bool("NANOVLLM_ENFORCE_EAGER", False)
    return LLM(
        model_path(),
        offload_mode=os.environ.get("NANOVLLM_OFFLOAD_MODE", "none"),
        enforce_eager=enforce_eager,
        decode_graph_capture_sizes=(max_num_decode_seqs_per_step,),
        tensor_parallel_size=env_int("NANOVLLM_TP_SIZE", DEFAULT_TP_SIZE),
        enable_expert_parallel=env_bool(
            "NANOVLLM_ENABLE_EXPERT_PARALLEL", True
        ),
        max_model_len=max_model_len,
        max_num_prefill_seqs_per_step=max_num_prefill_seqs_per_step,
        prefill_chunk_size=env_int("NANOVLLM_PREFILL_CHUNK_SIZE", 0),
        num_speculative_tokens=env_int(
            "NANOVLLM_NUM_SPECULATIVE_TOKENS", 0
        ),
        max_num_decode_seqs_per_step=max_num_decode_seqs_per_step,
        kvcache_block_size=env_int("NANOVLLM_KVCACHE_BLOCK_SIZE", 128),
        num_hbm_kvcache_blocks=env_int("NANOVLLM_HBM_NUM_BLOCKS", -1),
        num_dram_kvcache_blocks=env_int("NANOVLLM_DRAM_NUM_BLOCKS", -1),
        trust_remote_code=True,
    )


def print_outputs(
    prompts: list[str],
    prompt_token_ids: list[list[int]],
    outputs,
) -> None:
    for prompt, ids, output in zip(prompts, prompt_token_ids, outputs):
        one_line_prompt = " ".join(prompt.split())
        print("prompt_len:", len(ids))
        print("prompt    :", one_line_prompt[:300])
        print("response  :", repr(output["text"]))
        print("token_ids :", output["token_ids"])
        print()
