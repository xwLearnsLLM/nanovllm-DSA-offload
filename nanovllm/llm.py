from nanovllm.engine.llm_engine import LLMEngine


class LLM(LLMEngine):
    def __init__(
        self,
        model,
        prefill_chunk_size: int = 0,
        offload_mode: str = "none",
        enable_lidu_fused_attention_scatter: bool = False,
        **kwargs,
    ):
        super().__init__(
            model,
            prefill_chunk_size=prefill_chunk_size,
            offload_mode=offload_mode,
            enable_lidu_fused_attention_scatter=(
                enable_lidu_fused_attention_scatter
            ),
            **kwargs,
        )
