from nanovllm.engine.llm_engine import LLMEngine


class LLM(LLMEngine):
    def __init__(self, model, prefill_chunk_size: int = 0, **kwargs):
        super().__init__(model, prefill_chunk_size=prefill_chunk_size, **kwargs)
