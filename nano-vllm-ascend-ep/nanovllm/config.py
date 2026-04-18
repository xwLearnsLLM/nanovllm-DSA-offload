import os
from dataclasses import dataclass
from enum import Enum

from transformers import AutoConfig

from nanovllm.models.deepseek_v32 import DeepseekV32Config


class GraphMode(Enum):
    EAGER = "eager"
    MAX_AUTOTUNE = "max-autotune"  # Ascend IR 图模式
    REDUCE_OVERHEAD = "reduce-overhead"  # aclgraph 模式


@dataclass
class Config:
    model: str
    max_num_batched_tokens: int = 16384
    max_num_seqs: int = 256
    max_model_len: int = 4096
    gpu_memory_utilization: float = 0.7
    tensor_parallel_size: int = 1
    enable_expert_parallel: bool = False
    enforce_eager: bool = False
    hf_config: AutoConfig | None = None
    eos: int = -1
    kvcache_block_size: int = 256
    num_kvcache_blocks: int = -1
    use_graph_cache: bool = False
    hccl_port: int = 28000
    graph_mode: str = GraphMode.MAX_AUTOTUNE.value
    is_multimodal: bool = False
    skip_warmup: bool = False
    device = "npu"
    trust_remote_code: bool = False

    def __post_init__(self):
        assert os.path.isdir(self.model)
        assert self.kvcache_block_size % 16 == 0
        assert 1 <= self.tensor_parallel_size <= 8
        self.hf_config = self._load_hf_config()
        setattr(
            self.hf_config,
            "nanovllm_enable_expert_parallel",
            bool(self.enable_expert_parallel),
        )
        self._validate_model_format()
        if getattr(self.hf_config, "model_type", None) == "deepseek_v32":
            self.enforce_eager = True
        #self.max_model_len = min(self.max_model_len, self.hf_config.max_position_embeddings)
        # Multimodal models (e.g. Qwen3-VL) store the text settings in
        # hf_config.text_config.
        text_config = getattr(self.hf_config, "text_config", self.hf_config)

        max_position_embeddings = getattr(
            text_config,
            "max_position_embeddings",
            None,
        )
        if max_position_embeddings is not None:
            self.max_model_len = min(
                self.max_model_len,
                max_position_embeddings,
            )

        # eos may be defined within the text config
        eos_token_id = getattr(text_config, "eos_token_id", None)
        if eos_token_id is not None:
            self.eos = eos_token_id

        assert self.max_num_batched_tokens >= self.max_model_len

    def _validate_model_format(self):
        if getattr(self.hf_config, "model_type", None) != "deepseek_v32":
            return
        quantization_config = getattr(
            self.hf_config,
            "quantization_config",
            None,
        ) or {}
        quant_method = str(
            quantization_config.get("quant_method", "")
        ).lower()
        if quant_method != "fp8":
            return
        raise ValueError(
            "nano-vllm-ascend currently expects the BF16-exported "
            "DeepSeek-V3.2 checkpoint, not the original HF FP8 directory. "
            "Convert it first with "
            "`python scripts/export_deepseek_v32_to_hf_bf16.py "
            "<source_model> <output_model>` and then point `model=` to the "
            "BF16 output directory."
        )

    def _load_hf_config(self):
        try:
            return AutoConfig.from_pretrained(
                self.model,
                trust_remote_code=self.trust_remote_code,
            )
        except ValueError:
            config = DeepseekV32Config.from_pretrained(self.model)
            if getattr(config, "model_type", None) != "deepseek_v32":
                raise
            return config

    def __repr__(self):
        attrs = {k: v for k, v in self.__dict__.items() if k != 'hf_config'}
        attrs['hf_config'] = f"{self.hf_config.__class__.__name__}(...)"
        items = [f"{k}={v}" for k, v in attrs.items()]
        return f"Config({', '.join(items)})"
