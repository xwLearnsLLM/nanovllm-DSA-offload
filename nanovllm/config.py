import json
import os
from dataclasses import dataclass

from nanovllm.models.deepseek_v32 import DeepseekV32Config


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
    hf_config: DeepseekV32Config | None = None
    eos: int = -1
    kvcache_block_size: int = 256
    num_kvcache_blocks: int = -1
    hccl_port: int = 28000
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
        self.enforce_eager = True

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

        eos_token_id = getattr(text_config, "eos_token_id", None)
        if eos_token_id is not None:
            self.eos = eos_token_id

        assert self.max_num_batched_tokens >= self.max_model_len

    def _validate_model_format(self):
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
        config_path = os.path.join(self.model, "config.json")
        raw_config = {}
        if os.path.isfile(config_path):
            with open(config_path, "r", encoding="utf-8") as file:
                raw_config = json.load(file)

        deepseek_v32_like = (
            raw_config.get("model_type") == "deepseek_v32"
            or "DeepseekV32ForCausalLM"
            in (raw_config.get("architectures") or [])
            or all(
                key in raw_config
                for key in (
                    "first_k_dense_replace",
                    "q_lora_rank",
                    "kv_lora_rank",
                    "index_topk",
                )
            )
        )
        if deepseek_v32_like:
            return DeepseekV32Config.from_pretrained(self.model)

        raise ValueError(
            "nano-vllm-ascend only supports DeepSeek-V3.2 style model "
            "directories. Expected config.json model_type='deepseek_v32', "
            "DeepseekV32ForCausalLM architecture, or DeepSeek V3.2 fields "
            "such as first_k_dense_replace/q_lora_rank/kv_lora_rank/index_topk."
        )

    def __repr__(self):
        attrs = {k: v for k, v in self.__dict__.items() if k != "hf_config"}
        attrs["hf_config"] = f"{self.hf_config.__class__.__name__}(...)"
        items = [f"{k}={v}" for k, v in attrs.items()]
        return f"Config({', '.join(items)})"
