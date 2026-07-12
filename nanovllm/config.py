from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any

from nanovllm.engine.dsa_offload import DSA_SELECTION_TOPK_TOKENS
from nanovllm.engine.full_decode_graph import normalize_capture_sizes


@dataclass
class Config:
    model: str
    max_num_prefill_seqs_per_step: int = 1
    max_num_decode_seqs_per_step: int = 256
    max_model_len: int = 65536
    tensor_parallel_size: int = 1
    enable_expert_parallel: bool = False
    enforce_eager: bool = False
    decode_graph_capture_sizes: tuple[int, ...] | list[int] | None = None
    hf_config: Any = field(init=False)
    eos: int = field(init=False, default=-1)
    kvcache_block_size: int = 256
    num_hbm_kvcache_blocks: int = -1
    num_dram_kvcache_blocks: int = -1
    hccl_port: int = 28000
    device = "npu"
    trust_remote_code: bool = False
    prefill_chunk_size: int = 0

    def __post_init__(self):
        assert os.path.isdir(self.model)
        assert self.kvcache_block_size % 16 == 0
        assert 1 <= self.tensor_parallel_size
        if self.num_hbm_kvcache_blocks <= 2:
            raise ValueError(
                "num_hbm_kvcache_blocks must be > 2. The example scripts "
                "read it from NANOVLLM_HBM_NUM_BLOCKS."
            )
        if self.num_dram_kvcache_blocks <= 2:
            raise ValueError(
                "num_dram_kvcache_blocks must be > 2. The example scripts "
                "read it from NANOVLLM_DRAM_NUM_BLOCKS."
            )
        if self.max_num_prefill_seqs_per_step <= 0:
            raise ValueError("max_num_prefill_seqs_per_step must be > 0.")
        self._validate_prefill_chunking(
            self.prefill_chunk_size,
            self.max_num_prefill_seqs_per_step,
        )
        if self.max_num_decode_seqs_per_step <= 0:
            raise ValueError("max_num_decode_seqs_per_step must be > 0.")
        if self.max_model_len <= 0:
            raise ValueError("max_model_len must be > 0.")
        self.hf_config = self._load_hf_config()
        setattr(
            self.hf_config,
            "nanovllm_enable_expert_parallel",
            bool(self.enable_expert_parallel),
        )
        self._validate_model_format()

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
        self._configure_decode_graph()

    @staticmethod
    def _validate_prefill_chunking(
        prefill_chunk_size: int,
        max_num_prefill_seqs_per_step: int,
    ) -> None:
        if type(prefill_chunk_size) is not int:
            raise TypeError("prefill_chunk_size must be an int.")
        if prefill_chunk_size not in (0, 1024):
            raise ValueError("prefill_chunk_size must be either 0 or 1024.")
        if prefill_chunk_size and max_num_prefill_seqs_per_step != 1:
            raise ValueError(
                "Chunk prefill requires max_num_prefill_seqs_per_step=1."
            )

    def _configure_decode_graph(self) -> None:
        # There are exactly two execution modes. Prefill and first decode are
        # always eager; enforce_eager controls steady-state decode.
        if not isinstance(self.enforce_eager, bool):
            raise TypeError("enforce_eager must be a bool.")
        if self.enforce_eager:
            self.decode_graph_capture_sizes = ()
            return
        if os.environ.get("ASCEND_LAUNCH_BLOCKING") == "1":
            raise ValueError(
                "ASCEND_LAUNCH_BLOCKING=1 is incompatible with FULL_DECODE_ONLY."
            )
        if self.max_model_len < DSA_SELECTION_TOPK_TOKENS:
            raise ValueError(
                "DSA FULL_DECODE_ONLY requires max_model_len >= "
                f"{DSA_SELECTION_TOPK_TOKENS}, got {self.max_model_len}."
            )
        if DSA_SELECTION_TOPK_TOKENS % self.kvcache_block_size != 0:
            raise ValueError(
                "DSA FULL_DECODE_ONLY requires the 2048-token sparse budget "
                "to be divisible by kvcache_block_size, got "
                f"{self.kvcache_block_size}."
            )

        self.decode_graph_capture_sizes = normalize_capture_sizes(
            self.decode_graph_capture_sizes
            or (self.max_num_decode_seqs_per_step,)
        )
        if self.decode_graph_capture_sizes[-1] > self.max_num_decode_seqs_per_step:
            raise ValueError(
                "decode_graph_capture_sizes must not exceed "
                "max_num_decode_seqs_per_step: "
                f"sizes={self.decode_graph_capture_sizes}, "
                f"max={self.max_num_decode_seqs_per_step}."
            )

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
        from nanovllm.models.deepseek_v32 import DeepseekV32Config

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
