import json
import os
from dataclasses import dataclass

from nanovllm.engine.dsa_offload import DSA_SELECTION_TOPK_TOKENS
from nanovllm.engine.full_decode_graph import (
    FULL_DECODE_ONLY,
    normalize_capture_sizes,
)
from nanovllm.models.deepseek_v32 import DeepseekV32Config


@dataclass
class Config:
    model: str
    max_num_prefill_seqs_per_step: int = 1
    max_num_decode_seqs_per_step: int = 256
    max_model_len: int = 65536
    tensor_parallel_size: int = 1
    enable_expert_parallel: bool = False
    enforce_eager: bool = False
    decode_graph_mode: str = "none"
    decode_graph_capture_sizes: tuple[int, ...] | list[int] | None = None
    decode_graph_warmup_iters: int = 1
    hf_config: DeepseekV32Config | None = None
    eos: int = -1
    kvcache_block_size: int = 256
    num_kvcache_blocks: int = -1
    num_index_cache_blocks: int = -1
    num_hbm_kvcache_blocks: int = -1
    num_dram_kvcache_blocks: int = -1
    dsa_offload_max_sparse_tokens: int = -1
    hccl_port: int = 28000
    device = "npu"
    trust_remote_code: bool = False

    def __post_init__(self):
        assert os.path.isdir(self.model)
        assert self.kvcache_block_size % 16 == 0
        assert 1 <= self.tensor_parallel_size
        self.max_num_prefill_seqs_per_step = self._env_int(
            "NANOVLLM_MAX_PREFILL_SEQS_PER_STEP",
            self.max_num_prefill_seqs_per_step,
        )
        self.max_num_decode_seqs_per_step = self._env_int(
            "NANOVLLM_MAX_DECODE_SEQS_PER_STEP",
            self.max_num_decode_seqs_per_step,
        )
        self.max_model_len = self._env_int(
            "NANOVLLM_MAX_MODEL_LEN",
            self.max_model_len,
        )
        self.num_hbm_kvcache_blocks = self._env_int(
            "NANOVLLM_HBM_NUM_BLOCKS",
            self.num_hbm_kvcache_blocks,
        )
        self.num_dram_kvcache_blocks = self._env_int(
            "NANOVLLM_DRAM_NUM_BLOCKS",
            self.num_dram_kvcache_blocks,
        )
        if self.num_hbm_kvcache_blocks <= 2:
            raise ValueError(
                "NANOVLLM_HBM_NUM_BLOCKS must be set to a value > 2. "
                "It directly controls HBM KV cache block count."
            )
        if self.num_dram_kvcache_blocks <= 2:
            raise ValueError(
                "NANOVLLM_DRAM_NUM_BLOCKS must be set to a value > 2. "
                "It directly controls both DRAM KV cache and IndexCache block counts."
            )
        if self.max_num_prefill_seqs_per_step <= 0:
            raise ValueError("NANOVLLM_MAX_PREFILL_SEQS_PER_STEP must be > 0.")
        if self.max_num_decode_seqs_per_step <= 0:
            raise ValueError("NANOVLLM_MAX_DECODE_SEQS_PER_STEP must be > 0.")
        self.num_kvcache_blocks = self.num_hbm_kvcache_blocks
        self.num_index_cache_blocks = self.num_dram_kvcache_blocks
        self.dsa_offload_pool_capacity = self.max_num_decode_seqs_per_step
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
    def _env_int(name: str, default: int) -> int:
        value = os.environ.get(name)
        if value is None:
            return int(default)
        try:
            return int(value)
        except ValueError:
            raise ValueError(f"{name} must be an integer, got {value!r}.")

    @staticmethod
    def _env_bool(name: str, default: bool) -> bool:
        value = os.environ.get(name)
        if value is None:
            return bool(default)
        normalized = value.strip().lower()
        if normalized in ("1", "true", "yes", "on"):
            return True
        if normalized in ("0", "false", "no", "off"):
            return False
        raise ValueError(f"{name} must be a boolean, got {value!r}.")

    @staticmethod
    def _parse_capture_sizes(value: str) -> tuple[int, ...]:
        try:
            return tuple(
                int(item.strip())
                for item in value.split(",")
                if item.strip()
            )
        except ValueError as exc:
            raise ValueError(
                "NANOVLLM_DECODE_GRAPH_CAPTURE_SIZES must be a comma-separated "
                f"integer list, got {value!r}."
            ) from exc

    def _configure_decode_graph(self) -> None:
        requested_enforce_eager = bool(self.enforce_eager)
        mode_value = os.environ.get(
            "NANOVLLM_DECODE_GRAPH_MODE",
            self.decode_graph_mode,
        )
        mode = str(mode_value).strip().lower()
        if mode in ("", "none", "eager"):
            mode = "none"
        elif mode in ("full_decode_only", "full-decode-only"):
            mode = FULL_DECODE_ONLY
        else:
            raise ValueError(
                "NANOVLLM_DECODE_GRAPH_MODE only supports 'none' and "
                f"'full_decode_only', got {mode!r}."
            )

        self.decode_graph_mode = mode
        self.enforce_eager = mode == "none"
        if mode == "none":
            self.decode_graph_capture_sizes = ()
            return

        if requested_enforce_eager:
            raise ValueError(
                "enforce_eager=True conflicts with "
                "NANOVLLM_DECODE_GRAPH_MODE=full_decode_only."
            )
        if os.environ.get("ASCEND_LAUNCH_BLOCKING") == "1":
            raise ValueError(
                "ASCEND_LAUNCH_BLOCKING=1 is incompatible with FULL_DECODE_ONLY."
            )
        if not self._env_bool("NANOVLLM_ENABLE_DECODE_MLAPO", True):
            raise ValueError(
                "DSA FULL_DECODE_ONLY requires NANOVLLM_ENABLE_DECODE_MLAPO=1."
            )
        if self._env_bool("NANOVLLM_LOG_DECODE_LAYER_TIMING", False):
            raise ValueError(
                "NANOVLLM_LOG_DECODE_LAYER_TIMING must be 0 during graph "
                "capture. Use torch_npu.profiler to profile graph replay."
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

        sizes_env = os.environ.get("NANOVLLM_DECODE_GRAPH_CAPTURE_SIZES")
        sizes = (
            self._parse_capture_sizes(sizes_env)
            if sizes_env is not None
            else tuple(self.decode_graph_capture_sizes or ())
        )
        self.decode_graph_capture_sizes = normalize_capture_sizes(
            sizes,
            self.max_num_decode_seqs_per_step,
        )
        self.decode_graph_warmup_iters = self._env_int(
            "NANOVLLM_DECODE_GRAPH_WARMUP_ITERS",
            self.decode_graph_warmup_iters,
        )
        if self.decode_graph_warmup_iters < 1:
            raise ValueError("NANOVLLM_DECODE_GRAPH_WARMUP_ITERS must be >= 1.")

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
