from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any

from nanovllm.engine.dsa_offload import (
    DSA_SELECTION_TOPK_TOKENS,
    LIDU_MAX_SOURCE_TOKENS,
    LIDU_OFFLOAD_MODES,
    OFFLOAD_FUSE,
    OFFLOAD_NONE,
    OFFLOAD_SPLIT,
    IndexShareGroupManager,
    normalize_offload_mode,
    validate_lidu_cache_token_budgets,
)
from nanovllm.engine.full_decode_graph import normalize_capture_sizes

GLM_VERSION_51 = "5.1"
GLM_VERSION_52 = "5.2"


def glm52_indexer_types(num_hidden_layers: int) -> tuple[str, ...]:
    """Return the official GLM-5.2 target-layer IndexShare schedule."""

    return tuple(
        "full"
        if layer_idx < 3 or (layer_idx >= 6 and (layer_idx - 6) % 4 == 0)
        else "shared"
        for layer_idx in range(num_hidden_layers)
    )


def normalize_eos_token_ids(value: Any) -> tuple[int, ...]:
    if value is None:
        return ()
    if isinstance(value, int):
        return (int(value),)
    if isinstance(value, (list, tuple, set, frozenset)):
        return tuple(dict.fromkeys(int(token_id) for token_id in value))
    raise TypeError(
        "eos_token_id must be an int or a sequence of ints, got "
        f"{type(value).__name__}."
    )


def merge_eos_token_ids(*values: Any) -> tuple[int, ...]:
    merged = tuple(
        dict.fromkeys(
            token_id
            for value in values
            for token_id in normalize_eos_token_ids(value)
        )
    )
    if len(merged) > 1 and -1 in merged:
        merged = tuple(token_id for token_id in merged if token_id != -1)
    return merged or (-1,)


@dataclass
class Config:
    model: str
    max_num_prefill_seqs_per_step: int = 1
    max_num_decode_seqs_per_step: int = 256
    max_model_len: int = 65536
    tensor_parallel_size: int = 1
    enable_expert_parallel: bool = False
    offload_mode: str = OFFLOAD_NONE
    enforce_eager: bool = False
    decode_graph_capture_sizes: tuple[int, ...] | list[int] | None = None
    hf_config: Any = field(init=False)
    glm_version: str = field(init=False)
    glm_model_name: str = field(init=False)
    eos: tuple[int, ...] = field(init=False, default=(-1,))
    kvcache_block_size: int = 256
    num_hbm_kvcache_blocks: int = -1
    num_dram_kvcache_blocks: int = -1
    hccl_port: int = 28000
    device = "npu"
    trust_remote_code: bool = False
    prefill_chunk_size: int = 0
    num_speculative_tokens: int = 0

    def __post_init__(self):
        assert os.path.isdir(self.model)
        assert self.kvcache_block_size % 16 == 0
        assert 1 <= self.tensor_parallel_size
        self.offload_mode = normalize_offload_mode(self.offload_mode)
        if self.num_hbm_kvcache_blocks <= 2:
            raise ValueError(
                "num_hbm_kvcache_blocks must be > 2. The example scripts "
                "read it from NANOVLLM_HBM_NUM_BLOCKS."
            )
        if self.offload_mode != OFFLOAD_NONE and self.num_dram_kvcache_blocks <= 2:
            raise ValueError(
                "DSA offload requires num_dram_kvcache_blocks > 2. The "
                "example scripts read it from NANOVLLM_DRAM_NUM_BLOCKS."
            )
        if self.max_num_prefill_seqs_per_step <= 0:
            raise ValueError("max_num_prefill_seqs_per_step must be > 0.")
        self._validate_prefill_chunking(
            self.prefill_chunk_size,
            self.max_num_prefill_seqs_per_step,
        )
        self._validate_num_speculative_tokens(self.num_speculative_tokens)
        if self.max_num_decode_seqs_per_step <= 0:
            raise ValueError("max_num_decode_seqs_per_step must be > 0.")
        if self.max_model_len <= 0:
            raise ValueError("max_model_len must be > 0.")
        self.hf_config = self._load_hf_config()
        self._configure_glm_version()
        setattr(
            self.hf_config,
            "nanovllm_enable_expert_parallel",
            bool(self.enable_expert_parallel),
        )
        setattr(
            self.hf_config,
            "nanovllm_offload_mode",
            self.offload_mode,
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

        self._configure_glm_runtime()
        self._validate_glm52_phase1_runtime()
        self._validate_mtp_runtime()
        self._validate_lidu_runtime(text_config)
        eos_token_id = getattr(text_config, "eos_token_id", None)
        if eos_token_id is not None:
            self.eos = normalize_eos_token_ids(eos_token_id)
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

    @staticmethod
    def _validate_num_speculative_tokens(value: int) -> None:
        if type(value) is not int:
            raise TypeError("num_speculative_tokens must be an int.")
        if value not in (0, 3):
            raise ValueError(
                "num_speculative_tokens must be either 0 or 3."
            )

    def _validate_mtp_runtime(self) -> None:
        k = self.num_speculative_tokens
        setattr(self.hf_config, "nanovllm_num_speculative_tokens", k)
        if not k:
            return

        if getattr(self.hf_config, "model_type", "") != "glm_moe_dsa":
            raise ValueError(
                "Built-in MTP currently supports GLM W4A8 only."
            )
        if int(getattr(self.hf_config, "num_nextn_predict_layers", 0)) != 1:
            raise ValueError(
                "GLM MTP expects num_nextn_predict_layers=1 in config.json."
            )
        mtp_layer_idx = int(getattr(self.hf_config, "num_hidden_layers", -1))
        if mtp_layer_idx != 78:
            raise ValueError(
                "GLM MTP expects 78 target layers and checkpoint layer "
                f"78, got num_hidden_layers={mtp_layer_idx}."
            )

        metadata = getattr(self.hf_config, "nanovllm_quant_metadata", {})
        if metadata.get("is_rot_used") is not True:
            raise ValueError(
                "GLM MTP requires ModelSlim is_rot_used=true."
            )
        rot_path = os.path.join(self.model, "rot.safetensors")
        if not os.path.isfile(rot_path):
            raise ValueError(
                "GLM MTP requires root-level rot.safetensors containing "
                "rot.weight."
            )

        description_path = os.path.join(
            self.model, "quant_model_description.json"
        )
        with open(description_path, "r", encoding="utf-8") as file:
            description = json.load(file)
        mtp_entries = {
            name: quant_type
            for name, quant_type in description.items()
            if name.startswith(f"model.layers.{mtp_layer_idx}.")
        }
        if not mtp_entries:
            raise ValueError(
                "GLM checkpoint has no model.layers.78 MTP weights."
            )
        if self.glm_version == GLM_VERSION_52:
            expert_entries = {
                name: quant_type
                for name, quant_type in mtp_entries.items()
                if ".mlp.experts." in name
            }
            if not expert_entries or any(
                quant_type != "W4A8_DYNAMIC"
                for quant_type in expert_entries.values()
            ):
                raise ValueError(
                    "GLM-5.2 MTP experts must use ModelSlim W4A8_DYNAMIC "
                    "weights."
                )
            non_expert_w4 = sorted(
                name
                for name, quant_type in mtp_entries.items()
                if ".mlp.experts." not in name
                and quant_type == "W4A8_DYNAMIC"
            )
            if non_expert_w4:
                raise ValueError(
                    "GLM-5.2 MTP W4A8 is supported for routed experts "
                    f"only; unsupported entries include {non_expert_w4[:3]}."
                )
            setattr(
                self.hf_config,
                "nanovllm_mtp_uses_w4a8_experts",
                True,
            )
            return
        non_float = sorted(
            name
            for name, quant_type in mtp_entries.items()
            if quant_type != "FLOAT"
        )
        if non_float:
            raise ValueError(
                "GLM MTP layer must be FLOAT/BF16 in the ModelSlim "
                f"description; non-FLOAT entries include {non_float[:3]}."
            )

    def _configure_glm_version(self) -> None:
        indexer_types = getattr(self.hf_config, "indexer_types", None)
        if indexer_types is None:
            self.glm_version = GLM_VERSION_51
            self.glm_model_name = "GLM-5.1"
        else:
            num_layers = int(getattr(self.hf_config, "num_hidden_layers", 0))
            actual = tuple(str(value) for value in indexer_types)
            expected = glm52_indexer_types(num_layers)
            if num_layers != 78 or actual != expected:
                raise ValueError(
                    "GLM-5.2 requires the official 21-full/57-shared "
                    "IndexShare schedule."
                )
            self.glm_version = GLM_VERSION_52
            self.glm_model_name = "GLM-5.2"

        setattr(self.hf_config, "nanovllm_glm_version", self.glm_version)
        setattr(self.hf_config, "nanovllm_model_name", self.glm_model_name)
        num_layers = int(getattr(self.hf_config, "num_hidden_layers", 0))
        raw_indexer_types = getattr(self.hf_config, "indexer_types", None)
        index_share_groups = IndexShareGroupManager(
            num_hidden_layers=num_layers,
            indexer_types=(
                tuple(str(t) for t in raw_indexer_types)
                if raw_indexer_types is not None
                else None
            ),
        )
        setattr(
            self.hf_config,
            "nanovllm_index_share_groups",
            index_share_groups,
        )

    def _validate_glm52_phase1_runtime(self) -> None:
        if self.glm_version != GLM_VERSION_52:
            return
        if self.num_speculative_tokens and self.offload_mode != OFFLOAD_NONE:
            if self.offload_mode not in (OFFLOAD_SPLIT, OFFLOAD_FUSE):
                raise ValueError(
                    "GLM-5.2 MTP offload currently supports "
                    "offload_mode=offload_split or offload_fuse."
                )
    def _configure_decode_graph(self) -> None:
        # There are exactly two execution modes. Prefill and first decode are
        # always eager; enforce_eager controls steady-state decode.
        if not isinstance(self.enforce_eager, bool):
            raise TypeError("enforce_eager must be a bool.")
        if (
            os.environ.get(
                "NANOVLLM_LIDU_MISS_COUNT_ON_LAYERS", ""
            ).strip()
            and not self.enforce_eager
        ):
            raise ValueError(
                "NANOVLLM_LIDU_MISS_COUNT_ON_LAYERS is eager-only; set "
                "enforce_eager=True / NANOVLLM_ENFORCE_EAGER=1."
            )
        if self.enforce_eager:
            self.decode_graph_capture_sizes = ()
            return
        if os.environ.get("ASCEND_LAUNCH_BLOCKING") == "1":
            raise ValueError(
                "ASCEND_LAUNCH_BLOCKING=1 is incompatible with FULL_DECODE_ONLY."
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
        if (
            self.offload_mode == OFFLOAD_NONE
            and self.decode_graph_capture_sizes[-1] > self.kvcache_block_size
        ):
            raise ValueError(
                "Dense-MLA FULL_DECODE_ONLY requires capture sizes not to "
                "exceed kvcache_block_size so padded rows can use distinct "
                "null-block slots: "
                f"sizes={self.decode_graph_capture_sizes}, "
                f"block_size={self.kvcache_block_size}."
            )

    def _validate_model_format(self):
        dtype = getattr(
            self.hf_config,
            "torch_dtype",
            getattr(self.hf_config, "dtype", None),
        )
        if str(dtype).lower() not in (
            "bf16",
            "bfloat16",
            "torch.bfloat16",
        ):
            raise ValueError(
                "GLM W4A8 requires BF16 runtime dtype, got "
                f"{dtype!r}."
            )
        description_path = os.path.join(
            self.model, "quant_model_description.json"
        )
        if not os.path.isfile(description_path):
            raise ValueError(
                "GLM W4A8 requires quant_model_description.json "
                "from the ModelSlim checkpoint."
            )
        with open(description_path, "r", encoding="utf-8") as file:
            description = json.load(file)
        metadata = {
            key: description.get(key)
            for key in (
                "version",
                "model_quant_type",
                "group_size",
                "is_rot_used",
                "optional",
            )
        }
        if metadata["version"] != "1.0.0":
            raise ValueError(
                "GLM W4A8 support requires ModelSlim quant version 1.0.0 "
                f"only, got {metadata['version']!r}."
            )
        if metadata["model_quant_type"] != "W8A8_DYNAMIC":
            raise ValueError(
                "GLM W4A8 support expects model_quant_type="
                f"'W8A8_DYNAMIC', got {metadata['model_quant_type']!r}."
            )
        if metadata["group_size"] != 0:
            raise ValueError(
                "GLM W4A8 support expects per-channel W4A8 "
                f"(group_size=0), got {metadata['group_size']!r}."
            )
        setattr(self.hf_config, "nanovllm_quant_metadata", metadata)

    def _configure_glm_runtime(self) -> None:
        if not self.enable_expert_parallel:
            raise ValueError(
                "GLM W4A8 requires expert parallel; set "
                "enable_expert_parallel=True / "
                "NANOVLLM_ENABLE_EXPERT_PARALLEL=1."
            )
        index_topk = int(
            getattr(self.hf_config, "index_topk", DSA_SELECTION_TOPK_TOKENS)
        )
        if self.offload_mode != OFFLOAD_NONE and index_topk != DSA_SELECTION_TOPK_TOKENS:
            raise ValueError(
                "GLM DSA offload currently requires index_topk=2048, got "
                f"{index_topk}."
            )
        if (
            self.offload_mode != OFFLOAD_NONE
            and int(getattr(self.hf_config, "index_head_dim", 0)) <= 0
        ):
            raise ValueError("GLM DSA offload requires a positive index_head_dim.")
        if (
            self.offload_mode != OFFLOAD_NONE
            and int(getattr(self.hf_config, "index_n_heads", 0)) <= 0
        ):
            raise ValueError("GLM DSA offload requires a positive index_n_heads.")
        if self.offload_mode != OFFLOAD_NONE and not bool(
            getattr(self.hf_config, "indexer_rope_interleave", True)
        ):
            raise ValueError(
                "GLM DSA offload expects indexer_rope_interleave=true."
            )
        # Rotary caches are owned by every attention layer.  Limit them to the
        # configured runtime context instead of GLM's full 202K checkpoint cap.
        setattr(
            self.hf_config,
            "nanovllm_original_max_position_embeddings",
            int(self.hf_config.max_position_embeddings),
        )
        self.hf_config.max_position_embeddings = self.max_model_len

    def _validate_lidu_runtime(self, text_config: Any) -> None:
        if self.offload_mode not in LIDU_OFFLOAD_MODES:
            return
        if self.kvcache_block_size != 128:
            raise ValueError(
                "LIDU offload currently requires kvcache_block_size=128, got "
                f"{self.kvcache_block_size}."
            )
        if int(getattr(text_config, "kv_lora_rank", 0)) != 512:
            raise ValueError("LIDU SCATTER currently requires kv_lora_rank=512.")
        if int(getattr(text_config, "qk_rope_head_dim", 0)) != 64:
            raise ValueError(
                "LIDU SCATTER currently requires qk_rope_head_dim=64."
            )
        index_heads = int(getattr(text_config, "index_n_heads", 0))
        index_dim = int(getattr(text_config, "index_head_dim", 0))
        if index_heads != 32 or index_dim != 128:
            raise ValueError(
                "GLM-5.1 LIDU requires index_n_heads=32 and "
                f"index_head_dim=128, got heads={index_heads}, dim={index_dim}."
            )
        max_source_tokens = (
            self.max_model_len // self.kvcache_block_size
        ) * self.kvcache_block_size
        if max_source_tokens > LIDU_MAX_SOURCE_TOKENS:
            raise ValueError(
                "LIDU prefill-full-block source must contain fewer than 2^18 "
                f"tokens, got {max_source_tokens}."
            )
        validate_lidu_cache_token_budgets(self.kvcache_block_size)

    def _load_hf_config(self):
        config_path = os.path.join(self.model, "config.json")
        raw_config = {}
        if os.path.isfile(config_path):
            with open(config_path, "r", encoding="utf-8") as file:
                raw_config = json.load(file)

        glm_moe_dsa_like = (
            raw_config.get("model_type") == "glm_moe_dsa"
            or "GlmMoeDsaForCausalLM"
            in (raw_config.get("architectures") or [])
        )
        if glm_moe_dsa_like:
            from nanovllm.models.glm_moe_dsa_config import GlmMoeDsaConfig

            return GlmMoeDsaConfig.from_pretrained(self.model)

        raise ValueError(
            "nano-vllm-ascend supports GLM-5.1/5.2 W4A8 only. The model config "
            "must use model_type='glm_moe_dsa' or architecture "
            "'GlmMoeDsaForCausalLM'."
        )

    def __repr__(self):
        attrs = {k: v for k, v in self.__dict__.items() if k != "hf_config"}
        attrs["hf_config"] = f"{self.hf_config.__class__.__name__}(...)"
        items = [f"{k}={v}" for k, v in attrs.items()]
        return f"Config({', '.join(items)})"
