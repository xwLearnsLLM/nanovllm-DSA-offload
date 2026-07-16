from __future__ import annotations

import json
import os

from transformers import PretrainedConfig


class GlmMoeDsaConfig(PretrainedConfig):
    """Minimal local config for the GLM-5.1 DSA architecture."""

    model_type = "glm_moe_dsa"

    def __init__(self, **kwargs):
        self.max_position_embeddings = kwargs.get(
            "max_position_embeddings", 202752
        )
        # transformers does not currently normalize this model's
        # rope_parameters field for our local architecture.
        rope_parameters = dict(kwargs.get("rope_parameters") or {})
        if "rope_theta" not in rope_parameters:
            rope_parameters["rope_theta"] = kwargs.get(
                "rope_theta", 1_000_000.0
            )
        rope_parameters.setdefault("rope_type", "default")
        self.rope_parameters = rope_parameters
        # GLM's learned DSA indexer uses adjacent-pair (GPT-J style) RoPE.
        # Older exported configs may omit this field even though the upstream
        # vLLM-Ascend backend selects interleaved RoPE for model_type=glm_moe_dsa.
        self.indexer_rope_interleave = kwargs.get(
            "indexer_rope_interleave", True
        )
        super().__init__(**kwargs)
        for key, value in kwargs.items():
            setattr(self, key, value)
        self.rope_parameters = rope_parameters
        self.indexer_rope_interleave = kwargs.get(
            "indexer_rope_interleave", True
        )
        self.architectures = kwargs.get(
            "architectures", ["GlmMoeDsaForCausalLM"]
        )
        if getattr(self, "torch_dtype", None) is None and "dtype" in kwargs:
            self.torch_dtype = kwargs["dtype"]

    @classmethod
    def from_pretrained(cls, model_path: str) -> "GlmMoeDsaConfig":
        config_path = os.path.join(model_path, "config.json")
        with open(config_path, "r", encoding="utf-8") as file:
            return cls(**json.load(file))
