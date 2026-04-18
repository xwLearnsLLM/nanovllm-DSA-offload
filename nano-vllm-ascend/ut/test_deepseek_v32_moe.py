from __future__ import annotations

import unittest
from unittest.mock import patch

import torch

from nanovllm.models.deepseek_v32 import DeepseekV32Config, DeepseekV32SparseMoeBlock


def _build_config() -> DeepseekV32Config:
    return DeepseekV32Config(
        architectures=["DeepseekV32ForCausalLM"],
        vocab_size=64,
        hidden_size=12,
        intermediate_size=24,
        moe_intermediate_size=6,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        n_shared_experts=1,
        n_routed_experts=4,
        routed_scaling_factor=2.5,
        kv_lora_rank=4,
        q_lora_rank=8,
        qk_rope_head_dim=4,
        v_head_dim=4,
        qk_nope_head_dim=4,
        n_group=2,
        topk_group=1,
        num_experts_per_tok=2,
        first_k_dense_replace=0,
        norm_topk_prob=True,
        hidden_act="silu",
        max_position_embeddings=32,
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        use_cache=True,
        bos_token_id=0,
        eos_token_id=1,
        tie_word_embeddings=False,
        rope_theta=10000.0,
        rope_scaling=None,
        attention_bias=False,
        attention_dropout=0.0,
        index_n_heads=2,
        index_head_dim=8,
        index_topk=32,
        topk_method="noaux_tc",
        scoring_func="sigmoid",
    )


class TestDeepseekV32Moe(unittest.TestCase):
    def setUp(self):
        self.dist_patches = [
            patch("torch.distributed.get_rank", return_value=0),
            patch("torch.distributed.get_world_size", return_value=1),
            patch("torch.distributed.is_initialized", return_value=False),
            patch("torch.distributed.all_reduce", side_effect=lambda x: x),
        ]
        for patcher in self.dist_patches:
            patcher.start()

    def tearDown(self):
        for patcher in reversed(self.dist_patches):
            patcher.stop()

    def test_single_rank_matches_reference(self):
        torch.manual_seed(0)
        block = DeepseekV32SparseMoeBlock(_build_config(), layer_idx=0).to(torch.bfloat16)
        for param in block.parameters():
            if param.dtype == torch.float32:
                torch.nn.init.uniform_(param, -0.1, 0.1)
            else:
                torch.nn.init.uniform_(param, -0.1, 0.1)

        hidden_states = torch.randn(5, 12, dtype=torch.bfloat16)
        actual = block(hidden_states)

        shared_output = block.shared_experts(hidden_states)
        router_logits = block.gate(hidden_states)
        routing_weights, selected_experts = block._grouped_topk(
            hidden_states,
            router_logits,
        )
        routing_weights = routing_weights.to(hidden_states.dtype)

        expected = torch.zeros_like(hidden_states)
        for token_idx in range(hidden_states.shape[0]):
            for topk_idx in range(selected_experts.shape[1]):
                expert_idx = int(selected_experts[token_idx, topk_idx].item())
                expert = block.experts[str(expert_idx)]
                expert_out = expert(hidden_states[token_idx : token_idx + 1])[0]
                expected[token_idx] += (
                    expert_out * routing_weights[token_idx, topk_idx]
                )
        expected = expected + shared_output

        self.assertTrue(torch.allclose(actual, expected, atol=2e-2, rtol=2e-2))


if __name__ == "__main__":
    unittest.main()
