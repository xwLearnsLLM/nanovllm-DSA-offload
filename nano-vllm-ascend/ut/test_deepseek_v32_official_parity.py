from __future__ import annotations

import unittest
from contextlib import ExitStack
from unittest.mock import patch

import torch
from transformers.models.deepseek_v3 import DeepseekV3Config
from transformers.models.deepseek_v3.modeling_deepseek_v3 import (
    DeepseekV3Attention,
    DeepseekV3MoE,
    DeepseekV3RotaryEmbedding,
)

from nanovllm.models.deepseek_v32 import (
    DeepseekV32Config,
    DeepseekV32DSAAttention,
    DeepseekV32SparseMoeBlock,
)
from nanovllm.utils.context import reset_context, set_context


def _build_hf_config(*, n_routed_experts: int, num_experts_per_tok: int) -> DeepseekV3Config:
    return DeepseekV3Config(
        vocab_size=64,
        hidden_size=16,
        intermediate_size=32,
        moe_intermediate_size=8,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        n_shared_experts=1,
        n_routed_experts=n_routed_experts,
        routed_scaling_factor=1.0,
        kv_lora_rank=4,
        q_lora_rank=8,
        qk_rope_head_dim=4,
        v_head_dim=4,
        qk_nope_head_dim=4,
        n_group=1,
        topk_group=1,
        num_experts_per_tok=num_experts_per_tok,
        first_k_dense_replace=0,
        norm_topk_prob=True,
        hidden_act="silu",
        max_position_embeddings=32,
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        use_cache=False,
        bos_token_id=0,
        eos_token_id=1,
        tie_word_embeddings=False,
        rope_theta=10000.0,
        rope_scaling=None,
        rope_interleave=True,
        attention_bias=False,
        attention_dropout=0.0,
        topk_method="noaux_tc",
    )


def _build_nv_config(*, n_routed_experts: int, num_experts_per_tok: int) -> DeepseekV32Config:
    return DeepseekV32Config(
        architectures=["DeepseekV32ForCausalLM"],
        vocab_size=64,
        hidden_size=16,
        intermediate_size=32,
        moe_intermediate_size=8,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        n_shared_experts=1,
        n_routed_experts=n_routed_experts,
        routed_scaling_factor=1.0,
        kv_lora_rank=4,
        q_lora_rank=8,
        qk_rope_head_dim=4,
        v_head_dim=4,
        qk_nope_head_dim=4,
        n_group=1,
        topk_group=1,
        num_experts_per_tok=num_experts_per_tok,
        first_k_dense_replace=0,
        norm_topk_prob=True,
        hidden_act="silu",
        max_position_embeddings=32,
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        use_cache=False,
        bos_token_id=0,
        eos_token_id=1,
        tie_word_embeddings=False,
        rope_theta=10000.0,
        rope_scaling=None,
        rope_interleave=True,
        attention_bias=False,
        attention_dropout=0.0,
        index_n_heads=2,
        index_head_dim=8,
        index_topk=32,
        topk_method="noaux_tc",
        scoring_func="sigmoid",
    )


class TestDeepseekV32OfficialParity(unittest.TestCase):
    def setUp(self):
        self.dist_patches = [
            patch("torch.distributed.get_rank", return_value=0),
            patch("torch.distributed.get_world_size", return_value=1),
            patch("torch.distributed.is_initialized", return_value=False),
            patch("torch.distributed.all_reduce", side_effect=lambda x: x),
            patch("torch.distributed.gather", side_effect=lambda *args, **kwargs: None),
        ]
        for patcher in self.dist_patches:
            patcher.start()

    def tearDown(self):
        reset_context()
        for patcher in reversed(self.dist_patches):
            patcher.stop()

    @staticmethod
    def _assign_attention_cache(attn: DeepseekV32DSAAttention) -> None:
        attn.assign_dsa_cache(
            torch.zeros(2, 8, attn.kv_lora_rank),
            torch.zeros(2, 8, attn.qk_rope_head_dim),
            torch.zeros(2, 8, attn.indexer.head_dim),
        )
        attn.post_load_prepare()

    @staticmethod
    def _copy_attention_weights(
        dst: DeepseekV32DSAAttention,
        src: DeepseekV3Attention,
    ) -> None:
        dst_params = dict(dst.named_parameters())
        for name, param in src.named_parameters():
            dst_params[name].data.copy_(param.data)

    @staticmethod
    def _copy_moe_weights(
        dst: DeepseekV32SparseMoeBlock,
        src: DeepseekV3MoE,
    ) -> None:
        dst.gate.weight.data.copy_(src.gate.weight.data)
        if getattr(dst.gate, "e_score_correction_bias", None) is not None:
            dst.gate.e_score_correction_bias.data.copy_(
                src.gate.e_score_correction_bias.data
            )

        dst.shared_experts.gate_up_proj.weight.data[:8].copy_(
            src.shared_experts.gate_proj.weight.data
        )
        dst.shared_experts.gate_up_proj.weight.data[8:].copy_(
            src.shared_experts.up_proj.weight.data
        )
        dst.shared_experts.down_proj.weight.data.copy_(
            src.shared_experts.down_proj.weight.data
        )

        for expert_idx in range(dst.num_experts):
            src_expert = src.experts[expert_idx]
            dst_expert = dst.experts[str(expert_idx)]
            dst_expert.gate_up_proj.weight.data[:8].copy_(
                src_expert.gate_proj.weight.data
            )
            dst_expert.gate_up_proj.weight.data[8:].copy_(
                src_expert.up_proj.weight.data
            )
            dst_expert.down_proj.weight.data.copy_(
                src_expert.down_proj.weight.data
            )

    def test_attention_matches_huggingface_reference(self):
        torch.manual_seed(0)
        hf_config = _build_hf_config(n_routed_experts=1, num_experts_per_tok=1)
        nv_config = _build_nv_config(n_routed_experts=1, num_experts_per_tok=1)
        hf_attn = DeepseekV3Attention(hf_config, layer_idx=0)
        hf_rotary = DeepseekV3RotaryEmbedding(hf_config)
        nv_attn = DeepseekV32DSAAttention(nv_config, layer_idx=0)

        for param in hf_attn.parameters():
            torch.nn.init.uniform_(param, -0.1, 0.1)
        self._copy_attention_weights(nv_attn, hf_attn)
        self._assign_attention_cache(nv_attn)

        hidden_states = torch.randn(3, 16)
        positions = torch.arange(3, dtype=torch.long)
        position_ids = positions.unsqueeze(0)

        with torch.inference_mode():
            cos, sin = hf_rotary(hidden_states.unsqueeze(0), position_ids)
            hf_prefill = hf_attn(
                hidden_states=hidden_states.unsqueeze(0),
                attention_mask=None,
                position_embeddings=(cos, sin),
            )[0].squeeze(0)

            set_context(
                True,
                cu_seqlens_q=torch.tensor([0, 3], dtype=torch.int32),
                cu_seqlens_k=torch.tensor([0, 3], dtype=torch.int32),
                max_seqlen_q=3,
                max_seqlen_k=3,
                slot_mapping=torch.tensor([0, 1, 2], dtype=torch.int32),
                context_lens=None,
                block_tables=torch.tensor([[0]], dtype=torch.int32),
                block_size=8,
            )
            nv_prefill = nv_attn(positions, hidden_states)
            reset_context()

            self.assertTrue(torch.allclose(hf_prefill, nv_prefill, atol=1e-3, rtol=1e-3))

            decode_hidden_state = torch.randn(1, 16)
            full_hidden_states = torch.cat((hidden_states, decode_hidden_state), dim=0)
            full_position_ids = torch.arange(4, dtype=torch.long).unsqueeze(0)
            cos, sin = hf_rotary(full_hidden_states.unsqueeze(0), full_position_ids)
            hf_decode = hf_attn(
                hidden_states=full_hidden_states.unsqueeze(0),
                attention_mask=None,
                position_embeddings=(cos, sin),
            )[0].squeeze(0)[-1:]

            set_context(
                False,
                slot_mapping=torch.tensor([[0, 3]], dtype=torch.int32),
                context_lens=torch.tensor([4], dtype=torch.int32),
                block_tables=torch.tensor([[0]], dtype=torch.int32),
                is_enforce_eager=True,
                real_bs=1,
                block_size=8,
            )
            nv_decode = nv_attn(torch.tensor([3], dtype=torch.long), decode_hidden_state)
            reset_context()

            self.assertTrue(torch.allclose(hf_decode, nv_decode, atol=1e-6, rtol=1e-6))

    def test_moe_matches_huggingface_reference(self):
        torch.manual_seed(0)
        hf_config = _build_hf_config(n_routed_experts=4, num_experts_per_tok=2)
        nv_config = _build_nv_config(n_routed_experts=4, num_experts_per_tok=2)
        hf_moe = DeepseekV3MoE(hf_config)
        nv_moe = DeepseekV32SparseMoeBlock(nv_config, layer_idx=0)

        for param in hf_moe.parameters():
            torch.nn.init.uniform_(param, -0.1, 0.1)
        self._copy_moe_weights(nv_moe, hf_moe)

        hidden_states = torch.randn(5, 16)
        with torch.inference_mode():
            hf_out = hf_moe(hidden_states)
            nv_out = nv_moe(hidden_states)

        self.assertTrue(torch.allclose(hf_out, nv_out, atol=1e-6, rtol=1e-6))


if __name__ == "__main__":
    unittest.main()
