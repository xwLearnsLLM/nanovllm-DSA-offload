from __future__ import annotations

import unittest
from unittest.mock import patch

import torch

from nanovllm.models.deepseek_v32 import (
    DeepseekV32Config,
    DeepseekV32DSAAttention,
)
from nanovllm.utils.context import reset_context, set_context


def _build_config() -> DeepseekV32Config:
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
        n_routed_experts=1,
        routed_scaling_factor=1.0,
        kv_lora_rank=4,
        q_lora_rank=8,
        qk_rope_head_dim=4,
        v_head_dim=4,
        qk_nope_head_dim=4,
        n_group=1,
        topk_group=1,
        num_experts_per_tok=1,
        first_k_dense_replace=1,
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


class TestDeepseekV32Attention(unittest.TestCase):
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

    def _build_attention(self) -> tuple[DeepseekV32DSAAttention, int]:
        torch.manual_seed(0)
        config = _build_config()
        attn = DeepseekV32DSAAttention(config, layer_idx=0)
        for param in attn.parameters():
            torch.nn.init.uniform_(param, -0.1, 0.1)
        attn = attn.to(torch.bfloat16)
        attn.indexer.weights_proj = attn.indexer.weights_proj.to(torch.float32)
        block_size = 8
        num_blocks = 2
        attn.assign_dsa_cache(
            torch.zeros(
                num_blocks,
                block_size,
                attn.kv_lora_rank,
                dtype=torch.bfloat16,
            ),
            torch.zeros(
                num_blocks,
                block_size,
                attn.qk_rope_head_dim,
                dtype=torch.bfloat16,
            ),
            torch.zeros(
                num_blocks,
                block_size,
                attn.indexer.head_dim,
                dtype=torch.bfloat16,
            ),
        )
        attn.post_load_prepare()
        return attn, block_size

    @staticmethod
    def _reference_prefill(
        attn: DeepseekV32DSAAttention,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        qr = attn.q_a_layernorm(attn.q_a_proj(hidden_states))
        q = attn.q_b_proj(qr).view(-1, attn.num_local_heads, attn.qk_head_dim)
        q_nope, q_pe = torch.split(
            q,
            [attn.qk_nope_head_dim, attn.qk_rope_head_dim],
            dim=-1,
        )

        kv = attn.kv_a_proj_with_mqa(hidden_states)
        ckv, k_pe = torch.split(
            kv,
            [attn.kv_lora_rank, attn.qk_rope_head_dim],
            dim=-1,
        )
        ckv = attn.kv_a_layernorm(ckv)
        q_pe, k_pe = attn.rotary_emb(positions, q_pe, k_pe.unsqueeze(1))
        k_pe = k_pe.squeeze(1)

        wkv_b = attn.kv_b_proj.weight.data.view(
            attn.num_local_heads,
            attn.qk_nope_head_dim + attn.v_head_dim,
            attn.kv_lora_rank,
        )
        kv_full = torch.einsum("tc,hdc->thd", ckv, wkv_b)
        k_nope, v = torch.split(
            kv_full,
            [attn.qk_nope_head_dim, attn.v_head_dim],
            dim=-1,
        )
        k = torch.cat(
            (
                k_nope,
                k_pe.unsqueeze(1).expand(-1, attn.num_local_heads, -1),
            ),
            dim=-1,
        )
        q = torch.cat((q_nope, q_pe), dim=-1)

        outputs = []
        for token_idx in range(hidden_states.shape[0]):
            scores = torch.einsum(
                "hd,shd->hs",
                q[token_idx].float(),
                k[: token_idx + 1].float(),
            )
            scores = scores * attn.scale
            probs = torch.softmax(scores, dim=-1).to(v.dtype)
            out = torch.einsum("hs,shv->hv", probs, v[: token_idx + 1])
            outputs.append(out.reshape(-1))
        return attn.o_proj(torch.stack(outputs, dim=0))

    @staticmethod
    def _reference_decode(
        attn: DeepseekV32DSAAttention,
        prefix_hidden_states: torch.Tensor,
        decode_hidden_state: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        full_hidden_states = torch.cat((prefix_hidden_states, decode_hidden_state), dim=0)
        qr = attn.q_a_layernorm(attn.q_a_proj(full_hidden_states))
        q = attn.q_b_proj(qr).view(-1, attn.num_local_heads, attn.qk_head_dim)
        q_nope, q_pe = torch.split(
            q,
            [attn.qk_nope_head_dim, attn.qk_rope_head_dim],
            dim=-1,
        )

        kv = attn.kv_a_proj_with_mqa(full_hidden_states)
        ckv, k_pe = torch.split(
            kv,
            [attn.kv_lora_rank, attn.qk_rope_head_dim],
            dim=-1,
        )
        ckv = attn.kv_a_layernorm(ckv)
        q_pe, k_pe = attn.rotary_emb(positions, q_pe, k_pe.unsqueeze(1))
        k_pe = k_pe.squeeze(1)

        q_nope_last = q_nope[-1:]
        q_pe_last = q_pe[-1:]
        ckv_all = ckv
        k_pe_all = k_pe
        ql_nope = torch.einsum("thp,hpl->thl", q_nope_last, attn.w_uk_t)
        scores = torch.einsum("thl,sl->ths", ql_nope.float(), ckv_all.float())
        scores = scores + torch.einsum(
            "thr,sr->ths",
            q_pe_last.float(),
            k_pe_all.float(),
        )
        scores = scores * attn.scale
        probs = torch.softmax(scores, dim=-1).to(ckv_all.dtype)
        latent = torch.einsum("ths,sl->thl", probs, ckv_all)
        out = torch.einsum("thl,hlv->thv", latent, attn.w_uv).reshape(1, -1)
        return attn.o_proj(out)

    def test_prefill_matches_dense_reference_when_topk_covers_prefix(self):
        attn, block_size = self._build_attention()
        hidden_states = torch.randn(3, 16, dtype=torch.bfloat16)
        positions = torch.arange(3, dtype=torch.int64)

        set_context(
            True,
            cu_seqlens_q=torch.tensor([0, 3], dtype=torch.int32),
            cu_seqlens_k=torch.tensor([0, 3], dtype=torch.int32),
            max_seqlen_q=3,
            max_seqlen_k=3,
            slot_mapping=torch.tensor([0, 1, 2], dtype=torch.int32),
            context_lens=None,
            block_tables=torch.tensor([[0]], dtype=torch.int32),
            block_size=block_size,
        )
        actual = attn(positions, hidden_states)
        reset_context()

        expected = self._reference_prefill(attn, hidden_states, positions)
        self.assertTrue(torch.allclose(actual, expected, atol=2e-2, rtol=2e-2))

    def test_decode_matches_dense_reference_when_topk_covers_prefix(self):
        attn, block_size = self._build_attention()
        prefix_hidden_states = torch.randn(3, 16, dtype=torch.bfloat16)
        prefix_positions = torch.arange(3, dtype=torch.int64)

        set_context(
            True,
            cu_seqlens_q=torch.tensor([0, 3], dtype=torch.int32),
            cu_seqlens_k=torch.tensor([0, 3], dtype=torch.int32),
            max_seqlen_q=3,
            max_seqlen_k=3,
            slot_mapping=torch.tensor([0, 1, 2], dtype=torch.int32),
            context_lens=None,
            block_tables=torch.tensor([[0]], dtype=torch.int32),
            block_size=block_size,
        )
        _ = attn(prefix_positions, prefix_hidden_states)
        reset_context()

        decode_hidden_state = torch.randn(1, 16, dtype=torch.bfloat16)
        decode_position = torch.tensor([3], dtype=torch.int64)
        set_context(
            False,
            slot_mapping=torch.tensor([[0, 3]], dtype=torch.int32),
            context_lens=torch.tensor([4], dtype=torch.int32),
            block_tables=torch.tensor([[0]], dtype=torch.int32),
            is_enforce_eager=True,
            real_bs=1,
            block_size=block_size,
        )
        actual = attn(decode_position, decode_hidden_state)
        reset_context()

        expected = self._reference_decode(
            attn,
            prefix_hidden_states,
            decode_hidden_state,
            torch.arange(4, dtype=torch.int64),
        )
        self.assertTrue(torch.allclose(actual, expected, atol=2e-2, rtol=2e-2))

    def test_vectorized_decode_matches_loop_decode(self):
        attn, block_size = self._build_attention()
        prefix_hidden_states = torch.randn(5, 16, dtype=torch.bfloat16)
        prefix_positions = torch.tensor([0, 1, 2, 0, 1], dtype=torch.int64)

        set_context(
            True,
            cu_seqlens_q=torch.tensor([0, 3, 5], dtype=torch.int32),
            cu_seqlens_k=torch.tensor([0, 3, 5], dtype=torch.int32),
            max_seqlen_q=3,
            max_seqlen_k=3,
            slot_mapping=torch.tensor([0, 1, 2, 8, 9], dtype=torch.int32),
            context_lens=None,
            block_tables=torch.tensor([[0], [1]], dtype=torch.int32),
            block_size=block_size,
        )
        _ = attn(prefix_positions, prefix_hidden_states)
        reset_context()

        cached_ckv = attn.ckv_cache.clone()
        cached_kpe = attn.kpe_cache.clone()
        cached_index = attn.index_cache.clone()
        decode_hidden_states = torch.randn(2, 16, dtype=torch.bfloat16)
        decode_positions = torch.tensor([3, 2], dtype=torch.int64)

        set_context(
            False,
            slot_mapping=torch.tensor([[0, 3], [1, 2]], dtype=torch.int32),
            context_lens=torch.tensor([4, 3], dtype=torch.int32),
            block_tables=torch.tensor([[0], [1]], dtype=torch.int32),
            is_enforce_eager=True,
            real_bs=2,
            block_size=block_size,
        )
        loop_output = attn(decode_positions, decode_hidden_states)
        reset_context()

        attn.ckv_cache.copy_(cached_ckv)
        attn.kpe_cache.copy_(cached_kpe)
        attn.index_cache.copy_(cached_index)
        set_context(
            False,
            slot_mapping=torch.tensor([[0, 3], [1, 2]], dtype=torch.int32),
            context_lens=torch.tensor([4, 3], dtype=torch.int32),
            block_tables=torch.tensor([[0], [1]], dtype=torch.int32),
            is_enforce_eager=True,
            real_bs=2,
            block_size=block_size,
            decode_slots=torch.tensor(
                [[0, 1, 2, 3], [8, 9, 10, 0]],
                dtype=torch.int64,
            ),
            decode_mask=torch.tensor(
                [[True, True, True, True], [True, True, True, False]],
                dtype=torch.bool,
            ),
        )
        vectorized_output = attn(decode_positions, decode_hidden_states)
        reset_context()

        self.assertTrue(
            torch.allclose(loop_output, vectorized_output, atol=2e-2, rtol=2e-2)
        )


if __name__ == "__main__":
    unittest.main()
