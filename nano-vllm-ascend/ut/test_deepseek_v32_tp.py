from __future__ import annotations

import unittest
from contextlib import ExitStack
from unittest.mock import patch

import torch

from nanovllm.models.deepseek_v32 import DeepseekV32Config, DeepseekV32DSAAttention
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


def _instantiate_attention(
    config: DeepseekV32Config,
    *,
    rank: int,
    world_size: int,
) -> DeepseekV32DSAAttention:
    with ExitStack() as stack:
        stack.enter_context(patch("torch.distributed.get_rank", return_value=rank))
        stack.enter_context(
            patch("torch.distributed.get_world_size", return_value=world_size)
        )
        stack.enter_context(patch("torch.distributed.is_initialized", return_value=False))
        stack.enter_context(patch("torch.distributed.all_reduce", side_effect=lambda x: x))
        stack.enter_context(patch("torch.distributed.gather", side_effect=lambda *args, **kwargs: None))
        attn = DeepseekV32DSAAttention(config, layer_idx=0)
    return attn


def _init_attention(attn: DeepseekV32DSAAttention) -> None:
    for param in attn.parameters():
        torch.nn.init.uniform_(param, -0.1, 0.1)
    attn.to(torch.bfloat16)
    attn.indexer.weights_proj = attn.indexer.weights_proj.to(torch.float32)


def _assign_cache(attn: DeepseekV32DSAAttention) -> None:
    num_blocks = 2
    block_size = 8
    attn.assign_dsa_cache(
        torch.zeros(num_blocks, block_size, attn.kv_lora_rank, dtype=torch.bfloat16),
        torch.zeros(num_blocks, block_size, attn.qk_rope_head_dim, dtype=torch.bfloat16),
        torch.zeros(num_blocks, block_size, attn.indexer.head_dim, dtype=torch.bfloat16),
    )
    attn.post_load_prepare()


def _load_sharded_params(
    dst: DeepseekV32DSAAttention,
    src: DeepseekV32DSAAttention,
    *,
    rank: int,
    world_size: int,
) -> None:
    full_params = dict(src.named_parameters())
    with ExitStack() as stack:
        stack.enter_context(patch("torch.distributed.get_rank", return_value=rank))
        stack.enter_context(
            patch("torch.distributed.get_world_size", return_value=world_size)
        )
        for name, param in dst.named_parameters():
            src_param = full_params[name].detach().clone()
            weight_loader = getattr(param, "weight_loader", None)
            if weight_loader is not None:
                weight_loader(param, src_param)
            else:
                param.data.copy_(src_param)
    dst.to(torch.bfloat16)
    dst.indexer.weights_proj = dst.indexer.weights_proj.to(torch.float32)
    dst.post_load_prepare()


class TestDeepseekV32TensorParallel(unittest.TestCase):
    def tearDown(self):
        reset_context()

    @staticmethod
    def _run_with_mocked_collectives(fn):
        with patch("torch.distributed.all_reduce", side_effect=lambda x: x):
            return fn()

    def test_tp_decode_matches_single_rank_reference(self):
        torch.manual_seed(0)
        config = _build_config()

        ref_attn = _instantiate_attention(config, rank=0, world_size=1)
        _init_attention(ref_attn)
        _assign_cache(ref_attn)

        tp_attn_0 = _instantiate_attention(config, rank=0, world_size=2)
        tp_attn_1 = _instantiate_attention(config, rank=1, world_size=2)
        _assign_cache(tp_attn_0)
        _assign_cache(tp_attn_1)
        _load_sharded_params(tp_attn_0, ref_attn, rank=0, world_size=2)
        _load_sharded_params(tp_attn_1, ref_attn, rank=1, world_size=2)

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
            block_size=8,
        )
        ref_prefill = ref_attn(prefix_positions, prefix_hidden_states)
        reset_context()

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
        out0_prefill = self._run_with_mocked_collectives(
            lambda: tp_attn_0(prefix_positions, prefix_hidden_states)
        )
        reset_context()
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
        out1_prefill = self._run_with_mocked_collectives(
            lambda: tp_attn_1(prefix_positions, prefix_hidden_states)
        )
        reset_context()
        self.assertTrue(
            torch.allclose(out0_prefill + out1_prefill, ref_prefill, atol=2e-2, rtol=2e-2)
        )

        decode_hidden_state = torch.randn(1, 16, dtype=torch.bfloat16)
        decode_position = torch.tensor([3], dtype=torch.int64)

        set_context(
            False,
            slot_mapping=torch.tensor([[0, 3]], dtype=torch.int32),
            context_lens=torch.tensor([4], dtype=torch.int32),
            block_tables=torch.tensor([[0]], dtype=torch.int32),
            is_enforce_eager=True,
            real_bs=1,
            block_size=8,
        )
        ref_out = ref_attn(decode_position, decode_hidden_state)
        reset_context()

        set_context(
            False,
            slot_mapping=torch.tensor([[0, 3]], dtype=torch.int32),
            context_lens=torch.tensor([4], dtype=torch.int32),
            block_tables=torch.tensor([[0]], dtype=torch.int32),
            is_enforce_eager=True,
            real_bs=1,
            block_size=8,
        )
        out0 = self._run_with_mocked_collectives(
            lambda: tp_attn_0(decode_position, decode_hidden_state)
        )
        reset_context()
        set_context(
            False,
            slot_mapping=torch.tensor([[0, 3]], dtype=torch.int32),
            context_lens=torch.tensor([4], dtype=torch.int32),
            block_tables=torch.tensor([[0]], dtype=torch.int32),
            is_enforce_eager=True,
            real_bs=1,
            block_size=8,
        )
        out1 = self._run_with_mocked_collectives(
            lambda: tp_attn_1(decode_position, decode_hidden_state)
        )
        reset_context()

        self.assertTrue(
            torch.allclose(out0 + out1, ref_out, atol=2e-2, rtol=2e-2)
        )


if __name__ == "__main__":
    unittest.main()
