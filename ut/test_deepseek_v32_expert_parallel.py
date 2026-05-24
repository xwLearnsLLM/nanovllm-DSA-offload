from __future__ import annotations

import unittest
from contextlib import ExitStack
from unittest.mock import patch

import torch

from nanovllm.models.deepseek_v32 import DeepseekV32Config, DeepseekV32SparseMoeBlock


def _build_config(*, enable_expert_parallel: bool) -> DeepseekV32Config:
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
        n_routed_experts=4,
        routed_scaling_factor=1.0,
        kv_lora_rank=4,
        q_lora_rank=8,
        qk_rope_head_dim=4,
        v_head_dim=4,
        qk_nope_head_dim=4,
        n_group=1,
        topk_group=1,
        num_experts_per_tok=2,
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
        scoring_func="sigmoid",
        nanovllm_enable_expert_parallel=enable_expert_parallel,
    )


def _instantiate_block(
    config: DeepseekV32Config,
    *,
    rank: int,
    world_size: int,
) -> DeepseekV32SparseMoeBlock:
    with ExitStack() as stack:
        stack.enter_context(patch("torch.distributed.get_rank", return_value=rank))
        stack.enter_context(
            patch("torch.distributed.get_world_size", return_value=world_size)
        )
        stack.enter_context(patch("torch.distributed.is_initialized", return_value=False))
        stack.enter_context(patch("torch.distributed.all_reduce", side_effect=lambda x: x))
        stack.enter_context(patch("torch.distributed.gather", side_effect=lambda *args, **kwargs: None))
        return DeepseekV32SparseMoeBlock(config, layer_idx=0)


def _copy_merged_weight(
    dst_param: torch.nn.Parameter,
    src_param: torch.Tensor,
    *,
    rank: int,
    world_size: int,
    output_sizes: list[int],
    disable_tp: bool,
) -> None:
    if disable_tp:
        dst_param.data.copy_(src_param)
        return

    local_chunks = []
    offset = 0
    for size in output_sizes:
        shard = src_param[offset: offset + size]
        local_chunks.append(shard.chunk(world_size, dim=0)[rank])
        offset += size
    dst_param.data.copy_(torch.cat(local_chunks, dim=0))


def _load_sharded_params(
    dst: DeepseekV32SparseMoeBlock,
    src: DeepseekV32SparseMoeBlock,
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
            if name.endswith("gate_up_proj.weight"):
                _copy_merged_weight(
                    param,
                    src_param,
                    rank=rank,
                    world_size=world_size,
                    output_sizes=[dst.moe_intermediate_size, dst.moe_intermediate_size],
                    disable_tp=name.startswith("experts."),
                )
                continue
            weight_loader = getattr(param, "weight_loader", None)
            if weight_loader is not None:
                weight_loader(param, src_param)
            else:
                param.data.copy_(src_param)

        if getattr(dst.gate, "e_score_correction_bias", None) is not None:
            dst.gate.e_score_correction_bias.data.copy_(
                src.gate.e_score_correction_bias.data
            )


class TestDeepseekV32ExpertParallel(unittest.TestCase):
    def test_expert_parallel_matches_single_rank_reference(self):
        torch.manual_seed(0)
        ref_block = _instantiate_block(
            _build_config(enable_expert_parallel=False),
            rank=0,
            world_size=1,
        )
        for param in ref_block.parameters():
            torch.nn.init.uniform_(param, -0.1, 0.1)

        ep_block_0 = _instantiate_block(
            _build_config(enable_expert_parallel=True),
            rank=0,
            world_size=2,
        )
        ep_block_1 = _instantiate_block(
            _build_config(enable_expert_parallel=True),
            rank=1,
            world_size=2,
        )
        _load_sharded_params(ep_block_0, ref_block, rank=0, world_size=2)
        _load_sharded_params(ep_block_1, ref_block, rank=1, world_size=2)

        hidden_states = torch.randn(5, 16)
        with torch.inference_mode():
            ref_out = ref_block(hidden_states)
            with patch("torch.distributed.all_reduce", side_effect=lambda x: x):
                ep_out_0 = ep_block_0(hidden_states)
            with patch("torch.distributed.all_reduce", side_effect=lambda x: x):
                ep_out_1 = ep_block_1(hidden_states)

        self.assertTrue(
            torch.allclose(ep_out_0 + ep_out_1, ref_out, atol=1e-6, rtol=1e-6)
        )


if __name__ == "__main__":
    unittest.main()
