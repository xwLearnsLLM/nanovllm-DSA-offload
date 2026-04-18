from __future__ import annotations

import unittest
from contextlib import ExitStack
from unittest.mock import patch

import torch
import torch.nn.functional as F

from nanovllm.layers.embed_head import ParallelLMHead, VocabParallelEmbedding
from nanovllm.utils.context import reset_context, set_context


def _instantiate_module(module_cls, *args, rank: int, world_size: int, **kwargs):
    with ExitStack() as stack:
        stack.enter_context(patch("torch.distributed.get_rank", return_value=rank))
        stack.enter_context(
            patch("torch.distributed.get_world_size", return_value=world_size)
        )
        stack.enter_context(patch("torch.distributed.is_initialized", return_value=False))
        stack.enter_context(patch("torch.distributed.all_reduce", side_effect=lambda x: x))
        stack.enter_context(patch("torch.distributed.all_gather", side_effect=lambda *args, **kwargs: None))
        return module_cls(*args, **kwargs)


def _load_vocab_shard(dst, src) -> None:
    src_weight = src.weight.detach().clone()
    dst.weight_loader(dst.weight, src_weight)


class TestEmbedHeadTensorParallel(unittest.TestCase):
    def tearDown(self):
        reset_context()

    def test_vocab_parallel_embedding_matches_single_rank_reference(self):
        torch.manual_seed(0)
        ref_embed = _instantiate_module(
            VocabParallelEmbedding,
            8,
            4,
            rank=0,
            world_size=1,
        )
        torch.nn.init.uniform_(ref_embed.weight, -0.1, 0.1)

        tp_embed_0 = _instantiate_module(
            VocabParallelEmbedding,
            8,
            4,
            rank=0,
            world_size=2,
        )
        tp_embed_1 = _instantiate_module(
            VocabParallelEmbedding,
            8,
            4,
            rank=1,
            world_size=2,
        )
        _load_vocab_shard(tp_embed_0, ref_embed)
        _load_vocab_shard(tp_embed_1, ref_embed)

        input_ids = torch.tensor([0, 1, 4, 6, 7], dtype=torch.long)
        expected = ref_embed(input_ids)
        with patch("torch.distributed.all_reduce", side_effect=lambda x: x):
            actual = tp_embed_0(input_ids) + tp_embed_1(input_ids)
        self.assertTrue(torch.allclose(actual, expected, atol=1e-6, rtol=1e-6))

    def test_parallel_lm_head_matches_single_rank_reference(self):
        torch.manual_seed(0)
        ref_head = _instantiate_module(
            ParallelLMHead,
            8,
            4,
            rank=0,
            world_size=1,
        )
        torch.nn.init.uniform_(ref_head.weight, -0.1, 0.1)

        tp_head_0 = _instantiate_module(
            ParallelLMHead,
            8,
            4,
            rank=0,
            world_size=2,
        )
        tp_head_1 = _instantiate_module(
            ParallelLMHead,
            8,
            4,
            rank=1,
            world_size=2,
        )
        _load_vocab_shard(tp_head_0, ref_head)
        _load_vocab_shard(tp_head_1, ref_head)

        hidden_states = torch.randn(5, 4)
        set_context(
            True,
            cu_seqlens_q=torch.tensor([0, 2, 5], dtype=torch.int32),
            cu_seqlens_k=torch.tensor([0, 2, 5], dtype=torch.int32),
            max_seqlen_q=3,
            max_seqlen_k=3,
            slot_mapping=torch.tensor([0, 1, 2, 3, 4], dtype=torch.int32),
            context_lens=None,
            block_tables=torch.tensor([[0], [1]], dtype=torch.int32),
            block_size=8,
        )
        expected = ref_head(hidden_states)
        reset_context()

        last_hidden_states = hidden_states[torch.tensor([1, 4])]
        shard_0 = F.linear(last_hidden_states, tp_head_0.weight)
        shard_1 = F.linear(last_hidden_states, tp_head_1.weight)

        def _fake_all_gather(output_tensors, input_tensor):
            del input_tensor
            output_tensors[0].copy_(shard_0)
            output_tensors[1].copy_(shard_1)

        set_context(
            True,
            cu_seqlens_q=torch.tensor([0, 2, 5], dtype=torch.int32),
            cu_seqlens_k=torch.tensor([0, 2, 5], dtype=torch.int32),
            max_seqlen_q=3,
            max_seqlen_k=3,
            slot_mapping=torch.tensor([0, 1, 2, 3, 4], dtype=torch.int32),
            context_lens=None,
            block_tables=torch.tensor([[0], [1]], dtype=torch.int32),
            block_size=8,
        )
        with patch("torch.distributed.all_gather", side_effect=_fake_all_gather):
            actual_0 = tp_head_0(hidden_states)
        reset_context()

        set_context(
            True,
            cu_seqlens_q=torch.tensor([0, 2, 5], dtype=torch.int32),
            cu_seqlens_k=torch.tensor([0, 2, 5], dtype=torch.int32),
            max_seqlen_q=3,
            max_seqlen_k=3,
            slot_mapping=torch.tensor([0, 1, 2, 3, 4], dtype=torch.int32),
            context_lens=None,
            block_tables=torch.tensor([[0], [1]], dtype=torch.int32),
            block_size=8,
        )
        with patch("torch.distributed.all_gather", side_effect=_fake_all_gather):
            actual_1 = tp_head_1(hidden_states)
        reset_context()

        self.assertTrue(torch.allclose(actual_0, expected, atol=1e-6, rtol=1e-6))
        self.assertTrue(torch.allclose(actual_1, expected, atol=1e-6, rtol=1e-6))


if __name__ == "__main__":
    unittest.main()
