from __future__ import annotations

import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

import torch
from safetensors.torch import save_file

from nanovllm.models.deepseek_v32 import DeepseekV32Config, DeepseekV32ForCausalLM
from nanovllm.utils.loader import load_model


def _build_config(*, enable_expert_parallel: bool, world_size: int) -> DeepseekV32Config:
    if 2 % world_size != 0:
        raise ValueError("Test config expects num_attention_heads divisible by world size.")
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
        index_n_heads=2,
        index_head_dim=8,
        index_topk=32,
        topk_method="noaux_tc",
        scoring_func="sigmoid",
        nanovllm_enable_expert_parallel=enable_expert_parallel,
        nanovllm_pruned_keep_routed_experts=True,
    )


def _instantiate_model(
    config: DeepseekV32Config,
    *,
    rank: int,
    world_size: int,
) -> DeepseekV32ForCausalLM:
    with ExitStack() as stack:
        stack.enter_context(patch("torch.distributed.get_rank", return_value=rank))
        stack.enter_context(
            patch("torch.distributed.get_world_size", return_value=world_size)
        )
        stack.enter_context(patch("torch.distributed.is_initialized", return_value=False))
        stack.enter_context(patch("torch.distributed.all_reduce", side_effect=lambda x: x))
        stack.enter_context(patch("torch.distributed.all_gather", side_effect=lambda *args, **kwargs: None))
        stack.enter_context(patch("torch.distributed.gather", side_effect=lambda *args, **kwargs: None))
        return DeepseekV32ForCausalLM(config)


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


def _manually_load_expected_shards(
    dst: DeepseekV32ForCausalLM,
    src: DeepseekV32ForCausalLM,
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
                output_sizes = [src_param.shape[0] // 2, src_param.shape[0] // 2]
                _copy_merged_weight(
                    param,
                    src_param,
                    rank=rank,
                    world_size=world_size,
                    output_sizes=output_sizes,
                    disable_tp=name.startswith("model.layers.0.mlp.experts."),
                )
                continue
            weight_loader = getattr(param, "weight_loader", None)
            if weight_loader is not None:
                weight_loader(param, src_param)
            else:
                param.data.copy_(src_param)

        if hasattr(dst.model.layers[0].mlp.gate, "e_score_correction_bias"):
            dst.model.layers[0].mlp.gate.e_score_correction_bias.data.copy_(
                src.model.layers[0].mlp.gate.e_score_correction_bias.data
            )


def _build_loader_checkpoint(model: DeepseekV32ForCausalLM, output_dir: Path) -> None:
    tensors: dict[str, torch.Tensor] = {}
    for name, tensor in model.state_dict().items():
        if name.endswith("gate_up_proj.weight"):
            gate_proj, up_proj = tensor.chunk(2, dim=0)
            tensors[name.replace("gate_up_proj", "gate_proj")] = gate_proj.contiguous()
            tensors[name.replace("gate_up_proj", "up_proj")] = up_proj.contiguous()
        elif name in {"model.layers.0.self_attn.w_uk_t", "model.layers.0.self_attn.w_uv"}:
            continue
        else:
            tensors[name] = tensor.contiguous()
    save_file(tensors, str(output_dir / "model.safetensors"))


class TestDeepseekV32Loader(unittest.TestCase):
    def test_load_model_matches_expected_tp_ep_shards(self):
        torch.manual_seed(0)
        ref_model = _instantiate_model(
            _build_config(enable_expert_parallel=False, world_size=1),
            rank=0,
            world_size=1,
        )
        for param in ref_model.parameters():
            torch.nn.init.uniform_(param, -0.1, 0.1)

        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_dir = Path(tmpdir)
            _build_loader_checkpoint(ref_model, checkpoint_dir)

            actual_models = []
            expected_models = []
            for rank in range(2):
                actual = _instantiate_model(
                    _build_config(enable_expert_parallel=True, world_size=2),
                    rank=rank,
                    world_size=2,
                )
                expected = _instantiate_model(
                    _build_config(enable_expert_parallel=True, world_size=2),
                    rank=rank,
                    world_size=2,
                )

                load_model(
                    actual,
                    str(checkpoint_dir),
                    name_mapping=getattr(actual, "weight_name_mapping", None),
                )
                _manually_load_expected_shards(
                    expected,
                    ref_model,
                    rank=rank,
                    world_size=2,
                )
                actual_models.append(actual)
                expected_models.append(expected)

            for actual, expected in zip(actual_models, expected_models):
                for (actual_name, actual_param), (expected_name, expected_param) in zip(
                    actual.named_parameters(),
                    expected.named_parameters(),
                ):
                    self.assertEqual(actual_name, expected_name)
                    self.assertTrue(
                        torch.allclose(actual_param, expected_param, atol=1e-6, rtol=1e-6),
                        msg=f"Mismatch for parameter {actual_name}",
                    )


if __name__ == "__main__":
    unittest.main()
