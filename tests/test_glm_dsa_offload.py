import json
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file

from nanovllm.config import (
    Config,
    glm52_indexer_types,
    merge_eos_token_ids,
    normalize_eos_token_ids,
)
from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.sequence import FinishReason, Sequence
from nanovllm.sampling_params import SamplingParams
from nanovllm.utils.glm_quant import (
    balanced_moe_expert_ids,
    float32_scale_to_int64_bits,
    should_skip_glm_checkpoint_weight,
)
from nanovllm.utils.glm_tokenizer import (
    normalize_token_ids,
    require_glm_tokenizer_version,
)
from nanovllm.utils.loader import (
    dequantize_w8a8_weight,
    load_model,
    quant_tensor_types,
)


@pytest.mark.parametrize("version", ["5.5.0", "5.5.3", "5.6.0.dev0", "6.0.0"])
def test_glm_tokenizer_accepts_transformers_55_or_newer(version):
    require_glm_tokenizer_version(version)


@pytest.mark.parametrize("version", ["4.57.3", "5.4.0"])
def test_glm_tokenizer_rejects_old_transformers(version):
    with pytest.raises(RuntimeError, match="transformers==5.5.3"):
        require_glm_tokenizer_version(version)


def test_token_ids_are_extracted_from_v5_batch_encoding():
    class FakeEncoding:
        def __init__(self, ids):
            self.ids = ids

    class FakeBatchEncoding(dict):
        pass

    assert normalize_token_ids(FakeEncoding([11, 12, 13])) == [11, 12, 13]
    assert normalize_token_ids(
        FakeBatchEncoding(input_ids=[21, 22, 23])
    ) == [21, 22, 23]


def test_token_id_normalization_rejects_backend_encoding_elements():
    class FakeEncoding:
        ids = [31, 32]

    with pytest.raises(TypeError, match="must contain integers"):
        normalize_token_ids([FakeEncoding()])
    with pytest.raises(TypeError, match="must contain integers"):
        normalize_token_ids({"input_ids": [[41, 42]]})


def _write_glm_config(path, **overrides):
    raw = {
        "architectures": ["GlmMoeDsaForCausalLM"],
        "model_type": "glm_moe_dsa",
        "dtype": "bfloat16",
        "max_position_embeddings": 202752,
        "index_topk": 2048,
        "index_n_heads": 32,
        "index_head_dim": 128,
        "indexer_rope_interleave": True,
        "rope_parameters": {"rope_theta": 1_000_000},
        "rope_interleave": True,
        "eos_token_id": [154820, 154827, 154829],
        "num_hidden_layers": 78,
        "first_k_dense_replace": 3,
        "n_routed_experts": 256,
        "num_experts_per_tok": 8,
        "num_attention_heads": 64,
        "q_lora_rank": 2048,
        "kv_lora_rank": 512,
        "qk_rope_head_dim": 64,
    }
    raw.update(overrides)
    (path / "config.json").write_text(json.dumps(raw), encoding="utf-8")
    description = {
        "version": "1.0.0",
        "model_quant_type": "W8A8_DYNAMIC",
        "group_size": 0,
        "is_rot_used": True,
        "optional": {
            "quarot": {
                "rotation_map": {
                    "global_rotation": "optional/quarot.safetensors"
                }
            }
        },
    }
    (path / "quant_model_description.json").write_text(
        json.dumps(description), encoding="utf-8"
    )


def _write_glm52_config(path, **overrides):
    raw = {
        "head_dim": 192,
        "max_position_embeddings": 1_048_576,
        "rope_parameters": {
            "rope_theta": 8_000_000,
            "rope_type": "default",
        },
        "index_share_for_mtp_iteration": True,
        "index_skip_topk_offset": 3,
        "index_topk_freq": 4,
        "index_topk_pattern": None,
        "indexer_types": list(glm52_indexer_types(78)),
        "mlp_layer_types": ["dense"] * 3 + ["sparse"] * 75,
    }
    raw.update(overrides)
    _write_glm_config(path, **raw)


def _make_config(path, **overrides):
    kwargs = dict(
        max_model_len=2048,
        tensor_parallel_size=16,
        enable_expert_parallel=True,
        offload_mode="offload_split",
        enforce_eager=True,
        kvcache_block_size=128,
        num_hbm_kvcache_blocks=64,
        num_dram_kvcache_blocks=128,
    )
    kwargs.update(overrides)
    return Config(str(path), **kwargs)


def test_glm_config_is_loaded(tmp_path):
    _write_glm_config(
        tmp_path,
        q_lora_rank=2048,
        kv_lora_rank=512,
        first_k_dense_replace=3,
    )
    config = _make_config(tmp_path)

    assert config.glm_version == "5.1"
    assert config.glm_model_name == "GLM-5.1"
    assert config.hf_config.__class__.__name__ == "GlmMoeDsaConfig"
    assert config.hf_config.rope_parameters["rope_theta"] == 1_000_000
    assert config.hf_config.rope_parameters["rope_type"] == "default"
    assert config.eos == (154820, 154827, 154829)
    assert config.hf_config.max_position_embeddings == 2048
    assert (
        config.hf_config.nanovllm_original_max_position_embeddings
        == 202752
    )


def test_glm52_mtp0_nonoffload_eager_config_is_loaded(tmp_path):
    _write_glm52_config(tmp_path)

    config = _make_config(
        tmp_path,
        offload_mode="none",
        num_dram_kvcache_blocks=-1,
        enforce_eager=True,
        num_speculative_tokens=0,
        max_model_len=65536,
    )

    assert config.glm_version == "5.2"
    assert config.glm_model_name == "GLM-5.2"
    assert config.hf_config.nanovllm_glm_version == "5.2"
    assert config.hf_config.nanovllm_model_name == "GLM-5.2"
    assert config.hf_config.indexer_types.count("full") == 21
    assert config.hf_config.indexer_types.count("shared") == 57
    assert config.hf_config.rope_parameters["rope_theta"] == 8_000_000
    assert config.hf_config.max_position_embeddings == 65536
    assert (
        config.hf_config.nanovllm_original_max_position_embeddings
        == 1_048_576
    )


def test_glm52_stage3_accepts_offload_fuse(tmp_path):
    _write_glm52_config(tmp_path)

    config = _make_config(tmp_path, offload_mode="offload_fuse")

    assert config.offload_mode == "offload_fuse"
    assert config.glm_version == "5.2"


def test_glm52_phase1_rejects_mtp3(tmp_path):
    _write_glm52_config(tmp_path)

    with pytest.raises(ValueError, match="quantized MTP layer"):
        _make_config(
            tmp_path,
            offload_mode="none",
            num_dram_kvcache_blocks=-1,
            num_speculative_tokens=3,
        )


def test_glm52_phase1_rejects_full_decode_graph(tmp_path):
    _write_glm52_config(tmp_path)

    with pytest.raises(ValueError, match="offload graph is implemented in stage 4"):
        _make_config(
            tmp_path,
            offload_mode="offload_split",
            enforce_eager=False,
        )


def test_glm52_rejects_non_official_index_share_schedule(tmp_path):
    schedule = list(glm52_indexer_types(78))
    schedule[3] = "full"
    _write_glm52_config(tmp_path, indexer_types=schedule)

    with pytest.raises(ValueError, match="21-full/57-shared"):
        _make_config(
            tmp_path,
            offload_mode="none",
            num_dram_kvcache_blocks=-1,
        )


def test_glm_rejects_non_bf16_runtime(tmp_path):
    _write_glm_config(tmp_path, dtype="float16")
    with pytest.raises(ValueError, match="requires BF16 runtime dtype"):
        _make_config(tmp_path)


def test_glm_requires_expert_parallel(tmp_path):
    _write_glm_config(tmp_path)
    with pytest.raises(ValueError, match="requires expert parallel"):
        _make_config(tmp_path, enable_expert_parallel=False)


def test_glm_dsa_offload_accepts_context_above_index_topk(tmp_path):
    _write_glm_config(tmp_path)

    config = _make_config(tmp_path, max_model_len=16384)

    assert config.max_model_len == 16384
    assert config.hf_config.max_position_embeddings == 16384
    assert (
        config.hf_config.nanovllm_original_max_position_embeddings
        == 202752
    )
    assert config.hf_config.index_topk == 2048
    assert config.hf_config.index_n_heads == 32
    assert config.hf_config.indexer_rope_interleave is True


def test_glm_dsa_offload_keeps_native_indexer_at_index_topk(tmp_path):
    _write_glm_config(tmp_path)

    config = _make_config(tmp_path, max_model_len=2048)

    assert config.hf_config.index_topk == 2048


@pytest.mark.parametrize(
    "offload_mode",
    ["offload_split", "offload_fuse"],
)
def test_glm_offload_accepts_supported_cache_geometry(
    tmp_path,
    offload_mode,
):
    _write_glm_config(
        tmp_path,
        q_lora_rank=2048,
        kv_lora_rank=512,
        qk_rope_head_dim=64,
    )

    config = _make_config(
        tmp_path,
        offload_mode=offload_mode,
        max_model_len=16384,
        kvcache_block_size=128,
    )

    assert config.offload_mode == offload_mode
    assert config.hf_config.nanovllm_offload_mode == offload_mode
    assert not hasattr(
        config.hf_config,
        "nanovllm_enable_lidu_fused_attention_scatter",
    )


@pytest.mark.parametrize("batch_size", [24, 25, 48, 64])
def test_glm_offload_fuse_allows_full_decode_only(
    tmp_path,
    monkeypatch,
    batch_size,
):
    monkeypatch.delenv("ASCEND_LAUNCH_BLOCKING", raising=False)
    _write_glm_config(
        tmp_path,
        q_lora_rank=2048,
        kv_lora_rank=512,
        qk_rope_head_dim=64,
    )

    config = _make_config(
        tmp_path,
        offload_mode="offload_fuse",
        max_model_len=32768,
        kvcache_block_size=128,
        enforce_eager=False,
        max_num_decode_seqs_per_step=batch_size,
        decode_graph_capture_sizes=(batch_size,),
    )

    assert config.max_num_decode_seqs_per_step == batch_size
    assert config.decode_graph_capture_sizes == (batch_size,)


def test_glm_dense_mla_mode_needs_no_dram_cache(tmp_path):
    _write_glm_config(tmp_path)

    config = _make_config(
        tmp_path,
        offload_mode="none",
        num_dram_kvcache_blocks=-1,
        max_model_len=16384,
    )

    assert config.offload_mode == "none"
    assert config.hf_config.nanovllm_offload_mode == "none"


def test_glm_dense_mla_full_decode_graph_allows_short_context(tmp_path, monkeypatch):
    monkeypatch.delenv("ASCEND_LAUNCH_BLOCKING", raising=False)
    _write_glm_config(tmp_path)

    config = _make_config(
        tmp_path,
        offload_mode="none",
        num_dram_kvcache_blocks=-1,
        enforce_eager=False,
        max_model_len=512,
        decode_graph_capture_sizes=(3,),
    )

    assert config.decode_graph_capture_sizes == (3,)


def test_offload_mode_rejects_non_string(tmp_path):
    _write_glm_config(tmp_path)
    with pytest.raises(TypeError, match="offload_mode must be a string"):
        _make_config(tmp_path, offload_mode=1)


def test_glm_enables_full_decode_only(tmp_path, monkeypatch):
    monkeypatch.delenv("ASCEND_LAUNCH_BLOCKING", raising=False)
    _write_glm_config(tmp_path)

    config = _make_config(
        tmp_path,
        enforce_eager=False,
        decode_graph_capture_sizes=(3,),
    )

    assert config.enforce_eager is False
    assert config.decode_graph_capture_sizes == (3,)
    assert config.hf_config.num_hidden_layers == 78
    assert config.hf_config.max_position_embeddings == 2048
    assert config.hf_config.nanovllm_original_max_position_embeddings == 202752


def test_glm_weight_mapping_loads_native_dsa_indexer_and_skips_mtp():
    indexer_names = (
        "model.layers.3.self_attn.indexer.wq_b.weight",
        "model.layers.3.self_attn.indexer.wq_b.weight_scale",
        "model.layers.3.self_attn.indexer.wq_b.weight_offset",
        "model.layers.3.self_attn.indexer.wk.weight",
        "model.layers.3.self_attn.indexer.k_norm.weight",
        "model.layers.3.self_attn.indexer.k_norm.bias",
        "model.layers.3.self_attn.indexer.weights_proj.weight",
    )
    for name in indexer_names:
        assert not should_skip_glm_checkpoint_weight(name)
    assert should_skip_glm_checkpoint_weight(
        "model.layers.78.self_attn.indexer.wk.weight"
    )
    assert should_skip_glm_checkpoint_weight("rot.weight")


def test_glm_rejects_non_interleaved_indexer_rope(tmp_path):
    _write_glm_config(tmp_path, indexer_rope_interleave=False)
    with pytest.raises(ValueError, match="indexer_rope_interleave=true"):
        _make_config(tmp_path)


def test_eos_normalization_accepts_all_glm_stop_tokens():
    assert normalize_eos_token_ids([154820, 154827, 154829, 154820]) == (
        154820,
        154827,
        154829,
    )
    assert merge_eos_token_ids(
        [154820, 154827, 154829], 154820
    ) == (154820, 154827, 154829)

    scheduler_config = SimpleNamespace(
        max_num_prefill_seqs_per_step=1,
        prefill_chunk_size=0,
        max_num_decode_seqs_per_step=1,
        eos=[154820, 154827, 154829],
        num_hbm_kvcache_blocks=8,
        num_dram_kvcache_blocks=8,
        offload_mode="none",
        kvcache_block_size=16,
        max_model_len=64,
    )
    for eos in scheduler_config.eos:
        scheduler = Scheduler(scheduler_config)
        seq = Sequence(
            [1],
            SamplingParams(
                temperature=0.0, max_tokens=8, ignore_eos=False
            ),
            block_size=16,
        )
        assert scheduler._append_sampled_token(seq, eos)
        assert seq.finish_reason is FinishReason.EOS


def test_scheduler_stops_if_context_length_is_already_exceeded():
    scheduler_config = SimpleNamespace(
        max_num_prefill_seqs_per_step=1,
        prefill_chunk_size=0,
        max_num_decode_seqs_per_step=1,
        eos=[154820, 154827, 154829],
        num_hbm_kvcache_blocks=8,
        num_dram_kvcache_blocks=8,
        offload_mode="none",
        kvcache_block_size=16,
        max_model_len=4,
    )
    scheduler = Scheduler(scheduler_config)
    seq = Sequence(
        [1, 2, 3, 4],
        SamplingParams(temperature=0.0, max_tokens=8),
        block_size=16,
    )

    assert scheduler._append_sampled_token(seq, 5)
    assert seq.finish_reason is FinishReason.LENGTH


def test_glm_8200_prompt_really_crosses_dsa_offload_boundary():
    scheduler_config = SimpleNamespace(
        max_num_prefill_seqs_per_step=1,
        prefill_chunk_size=1024,
        max_num_decode_seqs_per_step=1,
        eos=[154820, 154827, 154829],
        num_hbm_kvcache_blocks=96,
        num_dram_kvcache_blocks=128,
        offload_mode="offload_split",
        kvcache_block_size=128,
        max_model_len=8224,
    )
    scheduler = Scheduler(scheduler_config)
    seq = Sequence(
        list(range(8200)),
        SamplingParams(temperature=0.0, max_tokens=8),
        block_size=128,
    )

    scheduler._prepare_prefill_metadata(seq)

    assert seq.num_prefill_blocks == 65
    assert seq.num_prefill_full_blocks == 64
    assert seq.num_prefill_tail_blocks == 1
    assert seq.prefill_tail_len == 8
    assert seq.num_sparse_blocks == 24
    assert seq.num_sparse_tokens == 3072
    candidate_len = seq.num_prefill_full_blocks * seq.block_size
    assert candidate_len == 8192
    assert candidate_len > 2048
    assert 2048 + seq.prefill_tail_len + 1 == 2057


def test_quant_description_filters_non_tensor_metadata():
    description = {
        "model.layers.0.self_attn.q_a_proj.weight": "W8A8_DYNAMIC",
        "model.layers.3.mlp.experts.0.gate_proj.weight": "W4A8_DYNAMIC",
        "model.norm.weight": "FLOAT",
        "model_quant_type": "W8A8_DYNAMIC",
        "version": "1.0.0",
    }
    assert quant_tensor_types(description) == {
        "model.layers.0.self_attn.q_a_proj.weight": "W8A8_DYNAMIC",
        "model.layers.3.mlp.experts.0.gate_proj.weight": "W4A8_DYNAMIC",
        "model.norm.weight": "FLOAT",
    }


def test_w8a8_loader_dequantizes_scale_and_offset(tmp_path):
    class TinyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = torch.nn.Linear(
                3, 2, bias=False, dtype=torch.bfloat16
            )

    weight = torch.tensor([[1, -2, 3], [-4, 5, -6]], dtype=torch.int8)
    scale = torch.tensor([[0.5], [0.25]], dtype=torch.float32)
    offset = torch.zeros_like(scale)
    save_file(
        {
            "proj.weight": weight,
            "proj.weight_scale": scale,
            "proj.weight_offset": offset,
        },
        str(tmp_path / "quant_model_weights-00001-of-00001.safetensors"),
    )
    description = {
        "proj.weight": "W8A8_DYNAMIC",
        "proj.weight_scale": "W8A8_DYNAMIC",
        "proj.weight_offset": "W8A8_DYNAMIC",
        "version": "1.0.0",
        "model_quant_type": "W8A8_DYNAMIC",
        "group_size": 0,
    }
    (tmp_path / "quant_model_description.json").write_text(
        json.dumps(description), encoding="utf-8"
    )
    model = TinyModel()
    load_model(model, str(tmp_path))
    expected = dequantize_w8a8_weight(weight, scale, offset).to(
        torch.bfloat16
    )
    torch.testing.assert_close(model.proj.weight, expected)


def test_w4a8_loader_rejects_asymmetric_offset(tmp_path):
    save_file(
        {
            "expert.weight_offset": torch.tensor(
                [[0.0], [1.0]], dtype=torch.float32
            )
        },
        str(tmp_path / "quant_model_weights-00001-of-00001.safetensors"),
    )
    description = {
        "expert.weight_offset": "W4A8_DYNAMIC",
        "version": "1.0.0",
        "model_quant_type": "W8A8_DYNAMIC",
        "group_size": 0,
    }
    (tmp_path / "quant_model_description.json").write_text(
        json.dumps(description), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="non-zero offset"):
        load_model(torch.nn.Module(), str(tmp_path))


def test_float_scale_is_bit_reinterpreted_not_numerically_cast():
    scale = torch.tensor([[[0.5, 1.0, 2.0]]], dtype=torch.float32)
    encoded = float32_scale_to_int64_bits(scale)
    assert encoded.dtype == torch.int64
    assert encoded.tolist() == [
        [[1056964608, 1065353216, 1073741824]]
    ]
    assert encoded.tolist() != scale.to(torch.int64).tolist()


def test_balanced_moe_warmup_routes_rotate_across_all_ep_ranks():
    first = balanced_moe_expert_ids(
        1, 8, 256, 16, route_offset=0, device="cpu"
    )
    second = balanced_moe_expert_ids(
        1, 8, 256, 16, route_offset=8, device="cpu"
    )

    assert first.shape == (1, 8)
    assert first.dtype == torch.int32
    all_ids = torch.cat((first.flatten(), second.flatten()))
    covered_ranks = torch.div(all_ids, 16, rounding_mode="floor")
    assert sorted(covered_ranks.tolist()) == list(range(16))


def test_balanced_moe_warmup_routes_cover_ep16_in_one_bs3_pass():
    expert_ids = balanced_moe_expert_ids(3, 8, 256, 16, device="cpu")
    covered_ranks = torch.div(
        expert_ids.flatten(), 16, rounding_mode="floor"
    ).unique()
    assert covered_ranks.tolist() == list(range(16))
