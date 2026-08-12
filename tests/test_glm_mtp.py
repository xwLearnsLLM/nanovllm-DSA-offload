import json
import pickle
from types import SimpleNamespace

import pytest
import torch

from nanovllm.config import Config, glm52_indexer_types
from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.dsa_offload import (
    finalize_prefill_hbm_layout,
    mtp_lidu_cache_tokens,
)
from nanovllm.engine.sequence import (
    DecodeBatchDelta,
    FinishReason,
    Sequence,
    SequenceStatus,
    SpeculativeStepOutput,
    apply_decode_batch_packet,
    build_decode_batch_packet,
)
from nanovllm.engine.speculative import (
    greedy_prefix_accept,
    materialize_accepted_tokens,
    shifted_mtp_prefill_tokens,
)
from nanovllm.sampling_params import SamplingParams


def _scheduler_config(*, max_model_len=256, block_size=8, eos=(-1,)):
    return SimpleNamespace(
        max_num_prefill_seqs_per_step=1,
        prefill_chunk_size=0,
        max_num_decode_seqs_per_step=8,
        eos=eos,
        num_hbm_kvcache_blocks=64,
        num_dram_kvcache_blocks=-1,
        offload_mode="none",
        kvcache_block_size=block_size,
        max_model_len=max_model_len,
        num_speculative_tokens=3,
    )


def _sequence(
    length=11,
    *,
    max_tokens=32,
    max_steps=None,
    ignore_eos=True,
    block_size=8,
    request_id="mtp",
):
    return Sequence(
        list(range(length)),
        SamplingParams(
            temperature=0.0,
            max_tokens=max_tokens,
            max_steps=max_steps,
            ignore_eos=ignore_eos,
        ),
        request_id=request_id,
        block_size=block_size,
    )


@pytest.mark.parametrize("value", [0, 3])
def test_num_speculative_tokens_accepts_disabled_or_mtp3(value):
    Config._validate_num_speculative_tokens(value)


@pytest.mark.parametrize("value", [1, 2, -1, 4, True, 1.0, "3"])
def test_num_speculative_tokens_rejects_other_values(value):
    with pytest.raises((TypeError, ValueError)):
        Config._validate_num_speculative_tokens(value)


@pytest.mark.parametrize("value", [None, 1, 2, 32])
def test_sampling_params_accepts_optional_positive_max_steps(value):
    assert SamplingParams(max_steps=value).max_steps == value


@pytest.mark.parametrize("value", [0, -1, True, 1.0, "3"])
def test_sampling_params_rejects_invalid_max_steps(value):
    with pytest.raises(ValueError, match="max_steps"):
        SamplingParams(max_steps=value)


@pytest.mark.parametrize(
    ("targets", "drafts", "counts", "next_tokens"),
    [
        ([[8, 9]], [[7]], [0], [8]),
        ([[7, 9]], [[7]], [1], [9]),
        ([[1, 9, 8], [1, 2, 8]], [[1, 2], [1, 2]], [1, 2], [9, 8]),
        (
            [[4, 5, 6, 7], [9, 5, 6, 7], [4, 9, 6, 7]],
            [[4, 5, 6], [4, 5, 6], [4, 5, 6]],
            [3, 0, 1],
            [7, 9, 9],
        ),
    ],
)
def test_greedy_prefix_accept_covers_mismatch_and_bonus(
    targets, drafts, counts, next_tokens
):
    actual_counts, actual_next = greedy_prefix_accept(
        torch.tensor(targets), torch.tensor(drafts)
    )
    assert actual_counts.tolist() == counts
    assert actual_next.tolist() == next_tokens


def test_materialize_accepted_tokens_commits_prefix_then_target():
    assert materialize_accepted_tokens(
        [[4, 5, 6], [4, 5, 6], [4, 5, 6]],
        [[4, 5, 6, 7], [9, 5, 6, 7], [4, 9, 6, 7]],
        [3, 0, 1],
    ) == [[4, 5, 6, 7], [9], [4, 9]]


def test_mtp_prefill_shift_crosses_chunks_without_losing_boundary_token():
    tokens = list(range(10_000))
    shifted = []
    for start in range(0, len(tokens), 1024):
        end = min(start + 1024, len(tokens))
        shifted.extend(
            shifted_mtp_prefill_tokens(
                tokens,
                start,
                end,
                sampled_token_id=99_999 if end == len(tokens) else None,
            )
        )
    assert shifted == tokens[1:] + [99_999]


def test_final_mtp_prefill_requires_target_sample_and_intermediate_forbids_it():
    with pytest.raises(ValueError, match="requires the target sampled token"):
        shifted_mtp_prefill_tokens([1, 2, 3], 0, 3)
    with pytest.raises(ValueError, match="must not have a sampled token"):
        shifted_mtp_prefill_tokens(
            [1, 2, 3], 0, 2, sampled_token_id=4
        )


def test_drafts_survive_sequence_and_compact_decode_ipc_round_trip():
    seq = _sequence(max_steps=7)
    seq.draft_token_ids = [101, 102, 103]
    seq.num_decode_steps = 2
    restored = pickle.loads(pickle.dumps(seq))
    assert restored.draft_token_ids == [101, 102, 103]
    assert restored.max_steps == 7
    assert restored.num_decode_steps == 2

    snapshot, key = build_decode_batch_packet([seq], None)
    cached, _ = apply_decode_batch_packet(snapshot, None)
    seq.append_token(101)
    seq.draft_token_ids = [201, 202, 203]
    delta, _ = build_decode_batch_packet([seq], key)
    assert isinstance(delta, DecodeBatchDelta)
    updated, _ = apply_decode_batch_packet(delta, cached)
    assert updated[0].last_token == 101
    assert updated[0].draft_token_ids == [201, 202, 203]


def test_scheduler_reserves_verify_and_worst_case_mtp_recurrence_slots():
    scheduler = Scheduler(_scheduler_config())
    seq = _sequence(length=11)
    scheduler.add(seq)

    seqs, is_prefill = scheduler.schedule()
    assert is_prefill and seqs == [seq]
    # Prefill needs positions 0..10 plus the initial K-token draft recurrence.
    assert len(seq.hbm_block_table) == 2
    scheduler.postprocess(
        seqs,
        SpeculativeStepOutput([[99]], [[1, 2, 3]], [0]),
        is_prefill=True,
    )
    assert seq.num_decode_steps == 0

    # L=12, K=3: target verification + full-accept recurrence can touch
    # through position L+2K-2=16, hence 17 slots / three 8-token blocks.
    assert scheduler._decode_growth_blocks(seq) == 1
    decode, is_prefill = scheduler.schedule()
    assert not is_prefill and decode == [seq]
    assert len(seq.hbm_block_table) == 3


@pytest.mark.parametrize(
    ("prompt_len", "expected"),
    [
        (2048, 0),
        (2049, 2048),
        (4097, 4096),
        (8192, 8192),
        (8193, 8192),
        (21_000, 8192),
        (65_537, 12_288),
    ],
)
def test_mtp_lidu_budget_can_hold_four_query_union(prompt_len, expected):
    assert mtp_lidu_cache_tokens(prompt_len, 128) == expected


def test_mtp_lidu_uses_independent_dense_mtp_block_pool():
    config = SimpleNamespace(
        max_num_prefill_seqs_per_step=1,
        prefill_chunk_size=0,
        max_num_decode_seqs_per_step=4,
        eos=(-1,),
        num_hbm_kvcache_blocks=64,
        num_dram_kvcache_blocks=64,
        offload_mode="offload_split",
        kvcache_block_size=128,
        max_model_len=4096,
        num_speculative_tokens=3,
    )
    scheduler = Scheduler(config)
    seq = _sequence(length=2050, block_size=128, request_id="mtp-lidu")
    scheduler.add(seq)

    seqs, is_prefill = scheduler.schedule()
    assert is_prefill and seqs == [seq]
    assert seq.lidu_cache_tokens == 2048
    assert len(seq.hbm_block_table) == 17
    assert len(seq.mtp_block_table) == 17
    mtp_blocks = tuple(seq.mtp_block_table)

    finalize_prefill_hbm_layout(seq, "offload_split")
    scheduler.release_prefill_hbm_blocks([seq])
    assert tuple(seq.mtp_block_table) == mtp_blocks
    scheduler.postprocess(
        [seq],
        SpeculativeStepOutput([[99]], [[1, 2, 3]], [0]),
        is_prefill=True,
    )
    decode, is_prefill = scheduler.schedule()
    assert not is_prefill and decode == [seq]
    assert tuple(seq.mtp_block_table) == mtp_blocks

    scheduler.deallocate(seq)
    assert not seq.hbm_block_table
    assert not seq.mtp_block_table
    assert not scheduler.mtp_block_manager.used_block_ids


def test_mtp_index_share_owns_independent_pool_row_and_cache_blocks():
    config = _scheduler_config()
    config.hf_config = SimpleNamespace(index_share_for_mtp_iteration=True)
    scheduler = Scheduler(config)
    seq = _sequence(length=17, block_size=8, request_id="mtp-index-share")
    scheduler.add(seq)

    seqs, is_prefill = scheduler.schedule()

    assert is_prefill and seqs == [seq]
    assert len(seq.hbm_block_table) == 3
    assert len(seq.mtp_block_table) == 3
    assert scheduler.mtp_block_manager is not scheduler.hbm_block_manager
    assert seq.mtp_index_pool_entry >= 0
    assert seq.mtp_lidu_cache_tokens == 16
    assert not seq.mtp_lidu_cache_initialized

    scheduler.deallocate(seq)

    assert seq.mtp_index_pool_entry == -1
    assert not seq.mtp_block_table
    assert scheduler.mtp_index_pool_entry_manager.used_entries == set()


def test_scheduler_commits_multiple_tokens_and_truncates_at_max_tokens():
    scheduler = Scheduler(_scheduler_config())
    seq = _sequence(max_tokens=3)
    scheduler._prepare_prefill_metadata(seq)
    scheduler._allocate_prefill(seq)
    seq.num_prefill_tokens_processed = len(seq)
    seq.status = SequenceStatus.RUNNING
    scheduler.running.append(seq)

    scheduler.postprocess(
        [seq],
        SpeculativeStepOutput(
            token_ids=[[10, 11, 12, 13]],
            draft_token_ids=[[20, 21, 22]],
            accepted_draft_counts=[3],
        ),
        is_prefill=False,
    )

    assert seq.completion_token_ids == [10, 11, 12]
    assert seq.is_finished
    assert seq.draft_token_ids == []
    assert scheduler.last_speculative_stats["emitted_tokens"] == 3
    assert scheduler.last_speculative_stats["accepted_drafts"] == 3


def test_mtp_max_steps_commits_the_complete_final_step():
    scheduler = Scheduler(_scheduler_config())
    seq = _sequence(max_tokens=16, max_steps=1)
    scheduler._prepare_prefill_metadata(seq)
    scheduler._allocate_prefill(seq)
    seq.num_prefill_tokens_processed = len(seq)
    seq.status = SequenceStatus.RUNNING
    scheduler.running.append(seq)

    scheduler.postprocess(
        [seq],
        SpeculativeStepOutput(
            token_ids=[[10, 11, 12, 13]],
            draft_token_ids=[[20, 21, 22]],
            accepted_draft_counts=[3],
        ),
        is_prefill=False,
    )

    assert seq.completion_token_ids == [10, 11, 12, 13]
    assert seq.num_decode_steps == 1
    assert seq.is_finished
    assert seq.finish_reason is FinishReason.LENGTH


def test_mtp_max_steps_finishes_mixed_acceptance_batch_together():
    scheduler = Scheduler(_scheduler_config())
    seqs = [
        _sequence(max_tokens=32, max_steps=2, request_id="short-accept"),
        _sequence(max_tokens=32, max_steps=2, request_id="full-accept"),
    ]
    for seq in seqs:
        scheduler._prepare_prefill_metadata(seq)
        scheduler._allocate_prefill(seq)
        seq.num_prefill_tokens_processed = len(seq)
        seq.status = SequenceStatus.RUNNING
        scheduler.running.append(seq)

    for step in range(2):
        scheduler.postprocess(
            seqs,
            SpeculativeStepOutput(
                token_ids=[[10 + step], [20, 21, 22, 23]],
                draft_token_ids=[[30, 31, 32], [40, 41, 42]],
                accepted_draft_counts=[0, 3],
            ),
            is_prefill=False,
        )
        assert [seq.num_decode_steps for seq in seqs] == [step + 1] * 2
        assert [seq.is_finished for seq in seqs] == [step == 1] * 2

    assert [len(seq.completion_token_ids) for seq in seqs] == [2, 8]
    assert not scheduler.running


def test_non_mtp_decode_honors_max_steps_without_counting_prefill():
    scheduler = Scheduler(_scheduler_config())
    seq = _sequence(max_tokens=16, max_steps=2)
    scheduler._prepare_prefill_metadata(seq)
    scheduler._allocate_prefill(seq)
    seq.status = SequenceStatus.RUNNING
    scheduler.running.append(seq)

    scheduler.postprocess([seq], [9], is_prefill=True)
    assert seq.num_decode_steps == 0
    scheduler.postprocess([seq], [10], is_prefill=False)
    assert seq.num_decode_steps == 1
    assert not seq.is_finished
    scheduler.postprocess([seq], [11], is_prefill=False)
    assert seq.num_decode_steps == 2
    assert seq.is_finished


def test_scheduler_stops_inside_an_accepted_prefix_at_eos():
    scheduler = Scheduler(_scheduler_config(eos=(11,)))
    seq = _sequence(ignore_eos=False)
    scheduler._prepare_prefill_metadata(seq)
    scheduler._allocate_prefill(seq)
    seq.num_prefill_tokens_processed = len(seq)
    seq.status = SequenceStatus.RUNNING
    scheduler.running.append(seq)

    scheduler.postprocess(
        [seq],
        SpeculativeStepOutput(
            token_ids=[[10, 11, 12]],
            draft_token_ids=[[20, 21, 22]],
            accepted_draft_counts=[2],
        ),
        is_prefill=False,
    )
    assert seq.completion_token_ids == [10, 11]
    assert seq.is_finished
    assert scheduler.last_speculative_stats["emitted_tokens"] == 2


def test_preemption_clears_drafts_and_rebuilds_mtp_prefill_state():
    scheduler = Scheduler(_scheduler_config())
    seq = _sequence()
    scheduler._prepare_prefill_metadata(seq)
    scheduler._allocate_prefill(seq)
    seq.draft_token_ids = [1, 2, 3]
    seq.status = SequenceStatus.RUNNING

    scheduler.preempt(seq)

    assert seq.draft_token_ids == []
    assert seq.num_prefill_tokens_processed == 0
    assert list(scheduler.waiting) == [seq]


def test_tail_decode_is_prioritized_and_not_mixed_with_mtp_ready_rows():
    scheduler = Scheduler(_scheduler_config(max_model_len=20))
    mtp = _sequence(length=8, request_id="mtp")
    tail = _sequence(length=16, request_id="tail")
    for seq in (mtp, tail):
        scheduler._prepare_prefill_metadata(seq)
        scheduler._allocate_prefill(seq)
        seq.num_prefill_tokens_processed = len(seq)
        seq.status = SequenceStatus.RUNNING
        seq.draft_token_ids = [1, 2, 3]
        scheduler.running.append(seq)

    scheduled, is_prefill = scheduler.schedule()
    assert not is_prefill
    assert scheduled == [tail]
    assert tail.draft_token_ids == []
    assert mtp.draft_token_ids == [1, 2, 3]


def _write_mtp_config(path, *, is_rot_used=True):
    config = {
        "architectures": ["GlmMoeDsaForCausalLM"],
        "model_type": "glm_moe_dsa",
        "dtype": "bfloat16",
        "max_position_embeddings": 202752,
        "index_topk": 2048,
        "index_n_heads": 32,
        "index_head_dim": 128,
        "rope_interleave": True,
        "num_hidden_layers": 78,
        "num_nextn_predict_layers": 1,
        "first_k_dense_replace": 3,
        "n_routed_experts": 256,
        "num_experts_per_tok": 8,
        "num_attention_heads": 64,
        "kv_lora_rank": 512,
        "qk_rope_head_dim": 64,
    }
    (path / "config.json").write_text(json.dumps(config), encoding="utf-8")
    description = {
        "version": "1.0.0",
        "model_quant_type": "W8A8_DYNAMIC",
        "group_size": 0,
        "is_rot_used": is_rot_used,
        "model.layers.78.enorm.weight": "FLOAT",
        "model.layers.78.hnorm.weight": "FLOAT",
    }
    (path / "quant_model_description.json").write_text(
        json.dumps(description), encoding="utf-8"
    )
    (path / "rot.safetensors").write_bytes(b"config-only-test")


def _write_glm52_mtp_config(path):
    _write_mtp_config(path)
    config_path = path / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config.update(
        head_dim=192,
        max_position_embeddings=1_048_576,
        rope_parameters={"rope_theta": 8_000_000, "rope_type": "default"},
        indexer_types=list(glm52_indexer_types(78)),
        mlp_layer_types=["dense"] * 3 + ["sparse"] * 75,
    )
    config_path.write_text(json.dumps(config), encoding="utf-8")

    description_path = path / "quant_model_description.json"
    description = json.loads(description_path.read_text(encoding="utf-8"))
    description.update(
        {
            "model.layers.78.self_attn.q_a_proj.weight": "W8A8_DYNAMIC",
            "model.layers.78.mlp.experts.0.gate_proj.weight": "W4A8_DYNAMIC",
            "model.layers.78.mlp.experts.0.gate_proj.weight_scale": "W4A8_DYNAMIC",
            "model.layers.78.mlp.experts.0.gate_proj.scale_bias": "W4A8_DYNAMIC",
        }
    )
    description_path.write_text(json.dumps(description), encoding="utf-8")


def _mtp_config(path, **overrides):
    kwargs = dict(
        max_model_len=256,
        tensor_parallel_size=16,
        enable_expert_parallel=True,
        offload_mode="none",
        enforce_eager=True,
        num_hbm_kvcache_blocks=64,
        num_dram_kvcache_blocks=-1,
        num_speculative_tokens=3,
    )
    kwargs.update(overrides)
    return Config(str(path), **kwargs)


@pytest.mark.parametrize("enforce_eager", [True, False])
def test_glm_mtp_runtime_accepts_nonoffload_k3(tmp_path, enforce_eager):
    _write_mtp_config(tmp_path)
    config = _mtp_config(tmp_path, enforce_eager=enforce_eager)
    assert config.num_speculative_tokens == 3
    assert config.hf_config.nanovllm_num_speculative_tokens == 3


@pytest.mark.parametrize("enforce_eager", [True, False])
@pytest.mark.parametrize("offload_mode", ["offload_split", "offload_fuse"])
def test_glm_mtp_runtime_accepts_offload(
    tmp_path, enforce_eager, offload_mode
):
    _write_mtp_config(tmp_path)
    config = _mtp_config(
        tmp_path,
        offload_mode=offload_mode,
        enforce_eager=enforce_eager,
        kvcache_block_size=128,
        num_dram_kvcache_blocks=64,
    )
    assert config.offload_mode == offload_mode
    assert config.decode_graph_capture_sizes == (
        () if enforce_eager else (config.max_num_decode_seqs_per_step,)
    )

@pytest.mark.parametrize("k", [1, 2])
@pytest.mark.parametrize("enforce_eager", [True, False])
def test_glm_mtp_rejects_k1_and_k2(tmp_path, k, enforce_eager):
    _write_mtp_config(tmp_path)
    with pytest.raises(ValueError, match="either 0 or 3"):
        _mtp_config(
            tmp_path,
            enforce_eager=enforce_eager,
            num_speculative_tokens=k,
        )


def test_glm_mtp_requires_root_rotation_and_float_layer_78(tmp_path):
    _write_mtp_config(tmp_path)
    (tmp_path / "rot.safetensors").unlink()
    with pytest.raises(ValueError, match="root-level rot.safetensors"):
        _mtp_config(tmp_path)

    _write_mtp_config(tmp_path)
    description_path = tmp_path / "quant_model_description.json"
    description = json.loads(description_path.read_text(encoding="utf-8"))
    description["model.layers.78.hnorm.weight"] = "W8A8_DYNAMIC"
    description_path.write_text(json.dumps(description), encoding="utf-8")
    with pytest.raises(ValueError, match="FLOAT/BF16"):
        _mtp_config(tmp_path)


def test_glm52_mtp_runtime_accepts_quantized_nonoffload_eager(tmp_path):
    _write_glm52_mtp_config(tmp_path)

    config = _mtp_config(tmp_path)

    assert config.glm_version == "5.2"
    assert config.num_speculative_tokens == 3
    assert config.hf_config.nanovllm_mtp_uses_w4a8_experts is True


def test_glm52_mtp_runtime_rejects_offload(tmp_path):
    _write_glm52_mtp_config(tmp_path)

    with pytest.raises(ValueError, match="MTP offload"):
        _mtp_config(
            tmp_path,
            offload_mode="offload_split",
            num_dram_kvcache_blocks=64,
        )


def test_glm52_mtp_runtime_accepts_nonoffload_graph(tmp_path):
    _write_glm52_mtp_config(tmp_path)

    config = _mtp_config(
        tmp_path,
        enforce_eager=False,
        max_num_decode_seqs_per_step=1,
    )

    assert config.decode_graph_capture_sizes == (1,)


def test_glm52_mtp_runtime_requires_w4a8_routed_experts(tmp_path):
    _write_glm52_mtp_config(tmp_path)
    description_path = tmp_path / "quant_model_description.json"
    description = json.loads(description_path.read_text(encoding="utf-8"))
    description["model.layers.78.mlp.experts.0.gate_proj.weight"] = "FLOAT"
    description_path.write_text(json.dumps(description), encoding="utf-8")

    with pytest.raises(ValueError, match="W4A8_DYNAMIC"):
        _mtp_config(tmp_path)
