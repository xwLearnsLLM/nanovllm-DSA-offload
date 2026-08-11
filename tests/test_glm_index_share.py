import json
import re
from types import SimpleNamespace

import pytest

from nanovllm.config import (
    Config,
    GLM_VERSION_51,
    GLM_VERSION_52,
    glm52_indexer_types,
)
from nanovllm.engine.dsa_offload import (
    IndexShareGroup,
    IndexShareGroupManager,
    OFFLOAD_NONE,
)
from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.sequence import FinishReason, Sequence
from nanovllm.sampling_params import SamplingParams


# Mirror of _INDEXER_WEIGHT_RE in glm_moe_dsa.py; kept local to avoid
# importing torch_npu on CPU-only test machines.
_INDEXER_WEIGHT_RE = re.compile(
    r"^model\.layers\.(?P<layer>\d+)\.self_attn\.indexer\."
)


# ---------------------------------------------------------------------------
# Test helpers (mirroring test_glm_dsa_offload.py conventions)
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# IndexShareGroupManager: GLM-5.2 topology
# ---------------------------------------------------------------------------


def test_glm52_group_manager_builds_21_groups():
    types = tuple(glm52_indexer_types(78))
    mgr = IndexShareGroupManager(78, types)

    assert mgr.num_hidden_layers == 78
    assert mgr.num_groups == 21
    assert len(mgr.owner_layer_idxs) == 21
    assert len(mgr.shared_layer_idxs) == 57


def test_glm52_group_manager_owner_layers_match_official_schedule():
    types = tuple(glm52_indexer_types(78))
    mgr = IndexShareGroupManager(78, types)

    expected_owners = tuple(
        idx for idx, t in enumerate(types) if t == "full"
    )
    assert mgr.owner_layer_idxs == expected_owners
    # Official schedule: 0, 1, 2, 6, 10, 14, ..., 74
    assert mgr.owner_layer_idxs == (
        0, 1, 2, 6, 10, 14, 18, 22, 26, 30,
        34, 38, 42, 46, 50, 54, 58, 62, 66, 70, 74,
    )


def test_glm52_group_manager_shared_layers_match_official_schedule():
    types = tuple(glm52_indexer_types(78))
    mgr = IndexShareGroupManager(78, types)

    expected_shared = tuple(
        idx for idx, t in enumerate(types) if t == "shared"
    )
    assert mgr.shared_layer_idxs == expected_shared


def test_glm52_group_manager_group_membership():
    types = tuple(glm52_indexer_types(78))
    mgr = IndexShareGroupManager(78, types)

    # First two groups: single-member (layers 0 and 1)
    assert mgr.group(0) == IndexShareGroup(
        group_id=0, owner_layer_idx=0, member_layer_idxs=(0,)
    )
    assert mgr.group(1) == IndexShareGroup(
        group_id=1, owner_layer_idx=1, member_layer_idxs=(1,)
    )
    # Third group: layers 2, 3, 4, 5
    assert mgr.group(2) == IndexShareGroup(
        group_id=2, owner_layer_idx=2, member_layer_idxs=(2, 3, 4, 5)
    )
    # Last group: layers 74, 75, 76, 77
    last_group = mgr.group(mgr.num_groups - 1)
    assert last_group.owner_layer_idx == 74
    assert last_group.member_layer_idxs == (74, 75, 76, 77)


def test_glm52_group_manager_all_members_cover_78_layers():
    types = tuple(glm52_indexer_types(78))
    mgr = IndexShareGroupManager(78, types)

    all_members = set()
    for group in mgr.groups():
        all_members.update(group.member_layer_idxs)
    assert all_members == set(range(78))


def test_glm52_group_manager_layer_to_owner_mapping():
    types = tuple(glm52_indexer_types(78))
    mgr = IndexShareGroupManager(78, types)

    # Full layers own themselves
    for owner_idx in mgr.owner_layer_idxs:
        assert mgr.owner_of(owner_idx) == owner_idx
        assert mgr.is_owner(owner_idx) is True
        assert mgr.is_shared(owner_idx) is False

    # Shared layers map to the nearest preceding full layer
    assert mgr.owner_of(3) == 2
    assert mgr.owner_of(4) == 2
    assert mgr.owner_of(5) == 2
    assert mgr.owner_of(7) == 6
    assert mgr.owner_of(9) == 6
    assert mgr.owner_of(75) == 74
    assert mgr.owner_of(77) == 74

    # Shared layers are not owners
    for shared_idx in mgr.shared_layer_idxs:
        assert mgr.is_owner(shared_idx) is False
        assert mgr.is_shared(shared_idx) is True


def test_glm52_group_manager_group_of_mapping():
    types = tuple(glm52_indexer_types(78))
    mgr = IndexShareGroupManager(78, types)

    # Each member maps to the correct group
    for group in mgr.groups():
        for layer_idx in group.member_layer_idxs:
            assert mgr.group_of(layer_idx) == group.group_id


def test_glm52_group_manager_group_state_uniqueness():
    types = tuple(glm52_indexer_types(78))
    mgr = IndexShareGroupManager(78, types)

    group_ids = [g.group_id for g in mgr.groups()]
    assert group_ids == list(range(21))

    owners = [g.owner_layer_idx for g in mgr.groups()]
    assert len(set(owners)) == 21  # all unique

    # No layer appears in multiple groups
    all_members = []
    for g in mgr.groups():
        all_members.extend(g.member_layer_idxs)
    assert len(all_members) == len(set(all_members)) == 78


# ---------------------------------------------------------------------------
# IndexShareGroupManager: GLM-5.1 degradation
# ---------------------------------------------------------------------------


def test_glm51_group_manager_degrades_to_single_layer_groups():
    mgr = IndexShareGroupManager(78, indexer_types=None)

    assert mgr.num_groups == 78
    assert len(mgr.owner_layer_idxs) == 78
    assert len(mgr.shared_layer_idxs) == 0

    # Every layer is its own owner and sole member
    for layer_idx in range(78):
        assert mgr.is_owner(layer_idx) is True
        assert mgr.is_shared(layer_idx) is False
        assert mgr.owner_of(layer_idx) == layer_idx
        assert mgr.group_of(layer_idx) == layer_idx
        group = mgr.group(layer_idx)
        assert group.member_layer_idxs == (layer_idx,)


def test_glm51_group_manager_preserves_original_behaviour():
    """GLM-5.1 has no indexer_types; every layer independently runs LIM."""

    mgr = IndexShareGroupManager(78, indexer_types=None)

    # All 78 layers are owners -> 78 LIM calls in the original schedule
    assert mgr.num_groups == 78
    assert all(mgr.is_owner(i) for i in range(78))
    assert mgr.shared_layer_idxs == ()


# ---------------------------------------------------------------------------
# IndexShareGroupManager: error handling
# ---------------------------------------------------------------------------


def test_group_manager_rejects_length_mismatch():
    with pytest.raises(ValueError, match="length must match"):
        IndexShareGroupManager(78, ("full", "shared"))


def test_group_manager_rejects_unknown_type():
    with pytest.raises(ValueError, match="Unknown indexer_type"):
        IndexShareGroupManager(4, ("full", "shared", "invalid", "shared"))


def test_group_manager_rejects_shared_without_preceding_full():
    with pytest.raises(ValueError, match="no preceding full"):
        IndexShareGroupManager(2, ("shared", "full"))


def test_group_manager_rejects_empty_types():
    with pytest.raises(ValueError, match="produced no IndexShare groups"):
        IndexShareGroupManager(0, ())


# ---------------------------------------------------------------------------
# Config integration
# ---------------------------------------------------------------------------


def test_glm52_config_builds_index_share_groups(tmp_path):
    _write_glm52_config(tmp_path)
    config = _make_config(
        tmp_path,
        offload_mode="none",
        num_dram_kvcache_blocks=-1,
        enforce_eager=True,
    )

    groups = getattr(config.hf_config, "nanovllm_index_share_groups", None)
    assert groups is not None
    assert isinstance(groups, IndexShareGroupManager)
    assert groups.num_groups == 21
    assert len(groups.shared_layer_idxs) == 57
    assert config.glm_version == GLM_VERSION_52


def test_glm51_config_builds_degenerate_index_share_groups(tmp_path):
    _write_glm_config(tmp_path)
    config = _make_config(tmp_path, max_model_len=2048)

    groups = getattr(config.hf_config, "nanovllm_index_share_groups", None)
    assert groups is not None
    assert isinstance(groups, IndexShareGroupManager)
    assert groups.num_groups == 78
    assert groups.shared_layer_idxs == ()
    assert config.glm_version == GLM_VERSION_51


def test_glm52_config_accepts_offload_split_in_stage2(tmp_path):
    _write_glm52_config(tmp_path)
    config = _make_config(
        tmp_path,
        offload_mode="offload_split",
        max_model_len=4096,
    )
    assert config.offload_mode == "offload_split"
    assert config.glm_version == GLM_VERSION_52


def test_glm52_config_rejects_offload_fuse_in_stage2(tmp_path):
    _write_glm52_config(tmp_path)
    with pytest.raises(ValueError, match="offload_fuse is implemented in stage 3"):
        _make_config(tmp_path, offload_mode="offload_fuse")


def test_glm52_config_rejects_graph_with_offload_in_stage2(tmp_path):
    _write_glm52_config(tmp_path)
    with pytest.raises(ValueError, match="offload graph is implemented in stage 4"):
        _make_config(
            tmp_path,
            offload_mode="offload_split",
            enforce_eager=False,
        )


# ---------------------------------------------------------------------------
# Weight name mapping: shared layer indexer skip
# ---------------------------------------------------------------------------


class _MockModelForWeightMapping:
    """Minimal stand-in for GlmMoeDsaForCausalLM.weight_name_mapping."""

    def __init__(self, config):
        self.config = config
        self.mtp = None

    def weight_name_mapping(self, weight_name):
        if ".self_attn.indexer." in weight_name:
            if (
                getattr(
                    self.config, "nanovllm_offload_mode", OFFLOAD_NONE
                ) == OFFLOAD_NONE
            ):
                return None
            index_share_groups = getattr(
                self.config, "nanovllm_index_share_groups", None
            )
            if index_share_groups is not None:
                indexer_match = _INDEXER_WEIGHT_RE.match(weight_name)
                if indexer_match is not None:
                    layer_idx = int(indexer_match.group("layer"))
                    if layer_idx < int(self.config.num_hidden_layers):
                        if not index_share_groups.is_owner(layer_idx):
                            return None
        return weight_name


def test_weight_mapping_skips_shared_layer_indexer_weights(tmp_path):
    _write_glm52_config(tmp_path)
    # Build a valid phase-1 config first, then override the offload mode
    # on hf_config to test the weight-mapping logic without the phase-1
    # runtime restriction.
    config = _make_config(
        tmp_path,
        offload_mode="none",
        num_dram_kvcache_blocks=-1,
        enforce_eager=True,
    )
    hf_config = config.hf_config
    setattr(hf_config, "nanovllm_offload_mode", "offload_split")
    model = _MockModelForWeightMapping(hf_config)

    # Owner layer indexer weights are loaded
    assert model.weight_name_mapping(
        "model.layers.0.self_attn.indexer.wq_b.weight"
    ) == "model.layers.0.self_attn.indexer.wq_b.weight"
    assert model.weight_name_mapping(
        "model.layers.6.self_attn.indexer.wk.weight"
    ) == "model.layers.6.self_attn.indexer.wk.weight"

    # Shared layer indexer weights are skipped
    assert model.weight_name_mapping(
        "model.layers.3.self_attn.indexer.wq_b.weight"
    ) is None
    assert model.weight_name_mapping(
        "model.layers.5.self_attn.indexer.wk.weight"
    ) is None
    assert model.weight_name_mapping(
        "model.layers.77.self_attn.indexer.weights_proj.weight"
    ) is None


def test_weight_mapping_loads_all_indexer_weights_for_glm51(tmp_path):
    _write_glm_config(tmp_path)
    config = _make_config(tmp_path, max_model_len=2048)
    # Config already has offload_mode="offload_split" from _make_config
    hf_config = config.hf_config
    model = _MockModelForWeightMapping(hf_config)

    # GLM-5.1 with offload: all layers are owners, all indexer weights load
    for layer_idx in [0, 3, 39, 77]:
        name = f"model.layers.{layer_idx}.self_attn.indexer.wq_b.weight"
        assert model.weight_name_mapping(name) == name


def test_weight_mapping_skips_all_indexer_weights_when_offload_none(tmp_path):
    _write_glm52_config(tmp_path)
    config = _make_config(
        tmp_path,
        offload_mode="none",
        num_dram_kvcache_blocks=-1,
        enforce_eager=True,
    )
    hf_config = config.hf_config
    model = _MockModelForWeightMapping(hf_config)

    # offload_mode=none: all indexer weights are skipped (both owner and shared)
    for layer_idx in [0, 3, 6, 77]:
        name = f"model.layers.{layer_idx}.self_attn.indexer.wq_b.weight"
        assert model.weight_name_mapping(name) is None


# ---------------------------------------------------------------------------
# Scheduler lifecycle: group manager is compatible with existing behaviour
# ---------------------------------------------------------------------------


def _make_scheduler_config(**overrides):
    raw = dict(
        max_num_prefill_seqs_per_step=1,
        prefill_chunk_size=0,
        max_num_decode_seqs_per_step=4,
        eos=[154820, 154827, 154829],
        num_hbm_kvcache_blocks=64,
        num_dram_kvcache_blocks=128,
        offload_mode="offload_split",
        kvcache_block_size=128,
        max_model_len=2048,
    )
    raw.update(overrides)
    return SimpleNamespace(**raw)


def test_scheduler_deallocate_frees_pool_entry_with_index_share_groups():
    """Pool entry lifecycle is unchanged when IndexShare groups are present.

    Stage 1 keeps the single per-request pool entry.  Group-level pool
    entries are introduced in stage 2.  This test verifies that the
    existing scheduler lifecycle (allocate, deallocate, reuse) is not
    broken by the group manager's presence.
    """

    mgr = IndexShareGroupManager(78, tuple(glm52_indexer_types(78)))
    assert mgr.num_groups == 21

    scheduler_config = _make_scheduler_config()
    scheduler = Scheduler(scheduler_config)

    seq = Sequence(
        list(range(256)),
        SamplingParams(temperature=0.0, max_tokens=8),
        block_size=128,
    )
    scheduler._prepare_prefill_metadata(seq)
    scheduler._allocate_prefill(seq)

    assert seq.offload_pool_entry >= 0
    entry = seq.offload_pool_entry
    assert entry in scheduler.pool_entry_manager.used_entries

    scheduler.deallocate(seq)

    assert seq.offload_pool_entry == -1
    assert entry not in scheduler.pool_entry_manager.used_entries
    assert entry in scheduler.pool_entry_manager.free_entries


def test_scheduler_preempt_recycles_pool_entry():
    mgr = IndexShareGroupManager(78, tuple(glm52_indexer_types(78)))
    assert mgr.num_groups == 21

    scheduler_config = _make_scheduler_config()
    scheduler = Scheduler(scheduler_config)

    seq = Sequence(
        list(range(256)),
        SamplingParams(temperature=0.0, max_tokens=8),
        block_size=128,
    )
    scheduler._prepare_prefill_metadata(seq)
    scheduler._allocate_prefill(seq)
    entry = seq.offload_pool_entry

    scheduler.preempt(seq)

    assert seq.status.name == "WAITING"
    assert seq.offload_pool_entry == -1
    assert entry in scheduler.pool_entry_manager.free_entries


def test_scheduler_abort_frees_pool_entry():
    mgr = IndexShareGroupManager(78, tuple(glm52_indexer_types(78)))
    assert mgr.num_groups == 21

    scheduler_config = _make_scheduler_config()
    scheduler = Scheduler(scheduler_config)

    seq = Sequence(
        list(range(256)),
        SamplingParams(temperature=0.0, max_tokens=8),
        block_size=128,
        request_id="req-1",
    )
    scheduler.add(seq)
    scheduler._prepare_prefill_metadata(seq)
    scheduler._allocate_prefill(seq)
    scheduler.running.append(seq)
    entry = seq.offload_pool_entry

    scheduler.abort_seq_group("req-1")

    assert seq.status.name == "FINISHED"
    assert seq.finish_reason is FinishReason.ABORTED
    assert entry in scheduler.pool_entry_manager.free_entries


def test_scheduler_pool_entry_reuse_after_deallocate():
    mgr = IndexShareGroupManager(78, tuple(glm52_indexer_types(78)))
    assert mgr.num_groups == 21

    scheduler_config = _make_scheduler_config(max_num_decode_seqs_per_step=2)
    scheduler = Scheduler(scheduler_config)

    seq1 = Sequence(
        list(range(256)),
        SamplingParams(temperature=0.0, max_tokens=8),
        block_size=128,
    )
    scheduler._prepare_prefill_metadata(seq1)
    scheduler._allocate_prefill(seq1)
    entry1 = seq1.offload_pool_entry
    assert entry1 in scheduler.pool_entry_manager.used_entries

    scheduler.deallocate(seq1)
    assert entry1 not in scheduler.pool_entry_manager.used_entries
    assert entry1 in scheduler.pool_entry_manager.free_entries

    seq2 = Sequence(
        list(range(256)),
        SamplingParams(temperature=0.0, max_tokens=8),
        block_size=128,
    )
    scheduler._prepare_prefill_metadata(seq2)
    scheduler._allocate_prefill(seq2)
    entry2 = seq2.offload_pool_entry
    assert entry2 in scheduler.pool_entry_manager.used_entries

    # The freed entry is available for reuse; PoolEntryManager uses a deque
    # with popleft/append, so the exact reuse order depends on how many
    # entries were already free.  Verify the entry is valid and the total
    # used count is correct.
    assert entry2 >= 0
    assert len(scheduler.pool_entry_manager.used_entries) == 1


def test_scheduler_batch_reorder_preserves_group_mapping():
    """Batch reorder does not affect the group topology.

    The IndexShareGroupManager is a static topology that maps layer
    indices to groups.  Request scheduling order (batch reorder) does
    not change which layers belong to which group.
    """

    mgr = IndexShareGroupManager(78, tuple(glm52_indexer_types(78)))

    # Simulate batch reorder: requests in different order
    seq_ids = [10, 20, 30, 40]
    for seq_id in seq_ids:
        # Every request sees the same layer->owner mapping
        assert mgr.owner_of(0) == 0
        assert mgr.owner_of(3) == 2
        assert mgr.owner_of(77) == 74
        assert mgr.is_owner(6) is True
        assert mgr.is_shared(7) is True


# ---------------------------------------------------------------------------
# GLM-5.1 regression: group manager does not change existing behaviour
# ---------------------------------------------------------------------------


def test_glm51_group_manager_has_no_shared_layers():
    mgr = IndexShareGroupManager(78, indexer_types=None)
    assert mgr.shared_layer_idxs == ()
    assert mgr.num_groups == 78


def test_glm51_pool_entry_lifecycle_unchanged():
    """GLM-5.1 pool entry lifecycle is identical with or without the
    group manager (which degrades to 78 single-layer groups)."""

    mgr = IndexShareGroupManager(78, indexer_types=None)
    assert mgr.num_groups == 78

    scheduler_config = _make_scheduler_config()
    scheduler = Scheduler(scheduler_config)

    seq = Sequence(
        list(range(256)),
        SamplingParams(temperature=0.0, max_tokens=8),
        block_size=128,
    )
    scheduler._prepare_prefill_metadata(seq)
    scheduler._allocate_prefill(seq)
    assert seq.offload_pool_entry >= 0

    scheduler.deallocate(seq)
    assert seq.offload_pool_entry == -1


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_group_manager_validate_layer():
    mgr = IndexShareGroupManager(78, tuple(glm52_indexer_types(78)))

    mgr.validate_layer(0)
    mgr.validate_layer(77)

    with pytest.raises(IndexError, match="not in any IndexShare group"):
        mgr.validate_layer(78)


def test_group_manager_repr():
    mgr = IndexShareGroupManager(78, tuple(glm52_indexer_types(78)))
    repr_str = repr(mgr)
    assert "num_hidden_layers=78" in repr_str
    assert "num_groups=21" in repr_str


def test_group_manager_handles_small_model():
    """A hypothetical 6-layer model with 3 full + 3 shared layers."""

    types = ("full", "shared", "shared", "full", "shared", "shared")
    mgr = IndexShareGroupManager(6, types)

    assert mgr.num_groups == 2
    assert mgr.owner_layer_idxs == (0, 3)
    assert mgr.shared_layer_idxs == (1, 2, 4, 5)
    assert mgr.group(0).member_layer_idxs == (0, 1, 2)
    assert mgr.group(1).member_layer_idxs == (3, 4, 5)


# ---------------------------------------------------------------------------
# Stage 2: offload_split + MTP0 + eager
# ---------------------------------------------------------------------------


def test_glm52_offload_split_config_builds_index_share_groups(tmp_path):
    _write_glm52_config(tmp_path)
    config = _make_config(
        tmp_path,
        offload_mode="offload_split",
        max_model_len=4096,
    )
    groups = config.hf_config.nanovllm_index_share_groups
    assert groups.num_groups == 21
    assert len(groups.owner_layer_idxs) == 21
    assert len(groups.shared_layer_idxs) == 57


def test_glm52_offload_split_enforces_eager(tmp_path):
    _write_glm52_config(tmp_path)
    with pytest.raises(ValueError, match="stage 4"):
        _make_config(
            tmp_path,
            offload_mode="offload_split",
            enforce_eager=False,
        )


def test_glm52_offload_split_rejects_mtp3(tmp_path):
    _write_glm52_config(tmp_path)
    with pytest.raises(ValueError, match="quantized MTP layer"):
        _make_config(
            tmp_path,
            offload_mode="offload_split",
            num_speculative_tokens=3,
        )


def test_glm52_offload_none_still_allowed(tmp_path):
    _write_glm52_config(tmp_path)
    config = _make_config(
        tmp_path,
        offload_mode="none",
        num_dram_kvcache_blocks=-1,
        enforce_eager=True,
    )
    assert config.offload_mode == "none"
    assert config.glm_version == GLM_VERSION_52


def test_initialize_lidu_row_shared_extracts_owner_mapping():
    """Shared layer init reads cache_slots_row filled by the owner."""

    import torch
    from nanovllm.models.dsa_offload_ops import (
        initialize_lidu_row_shared,
        LIDU_TOPK,
    )

    # scatter_copy is a custom Ascend operator not available on CPU.
    if not hasattr(torch.ops, "nanovllm_dsa") or not hasattr(
        torch.ops.nanovllm_dsa, "scatter_copy"
    ):
        pytest.skip("scatter_copy operator is not registered on this machine.")

    cache_tokens = LIDU_TOPK
    source_capacity = 4096
    cache_slots_row = torch.full(
        (source_capacity,), -1, dtype=torch.int32
    )
    # Simulate owner's mapping: source_ids within [0, source_capacity) -> slots [0, 1, 2, ...]
    source_ids = torch.tensor(
        [(i * 2) % source_capacity for i in range(cache_tokens)],
        dtype=torch.int32,
    )
    dest_slots = torch.arange(cache_tokens, dtype=torch.int32)
    cache_slots_row[source_ids.long()] = dest_slots

    # Create small dummy KV caches for scatter_copy
    block_size = 128
    hbm_kpe = torch.zeros(64, block_size, 64, dtype=torch.bfloat16)
    hbm_ckv = torch.zeros(64, block_size, 512, dtype=torch.bfloat16)
    dram_kpe = torch.ones(64, block_size, 64, dtype=torch.bfloat16)
    dram_ckv = torch.ones(64, block_size, 512, dtype=torch.bfloat16)
    hbm_bt = torch.tensor([0, 1, 2, 3], dtype=torch.int32)
    dram_bt = torch.tensor([0, 1, 2, 3], dtype=torch.int32)

    hbm_kpe_out, hbm_ckv_out = initialize_lidu_row_shared(
        cache_slots_row=cache_slots_row,
        cache_tokens=cache_tokens,
        hbm_kpe=hbm_kpe,
        hbm_ckv=hbm_ckv,
        dram_kpe=dram_kpe,
        dram_ckv=dram_ckv,
        hbm_block_table=hbm_bt,
        dram_block_table=dram_bt,
    )

    # scatter_copy should have written DRAM data (all ones) to HBM at the
    # destination slots.  Verify at least one position was written.
    assert hbm_ckv_out.sum() > 0
    assert hbm_kpe_out.sum() > 0


def test_initialize_lidu_row_shared_noop_for_zero_cache():
    import torch
    from nanovllm.models.dsa_offload_ops import initialize_lidu_row_shared

    hbm_kpe = torch.zeros(4, 128, 64, dtype=torch.bfloat16)
    hbm_ckv = torch.zeros(4, 128, 512, dtype=torch.bfloat16)
    cache_slots_row = torch.full((1024,), -1, dtype=torch.int32)

    result = initialize_lidu_row_shared(
        cache_slots_row=cache_slots_row,
        cache_tokens=0,
        hbm_kpe=hbm_kpe,
        hbm_ckv=hbm_ckv,
        dram_kpe=hbm_kpe.clone(),
        dram_ckv=hbm_ckv.clone(),
        hbm_block_table=torch.tensor([0], dtype=torch.int32),
        dram_block_table=torch.tensor([0], dtype=torch.int32),
    )
    # No copy should have occurred
    assert torch.equal(result[0], hbm_kpe)
    assert torch.equal(result[1], hbm_ckv)


def test_initialize_lidu_row_shared_detects_unfilled_mapping():
    import torch
    from nanovllm.models.dsa_offload_ops import initialize_lidu_row_shared

    cache_slots_row = torch.full((1024,), -1, dtype=torch.int32)
    # Don't fill any mapping; expect error when cache_tokens > 0

    with pytest.raises(RuntimeError, match="found 0 cached tokens"):
        initialize_lidu_row_shared(
            cache_slots_row=cache_slots_row,
            cache_tokens=2048,
            hbm_kpe=torch.zeros(4, 128, 64, dtype=torch.bfloat16),
            hbm_ckv=torch.zeros(4, 128, 512, dtype=torch.bfloat16),
            dram_kpe=torch.zeros(4, 128, 64, dtype=torch.bfloat16),
            dram_ckv=torch.zeros(4, 128, 512, dtype=torch.bfloat16),
            hbm_block_table=torch.tensor([0], dtype=torch.int32),
            dram_block_table=torch.tensor([0], dtype=torch.int32),
        )


def test_scheduler_offload_split_lifecycle_with_index_share():
    """Scheduler allocate/deallocate/preempt/abort work with offload_split
    and IndexShare groups present."""

    mgr = IndexShareGroupManager(78, tuple(glm52_indexer_types(78)))
    assert mgr.num_groups == 21

    scheduler_config = _make_scheduler_config(
        offload_mode="offload_split",
        max_model_len=8224,
        num_hbm_kvcache_blocks=96,
        num_dram_kvcache_blocks=128,
    )
    scheduler = Scheduler(scheduler_config)

    seq = Sequence(
        list(range(8200)),
        SamplingParams(temperature=0.0, max_tokens=8),
        block_size=128,
    )
    scheduler._prepare_prefill_metadata(seq)
    scheduler._allocate_prefill(seq)
    assert seq.offload_pool_entry >= 0
    assert seq.index_block_table
    assert seq.dram_block_table

    # Preempt recycles all resources
    scheduler.preempt(seq)
    assert seq.offload_pool_entry == -1
    assert not seq.index_block_table
    assert not seq.dram_block_table

    # Abort after re-allocate
    scheduler._prepare_prefill_metadata(seq)
    scheduler._allocate_prefill(seq)
    scheduler.running.append(seq)
    scheduler.abort_seq_group(seq.request_id)
    assert seq.finish_reason is FinishReason.ABORTED
