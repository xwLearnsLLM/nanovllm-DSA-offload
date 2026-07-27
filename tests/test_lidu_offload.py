import pickle
from types import SimpleNamespace

import pytest

import nanovllm.engine.dsa_offload as dsa_offload
from nanovllm.engine.dsa_offload import (
    LIDU_CACHE_TOKEN_BUDGETS,
    OFFLOAD_GS,
    OFFLOAD_LIDU,
    OFFLOAD_MODES,
    finalize_prefill_hbm_layout,
    lidu_cache_tokens,
    normalize_offload_mode,
    parse_gs_miss_rate_layers,
    validate_lidu_cache_token_budgets,
)
from nanovllm.config import Config
from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.sequence import (
    DecodeSequenceMetadata,
    Sequence,
    SequenceStatus,
)
from nanovllm.sampling_params import SamplingParams


def _config(
    *,
    max_decode_seqs: int = 4,
    num_hbm_blocks: int = 800,
    num_dram_blocks: int = 800,
    offload_mode: str = OFFLOAD_LIDU,
    prefill_chunk_size: int = 0,
):
    return SimpleNamespace(
        max_num_prefill_seqs_per_step=1,
        prefill_chunk_size=prefill_chunk_size,
        max_num_decode_seqs_per_step=max_decode_seqs,
        eos=-1,
        num_hbm_kvcache_blocks=num_hbm_blocks,
        num_dram_kvcache_blocks=num_dram_blocks,
        offload_mode=offload_mode,
        kvcache_block_size=128,
        max_model_len=70_000,
    )


def _seq(length: int, request_id: str) -> Sequence:
    return Sequence(
        list(range(length)),
        SamplingParams(temperature=0.0, max_tokens=4, ignore_eos=True),
        request_id=request_id,
        block_size=128,
    )


def test_three_modes_and_default_public_api():
    assert OFFLOAD_MODES == ("none", "gs", "lidu")
    assert Config.__dataclass_fields__["offload_mode"].default == "none"
    for mode in OFFLOAD_MODES:
        assert normalize_offload_mode(mode.upper()) == mode
    with pytest.raises(ValueError, match="none.*gs.*lidu"):
        normalize_offload_mode("legacy")
    with pytest.raises(TypeError, match="must be a string"):
        normalize_offload_mode(True)
    with pytest.raises(TypeError, match="enable_dsa_offload"):
        Config("unused", enable_dsa_offload=True)


@pytest.mark.parametrize(
    ("prompt_len", "expected"),
    [
        (0, 0),
        (2048, 0),
        (2049, 2048),
        (8192, 2048),
        (8193, 3072),
        (16384, 3072),
        (16385, 6144),
        (32768, 6144),
        (32769, 8192),
        (65536, 8192),
        (65537, 12288),
    ],
)
def test_lidu_cache_tiers_use_original_prompt_length(prompt_len, expected):
    assert lidu_cache_tokens(prompt_len) == expected


def test_four_long_prompt_cache_budgets_are_centralized_and_tunable(
    monkeypatch,
):
    assert LIDU_CACHE_TOKEN_BUDGETS == (3072, 6144, 8192, 12288)
    tuned = (5120, 8192, 16384, 24576)
    monkeypatch.setattr(
        dsa_offload,
        "LIDU_CACHE_TOKEN_BUDGETS",
        tuned,
    )

    assert validate_lidu_cache_token_budgets(128) == tuned
    assert lidu_cache_tokens(8192) == 2048
    assert lidu_cache_tokens(8193) == 5120
    assert lidu_cache_tokens(16385) == 8192
    assert lidu_cache_tokens(32769) == 16384
    assert lidu_cache_tokens(65537) == 24576
    assert max(
        lidu_cache_tokens(length)
        for length in (8193, 16385, 32769, 65537)
    ) == 24576

    seq = _seq(21_000, "tuned-21k")
    Scheduler(_config())._prepare_prefill_metadata(seq)
    assert seq.lidu_cache_tokens == 8192
    assert seq.num_sparse_tokens == 8192


@pytest.mark.parametrize(
    ("budgets", "message"),
    [
        ((5120, 3072, 8192, 12288), "nondecreasing"),
        ((12288, 12288, 12288, 12288), "exceeds the complete source"),
        ((1024, 5120, 8192, 12288), "at least 2048"),
        ((3073, 5120, 8192, 12288), "divisible"),
        ((3072, 5120, 8192), "exactly four integers"),
    ],
)
def test_invalid_tuned_lidu_budgets_fail_at_startup(
    monkeypatch,
    budgets,
    message,
):
    monkeypatch.setattr(
        dsa_offload,
        "LIDU_CACHE_TOKEN_BUDGETS",
        budgets,
    )
    with pytest.raises(ValueError, match=message):
        validate_lidu_cache_token_budgets(128)


def test_lidu_reuses_gs_miss_rate_layer_switch():
    assert parse_gs_miss_rate_layers(None, 78) == frozenset()
    assert parse_gs_miss_rate_layers("0, 30,77,30", 78) == frozenset(
        {0, 30, 77}
    )
    for value in ("0,,30", "layer0", "-1", "78"):
        with pytest.raises(
            ValueError,
            match="NANOVLLM_GS_MISS_RATE_ON_LAYERS",
        ):
            parse_gs_miss_rate_layers(value, 78)


def test_mixed_short_and_long_requests_get_unique_persistent_pool_rows():
    scheduler = Scheduler(_config())
    short = _seq(1024, "short")
    long = _seq(9000, "long")
    for seq in (short, long):
        scheduler._prepare_prefill_metadata(seq)
        scheduler._allocate_prefill(seq)

    assert short.lidu_cache_tokens == 0
    assert short.lidu_cache_initialized
    assert short.num_sparse_tokens == 1024
    assert long.lidu_cache_tokens == 3072
    assert not long.lidu_cache_initialized
    assert long.num_sparse_tokens == 3072
    assert short.offload_pool_entry >= 0
    assert long.offload_pool_entry >= 0
    assert short.offload_pool_entry != long.offload_pool_entry
    assert short.index_block_table == []
    assert short.dram_block_table == []
    assert long.index_block_table
    assert long.dram_block_table

    rows_before_reorder = {
        seq.request_id: seq.offload_pool_entry for seq in (short, long)
    }
    reordered = (long, short)
    assert {
        seq.request_id: seq.offload_pool_entry for seq in reordered
    } == rows_before_reorder


def test_lidu_state_survives_sequence_and_decode_snapshot_serialization():
    seq = _seq(9000, "serialize")
    seq.offload_pool_entry = 3
    seq.lidu_cache_tokens = 3072
    seq.lidu_cache_initialized = True
    seq.num_prefill_full_blocks = 70
    seq.num_sparse_blocks = 24
    seq.num_sparse_tokens = 3072
    seq.num_prefill_tokens_processed = 9000
    seq.append_token(42)
    seq.lidu_decode_hbm_pending = True

    restored = pickle.loads(pickle.dumps(seq))
    assert restored.lidu_decode_hbm_pending
    with pytest.raises(RuntimeError, match="before its HBM cache arena"):
        DecodeSequenceMetadata.from_sequence(restored)
    restored.lidu_decode_hbm_pending = False
    snapshot = DecodeSequenceMetadata.from_sequence(restored)
    assert snapshot.offload_pool_entry == 3
    assert snapshot.lidu_cache_tokens == 3072
    assert snapshot.lidu_cache_initialized


def test_preemption_releases_pool_row_and_resets_lidu_initialization():
    scheduler = Scheduler(_config(max_decode_seqs=1))
    seq = _seq(9000, "preempt")
    scheduler._prepare_prefill_metadata(seq)
    scheduler._allocate_prefill(seq)
    entry = seq.offload_pool_entry
    finalize_prefill_hbm_layout(seq, OFFLOAD_LIDU)
    scheduler.release_prefill_hbm_blocks([seq])
    assert seq.lidu_decode_hbm_pending
    seq.status = SequenceStatus.RUNNING

    scheduler.preempt(seq)

    assert entry not in scheduler.pool_entry_manager.used_entries
    assert seq.offload_pool_entry == -1
    assert not seq.lidu_cache_initialized
    assert not seq.lidu_decode_hbm_pending
    assert seq.status is SequenceStatus.WAITING


def test_preemption_keeps_generated_tokens_out_of_the_lidu_source():
    scheduler = Scheduler(_config(max_decode_seqs=1))
    seq = _seq(2049, "preempt-source")
    scheduler._prepare_prefill_metadata(seq)
    scheduler._allocate_prefill(seq)
    for token_id in range(128):
        seq.append_token(token_id)
    seq.status = SequenceStatus.RUNNING

    scheduler.preempt(seq)
    scheduler._prepare_prefill_metadata(seq)

    assert seq.lidu_cache_tokens == 2048
    assert seq.num_prefill_full_blocks == 16
    assert seq.prefill_tail_len == 129
    assert seq.num_prefill_tail_blocks == 2


def test_decode_growth_does_not_allocate_index_cache_blocks():
    scheduler = Scheduler(_config(max_decode_seqs=1))
    seq = _seq(2048, "decode-growth")
    scheduler._prepare_prefill_metadata(seq)
    scheduler._allocate_prefill(seq)
    index_blocks = list(seq.index_block_table)
    hbm_block_count = len(seq.hbm_block_table)

    seq.append_token(42)
    assert scheduler.can_append(seq)
    scheduler.may_append(seq)

    assert seq.index_block_table == index_blocks
    assert len(seq.hbm_block_table) == hbm_block_count + 1


@pytest.mark.parametrize(
    ("prompt_len", "expected_tail_blocks", "expected_new_blocks"),
    [
        (9000, 1, 24),
        # A block-aligned prompt needs both its delayed C arena and the first
        # decode-tail block in the same atomic allocation.
        (9216, 0, 25),
    ],
)
def test_lidu_releases_all_full_prefill_blocks_until_first_decode(
    prompt_len,
    expected_tail_blocks,
    expected_new_blocks,
):
    scheduler = Scheduler(_config(max_decode_seqs=1))
    seq = _seq(prompt_len, f"lazy-{prompt_len}")
    scheduler._prepare_prefill_metadata(seq)
    scheduler._allocate_prefill(seq)
    old_hbm_blocks = list(seq.hbm_block_table)
    old_metadata_version = seq.decode_metadata_version

    finalize_prefill_hbm_layout(seq, OFFLOAD_LIDU)

    assert seq.hbm_block_table == old_hbm_blocks[
        seq.num_prefill_full_blocks:seq.num_prefill_blocks
    ]
    assert len(seq.hbm_block_table) == expected_tail_blocks
    assert seq.hbm_blocks_to_release == old_hbm_blocks[
        :seq.num_prefill_full_blocks
    ]
    assert seq.lidu_decode_hbm_pending
    assert not seq.lidu_cache_initialized
    scheduler.release_prefill_hbm_blocks([seq])
    assert len(scheduler.hbm_block_manager.used_block_ids) == expected_tail_blocks

    seq.num_prefill_tokens_processed = len(seq)
    seq.append_token(42)
    seq.status = SequenceStatus.RUNNING
    scheduler.running.append(seq)
    scheduled, is_prefill = scheduler.schedule()

    assert not is_prefill
    assert scheduled == [seq]
    assert not seq.lidu_decode_hbm_pending
    assert not seq.lidu_cache_initialized
    assert len(seq.hbm_block_table) == expected_tail_blocks + expected_new_blocks
    if expected_tail_blocks:
        assert seq.hbm_block_table[-expected_tail_blocks:] == (
            old_hbm_blocks[-expected_tail_blocks:]
        )
    else:
        assert old_hbm_blocks[
            seq.num_prefill_full_blocks:seq.num_prefill_blocks
        ] == []
    assert seq.decode_metadata_version == old_metadata_version + 2


def _finish_lidu_prefill_without_running_decode(
    scheduler: Scheduler,
    seq: Sequence,
) -> None:
    assert scheduler._can_allocate_prefill(seq)
    scheduler._allocate_prefill(seq)
    finalize_prefill_hbm_layout(seq, OFFLOAD_LIDU)
    scheduler.release_prefill_hbm_blocks([seq])
    seq.num_prefill_tokens_processed = len(seq)
    seq.append_token(42)
    seq.status = SequenceStatus.RUNNING
    scheduler.running.append(seq)


def test_lidu_prefill_borrows_pending_decode_arenas_without_stranding():
    # Scheduler reserves two blocks outside this manager.  102 configured HBM
    # blocks therefore provide exactly 100 usable blocks.  A 9K request needs
    # 71 blocks for prefill but only 24 sparse + 1 tail blocks for decode.
    scheduler = Scheduler(
        _config(
            max_decode_seqs=4,
            num_hbm_blocks=102,
            num_dram_blocks=800,
        )
    )
    seqs = [_seq(9000, f"borrow-{index}") for index in range(4)]
    for seq in seqs:
        _finish_lidu_prefill_without_running_decode(scheduler, seq)

    # All four requests are admitted while only their tails are physically
    # resident.  Their 4 * 24 pending sparse blocks remain borrowable.
    assert len(scheduler.hbm_block_manager.used_block_ids) == 4
    assert all(seq.lidu_decode_hbm_pending for seq in seqs)

    scheduled, is_prefill = scheduler.schedule()

    assert not is_prefill
    assert scheduled == seqs
    assert all(not seq.lidu_decode_hbm_pending for seq in seqs)
    assert all(len(seq.hbm_block_table) == 25 for seq in seqs)
    assert len(scheduler.hbm_block_manager.used_block_ids) == 100


def test_lidu_decode_reservation_rejects_oversubscription_before_prefill():
    # With 99 usable blocks, physical HBM can still run the fourth 71-block
    # prefill after three tails are retained, but 4 * 25 decode blocks cannot
    # coexist.  Reject at admission instead of failing at first decode.
    scheduler = Scheduler(
        _config(
            max_decode_seqs=4,
            num_hbm_blocks=101,
            num_dram_blocks=800,
        )
    )
    for index in range(3):
        _finish_lidu_prefill_without_running_decode(
            scheduler,
            _seq(9000, f"reserved-{index}"),
        )
    fourth = _seq(9000, "reserved-3")
    scheduler._prepare_prefill_metadata(fourth)
    assert scheduler.hbm_block_manager.can_allocate_blocks(
        fourth.num_prefill_blocks
    )
    assert not scheduler._can_allocate_prefill(fourth)


def test_chunk_prefill_keeps_sparse_arena_delayed_until_decode():
    scheduler = Scheduler(
        _config(
            max_decode_seqs=1,
            prefill_chunk_size=1024,
        )
    )
    seq = _seq(9000, "chunk-lazy")
    scheduler.add(seq)
    num_chunks = 0
    while scheduler.prefilling is not None or scheduler.waiting:
        scheduled, is_prefill = scheduler.schedule()
        assert is_prefill and scheduled == [seq]
        num_chunks += 1
        is_last = (
            seq.num_prefill_tokens_processed + seq.num_scheduled_tokens
            == len(seq)
        )
        if is_last:
            finalize_prefill_hbm_layout(seq, OFFLOAD_LIDU)
            scheduler.release_prefill_hbm_blocks([seq])
            scheduler.postprocess(scheduled, [42], is_prefill)
        else:
            scheduler.postprocess(scheduled, None, is_prefill)

    assert num_chunks == 9
    assert seq.lidu_decode_hbm_pending
    assert len(seq.hbm_block_table) == 1

    scheduled, is_prefill = scheduler.schedule()

    assert not is_prefill
    assert scheduled == [seq]
    assert not seq.lidu_decode_hbm_pending
    assert len(seq.hbm_block_table) == 25


def test_abort_releases_tail_and_pending_lidu_state():
    scheduler = Scheduler(_config(max_decode_seqs=1))
    seq = _seq(9000, "abort-pending")
    _finish_lidu_prefill_without_running_decode(scheduler, seq)
    assert seq.lidu_decode_hbm_pending
    assert scheduler.hbm_block_manager.used_block_ids

    scheduler.abort_seq_group(seq.request_id)

    assert seq.is_finished
    assert not seq.lidu_decode_hbm_pending
    assert not scheduler.hbm_block_manager.used_block_ids
    assert not scheduler.index_block_manager.used_block_ids
    assert not scheduler.dram_block_manager.used_block_ids
    assert not scheduler.pool_entry_manager.used_entries


def test_short_lidu_and_gs_final_layouts_are_unchanged():
    short_scheduler = Scheduler(_config(max_decode_seqs=1))
    short = _seq(1024, "short-layout")
    short_scheduler._prepare_prefill_metadata(short)
    short_scheduler._allocate_prefill(short)
    short_hbm = list(short.hbm_block_table)
    finalize_prefill_hbm_layout(short, OFFLOAD_LIDU)
    assert short.hbm_block_table == short_hbm
    assert short.hbm_blocks_to_release == []
    assert not short.lidu_decode_hbm_pending

    gs_scheduler = Scheduler(
        _config(max_decode_seqs=1, offload_mode=OFFLOAD_GS)
    )
    gs = _seq(9000, "gs-layout")
    gs_scheduler._prepare_prefill_metadata(gs)
    gs_scheduler._allocate_prefill(gs)
    gs_hbm = list(gs.hbm_block_table)
    finalize_prefill_hbm_layout(gs, OFFLOAD_GS)
    prefix_blocks = 1
    suffix_blocks = gs.num_sparse_blocks - prefix_blocks
    assert gs.hbm_block_table == (
        gs_hbm[:prefix_blocks]
        + gs_hbm[
            gs.num_prefill_full_blocks - suffix_blocks:
            gs.num_prefill_blocks
        ]
    )
    assert not gs.lidu_decode_hbm_pending
