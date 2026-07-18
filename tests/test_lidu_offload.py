import pickle
from types import SimpleNamespace

import pytest

from nanovllm.engine.dsa_offload import (
    OFFLOAD_MODES,
    lidu_cache_tokens,
    normalize_offload_mode,
    parse_gs_miss_rate_layers,
)
from nanovllm.config import Config
from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.sequence import (
    DecodeSequenceMetadata,
    Sequence,
    SequenceStatus,
)
from nanovllm.sampling_params import SamplingParams


def _config(*, max_decode_seqs: int = 4):
    return SimpleNamespace(
        max_num_prefill_seqs_per_step=1,
        prefill_chunk_size=0,
        max_num_decode_seqs_per_step=max_decode_seqs,
        eos=-1,
        num_hbm_kvcache_blocks=800,
        num_dram_kvcache_blocks=800,
        offload_mode="lidu",
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
        (16385, 5120),
        (32768, 5120),
        (32769, 8192),
        (65536, 8192),
        (65537, 12288),
    ],
)
def test_lidu_cache_tiers_use_original_prompt_length(prompt_len, expected):
    assert lidu_cache_tokens(prompt_len) == expected


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

    restored = pickle.loads(pickle.dumps(seq))
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
    seq.lidu_cache_initialized = True
    seq.status = SequenceStatus.RUNNING

    scheduler.preempt(seq)

    assert entry not in scheduler.pool_entry_manager.used_entries
    assert seq.offload_pool_entry == -1
    assert not seq.lidu_cache_initialized
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
