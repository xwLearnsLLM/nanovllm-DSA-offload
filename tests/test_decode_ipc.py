import pickle

import pytest

from nanovllm.engine.sequence import (
    DecodeBatchDelta,
    DecodeBatchSnapshot,
    Sequence,
    apply_decode_batch_packet,
    build_decode_batch_packet,
)
from nanovllm.sampling_params import SamplingParams


def _sequence(offset: int) -> Sequence:
    seq = Sequence(
        [offset + 1, offset + 2],
        SamplingParams(temperature=0.0, max_tokens=8, ignore_eos=True),
        block_size=128,
    )
    seq.num_prefill_tokens_processed = len(seq)
    seq.hbm_block_table = list(range(offset, offset + 41))
    seq.block_table = seq.hbm_block_table
    seq.index_block_table = list(range(offset + 1000, offset + 1164))
    seq.dram_block_table = list(range(offset + 2000, offset + 2164))
    seq.offload_pool_entry = offset // 10_000
    seq.lidu_cache_tokens = 5120
    seq.lidu_cache_initialized = True
    seq.num_prefill_full_blocks = 164
    seq.num_sparse_blocks = 40
    seq.num_sparse_tokens = 5120
    seq.prefill_tail_len = 8
    seq.decode_metadata_version = 3
    return seq


def _round_trip(value):
    return pickle.loads(
        pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)
    )


def test_stable_decode_uses_small_delta_and_updates_worker_cache():
    seqs = [_sequence(0), _sequence(10_000)]
    snapshot, key = build_decode_batch_packet(seqs, None)
    assert isinstance(snapshot, DecodeBatchSnapshot)
    cached, cached_key = apply_decode_batch_packet(
        _round_trip(snapshot),
        None,
    )
    assert cached_key == key
    original_tables = [
        (
            list(seq.hbm_block_table),
            list(seq.index_block_table),
            list(seq.dram_block_table),
        )
        for seq in cached
    ]

    for row, seq in enumerate(seqs):
        seq.append_token(90 + row)
    delta, next_key = build_decode_batch_packet(seqs, key)
    assert isinstance(delta, DecodeBatchDelta)
    delta = _round_trip(delta)
    updated, updated_key = apply_decode_batch_packet(delta, cached)

    assert updated is cached
    assert updated_key == next_key == key
    assert [seq.num_tokens for seq in updated] == [3, 3]
    assert [seq.last_token for seq in updated] == [90, 91]
    for seq, tables in zip(updated, original_tables):
        assert seq.hbm_block_table == tables[0]
        assert seq.index_block_table == tables[1]
        assert seq.dram_block_table == tables[2]

    snapshot_bytes = len(
        pickle.dumps(snapshot, protocol=pickle.HIGHEST_PROTOCOL)
    )
    delta_bytes = len(pickle.dumps(delta, protocol=pickle.HIGHEST_PROTOCOL))
    assert delta_bytes < snapshot_bytes // 10


def test_metadata_version_change_forces_complete_snapshot():
    seqs = [_sequence(0), _sequence(10_000)]
    first, key = build_decode_batch_packet(seqs, None)
    cached, _ = apply_decode_batch_packet(_round_trip(first), None)

    seqs[0].hbm_block_table.append(999)
    seqs[0].bump_decode_metadata_version()
    refresh, next_key = build_decode_batch_packet(seqs, key)

    assert isinstance(refresh, DecodeBatchSnapshot)
    assert next_key != key
    refreshed, refreshed_key = apply_decode_batch_packet(
        _round_trip(refresh),
        cached,
    )
    assert refreshed is not cached
    assert refreshed_key == next_key
    assert refreshed[0].hbm_block_table[-1] == 999


@pytest.mark.parametrize("mutation", ["reorder", "shrink"])
def test_batch_shape_or_order_change_forces_complete_snapshot(mutation):
    seqs = [_sequence(0), _sequence(10_000), _sequence(20_000)]
    _, key = build_decode_batch_packet(seqs, None)
    next_seqs = list(reversed(seqs)) if mutation == "reorder" else seqs[:-1]

    packet, next_key = build_decode_batch_packet(next_seqs, key)

    assert isinstance(packet, DecodeBatchSnapshot)
    assert next_key != key


def test_worker_rejects_delta_without_matching_snapshot():
    seqs = [_sequence(0), _sequence(10_000)]
    _, key = build_decode_batch_packet(seqs, None)
    delta, _ = build_decode_batch_packet(seqs, key)
    assert isinstance(delta, DecodeBatchDelta)

    with pytest.raises(RuntimeError, match="before a complete snapshot"):
        apply_decode_batch_packet(delta, None)

    snapshot, _ = build_decode_batch_packet(seqs, None)
    cached, _ = apply_decode_batch_packet(snapshot, None)
    bad_delta = DecodeBatchDelta(
        key=((seqs[0].seq_id, 99), (seqs[1].seq_id, 99)),
        num_tokens=delta.num_tokens,
        last_tokens=delta.last_tokens,
    )
    with pytest.raises(RuntimeError, match="does not match"):
        apply_decode_batch_packet(bad_delta, cached)

    wrong_size = DecodeBatchDelta(
        key=key,
        num_tokens=delta.num_tokens[:-1],
        last_tokens=delta.last_tokens,
    )
    with pytest.raises(RuntimeError, match="batch size"):
        apply_decode_batch_packet(wrong_size, cached)
