import pickle
from types import SimpleNamespace

import pytest

from nanovllm.config import Config
from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.sequence import FinishReason, Sequence, SequenceStatus
from nanovllm.sampling_params import SamplingParams


def make_config(
    *,
    prefill_chunk_size: int = 1024,
    max_num_prefill_seqs_per_step: int = 1,
    block_size: int = 128,
    num_hbm_blocks: int = 256,
    num_dram_blocks: int = 256,
):
    return SimpleNamespace(
        max_num_prefill_seqs_per_step=max_num_prefill_seqs_per_step,
        prefill_chunk_size=prefill_chunk_size,
        max_num_decode_seqs_per_step=16,
        eos=-1,
        num_hbm_kvcache_blocks=num_hbm_blocks,
        num_dram_kvcache_blocks=num_dram_blocks,
        kvcache_block_size=block_size,
        max_model_len=20_000,
    )


def make_sequence(
    length: int,
    *,
    request_id: str = "request",
    block_size: int = 128,
    max_tokens: int = 4,
) -> Sequence:
    return Sequence(
        list(range(length)),
        SamplingParams(temperature=0.0, max_tokens=max_tokens, ignore_eos=True),
        request_id=request_id,
        block_size=block_size,
    )


@pytest.mark.parametrize("value", [0, 1024])
def test_prefill_chunk_size_accepts_only_supported_values(value):
    Config._validate_prefill_chunking(value, 1)


@pytest.mark.parametrize("value", [-1, 1, 512, 2048, True, 1024.0])
def test_prefill_chunk_size_rejects_other_values(value):
    with pytest.raises((TypeError, ValueError)):
        Config._validate_prefill_chunking(value, 1)


def test_chunk_prefill_requires_one_prefill_sequence_per_step():
    with pytest.raises(ValueError, match="max_num_prefill_seqs_per_step=1"):
        Config._validate_prefill_chunking(1024, 2)
    Config._validate_prefill_chunking(0, 2)


def test_ten_thousand_token_prompt_is_chunked_and_samples_once():
    scheduler = Scheduler(make_config())
    seq = make_sequence(10_000)
    scheduler.add(seq)

    chunk_sizes = []
    sample_calls = 0
    while seq.num_prefill_tokens_processed < 10_000:
        seqs, is_prefill = scheduler.schedule()
        assert is_prefill
        assert seqs == [seq]
        chunk_sizes.append(seq.num_scheduled_tokens)

        chunk_end = (
            seq.num_prefill_tokens_processed + seq.num_scheduled_tokens
        )
        if chunk_end < 10_000:
            scheduler.postprocess(seqs, None, is_prefill=True)
            assert len(seq) == 10_000
            assert scheduler.prefilling is seq
            assert seq not in scheduler.running
            assert not seq.offload_finalized
        else:
            sample_calls += 1
            scheduler.postprocess(seqs, [42], is_prefill=True)

    assert chunk_sizes == [1024] * 9 + [784]
    assert sample_calls == 1
    assert seq.completion_token_ids == [42]
    assert seq.is_first_decode_after_prefill
    assert seq.num_decode_tokens_since_prefill == 1
    assert scheduler.prefilling is None
    assert list(scheduler.running) == [seq]


def test_scheduled_chunk_maps_hbm_and_index_slots_for_current_tokens_only():
    seq = make_sequence(10, block_size=4)
    seq.hbm_block_table = [5, 8, 2]
    seq.block_table = seq.hbm_block_table
    seq.index_block_table = [11, 12, 13]
    seq.num_prefill_tokens_processed = 3
    seq.num_scheduled_tokens = 5

    input_ids, positions, hbm_slots, index_slots, kv_length, is_last = (
        seq.scheduled_prefill_chunk()
    )

    assert input_ids == [3, 4, 5, 6, 7]
    assert positions == [3, 4, 5, 6, 7]
    assert hbm_slots == [23, 32, 33, 34, 35]
    assert index_slots == [47, 48, 49, 50, 51]
    assert kv_length == 8
    assert not is_last

    seq.num_prefill_tokens_processed = 8
    seq.num_scheduled_tokens = 2
    input_ids, positions, hbm_slots, index_slots, kv_length, is_last = (
        seq.scheduled_prefill_chunk()
    )
    assert input_ids == [8, 9]
    assert positions == [8, 9]
    assert hbm_slots == [8, 9]
    assert index_slots == [52, 53]
    assert kv_length == 10
    assert is_last


def test_partial_prefill_state_and_dsa_tables_survive_serialization():
    seq = make_sequence(32, block_size=8)
    seq.hbm_block_table = [3, 7, 9, 11]
    seq.block_table = seq.hbm_block_table
    seq.index_block_table = [13, 15, 17, 19]
    seq.dram_block_table = [21, 23, 25, 27]
    seq.num_prefill_tokens_processed = 16
    seq.num_scheduled_tokens = 8

    restored = pickle.loads(pickle.dumps(seq))

    assert restored.num_prefill_tokens_processed == 16
    assert restored.num_scheduled_tokens == 8
    assert restored.hbm_block_table == [3, 7, 9, 11]
    assert restored.index_block_table == [13, 15, 17, 19]
    assert restored.dram_block_table == [21, 23, 25, 27]
    assert restored.scheduled_prefill_chunk()[4:] == (24, False)


def test_abort_releases_all_partially_prefilled_dsa_resources():
    scheduler = Scheduler(make_config())
    seq = make_sequence(1500, request_id="abort-me")
    scheduler.add(seq)

    seqs, is_prefill = scheduler.schedule()
    scheduler.postprocess(seqs, None, is_prefill)
    assert seq.num_prefill_tokens_processed == 1024
    assert all(used > 0 for used, _ in scheduler.dsa_block_usage())
    assert scheduler.pool_entry_manager.used_entries

    scheduler.abort_seq_group("abort-me")

    assert scheduler.prefilling is None
    assert scheduler.is_finished()
    assert seq.status is SequenceStatus.FINISHED
    assert seq.finish_reason is FinishReason.ABORTED
    assert seq.hbm_block_table == []
    assert seq.index_block_table == []
    assert seq.dram_block_table == []
    assert all(used == 0 for used, _ in scheduler.dsa_block_usage())
    assert not scheduler.pool_entry_manager.used_entries


def test_active_chunk_prefill_is_not_interleaved_with_decode():
    scheduler = Scheduler(make_config())
    decode_seq = make_sequence(16, request_id="decode")
    scheduler._prepare_prefill_metadata(decode_seq)
    scheduler._allocate_prefill(decode_seq)
    decode_seq.num_prefill_tokens_processed = len(decode_seq)
    decode_seq.append_token(7)
    decode_seq.status = SequenceStatus.RUNNING
    scheduler.running.append(decode_seq)

    prefill_seq = make_sequence(1500, request_id="prefill")
    scheduler.add(prefill_seq)

    seqs, is_prefill = scheduler.schedule()
    assert is_prefill
    assert seqs == [prefill_seq]
    scheduler.postprocess(seqs, None, is_prefill)

    seqs, is_prefill = scheduler.schedule()
    assert is_prefill
    assert seqs == [prefill_seq]
    assert decode_seq in scheduler.running


def test_decode_preemption_resets_progress_and_recompute_boundary():
    scheduler = Scheduler(make_config())
    seq = make_sequence(1500)
    seq.append_token(99)
    scheduler._prepare_prefill_metadata(seq)
    scheduler._allocate_prefill(seq)
    seq.status = SequenceStatus.RUNNING
    seq.num_prefill_tokens_processed = 1024
    seq.num_scheduled_tokens = 128
    scheduler.running.append(seq)

    scheduler.preempt(scheduler.running.pop())

    assert seq.status is SequenceStatus.WAITING
    assert seq.finish_reason is FinishReason.PREEMPTED
    assert seq.num_prefill_tokens_processed == 0
    assert seq.num_scheduled_tokens == 0
    assert seq.hbm_block_table == []
    assert seq.index_block_table == []
    assert seq.dram_block_table == []

    seqs, is_prefill = scheduler.schedule()
    assert is_prefill
    assert seqs == [seq]
    assert seq.num_scheduled_tokens == 1024
    assert seq.finish_reason is None

    scheduler.postprocess(seqs, None, is_prefill)
    seqs, is_prefill = scheduler.schedule()
    assert seq.num_scheduled_tokens == 477
    scheduler.postprocess(seqs, [100], is_prefill)
    assert seq.is_first_decode_after_prefill
    assert seq.num_decode_tokens_since_prefill == 1


def test_prefill_chunk_size_zero_preserves_batched_prefill_behavior():
    scheduler = Scheduler(
        make_config(
            prefill_chunk_size=0,
            max_num_prefill_seqs_per_step=2,
            block_size=16,
            num_hbm_blocks=32,
            num_dram_blocks=32,
        )
    )
    seq1 = make_sequence(4, request_id="one", block_size=16, max_tokens=2)
    seq2 = make_sequence(6, request_id="two", block_size=16, max_tokens=2)
    scheduler.add(seq1)
    scheduler.add(seq2)

    seqs, is_prefill = scheduler.schedule()

    assert is_prefill
    assert seqs == [seq1, seq2]
    assert list(scheduler.running) == [seq1, seq2]
    assert scheduler.prefilling is None
    assert all(seq.num_scheduled_tokens == 0 for seq in seqs)

    scheduler.postprocess(seqs, [101, 102], is_prefill)
    assert seq1.completion_token_ids == [101]
    assert seq2.completion_token_ids == [102]
    assert seq1.is_first_decode_after_prefill
    assert seq2.is_first_decode_after_prefill

    decode_seqs, is_prefill = scheduler.schedule()
    assert not is_prefill
    assert decode_seqs == [seq1, seq2]

    scheduler.postprocess(decode_seqs, [103, 104], is_prefill)
    assert not seq1.is_first_decode_after_prefill
    assert not seq2.is_first_decode_after_prefill
