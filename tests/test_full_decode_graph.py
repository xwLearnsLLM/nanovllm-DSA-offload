from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch

from nanovllm.engine.full_decode_graph import (
    FullDecodeGraphEntry,
    FullDecodeOnlyGraphManager,
    MLAGraphTask,
    MTPDecodeGraphEntry,
    MTPDecodeOnlyGraphManager,
    normalize_capture_sizes,
)
from nanovllm.utils.context import (
    Context,
    get_context,
    reset_context,
    set_context,
)


def _decode_context(
    batch_size: int = 2,
    metadata_key=None,
) -> Context:
    block_tables = torch.tensor(
        [[row + 1, row + 3, 0] for row in range(batch_size)],
        dtype=torch.int32,
    )
    return Context(
        is_prefill=False,
        flat_slot_mapping_i32=torch.arange(
            10,
            10 + batch_size,
            dtype=torch.int32,
        ),
        actual_seq_lengths_kv=list(range(2200, 2200 + batch_size)),
        actual_seq_lengths_kv_tensor=torch.arange(
            2200, 2200 + batch_size, dtype=torch.int32
        ),
        block_tables=block_tables,
        index_block_tables=block_tables + 10,
        dram_block_tables=block_tables + 20,
        req_pool_entries=torch.arange(batch_size, dtype=torch.int32),
        candidate_lens=torch.arange(4096, 4096 + batch_size, dtype=torch.int32),
        candidate_query_lens=torch.arange(1, batch_size + 1, dtype=torch.int32),
        lidu_cache_tokens=torch.tensor(
            [0 if row == 0 else 3072 for row in range(batch_size)],
            dtype=torch.int32,
        ),
        needs_dsa_update=True,
        lidu_all_rows_ready=True,
        decode_metadata_key=metadata_key,
    )


def test_normalize_capture_sizes():
    assert normalize_capture_sizes([16, 4, 16, 8]) == (4, 8, 16)


@pytest.mark.parametrize("values", [[], [0], [-1, 4]])
def test_normalize_capture_sizes_rejects_invalid_values(values):
    with pytest.raises(ValueError):
        normalize_capture_sizes(values)


def test_mtp_entry_copies_exact_target_inputs():
    entry = MTPDecodeGraphEntry.allocate(
        batch_size=2,
        speculative_tokens=3,
        max_block_columns=4,
        device=torch.device("cpu"),
    )
    context = Context(
        flat_slot_mapping=torch.arange(8, dtype=torch.int64) + 20,
        block_tables=torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.int32),
    )
    entry.copy_runtime_inputs(
        torch.arange(8, dtype=torch.int64) + 100,
        torch.arange(8, dtype=torch.int64) + 1000,
        torch.tensor([[7, 8, 9], [10, 11, 12]], dtype=torch.int64),
        context,
    )

    assert entry.input_ids.tolist() == list(range(100, 108))
    assert entry.positions.tolist() == list(range(1000, 1008))
    assert entry.draft_token_ids.tolist() == [[7, 8, 9], [10, 11, 12]]
    assert entry.flat_slot_mapping_i32.tolist() == list(range(20, 28))
    assert entry.cu_seqlens_q.tolist() == [0, 4, 8]
    assert entry.block_tables[:, :3].equal(context.block_tables)
    assert entry.block_tables[:, 3].count_nonzero().item() == 0


def test_mtp_entry_copies_offload_and_separate_draft_metadata():
    entry = MTPDecodeGraphEntry.allocate(
        batch_size=2,
        speculative_tokens=3,
        max_block_columns=4,
        device=torch.device("cpu"),
    )
    target_tables = torch.tensor(
        [[1, 2, 3], [4, 5, 6]], dtype=torch.int32
    )
    mtp_tables = target_tables + 100
    original_target_tables = target_tables.clone()
    original_mtp_tables = mtp_tables.clone()
    context = Context(
        flat_slot_mapping=torch.arange(8, dtype=torch.int64) + 20,
        flat_slot_mapping_i32=torch.arange(8, dtype=torch.int32) + 20,
        actual_seq_lengths_kv=[5188, 5192],
        actual_seq_lengths_kv_tensor=torch.tensor(
            [5188, 5192], dtype=torch.int32
        ),
        block_tables=target_tables,
        index_block_tables=target_tables + 10,
        dram_block_tables=target_tables + 20,
        req_pool_entries=torch.tensor([7, 3], dtype=torch.int32),
        candidate_lens=torch.tensor([20992, 32768], dtype=torch.int32),
        candidate_query_lens=torch.tensor([4, 8], dtype=torch.int32),
        lidu_cache_tokens=torch.tensor([8192, 12288], dtype=torch.int32),
        decode_metadata_key=((10, 3), (11, 7)),
    )

    entry.copy_runtime_inputs(
        torch.arange(8, dtype=torch.int64) + 100,
        torch.arange(8, dtype=torch.int64) + 1000,
        torch.tensor([[7, 8, 9], [10, 11, 12]], dtype=torch.int64),
        context,
        mtp_tables,
        offload_mode="offload_split",
    )

    assert entry.block_tables[:, :3].equal(original_target_tables)
    assert entry.mtp_block_tables[:, :3].equal(original_mtp_tables)
    assert entry.index_block_tables[:, :3].equal(target_tables + 10)
    assert entry.dram_block_tables[:, :3].equal(target_tables + 20)
    assert entry.actual_seq_lengths_kv.tolist() == [5188, 5192]
    assert entry.req_pool_entries.tolist() == [7, 3]
    assert entry.candidate_lens.tolist() == [20992, 32768]
    assert entry.candidate_query_lens.tolist() == [4, 8]
    assert entry.lidu_cache_tokens.tolist() == [8192, 12288]

    # Dynamic lengths always refresh, while unchanged tables/state are reused.
    context.actual_seq_lengths_kv_tensor.add_(1)
    context.block_tables.add_(1000)
    entry.copy_runtime_inputs(
        torch.arange(8, dtype=torch.int64) + 200,
        torch.arange(8, dtype=torch.int64) + 2000,
        torch.tensor([[17, 18, 19], [20, 21, 22]], dtype=torch.int64),
        context,
        mtp_tables + 1000,
        offload_mode="offload_split",
    )
    assert entry.actual_seq_lengths_kv.tolist() == [5189, 5193]
    assert entry.block_tables[:, :3].equal(original_target_tables)
    assert entry.mtp_block_tables[:, :3].equal(original_mtp_tables)
    assert entry.metadata_refresh_count == 1
    assert entry.metadata_reuse_count == 1


def test_mtp_graph_uses_only_exact_batch_after_one_eager_step():
    manager = MTPDecodeOnlyGraphManager(
        target_forward=lambda *_args: (),
        draft_forward=lambda *_args: torch.empty(0),
        target_warmup=None,
        draft_warmup=None,
        capture_sizes=(8,),
        max_model_len=4096,
        block_size=128,
        device="cpu",
        speculative_tokens=3,
        expected_target_tasks=78,
        log_enabled=False,
    )

    assert not manager.should_use_graph(8)
    assert manager.should_use_graph(8)
    assert not manager.should_use_graph(7)
    stats = manager.stats()
    assert stats["eager_first_decode"] == 1
    assert stats["eager_uncaptured_batch"] == 1
    assert stats["exact_size_only"] is True


def test_mtp_offload_graph_waits_for_initialized_rows():
    manager = MTPDecodeOnlyGraphManager(
        target_forward=lambda *_args: (),
        draft_forward=lambda *_args: torch.empty(0),
        target_warmup=None,
        draft_warmup=None,
        capture_sizes=(8,),
        max_model_len=32768,
        block_size=128,
        device="cpu",
        speculative_tokens=3,
        expected_target_tasks=0,
        offload_mode="offload_split",
        log_enabled=False,
    )

    assert not manager.should_use_graph(
        8, Context(has_first_decode=True, lidu_all_rows_ready=False)
    )
    assert not manager.should_use_graph(
        8, Context(has_first_decode=False, lidu_all_rows_ready=False)
    )
    assert manager.should_use_graph(
        8, Context(has_first_decode=False, lidu_all_rows_ready=True)
    )
    stats = manager.stats()
    assert stats["offload_mode"] == "offload_split"
    assert stats["eager_first_decode"] == 1
    assert stats["eager_lidu_uninitialized"] == 1


def test_mtp_offload_target_context_uses_fixed_graph_metadata():
    manager = MTPDecodeOnlyGraphManager(
        target_forward=lambda *_args: (),
        draft_forward=lambda *_args: torch.empty(0),
        target_warmup=None,
        draft_warmup=None,
        capture_sizes=(2,),
        max_model_len=32768,
        block_size=128,
        device="cpu",
        speculative_tokens=3,
        expected_target_tasks=0,
        offload_mode="offload_split",
        log_enabled=False,
    )
    entry = manager._allocate_entry(2)
    entry.actual_seq_lengths_kv.copy_(torch.tensor([5188, 5192]))
    entry.req_pool_entries.copy_(torch.tensor([7, 3]))
    entry.candidate_lens.copy_(torch.tensor([20992, 32768]))
    entry.lidu_cache_tokens.copy_(torch.tensor([8192, 12288]))

    manager._set_target_context(entry, [21003, 32779])
    try:
        context = get_context()
        assert context.is_spec_decode is True
        assert context.full_decode_graph is True
        assert context.needs_dsa_update is True
        assert context.lidu_all_rows_ready is True
        assert context.actual_seq_lengths_kv == [21003, 32779]
        assert context.actual_seq_lengths_kv_tensor is entry.actual_seq_lengths_kv
        assert context.block_tables is entry.block_tables
        assert context.index_block_tables is entry.index_block_tables
        assert context.dram_block_tables is entry.dram_block_tables
        assert context.req_pool_entries is entry.req_pool_entries
        assert context.candidate_lens is entry.candidate_lens
        assert context.candidate_query_lens is entry.candidate_query_lens
        assert context.lidu_cache_tokens is entry.lidu_cache_tokens
    finally:
        reset_context()


def test_static_entry_copies_all_lidu_metadata():
    entry = FullDecodeGraphEntry.allocate(2, 4, torch.device("cpu"))
    context = _decode_context()

    seq_lens = entry.copy_runtime_inputs(
        torch.tensor([11, 12], dtype=torch.int64),
        torch.tensor([2199, 2200], dtype=torch.int64),
        context,
        offload_mode="offload_split",
    )

    assert seq_lens == [2200, 2201]
    assert entry.input_ids.tolist() == [11, 12]
    assert entry.positions.tolist() == [2199, 2200]
    assert entry.flat_slot_mapping_i32.tolist() == [10, 11]
    assert entry.block_tables[:, :3].equal(context.block_tables)
    assert entry.block_tables[:, 3].count_nonzero().item() == 0
    assert entry.index_block_tables[:, :3].equal(context.index_block_tables)
    assert entry.dram_block_tables[:, :3].equal(context.dram_block_tables)
    assert entry.req_pool_entries.tolist() == [0, 1]
    assert entry.candidate_lens.tolist() == [4096, 4097]


def test_static_entry_copies_lidu_tiers_for_mixed_batch():
    entry = FullDecodeGraphEntry.allocate(2, 4, torch.device("cpu"))
    context = _decode_context()

    entry.copy_runtime_inputs(
        torch.tensor([11, 12], dtype=torch.int64),
        torch.tensor([2199, 2200], dtype=torch.int64),
        context,
        offload_mode="offload_split",
    )

    assert entry.lidu_cache_tokens.tolist() == [0, 3072]


def test_static_entry_refreshes_tensor_mla_lengths_every_step():
    entry = FullDecodeGraphEntry.allocate(2, 4, torch.device("cpu"))
    key = ((10, 3), (11, 7))
    context = _decode_context(metadata_key=key)

    entry.copy_runtime_inputs(
        torch.tensor([11, 12], dtype=torch.int64),
        torch.tensor([2199, 2200], dtype=torch.int64),
        context,
        offload_mode="offload_split",
        uses_tensor_mla_lengths=True,
    )
    assert entry.actual_seq_lengths_kv.tolist() == [2200, 2201]

    context.actual_seq_lengths_kv = [2201, 2202]
    context.actual_seq_lengths_kv_tensor.add_(1)
    entry.copy_runtime_inputs(
        torch.tensor([21, 22], dtype=torch.int64),
        torch.tensor([2200, 2201], dtype=torch.int64),
        context,
        offload_mode="offload_split",
        uses_tensor_mla_lengths=True,
    )
    assert entry.metadata_reuse_count == 1
    assert entry.actual_seq_lengths_kv.tolist() == [2201, 2202]


def test_static_entry_reuses_unchanged_decode_metadata():
    entry = FullDecodeGraphEntry.allocate(2, 4, torch.device("cpu"))
    key = ((10, 3), (11, 7))
    first = _decode_context(metadata_key=key)
    entry.copy_runtime_inputs(
        torch.tensor([11, 12], dtype=torch.int64),
        torch.tensor([2199, 2200], dtype=torch.int64),
        first,
        offload_mode="offload_split",
    )
    original_tables = entry.block_tables.clone()

    same_revision = _decode_context(metadata_key=key)
    same_revision.block_tables.add_(1000)
    entry.copy_runtime_inputs(
        torch.tensor([21, 22], dtype=torch.int64),
        torch.tensor([2200, 2201], dtype=torch.int64),
        same_revision,
        offload_mode="offload_split",
    )

    assert entry.input_ids.tolist() == [21, 22]
    assert entry.positions.tolist() == [2200, 2201]
    assert entry.block_tables.equal(original_tables)
    assert entry.metadata_refresh_count == 1
    assert entry.metadata_reuse_count == 1

    next_revision = _decode_context(metadata_key=((10, 4), (11, 7)))
    next_revision.block_tables.add_(2000)
    entry.copy_runtime_inputs(
        torch.tensor([31, 32], dtype=torch.int64),
        torch.tensor([2201, 2202], dtype=torch.int64),
        next_revision,
        offload_mode="offload_split",
    )
    assert entry.block_tables[:, :3].equal(next_revision.block_tables)
    assert entry.metadata_refresh_count == 2


def test_static_entry_rejects_bucket_padding():
    entry = FullDecodeGraphEntry.allocate(4, 4, torch.device("cpu"))
    with pytest.raises(ValueError, match="exact capture sizes"):
        entry.copy_runtime_inputs(
            torch.tensor([11, 12], dtype=torch.int64),
            torch.tensor([100, 200], dtype=torch.int64),
            _decode_context(),
            offload_mode="offload_split",
        )


def test_mla_task_refreshes_host_sequence_lengths(monkeypatch):
    calls = []
    fake_npu = SimpleNamespace(
        graph_task_update_begin=lambda stream, handle: calls.append(
            ("begin", stream, handle)
        ),
        graph_task_update_end=lambda stream: calls.append(("end", stream)),
    )
    monkeypatch.setattr(torch, "npu", fake_npu, raising=False)

    class FakeEvent:
        def record(self, stream):
            calls.append(("event", stream))

    def fake_attention(*args, **kwargs):
        calls.append(("attention", args, kwargs))

    tensor = torch.empty(1)
    task = MLAGraphTask(
        handle="handle",
        event=FakeEvent(),
        op=fake_attention,
        query=tensor,
        key_cache=tensor,
        query_rope=tensor,
        key_rope_cache=tensor,
        block_table=torch.zeros(2, 1, dtype=torch.int32),
        workspace=tensor,
        output=tensor,
        softmax_lse=tensor,
        num_query_heads=4,
        block_size=128,
        softmax_scale=0.25,
    )

    task.update("update-stream", [16001, 16002])

    assert [call[0] for call in calls] == ["begin", "attention", "end", "event"]
    attention_kwargs = calls[1][2]
    assert attention_kwargs["actual_seq_kvlen"] == [16001, 16002]
    assert attention_kwargs["block_table"] is task.block_table


def test_mla_task_refresh_preserves_tnd_decode_arguments(monkeypatch):
    calls = []
    fake_npu = SimpleNamespace(
        graph_task_update_begin=lambda stream, handle: calls.append(
            ("begin", stream, handle)
        ),
        graph_task_update_end=lambda stream: calls.append(("end", stream)),
    )
    monkeypatch.setattr(torch, "npu", fake_npu, raising=False)

    class FakeEvent:
        def record(self, stream):
            calls.append(("event", stream))

    def fake_attention(*args, **kwargs):
        calls.append(("attention", args, kwargs))

    tensor = torch.empty(1)
    mask = torch.ones(4, 4, dtype=torch.bool)
    task = MLAGraphTask(
        handle="tnd-handle",
        event=FakeEvent(),
        op=fake_attention,
        query=tensor,
        key_cache=tensor,
        query_rope=tensor,
        key_rope_cache=tensor,
        block_table=torch.zeros(2, 2, dtype=torch.int32),
        workspace=tensor,
        output=tensor,
        softmax_lse=tensor,
        num_query_heads=4,
        block_size=128,
        softmax_scale=0.25,
        input_layout="TND_NTD",
        atten_mask=mask,
        sparse_mode=3,
        actual_seq_qlen=[4, 8],
    )

    task.update("update-stream", [16003, 32006])

    attention_kwargs = calls[1][2]
    assert attention_kwargs["input_layout"] == "TND_NTD"
    assert attention_kwargs["atten_mask"] is mask
    assert attention_kwargs["sparse_mode"] == 3
    assert attention_kwargs["actual_seq_qlen"] == [4, 8]
    assert attention_kwargs["actual_seq_kvlen"] == [16003, 32006]


def test_lidu_noop_and_uninitialized_batches_stay_eager():
    manager = object.__new__(FullDecodeOnlyGraphManager)
    manager.model = lambda input_ids, positions: input_ids + positions
    manager.capture_sizes = (2,)
    manager.offload_mode = "offload_split"
    manager.stateful_offload = True
    manager.eager_prefill_count = 0
    manager.eager_first_decode_count = 0
    manager.eager_no_dsa_count = 0
    manager.eager_lidu_uninitialized_count = 0
    manager.eager_lidu_capture_count = 0
    manager.eager_uncaptured_batch_count = 0
    manager.replay_count = 0
    manager.log_enabled = False

    set_context(False, needs_dsa_update=False, lidu_all_rows_ready=True)
    try:
        output = manager.run(torch.tensor([2, 3]), torch.tensor([4, 5]))
    finally:
        reset_context()
    assert output.tolist() == [6, 8]
    assert manager.eager_no_dsa_count == 1

    set_context(False, needs_dsa_update=True, lidu_all_rows_ready=False)
    try:
        output = manager.run(torch.tensor([2, 3]), torch.tensor([4, 5]))
    finally:
        reset_context()
    assert output.tolist() == [6, 8]
    assert manager.eager_lidu_uninitialized_count == 1


def test_first_initialized_lidu_batch_is_captured_but_runs_eager():
    calls = []
    entry = SimpleNamespace(graph=None, output=None)
    manager = object.__new__(FullDecodeOnlyGraphManager)
    manager.model = lambda input_ids, positions: input_ids + positions
    manager.capture_sizes = (2,)
    manager.offload_mode = "offload_split"
    manager.stateful_offload = True
    manager._entries = {}
    manager.eager_prefill_count = 0
    manager.eager_first_decode_count = 0
    manager.eager_no_dsa_count = 0
    manager.eager_lidu_uninitialized_count = 0
    manager.eager_lidu_capture_count = 0
    manager.eager_uncaptured_batch_count = 0
    manager.replay_count = 0
    manager.log_enabled = False

    def allocate(batch_size):
        assert batch_size == 2
        manager._entries[batch_size] = entry
        return entry

    def capture(capture_entry, input_ids, positions, runtime_context):
        calls.append(
            (
                "capture",
                input_ids.tolist(),
                positions.tolist(),
                runtime_context.lidu_all_rows_ready,
            )
        )
        capture_entry.graph = object()
        capture_entry.output = torch.empty(2, 1)

    manager._allocate_entry = allocate
    manager._capture_lidu_runtime = capture

    set_context(False, needs_dsa_update=True, lidu_all_rows_ready=True)
    try:
        output = manager.run(torch.tensor([2, 3]), torch.tensor([4, 5]))
    finally:
        reset_context()

    assert output.tolist() == [6, 8]
    assert calls == [("capture", [2, 3], [4, 5], True)]
    assert manager.eager_lidu_capture_count == 1
    assert manager.replay_count == 0


def test_exact_eligible_lidu_batch_replays_graph(monkeypatch):
    calls = []

    class FakeGraph:
        def replay(self):
            calls.append("replay")

    class FakeStream:
        def synchronize(self):
            calls.append("synchronize")

    class FakeTask:
        def update(self, _stream, actual_seq_kvlen):
            calls.append(("update", actual_seq_kvlen))

    monkeypatch.setattr(
        torch,
        "npu",
        SimpleNamespace(
            current_stream=lambda: FakeStream(),
            stream=lambda _stream: nullcontext(),
        ),
        raising=False,
    )

    entry = FullDecodeGraphEntry.allocate(2, 4, torch.device("cpu"))
    entry.graph = FakeGraph()
    entry.output = torch.arange(6).view(2, 3)
    entry.mla_tasks = [FakeTask()]

    manager = object.__new__(FullDecodeOnlyGraphManager)
    manager.model = lambda input_ids, positions: input_ids + positions
    manager.capture_sizes = (2,)
    manager._entries = {2: entry}
    manager._update_stream = object()
    manager.replay_count = 0
    manager.eager_prefill_count = 0
    manager.eager_first_decode_count = 0
    manager.eager_no_dsa_count = 0
    manager.eager_uncaptured_batch_count = 0
    manager.log_enabled = False
    manager.offload_mode = "offload_split"
    manager.stateful_offload = True
    manager.eager_lidu_uninitialized_count = 0
    manager.eager_lidu_capture_count = 0

    context = _decode_context()
    set_context(
        False,
        flat_slot_mapping_i32=context.flat_slot_mapping_i32,
        actual_seq_lengths_kv=context.actual_seq_lengths_kv,
        block_tables=context.block_tables,
        index_block_tables=context.index_block_tables,
        dram_block_tables=context.dram_block_tables,
        req_pool_entries=context.req_pool_entries,
        candidate_lens=context.candidate_lens,
        candidate_query_lens=context.candidate_query_lens,
        lidu_cache_tokens=context.lidu_cache_tokens,
        needs_dsa_update=True,
        lidu_all_rows_ready=True,
    )
    try:
        output = manager.run(torch.tensor([7, 8]), torch.tensor([99, 199]))
    finally:
        reset_context()

    assert output.tolist() == [[0, 1, 2], [3, 4, 5]]
    assert calls == ["synchronize", "replay", ("update", [2200, 2201])]
    assert manager.replay_count == 1


def test_dense_mla_graph_pads_to_the_smallest_capture(monkeypatch):
    calls = []

    class FakeGraph:
        def replay(self):
            calls.append("replay")

    class FakeStream:
        def synchronize(self):
            calls.append("synchronize")

    class FakeTask:
        def update(self, _stream, actual_seq_kvlen):
            calls.append(("update", actual_seq_kvlen))

    monkeypatch.setattr(
        torch,
        "npu",
        SimpleNamespace(
            current_stream=lambda: FakeStream(),
            stream=lambda _stream: nullcontext(),
        ),
        raising=False,
    )

    entry = FullDecodeGraphEntry.allocate(4, 4, torch.device("cpu"))
    entry.graph = FakeGraph()
    entry.output = torch.arange(12).view(4, 3)
    entry.mla_tasks = [FakeTask()]

    manager = object.__new__(FullDecodeOnlyGraphManager)
    manager.model = lambda input_ids, positions: input_ids + positions
    manager.capture_sizes = (4,)
    manager.offload_mode = "none"
    manager.stateful_offload = False
    manager.eager_lidu_uninitialized_count = 0
    manager.eager_lidu_capture_count = 0
    manager._entries = {4: entry}
    manager._update_stream = object()
    manager.replay_count = 0
    manager.eager_prefill_count = 0
    manager.eager_first_decode_count = 0
    manager.eager_no_dsa_count = 0
    manager.eager_uncaptured_batch_count = 0
    manager.log_enabled = False

    context = _decode_context()
    set_context(
        False,
        flat_slot_mapping_i32=context.flat_slot_mapping_i32,
        actual_seq_lengths_kv=context.actual_seq_lengths_kv,
        block_tables=context.block_tables,
    )
    try:
        output = manager.run(
            torch.tensor([7, 8]), torch.tensor([99, 199])
        )
    finally:
        reset_context()

    assert output.tolist() == [[0, 1, 2], [3, 4, 5]]
    assert entry.input_ids.tolist() == [7, 8, 0, 0]
    assert entry.positions.tolist() == [99, 199, 0, 0]
    assert entry.flat_slot_mapping_i32.tolist() == [10, 11, 2, 3]
    assert entry.block_tables[:2, :3].equal(context.block_tables)
    assert entry.block_tables[2:].count_nonzero().item() == 0
    assert calls == [
        "synchronize",
        "replay",
        ("update", [2200, 2201, 0, 0]),
    ]


def test_mtp_graph_replays_target_then_refreshes_three_draft_steps(
    monkeypatch,
):
    calls = []

    class FakeGraph:
        def __init__(self, name):
            self.name = name

        def replay(self):
            calls.append(("replay", self.name))

    class FakeStream:
        def synchronize(self):
            calls.append("synchronize")

    class FakeTask:
        def __init__(self, name):
            self.name = name

        def update(self, _stream, seq_lengths):
            calls.append(("update", self.name, list(seq_lengths)))

    monkeypatch.setattr(
        torch,
        "npu",
        SimpleNamespace(
            current_stream=lambda: FakeStream(),
            stream=lambda _stream: nullcontext(),
        ),
        raising=False,
    )

    entry = MTPDecodeGraphEntry.allocate(2, 3, 4, torch.device("cpu"))
    entry.target_graph = FakeGraph("target")
    entry.draft_graph = FakeGraph("draft")
    entry.target_tokens = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]])
    entry.accepted_counts = torch.tensor([0, 2])
    entry.next_token_ids = torch.tensor([1, 7])
    entry.selected_hidden_states = torch.zeros(2, 4)
    entry.selected_positions = torch.tensor([99, 201])
    entry.next_drafts = torch.tensor([[9, 10, 11], [12, 13, 14]])
    entry.target_tasks = [FakeTask("target")]
    entry.draft_tasks = [FakeTask(f"draft-{step}") for step in range(3)]

    manager = MTPDecodeOnlyGraphManager(
        target_forward=lambda *_args: (),
        draft_forward=lambda *_args: torch.empty(0),
        target_warmup=None,
        draft_warmup=None,
        capture_sizes=(2,),
        max_model_len=512,
        block_size=128,
        device="cpu",
        speculative_tokens=3,
        expected_target_tasks=1,
        log_enabled=False,
    )
    manager._entries[2] = entry
    manager._update_stream = object()
    context = Context(
        flat_slot_mapping=torch.arange(8, dtype=torch.int64),
        block_tables=torch.tensor([[1, 2], [3, 4]], dtype=torch.int32),
    )

    outputs = manager.run(
        torch.arange(8, dtype=torch.int64),
        torch.arange(8, dtype=torch.int64) + 99,
        torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.int64),
        context,
        [100, 200],
    )

    assert outputs[0] is entry.target_tokens
    assert outputs[1] is entry.accepted_counts
    assert outputs[2] is entry.next_drafts
    assert calls == [
        "synchronize",
        ("replay", "target"),
        ("update", "target", [103, 203]),
        ("replay", "draft"),
        ("update", "draft-0", [100, 202]),
        ("update", "draft-1", [101, 203]),
        ("update", "draft-2", [102, 204]),
    ]
    assert manager.replay_count == 1


def test_mtp_graph_serial_target_refreshes_each_verification_step(
    monkeypatch,
):
    calls = []

    class FakeGraph:
        def __init__(self, name):
            self.name = name

        def replay(self):
            calls.append(("replay", self.name))

    class FakeStream:
        def synchronize(self):
            calls.append("synchronize")

    class FakeTask:
        def __init__(self, name):
            self.name = name

        def update(self, _stream, seq_lengths):
            calls.append(("update", self.name, list(seq_lengths)))

    monkeypatch.setattr(
        torch,
        "npu",
        SimpleNamespace(
            current_stream=lambda: FakeStream(),
            stream=lambda _stream: nullcontext(),
        ),
        raising=False,
    )

    entry = MTPDecodeGraphEntry.allocate(2, 3, 4, torch.device("cpu"))
    entry.target_graph = FakeGraph("target")
    entry.draft_graph = FakeGraph("draft")
    entry.target_tokens = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]])
    entry.accepted_counts = torch.tensor([0, 2])
    entry.next_token_ids = torch.tensor([1, 7])
    entry.selected_hidden_states = torch.zeros(2, 4)
    entry.selected_positions = torch.tensor([99, 201])
    entry.next_drafts = torch.tensor([[9, 10, 11], [12, 13, 14]])
    entry.target_tasks = [FakeTask(f"target-{step}") for step in range(4)]
    entry.draft_tasks = [FakeTask(f"draft-{step}") for step in range(3)]

    manager = MTPDecodeOnlyGraphManager(
        target_forward=lambda *_args: (),
        draft_forward=lambda *_args: torch.empty(0),
        target_warmup=None,
        draft_warmup=None,
        capture_sizes=(2,),
        max_model_len=512,
        block_size=128,
        device="cpu",
        speculative_tokens=3,
        expected_target_tasks=4,
        serial_target_verification=True,
        log_enabled=False,
    )
    manager._entries[2] = entry
    manager._update_stream = object()
    context = Context(
        flat_slot_mapping=torch.arange(8, dtype=torch.int64),
        block_tables=torch.tensor([[1, 2], [3, 4]], dtype=torch.int32),
    )

    manager.run(
        torch.arange(8, dtype=torch.int64),
        torch.arange(8, dtype=torch.int64) + 99,
        torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.int64),
        context,
        [100, 200],
    )

    assert calls == [
        "synchronize",
        ("replay", "target"),
        ("update", "target-0", [100, 200]),
        ("update", "target-1", [101, 201]),
        ("update", "target-2", [102, 202]),
        ("update", "target-3", [103, 203]),
        ("replay", "draft"),
        ("update", "draft-0", [100, 202]),
        ("update", "draft-1", [101, 203]),
        ("update", "draft-2", [102, 204]),
    ]
