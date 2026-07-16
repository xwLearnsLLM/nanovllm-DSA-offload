from contextlib import nullcontext
import sys
from types import SimpleNamespace

import pytest
import torch

from nanovllm.engine.full_decode_graph import (
    FullDecodeGraphEntry,
    FullDecodeOnlyGraphManager,
    MLAGraphTask,
    normalize_capture_sizes,
)
from nanovllm.utils.context import Context, reset_context, set_context


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
        block_tables=block_tables,
        index_block_tables=block_tables + 10,
        dram_block_tables=block_tables + 20,
        selection_block_tables=torch.tensor(
            [[row + 1, row + 2] for row in range(batch_size)],
            dtype=torch.int32,
        ),
        req_pool_entries=torch.arange(batch_size, dtype=torch.int32),
        candidate_lens=torch.arange(4096, 4096 + batch_size, dtype=torch.int32),
        candidate_query_lens=torch.arange(1, batch_size + 1, dtype=torch.int32),
        needs_dsa_update=True,
        dsa_offload_all_rows=True,
        decode_metadata_key=metadata_key,
    )


def test_normalize_capture_sizes():
    assert normalize_capture_sizes([16, 4, 16, 8]) == (4, 8, 16)


@pytest.mark.parametrize("values", [[], [0], [-1, 4]])
def test_normalize_capture_sizes_rejects_invalid_values(values):
    with pytest.raises(ValueError):
        normalize_capture_sizes(values)


def test_static_entry_copies_all_dsa_metadata():
    entry = FullDecodeGraphEntry.allocate(2, 4, 2, torch.device("cpu"))
    context = _decode_context()

    seq_lens = entry.copy_runtime_inputs(
        torch.tensor([11, 12], dtype=torch.int64),
        torch.tensor([2199, 2200], dtype=torch.int64),
        context,
    )

    assert seq_lens == [2200, 2201]
    assert entry.input_ids.tolist() == [11, 12]
    assert entry.positions.tolist() == [2199, 2200]
    assert entry.flat_slot_mapping_i32.tolist() == [10, 11]
    assert entry.block_tables[:, :3].equal(context.block_tables)
    assert entry.block_tables[:, 3].count_nonzero().item() == 0
    assert entry.index_block_tables[:, :3].equal(context.index_block_tables)
    assert entry.dram_block_tables[:, :3].equal(context.dram_block_tables)
    assert entry.selection_block_tables.equal(context.selection_block_tables)
    assert entry.req_pool_entries.tolist() == [0, 1]
    assert entry.candidate_lens.tolist() == [4096, 4097]


def test_static_entry_reuses_unchanged_decode_metadata():
    entry = FullDecodeGraphEntry.allocate(2, 4, 2, torch.device("cpu"))
    key = ((10, 3), (11, 7))
    first = _decode_context(metadata_key=key)
    entry.copy_runtime_inputs(
        torch.tensor([11, 12], dtype=torch.int64),
        torch.tensor([2199, 2200], dtype=torch.int64),
        first,
    )
    original_tables = entry.block_tables.clone()

    same_revision = _decode_context(metadata_key=key)
    same_revision.block_tables.add_(1000)
    entry.copy_runtime_inputs(
        torch.tensor([21, 22], dtype=torch.int64),
        torch.tensor([2200, 2201], dtype=torch.int64),
        same_revision,
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
    )
    assert entry.block_tables[:, :3].equal(next_revision.block_tables)
    assert entry.metadata_refresh_count == 2


def test_static_entry_rejects_bucket_padding():
    entry = FullDecodeGraphEntry.allocate(4, 4, 2, torch.device("cpu"))
    with pytest.raises(ValueError, match="exact capture sizes"):
        entry.copy_runtime_inputs(
            torch.tensor([11, 12], dtype=torch.int64),
            torch.tensor([100, 200], dtype=torch.int64),
            _decode_context(),
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


def test_ineligible_decode_paths_stay_eager():
    manager = object.__new__(FullDecodeOnlyGraphManager)
    manager.model = lambda input_ids, positions: input_ids + positions
    manager.eager_prefill_count = 0
    manager.eager_first_decode_count = 0
    manager.eager_no_dsa_count = 0
    manager.eager_mixed_batch_count = 0
    manager.eager_uncaptured_batch_count = 0
    manager.log_enabled = False
    manager.capture_sizes = (16,)

    set_context(False, has_first_decode=True)
    try:
        assert manager.run(torch.tensor([2]), torch.tensor([3])).tolist() == [5]
    finally:
        reset_context()
    assert manager.eager_first_decode_count == 1

    set_context(False, needs_dsa_update=False)
    try:
        manager.run(torch.tensor([2]), torch.tensor([3]))
    finally:
        reset_context()
    assert manager.eager_no_dsa_count == 1

    set_context(False, needs_dsa_update=True, dsa_offload_all_rows=False)
    try:
        manager.run(torch.tensor([2]), torch.tensor([3]))
    finally:
        reset_context()
    assert manager.eager_mixed_batch_count == 1

    set_context(False, needs_dsa_update=True, dsa_offload_all_rows=True)
    try:
        manager.run(torch.tensor([2]), torch.tensor([3]))
    finally:
        reset_context()
    assert manager.eager_uncaptured_batch_count == 1


def test_exact_eligible_batch_replays_graph(monkeypatch):
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

    entry = FullDecodeGraphEntry.allocate(2, 4, 2, torch.device("cpu"))
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
    manager.eager_mixed_batch_count = 0
    manager.eager_uncaptured_batch_count = 0
    manager.log_enabled = False

    context = _decode_context()
    set_context(
        False,
        flat_slot_mapping_i32=context.flat_slot_mapping_i32,
        actual_seq_lengths_kv=context.actual_seq_lengths_kv,
        block_tables=context.block_tables,
        index_block_tables=context.index_block_tables,
        dram_block_tables=context.dram_block_tables,
        selection_block_tables=context.selection_block_tables,
        req_pool_entries=context.req_pool_entries,
        candidate_lens=context.candidate_lens,
        candidate_query_lens=context.candidate_query_lens,
        needs_dsa_update=True,
        dsa_offload_all_rows=True,
    )
    try:
        output = manager.run(torch.tensor([7, 8]), torch.tensor([99, 199]))
    finally:
        reset_context()

    assert output.tolist() == [[0, 1, 2], [3, 4, 5]]
    assert calls == ["synchronize", "replay", ("update", [2200, 2201])]
    assert manager.replay_count == 1


def test_decode_callable_uses_npugraph_ex(monkeypatch):
    calls = {}
    fake_dynamo = SimpleNamespace(
        config=SimpleNamespace(
            cache_size_limit=64,
            accumulated_cache_size_limit=256,
        )
    )

    class FakeCompilerConfig:
        def __init__(self):
            self.mode = None
            self.debug = SimpleNamespace(
                run_eagerly=False,
                aclgraph=SimpleNamespace(
                    disable_reinplace_inplaceable_ops_pass=False
                ),
            )

    def get_npu_backend(*, compiler_config):
        calls["compiler_config"] = compiler_config
        return "npugraph_ex-backend"

    def compile_model(model, **kwargs):
        calls["compile"] = (model, kwargs)
        return "compiled-model"

    fake_torchair = SimpleNamespace(
        CompilerConfig=FakeCompilerConfig,
        get_npu_backend=get_npu_backend,
    )
    fake_npu = SimpleNamespace(
        set_compile_mode=lambda **kwargs: calls.setdefault("compile_mode", kwargs)
    )
    monkeypatch.setitem(torch.__dict__, "_dynamo", fake_dynamo)
    monkeypatch.setitem(sys.modules, "torchair", fake_torchair)
    monkeypatch.setattr(torch, "npu", fake_npu, raising=False)
    monkeypatch.setattr(torch, "compile", compile_model)

    manager = object.__new__(FullDecodeOnlyGraphManager)
    manager.log_enabled = False
    model = object()
    compiled = manager._build_decode_callable(model)

    compiler_config = calls["compiler_config"]
    assert compiled == "compiled-model"
    assert calls["compile_mode"] == {"jit_compile": False}
    assert compiler_config.mode == "reduce-overhead"
    assert compiler_config.debug.run_eagerly is True
    assert (
        compiler_config.debug.aclgraph.disable_reinplace_inplaceable_ops_pass
        is True
    )
    assert calls["compile"] == (
        model,
        {
            "backend": "npugraph_ex-backend",
            "fullgraph": False,
            "dynamic": False,
        },
    )


def test_disabled_npugraph_ex_returns_raw_model(monkeypatch):
    model = object()
    manager = object.__new__(FullDecodeOnlyGraphManager)
    manager.enable_npugraph_ex = False
    manager.log_enabled = False
    calls = {}
    monkeypatch.setattr(
        torch,
        "npu",
        SimpleNamespace(
            set_compile_mode=lambda **kwargs: calls.setdefault(
                "compile_mode", kwargs
            )
        ),
        raising=False,
    )

    assert manager._build_decode_callable(model) is model
    assert calls["compile_mode"] == {"jit_compile": False}
