from __future__ import annotations

from collections.abc import Callable, Iterable
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
import gc
from typing import Any

import torch
import torch.distributed as dist

from nanovllm.engine.dsa_offload import (
    OFFLOAD_NONE,
    normalize_offload_mode,
)
from nanovllm.utils.context import (
    Context,
    get_context,
    preserve_context,
    reset_context,
    set_context,
)
from nanovllm.utils.logger import init_logger


logger = init_logger(__name__)


FULL_DECODE_ONLY = "full_decode_only"


def normalize_capture_sizes(
    values: Iterable[int],
) -> tuple[int, ...]:
    """Validate, sort, and deduplicate full-decode graph batch sizes."""
    sizes = tuple(sorted({int(value) for value in values}))
    if not sizes:
        raise ValueError(
            "FULL_DECODE_ONLY requires at least one decode graph capture "
            "size. Pass decode_graph_capture_sizes, for example (16,)."
        )
    if sizes[0] <= 0:
        raise ValueError(f"Decode graph capture sizes must be positive, got {sizes}.")
    return sizes


def select_capture_size(
    real_batch_size: int,
    capture_sizes: Iterable[int],
) -> int | None:
    for size in capture_sizes:
        if int(size) >= int(real_batch_size):
            return int(size)
    return None


@dataclass
class MLAGraphTask:
    """One captured FIA-v2 task whose host sequence-length attrs are replayed."""

    handle: Any
    event: Any
    op: Callable[..., Any]
    query: torch.Tensor
    key_cache: torch.Tensor
    query_rope: torch.Tensor
    key_rope_cache: torch.Tensor
    block_table: torch.Tensor
    workspace: torch.Tensor
    output: torch.Tensor
    softmax_lse: torch.Tensor
    num_query_heads: int
    block_size: int
    softmax_scale: float
    input_layout: str = "BNSD_NBSD"
    atten_mask: torch.Tensor | None = None
    sparse_mode: int = 0
    actual_seq_qlen: list[int] | None = None

    def update(self, update_stream: Any, actual_seq_kvlen: list[int]) -> None:
        torch.npu.graph_task_update_begin(update_stream, self.handle)
        try:
            self.op(
                self.query,
                self.key_cache,
                self.key_cache,
                query_rope=self.query_rope,
                key_rope=self.key_rope_cache,
                num_query_heads=self.num_query_heads,
                num_key_value_heads=1,
                input_layout=self.input_layout,
                atten_mask=self.atten_mask,
                sparse_mode=self.sparse_mode,
                softmax_scale=self.softmax_scale,
                block_table=self.block_table,
                block_size=self.block_size,
                actual_seq_qlen=self.actual_seq_qlen,
                actual_seq_kvlen=actual_seq_kvlen,
                workspace=self.workspace,
                out=[self.output, self.softmax_lse],
            )
        finally:
            torch.npu.graph_task_update_end(update_stream)
        self.event.record(update_stream)


@dataclass
class _CaptureState:
    tasks: list[MLAGraphTask]


_ACTIVE_CAPTURE: ContextVar[_CaptureState | None] = ContextVar(
    "nanovllm_dsa_full_decode_graph_capture",
    default=None,
)


def is_full_decode_graph_capturing() -> bool:
    return _ACTIVE_CAPTURE.get() is not None


def record_mla_graph_task(task: MLAGraphTask) -> None:
    capture = _ACTIVE_CAPTURE.get()
    if capture is None:
        raise RuntimeError("MLA graph task was recorded outside FULL_DECODE_ONLY capture.")
    capture.tasks.append(task)


@contextmanager
def _record_graph_tasks(tasks: list[MLAGraphTask]):
    token = _ACTIVE_CAPTURE.set(_CaptureState(tasks))
    try:
        yield
    finally:
        _ACTIVE_CAPTURE.reset(token)


@dataclass
class FullDecodeGraphEntry:
    """Static inputs for one steady-state decode graph."""

    batch_size: int
    max_block_columns: int
    input_ids: torch.Tensor
    positions: torch.Tensor
    flat_slot_mapping_i32: torch.Tensor
    actual_seq_lengths_kv: torch.Tensor
    block_tables: torch.Tensor
    index_block_tables: torch.Tensor
    dram_block_tables: torch.Tensor
    req_pool_entries: torch.Tensor
    candidate_lens: torch.Tensor
    candidate_query_lens: torch.Tensor
    lidu_cache_tokens: torch.Tensor
    graph: Any | None = None
    output: torch.Tensor | None = None
    mla_tasks: list[MLAGraphTask] = field(default_factory=list)
    decode_metadata_key: tuple[tuple[int, int], ...] | None = None
    metadata_refresh_count: int = 0
    metadata_reuse_count: int = 0
    replay_count: int = 0

    @classmethod
    def allocate(
        cls,
        batch_size: int,
        max_block_columns: int,
        device: torch.device,
    ) -> "FullDecodeGraphEntry":
        flat_slot_mapping_i32 = torch.arange(
            batch_size,
            dtype=torch.int32,
            device=device,
        )
        candidate_query_lens = torch.arange(
            1,
            batch_size + 1,
            dtype=torch.int32,
            device=device,
        )
        return cls(
            batch_size=batch_size,
            max_block_columns=max_block_columns,
            input_ids=torch.zeros(batch_size, dtype=torch.int64, device=device),
            positions=torch.zeros(batch_size, dtype=torch.int64, device=device),
            flat_slot_mapping_i32=flat_slot_mapping_i32,
            actual_seq_lengths_kv=torch.ones(
                batch_size, dtype=torch.int32, device=device
            ),
            block_tables=torch.zeros(
                batch_size,
                max_block_columns,
                dtype=torch.int32,
                device=device,
            ),
            index_block_tables=torch.zeros(
                batch_size,
                max_block_columns,
                dtype=torch.int32,
                device=device,
            ),
            dram_block_tables=torch.zeros(
                batch_size,
                max_block_columns,
                dtype=torch.int32,
                device=device,
            ),
            req_pool_entries=torch.arange(
                batch_size,
                dtype=torch.int32,
                device=device,
            ),
            candidate_lens=torch.full(
                (batch_size,),
                1,
                dtype=torch.int32,
                device=device,
            ),
            candidate_query_lens=candidate_query_lens,
            lidu_cache_tokens=torch.zeros(
                batch_size, dtype=torch.int32, device=device
            ),
        )

    @staticmethod
    def _copy_table(
        destination: torch.Tensor,
        source: torch.Tensor,
        name: str,
    ) -> None:
        if source.ndim != 2:
            raise ValueError(f"{name} must be rank 2, got shape={tuple(source.shape)}.")
        if source.shape[0] != destination.shape[0]:
            raise ValueError(
                f"{name} batch mismatch: runtime={source.shape[0]}, "
                f"graph={destination.shape[0]}."
            )
        runtime_columns = int(source.shape[1])
        graph_columns = int(destination.shape[1])
        if runtime_columns > graph_columns:
            raise ValueError(
                f"{name} is wider than its graph buffer: "
                f"runtime={runtime_columns}, graph={graph_columns}."
            )
        if runtime_columns < graph_columns:
            destination[:, runtime_columns:].zero_()
        destination[:, :runtime_columns].copy_(source)

    def copy_runtime_inputs(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        context: Context,
        *,
        offload_mode: str = OFFLOAD_NONE,
        uses_tensor_mla_lengths: bool = False,
    ) -> list[int]:
        if input_ids.ndim != 1 or positions.ndim != 1:
            raise ValueError(
                "FULL_DECODE_ONLY expects one token and one position per request: "
                f"input_ids.shape={tuple(input_ids.shape)}, "
                f"positions.shape={tuple(positions.shape)}."
            )
        runtime_batch_size = int(input_ids.shape[0])
        stateful_offload = offload_mode != OFFLOAD_NONE
        if stateful_offload and runtime_batch_size != self.batch_size:
            raise ValueError(
                "DSA FULL_DECODE_ONLY uses exact capture sizes because padding "
                "would mutate another request's persistent LIDU state: "
                f"runtime={runtime_batch_size}, graph={self.batch_size}."
            )
        if runtime_batch_size > self.batch_size:
            raise ValueError(
                f"Runtime batch {runtime_batch_size} does not fit graph "
                f"batch {self.batch_size}."
            )
        if int(positions.shape[0]) != runtime_batch_size:
            raise ValueError(
                "Decode input_ids and positions must have the same batch size: "
                f"{runtime_batch_size} != {positions.shape[0]}."
            )
        required_metadata = {
            "flat_slot_mapping_i32": context.flat_slot_mapping_i32,
            "block_tables": context.block_tables,
        }
        if stateful_offload:
            required_metadata.update(
                index_block_tables=context.index_block_tables,
                dram_block_tables=context.dram_block_tables,
                req_pool_entries=context.req_pool_entries,
                candidate_lens=context.candidate_lens,
                candidate_query_lens=context.candidate_query_lens,
                lidu_cache_tokens=context.lidu_cache_tokens,
            )
        if uses_tensor_mla_lengths:
            required_metadata["actual_seq_lengths_kv_tensor"] = (
                context.actual_seq_lengths_kv_tensor
            )
        missing = [name for name, value in required_metadata.items() if value is None]
        if missing:
            raise RuntimeError(
                "FULL_DECODE_ONLY is missing decode metadata tensors: "
                + ", ".join(missing)
            )

        if int(context.block_tables.shape[1]) > self.max_block_columns:
            raise ValueError(
                "Runtime block table is wider than its graph buffer: "
                f"runtime={context.block_tables.shape[1]}, "
                f"graph={self.max_block_columns}."
            )

        if not stateful_offload and runtime_batch_size < self.batch_size:
            padded = slice(runtime_batch_size, self.batch_size)
            self.input_ids[padded].zero_()
            self.positions[padded].zero_()
            self.flat_slot_mapping_i32[padded].copy_(
                torch.arange(
                    runtime_batch_size,
                    self.batch_size,
                    dtype=torch.int32,
                    device=self.input_ids.device,
                )
            )

        if not stateful_offload:
            self.input_ids[:runtime_batch_size].copy_(input_ids)
            self.positions[:runtime_batch_size].copy_(positions)
            self.flat_slot_mapping_i32[:runtime_batch_size].copy_(
                context.flat_slot_mapping_i32
            )
        else:
            self.input_ids.copy_(input_ids)
            self.positions.copy_(positions)
            self.flat_slot_mapping_i32.copy_(context.flat_slot_mapping_i32)
        metadata_key = context.decode_metadata_key
        refresh_metadata = (
            metadata_key is None or metadata_key != self.decode_metadata_key
        )
        if refresh_metadata:
            if stateful_offload:
                self.req_pool_entries.copy_(context.req_pool_entries)
                self.candidate_lens.copy_(context.candidate_lens)
                self.candidate_query_lens.copy_(context.candidate_query_lens)
                self.lidu_cache_tokens.copy_(context.lidu_cache_tokens)
                self._copy_table(
                    self.block_tables, context.block_tables, "block_tables"
                )
                self._copy_table(
                    self.index_block_tables,
                    context.index_block_tables,
                    "index_block_tables",
                )
                self._copy_table(
                    self.dram_block_tables,
                    context.dram_block_tables,
                    "dram_block_tables",
                )
            else:
                columns = int(context.block_tables.shape[1])
                self.block_tables.zero_()
                self.block_tables[
                    :runtime_batch_size, :columns
                ].copy_(context.block_tables)
            self.decode_metadata_key = metadata_key
            self.metadata_refresh_count += 1
        else:
            self.metadata_reuse_count += 1

        seq_lens = list(context.actual_seq_lengths_kv or ())
        if len(seq_lens) != runtime_batch_size:
            raise ValueError(
                "Decode actual_seq_lengths_kv must have one value per request: "
                f"got {len(seq_lens)} for batch {runtime_batch_size}."
            )
        if uses_tensor_mla_lengths:
            self.actual_seq_lengths_kv.copy_(
                context.actual_seq_lengths_kv_tensor
            )
        if stateful_offload:
            return seq_lens
        return seq_lens + [0] * (self.batch_size - runtime_batch_size)


class FullDecodeOnlyGraphManager:
    """Capture/replay the complete steady-state decode model.

    LIDU uses exact-size captures because its request-pool state is persistent.
    Dense MLA may pad into reserved null-block slots and select the smallest
    configured capture that fits the runtime batch.
    """

    def __init__(
        self,
        model: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        *,
        capture_sizes: Iterable[int],
        max_model_len: int,
        block_size: int,
        device: str | torch.device,
        expected_mla_tasks: int,
        offload_mode: str = OFFLOAD_NONE,
        uses_tensor_mla_lengths: bool = False,
        log_enabled: bool = True,
    ) -> None:
        self.model = model
        self.capture_sizes = tuple(int(size) for size in capture_sizes)
        self.block_size = int(block_size)
        self.offload_mode = normalize_offload_mode(offload_mode)
        self.stateful_offload = self.offload_mode != OFFLOAD_NONE
        self.max_block_columns = (
            int(max_model_len) + self.block_size - 1
        ) // self.block_size
        if (
            not self.stateful_offload
            and max(self.capture_sizes) > self.block_size
        ):
            raise ValueError(
                "Dense-MLA FULL_DECODE_ONLY requires capture sizes not to "
                "exceed block_size so padded rows can use distinct offsets "
                f"inside null block 0: sizes={self.capture_sizes}, "
                f"block_size={self.block_size}."
            )
        self.device = torch.device(device)
        self.expected_mla_tasks = int(expected_mla_tasks)
        self.uses_tensor_mla_lengths = bool(uses_tensor_mla_lengths)
        self.log_enabled = bool(log_enabled)
        self._entries: dict[int, FullDecodeGraphEntry] = {}
        self._graph_pool = None
        self._update_stream = None
        self.capture_count = 0
        self.replay_count = 0
        self.eager_prefill_count = 0
        self.eager_first_decode_count = 0
        self.eager_no_dsa_count = 0
        self.eager_lidu_uninitialized_count = 0
        self.eager_lidu_capture_count = 0
        self.eager_uncaptured_batch_count = 0

        self._validate_runtime()
        torch.npu.set_compile_mode(jit_compile=False)
        self._callable = model

    def _validate_runtime(self) -> None:
        npu = getattr(torch, "npu", None)
        required = (
            "NPUGraph",
            "ExternalEvent",
            "Stream",
            "current_stream",
            "graph",
            "graph_pool_handle",
            "stream",
            "graph_task_group_begin",
            "graph_task_group_end",
            "graph_task_update_begin",
            "graph_task_update_end",
        )
        missing = [name for name in required if npu is None or not hasattr(npu, name)]
        if missing:
            raise RuntimeError(
                "The installed torch-npu does not provide the APIs required by "
                f"FULL_DECODE_ONLY: {', '.join(missing)}."
            )

    def _allocate_entry(self, batch_size: int) -> FullDecodeGraphEntry:
        entry = FullDecodeGraphEntry.allocate(
            batch_size,
            self.max_block_columns,
            self.device,
        )
        self._entries[batch_size] = entry
        return entry

    def _set_capture_context(
        self,
        entry: FullDecodeGraphEntry,
        actual_seq_kvlen: list[int],
    ) -> None:
        uses_tensor_mla_lengths = getattr(
            self, "uses_tensor_mla_lengths", False
        )
        if uses_tensor_mla_lengths:
            entry.actual_seq_lengths_kv.copy_(
                torch.tensor(
                    actual_seq_kvlen,
                    dtype=torch.int32,
                    device=self.device,
                )
            )
        kwargs = {}
        if self.stateful_offload:
            kwargs.update(
                index_block_tables=entry.index_block_tables,
                dram_block_tables=entry.dram_block_tables,
                req_pool_entries=entry.req_pool_entries,
                candidate_lens=entry.candidate_lens,
                candidate_query_lens=entry.candidate_query_lens,
                needs_dsa_update=True,
                full_decode_graph=True,
                lidu_cache_tokens=entry.lidu_cache_tokens,
                lidu_all_rows_ready=True,
            )
        set_context(
            False,
            flat_slot_mapping_i32=entry.flat_slot_mapping_i32,
            actual_seq_lengths_kv=actual_seq_kvlen,
            actual_seq_lengths_kv_tensor=(
                entry.actual_seq_lengths_kv
                if uses_tensor_mla_lengths
                else None
            ),
            block_tables=entry.block_tables,
            has_first_decode=False,
            **kwargs,
        )

    def _ensure_capture_resources(self) -> None:
        if self._graph_pool is None:
            self._graph_pool = torch.npu.graph_pool_handle()
        if self._update_stream is None:
            self._update_stream = torch.npu.Stream()

    def capture_all(self) -> None:
        self._ensure_capture_resources()

        if self.log_enabled:
            logger.info(
                "FULL_DECODE_ONLY: pre-capturing %s for sizes=%s, "
                "offload_mode=%s",
                "raw decode model in one outer ACLGraph",
                self.capture_sizes,
                self.offload_mode,
            )
        try:
            with torch.inference_mode():
                for batch_size in self.capture_sizes:
                    entry = self._entries.get(batch_size) or self._allocate_entry(
                        batch_size
                    )
                    self._capture(entry)
        finally:
            reset_context()

    def _capture(
        self,
        entry: FullDecodeGraphEntry,
        actual_seq_lens: list[int] | None = None,
    ) -> None:
        if actual_seq_lens is None:
            actual_seq_lens = [1] * entry.batch_size
        self._set_capture_context(entry, actual_seq_lens)
        capture_context = get_context()
        model_warmup = getattr(
            self.model, "full_decode_graph_eager_warmup", None
        )
        if callable(model_warmup):
            if self.log_enabled:
                logger.info(
                    "FULL_DECODE_ONLY: running model-specific balanced "
                    "eager warmup for batch_size=%d.",
                    entry.batch_size,
                )
            warmup_passes = model_warmup(
                entry.input_ids, entry.positions
            )
            if self.log_enabled:
                logger.info(
                    "FULL_DECODE_ONLY: balanced eager warmup completed "
                    "in %d pass(es).",
                    int(warmup_passes),
                )

        capture_context.scratch.clear()
        self._callable(entry.input_ids, entry.positions)
        torch.npu.synchronize()
        gc.collect()
        torch.npu.empty_cache()

        entry.mla_tasks.clear()
        entry.graph = torch.npu.NPUGraph()
        capture_context.scratch.clear()
        with _record_graph_tasks(entry.mla_tasks):
            with torch.npu.graph(entry.graph, pool=self._graph_pool):
                entry.output = self._callable(entry.input_ids, entry.positions)
        torch.npu.synchronize()

        if not isinstance(entry.output, torch.Tensor):
            raise RuntimeError(
                "FULL_DECODE_ONLY model forward must return one hidden-state "
                f"Tensor, got {type(entry.output).__name__}."
            )
        if len(entry.mla_tasks) != self.expected_mla_tasks:
            raise RuntimeError(
                "FULL_DECODE_ONLY captured an unexpected number of MLA tasks: "
                f"expected={self.expected_mla_tasks}, actual={len(entry.mla_tasks)}, "
                f"batch_size={entry.batch_size}. This graph cannot safely refresh "
                "actual_seq_kvlen."
            )
        self.capture_count += 1
        if self.log_enabled:
            logger.info(
                "FULL_DECODE_ONLY: captured complete decode graph for "
                "batch_size=%d with %d refreshable MLA tasks.",
                entry.batch_size,
                len(entry.mla_tasks),
            )

    def _capture_lidu_runtime(
        self,
        entry: FullDecodeGraphEntry,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        runtime_context: Context,
    ) -> None:
        """Capture LIDU only after every request row has real initialized state."""

        self._ensure_capture_resources()
        seq_lens = entry.copy_runtime_inputs(
            input_ids,
            positions,
            runtime_context,
            offload_mode=self.offload_mode,
            uses_tensor_mla_lengths=getattr(
                self, "uses_tensor_mla_lengths", False
            ),
        )
        torch.npu.current_stream().synchronize()
        if self.log_enabled:
            logger.info(
                "FULL_DECODE_ONLY: lazily capturing initialized LIDU decode "
                "for batch_size=%d; this decode step remains eager.",
                entry.batch_size,
            )
        distributed = dist.is_initialized() and dist.get_world_size() > 1
        if distributed:
            dist.barrier()
        with preserve_context():
            self._capture(entry, seq_lens)
        if distributed:
            dist.barrier()

    def _run_eager(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        return self.model(input_ids, positions)

    def run(self, input_ids: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        runtime_context = get_context()
        if runtime_context.is_prefill:
            self.eager_prefill_count += 1
            return self._run_eager(input_ids, positions)
        if runtime_context.has_first_decode:
            self.eager_first_decode_count += 1
            return self._run_eager(input_ids, positions)
        batch_size = int(input_ids.shape[0])
        if self.stateful_offload:
            if not runtime_context.needs_dsa_update:
                self.eager_no_dsa_count += 1
                return self._run_eager(input_ids, positions)
            if not runtime_context.lidu_all_rows_ready:
                self.eager_lidu_uninitialized_count += 1
                return self._run_eager(input_ids, positions)
            graph_batch_size = (
                batch_size if batch_size in self.capture_sizes else None
            )
        else:
            graph_batch_size = select_capture_size(
                batch_size, self.capture_sizes
            )
        if graph_batch_size is None:
            self.eager_uncaptured_batch_count += 1
            return self._run_eager(input_ids, positions)
        entry = self._entries.get(graph_batch_size)
        if entry is None or entry.graph is None or entry.output is None:
            if self.stateful_offload:
                entry = entry or self._allocate_entry(graph_batch_size)
                self._capture_lidu_runtime(
                    entry,
                    input_ids,
                    positions,
                    runtime_context,
                )
                self.eager_lidu_capture_count += 1
                return self._run_eager(input_ids, positions)
            raise RuntimeError(
                "FULL_DECODE_ONLY graph was not pre-captured for exact batch "
                f"size {graph_batch_size}."
            )

        seq_lens = entry.copy_runtime_inputs(
            input_ids,
            positions,
            runtime_context,
            offload_mode=self.offload_mode,
            uses_tensor_mla_lengths=getattr(
                self, "uses_tensor_mla_lengths", False
            ),
        )

        # Wait for H2D metadata staging and the previous replay before changing
        # task parameters. The graph itself is enqueued first; captured external
        # events then let early graph work overlap later-layer host task updates.
        torch.npu.current_stream().synchronize()
        entry.graph.replay()
        with torch.npu.stream(self._update_stream):
            for task in entry.mla_tasks:
                task.update(self._update_stream, seq_lens)
        if self.log_enabled and self.replay_count == 0:
            logger.info(
                "FULL_DECODE_ONLY: first complete graph replay entered for "
                "runtime_batch=%d, graph_batch=%d, offload_mode=%s.",
                batch_size,
                graph_batch_size,
                self.offload_mode,
            )
        entry.replay_count += 1
        self.replay_count += 1
        return entry.output[:batch_size]

    def is_stable_replay_ready(self, runtime_batch_size: int) -> bool:
        """Return whether the next decode uses an already-warmed graph entry."""

        if self.stateful_offload:
            graph_batch_size = (
                runtime_batch_size
                if runtime_batch_size in self.capture_sizes
                else None
            )
        else:
            graph_batch_size = select_capture_size(
                runtime_batch_size,
                self.capture_sizes,
            )
        if graph_batch_size is None:
            return False
        entry = self._entries.get(graph_batch_size)
        return bool(
            entry is not None
            and entry.graph is not None
            and entry.output is not None
            and entry.replay_count > 0
        )

    def stats(self) -> dict[str, Any]:
        return {
            "enabled": True,
            "mode": FULL_DECODE_ONLY,
            "capture_sizes": list(self.capture_sizes),
            "offload_mode": self.offload_mode,
            "exact_size_only": self.stateful_offload,
            "captures": self.capture_count,
            "replays": self.replay_count,
            "eager_prefill": self.eager_prefill_count,
            "eager_first_decode": self.eager_first_decode_count,
            "eager_no_dsa": self.eager_no_dsa_count,
            "eager_lidu_uninitialized": self.eager_lidu_uninitialized_count,
            "eager_lidu_capture": self.eager_lidu_capture_count,
            "eager_uncaptured_batch": self.eager_uncaptured_batch_count,
            "metadata_refreshes": sum(
                entry.metadata_refresh_count for entry in self._entries.values()
            ),
            "metadata_reuses": sum(
                entry.metadata_reuse_count for entry in self._entries.values()
            ),
        }


@dataclass
class MTPDecodeGraphEntry:
    """Fixed-address inputs and outputs for one exact MTP batch size."""

    batch_size: int
    speculative_tokens: int
    max_block_columns: int
    input_ids: torch.Tensor
    positions: torch.Tensor
    draft_token_ids: torch.Tensor
    flat_slot_mapping: torch.Tensor
    flat_slot_mapping_i32: torch.Tensor
    cu_seqlens_q: torch.Tensor
    block_tables: torch.Tensor
    mtp_block_tables: torch.Tensor
    mtp_index_block_tables: torch.Tensor
    mtp_req_pool_entries: torch.Tensor
    mtp_candidate_lens: torch.Tensor
    mtp_lidu_cache_tokens: torch.Tensor
    mtp_actual_seq_lengths_by_step: list[torch.Tensor]
    mtp_target_actual_seq_lengths_by_step: list[torch.Tensor]
    actual_seq_lengths_kv: torch.Tensor
    index_block_tables: torch.Tensor
    dram_block_tables: torch.Tensor
    req_pool_entries: torch.Tensor
    candidate_lens: torch.Tensor
    candidate_query_lens: torch.Tensor
    lidu_cache_tokens: torch.Tensor
    decode_metadata_key: tuple[tuple[int, int], ...] | None = None
    target_graph: Any | None = None
    draft_graph: Any | None = None
    target_tokens: torch.Tensor | None = None
    accepted_counts: torch.Tensor | None = None
    next_token_ids: torch.Tensor | None = None
    selected_hidden_states: torch.Tensor | None = None
    selected_positions: torch.Tensor | None = None
    next_drafts: torch.Tensor | None = None
    target_tasks: list[MLAGraphTask] = field(default_factory=list)
    draft_tasks: list[MLAGraphTask] = field(default_factory=list)
    replay_count: int = 0
    metadata_refresh_count: int = 0
    metadata_reuse_count: int = 0

    @property
    def query_len(self) -> int:
        return self.speculative_tokens + 1

    @classmethod
    def allocate(
        cls,
        batch_size: int,
        speculative_tokens: int,
        max_block_columns: int,
        device: torch.device,
    ) -> "MTPDecodeGraphEntry":
        query_len = speculative_tokens + 1
        total_tokens = batch_size * query_len
        return cls(
            batch_size=batch_size,
            speculative_tokens=speculative_tokens,
            max_block_columns=max_block_columns,
            input_ids=torch.zeros(
                total_tokens, dtype=torch.int64, device=device
            ),
            positions=torch.zeros(
                total_tokens, dtype=torch.int64, device=device
            ),
            draft_token_ids=torch.zeros(
                batch_size,
                speculative_tokens,
                dtype=torch.int64,
                device=device,
            ),
            flat_slot_mapping=torch.zeros(
                total_tokens, dtype=torch.int64, device=device
            ),
            flat_slot_mapping_i32=torch.zeros(
                total_tokens, dtype=torch.int32, device=device
            ),
            cu_seqlens_q=torch.arange(
                0,
                total_tokens + 1,
                query_len,
                dtype=torch.int32,
                device=device,
            ),
            block_tables=torch.zeros(
                batch_size,
                max_block_columns,
                dtype=torch.int32,
                device=device,
            ),
            mtp_block_tables=torch.zeros(
                batch_size,
                max_block_columns,
                dtype=torch.int32,
                device=device,
            ),
            mtp_index_block_tables=torch.zeros(
                batch_size,
                max_block_columns,
                dtype=torch.int32,
                device=device,
            ),
            mtp_req_pool_entries=torch.zeros(
                batch_size, dtype=torch.int32, device=device
            ),
            mtp_candidate_lens=torch.zeros(
                batch_size, dtype=torch.int32, device=device
            ),
            mtp_lidu_cache_tokens=torch.zeros(
                batch_size, dtype=torch.int32, device=device
            ),
            mtp_actual_seq_lengths_by_step=[
                torch.zeros(batch_size, dtype=torch.int32, device=device)
                for _ in range(speculative_tokens)
            ],
            mtp_target_actual_seq_lengths_by_step=[
                torch.zeros(batch_size, dtype=torch.int32, device=device)
                for _ in range(query_len)
            ],
            actual_seq_lengths_kv=torch.zeros(
                batch_size, dtype=torch.int32, device=device
            ),
            index_block_tables=torch.zeros(
                batch_size,
                max_block_columns,
                dtype=torch.int32,
                device=device,
            ),
            dram_block_tables=torch.zeros(
                batch_size,
                max_block_columns,
                dtype=torch.int32,
                device=device,
            ),
            req_pool_entries=torch.zeros(
                batch_size, dtype=torch.int32, device=device
            ),
            candidate_lens=torch.zeros(
                batch_size, dtype=torch.int32, device=device
            ),
            candidate_query_lens=torch.arange(
                query_len,
                (batch_size + 1) * query_len,
                query_len,
                dtype=torch.int32,
                device=device,
            ),
            lidu_cache_tokens=torch.zeros(
                batch_size, dtype=torch.int32, device=device
            ),
        )

    def _copy_table(
        self,
        destination: torch.Tensor,
        source: torch.Tensor,
        name: str,
    ) -> None:
        if int(source.shape[0]) != self.batch_size:
            raise ValueError(
                f"MTP {name} batch changed: expected={self.batch_size}, "
                f"actual={source.shape[0]}."
            )
        columns = int(source.shape[1])
        if columns > self.max_block_columns:
            raise ValueError(
                f"MTP {name} is wider than its graph buffer: "
                f"runtime={columns}, graph={self.max_block_columns}."
            )
        destination.zero_()
        destination[:, :columns].copy_(source)

    def copy_runtime_inputs(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        draft_token_ids: torch.Tensor,
        context: Context,
        mtp_block_tables: torch.Tensor | None = None,
        mtp_index_share: Any | None = None,
        *,
        offload_mode: str = OFFLOAD_NONE,
    ) -> None:
        expected_tokens = self.batch_size * self.query_len
        if tuple(input_ids.shape) != (expected_tokens,):
            raise ValueError(
                "MTP target graph input shape changed: "
                f"expected={(expected_tokens,)}, actual={tuple(input_ids.shape)}."
            )
        if tuple(positions.shape) != (expected_tokens,):
            raise ValueError(
                "MTP target graph position shape changed: "
                f"expected={(expected_tokens,)}, actual={tuple(positions.shape)}."
            )
        if tuple(draft_token_ids.shape) != (
            self.batch_size,
            self.speculative_tokens,
        ):
            raise ValueError(
                "MTP draft shape changed: expected="
                f"{(self.batch_size, self.speculative_tokens)}, "
                f"actual={tuple(draft_token_ids.shape)}."
            )
        if context.flat_slot_mapping is None or context.block_tables is None:
            raise RuntimeError(
                "MTP target graph requires slot mapping and block tables."
            )
        if tuple(context.flat_slot_mapping.shape) != (expected_tokens,):
            raise ValueError(
                "MTP target graph slot shape changed: "
                f"expected={(expected_tokens,)}, "
                f"actual={tuple(context.flat_slot_mapping.shape)}."
            )
        self.input_ids.copy_(input_ids)
        self.positions.copy_(positions)
        self.draft_token_ids.copy_(draft_token_ids)
        self.flat_slot_mapping.copy_(context.flat_slot_mapping)
        runtime_slots_i32 = context.flat_slot_mapping_i32
        if runtime_slots_i32 is None:
            runtime_slots_i32 = context.flat_slot_mapping.to(torch.int32)
        self.flat_slot_mapping_i32.copy_(runtime_slots_i32)
        stateful_offload = normalize_offload_mode(offload_mode) != OFFLOAD_NONE
        if stateful_offload:
            required = {
                "actual_seq_lengths_kv_tensor": (
                    context.actual_seq_lengths_kv_tensor
                ),
                "index_block_tables": context.index_block_tables,
                "dram_block_tables": context.dram_block_tables,
                "req_pool_entries": context.req_pool_entries,
                "candidate_lens": context.candidate_lens,
                "candidate_query_lens": context.candidate_query_lens,
                "lidu_cache_tokens": context.lidu_cache_tokens,
            }
            missing = [
                name for name, value in required.items() if value is None
            ]
            if missing:
                raise RuntimeError(
                    "MTP FULL_DECODE_ONLY is missing offload metadata: "
                    + ", ".join(missing)
                )
            self.actual_seq_lengths_kv.copy_(
                context.actual_seq_lengths_kv_tensor
            )

        if mtp_block_tables is None:
            mtp_block_tables = context.block_tables
        metadata_key = context.decode_metadata_key
        refresh_metadata = (
            metadata_key is None or metadata_key != self.decode_metadata_key
        )
        if refresh_metadata:
            self._copy_table(
                self.block_tables, context.block_tables, "block_tables"
            )
            self._copy_table(
                self.mtp_block_tables,
                mtp_block_tables,
                "mtp_block_tables",
            )
            if mtp_index_share is not None:
                required = {
                    "block_tables": getattr(
                        mtp_index_share, "block_tables", None
                    ),
                    "req_pool_entries": getattr(
                        mtp_index_share, "req_pool_entries", None
                    ),
                    "candidate_lens": getattr(
                        mtp_index_share, "candidate_lens", None
                    ),
                    "lidu_cache_tokens": getattr(
                        mtp_index_share, "lidu_cache_tokens", None
                    ),
                }
                missing = [
                    name for name, value in required.items()
                    if value is None
                ]
                if missing:
                    raise RuntimeError(
                        "MTP graph IndexShare metadata is missing: "
                        + ", ".join(missing)
                    )
                self._copy_table(
                    self.mtp_index_block_tables,
                    required["block_tables"],
                    "mtp_index_block_tables",
                )
                self.mtp_req_pool_entries.copy_(
                    required["req_pool_entries"]
                )
                self.mtp_candidate_lens.copy_(
                    required["candidate_lens"]
                )
                self.mtp_lidu_cache_tokens.copy_(
                    required["lidu_cache_tokens"]
                )
            if stateful_offload:
                self._copy_table(
                    self.index_block_tables,
                    context.index_block_tables,
                    "index_block_tables",
                )
                self._copy_table(
                    self.dram_block_tables,
                    context.dram_block_tables,
                    "dram_block_tables",
                )
                self.req_pool_entries.copy_(context.req_pool_entries)
                self.candidate_lens.copy_(context.candidate_lens)
                self.candidate_query_lens.copy_(
                    context.candidate_query_lens
                )
                self.lidu_cache_tokens.copy_(context.lidu_cache_tokens)
            self.decode_metadata_key = metadata_key
            self.metadata_refresh_count += 1
        else:
            self.metadata_reuse_count += 1

    def stage_mtp_actual_seq_lengths(
        self,
        values_by_step: list[list[int]],
    ) -> None:
        if len(values_by_step) != self.speculative_tokens:
            raise ValueError("MTP graph draft length step count changed.")
        for destination, values in zip(
            self.mtp_actual_seq_lengths_by_step, values_by_step
        ):
            if len(values) != self.batch_size:
                raise ValueError("MTP graph draft length batch changed.")
            destination.copy_(
                torch.tensor(
                    values,
                    dtype=torch.int32,
                    device=destination.device,
                )
            )

    def stage_mtp_target_actual_seq_lengths(self) -> None:
        """Stage the four serial target SFA KV lengths at fixed addresses."""

        base = self.actual_seq_lengths_kv - self.speculative_tokens
        for step, destination in enumerate(
            self.mtp_target_actual_seq_lengths_by_step
        ):
            destination.copy_(base + step)


class MTPDecodeOnlyGraphManager:
    """Two exact-size graphs for steady GLM MTP verification and drafting.

    Target verification determines how many draft tokens were accepted. FIA-v2
    exposes KV lengths as host attributes, so one synchronization between the
    target and draft graphs is intentional: it refreshes the three draft FIA
    tasks with the current accepted prefix lengths.
    """

    def __init__(
        self,
        *,
        target_forward: Callable[..., tuple[torch.Tensor, ...]],
        draft_forward: Callable[..., torch.Tensor],
        target_warmup: Callable[..., int] | None,
        draft_warmup: Callable[..., int] | None,
        capture_sizes: Iterable[int],
        max_model_len: int,
        block_size: int,
        device: str,
        speculative_tokens: int,
        expected_target_tasks: int,
        serial_target_verification: bool = False,
        offload_mode: str = OFFLOAD_NONE,
        log_enabled: bool = True,
    ) -> None:
        if int(speculative_tokens) != 3:
            raise ValueError(
                "MTP FULL_DECODE_ONLY currently supports K=3 only."
            )
        self.target_forward = target_forward
        self.draft_forward = draft_forward
        self.target_warmup = target_warmup
        self.draft_warmup = draft_warmup
        self.capture_sizes = normalize_capture_sizes(capture_sizes)
        self.max_block_columns = (
            int(max_model_len) + int(block_size) - 1
        ) // int(block_size)
        self.block_size = int(block_size)
        self.device = torch.device(device)
        self.speculative_tokens = int(speculative_tokens)
        self.expected_target_tasks = int(expected_target_tasks)
        self.serial_target_verification = bool(serial_target_verification)
        self.log_enabled = bool(log_enabled)
        self.offload_mode = normalize_offload_mode(offload_mode)
        self.stateful_offload = self.offload_mode != OFFLOAD_NONE
        self._entries: dict[int, MTPDecodeGraphEntry] = {}
        self._eager_seen: set[int] = set()
        self._graph_pool = None
        self._update_stream = None
        self.capture_count = 0
        self.replay_count = 0
        self.target_replay_count = 0
        self.draft_replay_count = 0
        self.eager_first_decode_count = 0
        self.eager_lidu_uninitialized_count = 0
        self.eager_uncaptured_batch_count = 0
        self.eager_capture_count = 0

    def should_use_graph(
        self,
        batch_size: int,
        runtime_context: Context | None = None,
    ) -> bool:
        """Keep the first exact-size verification eager, then graph it."""

        batch_size = int(batch_size)
        if batch_size not in self.capture_sizes:
            self.eager_uncaptured_batch_count += 1
            return False
        if self.stateful_offload and runtime_context is not None:
            if runtime_context.has_first_decode:
                self._eager_seen.add(batch_size)
                self.eager_first_decode_count += 1
                return False
            if not runtime_context.lidu_all_rows_ready:
                self.eager_lidu_uninitialized_count += 1
                return False
        if batch_size not in self._eager_seen:
            self._eager_seen.add(batch_size)
            self.eager_first_decode_count += 1
            return False
        return True

    def _ensure_capture_resources(self) -> None:
        if self._graph_pool is None:
            self._graph_pool = torch.npu.graph_pool_handle()
        if self._update_stream is None:
            self._update_stream = torch.npu.Stream()

    def _allocate_entry(self, batch_size: int) -> MTPDecodeGraphEntry:
        entry = MTPDecodeGraphEntry.allocate(
            batch_size,
            self.speculative_tokens,
            self.max_block_columns,
            self.device,
        )
        self._entries[batch_size] = entry
        return entry

    @staticmethod
    def _target_seq_lengths(
        base_seq_lengths: list[int], speculative_tokens: int
    ) -> list[int]:
        return [
            int(length) + int(speculative_tokens)
            for length in base_seq_lengths
        ]

    def _target_task_seq_lengths(
        self, base_seq_lengths: list[int]
    ) -> list[int] | list[list[int]]:
        """Return the KV length each captured target attention task observes."""
        if not self.serial_target_verification:
            return self._target_seq_lengths(
                base_seq_lengths, self.speculative_tokens
            )
        return [
            [int(length) + step for length in base_seq_lengths]
            for step in range(self.speculative_tokens + 1)
        ]

    @staticmethod
    def _draft_seq_lengths(
        base_seq_lengths: list[int],
        accepted_counts: list[int],
        speculative_tokens: int,
    ) -> list[list[int]]:
        if len(base_seq_lengths) != len(accepted_counts):
            raise ValueError("MTP accepted-count batch size changed.")
        return [
            [
                int(length) + int(accepted) + step
                for length, accepted in zip(
                    base_seq_lengths, accepted_counts
                )
            ]
            for step in range(int(speculative_tokens))
        ]

    def _set_target_context(
        self,
        entry: MTPDecodeGraphEntry,
        target_seq_lengths: list[int],
    ) -> None:
        query_len = entry.query_len
        actual_seq_lengths_q = [
            (row + 1) * query_len for row in range(entry.batch_size)
        ]
        set_context(
            False,
            is_spec_decode=True,
            cu_seqlens_q=entry.cu_seqlens_q,
            actual_seq_lengths_q=actual_seq_lengths_q,
            flat_slot_mapping=entry.flat_slot_mapping,
            flat_slot_mapping_i32=entry.flat_slot_mapping_i32,
            actual_seq_lengths_kv=target_seq_lengths,
            actual_seq_lengths_kv_tensor=(
                entry.actual_seq_lengths_kv
                if self.stateful_offload
                else None
            ),
            mtp_target_actual_seq_lengths_by_step=(
                entry.mtp_target_actual_seq_lengths_by_step
                if self.stateful_offload
                else None
            ),
            block_tables=entry.block_tables,
            index_block_tables=(
                entry.index_block_tables if self.stateful_offload else None
            ),
            dram_block_tables=(
                entry.dram_block_tables if self.stateful_offload else None
            ),
            req_pool_entries=(
                entry.req_pool_entries if self.stateful_offload else None
            ),
            candidate_lens=(
                entry.candidate_lens if self.stateful_offload else None
            ),
            candidate_query_lens=(
                entry.candidate_query_lens
                if self.stateful_offload
                else None
            ),
            lidu_cache_tokens=(
                entry.lidu_cache_tokens if self.stateful_offload else None
            ),
            needs_dsa_update=self.stateful_offload,
            lidu_all_rows_ready=self.stateful_offload,
            has_first_decode=False,
            full_decode_graph=True,
        )

    @staticmethod
    def _validate_target_outputs(
        entry: MTPDecodeGraphEntry,
        outputs: tuple[torch.Tensor, ...],
    ) -> None:
        if not isinstance(outputs, tuple) or len(outputs) != 5:
            raise RuntimeError(
                "MTP target graph must return target tokens, accepted counts, "
                "next token IDs, selected hidden states and positions."
            )
        (
            entry.target_tokens,
            entry.accepted_counts,
            entry.next_token_ids,
            entry.selected_hidden_states,
            entry.selected_positions,
        ) = outputs
        if tuple(entry.target_tokens.shape) != (
            entry.batch_size,
            entry.query_len,
        ):
            raise RuntimeError(
                "MTP target graph returned an unexpected token shape: "
                f"{tuple(entry.target_tokens.shape)}."
            )
        if tuple(entry.accepted_counts.shape) != (entry.batch_size,):
            raise RuntimeError(
                "MTP target graph returned an unexpected acceptance shape: "
                f"{tuple(entry.accepted_counts.shape)}."
            )

    def _replay_target_graph(
        self,
        entry: MTPDecodeGraphEntry,
        target_task_seq_lengths: list[int] | list[list[int]],
    ) -> None:
        torch.npu.current_stream().synchronize()
        entry.target_graph.replay()
        with torch.npu.stream(self._update_stream):
            if not self.serial_target_verification:
                for task in entry.target_tasks:
                    task.update(self._update_stream, target_task_seq_lengths)
                return
            target_steps = self.speculative_tokens + 1
            if (
                not isinstance(target_task_seq_lengths[0], list)
                or len(target_task_seq_lengths) != target_steps
                or len(entry.target_tasks) % target_steps != 0
            ):
                raise RuntimeError(
                    "Serial MTP target graph task metadata is inconsistent "
                    "with the configured speculative depth."
                )
            tasks_per_step = len(entry.target_tasks) // target_steps
            for step, task_seq_lengths in enumerate(target_task_seq_lengths):
                start = step * tasks_per_step
                for task in entry.target_tasks[start : start + tasks_per_step]:
                    task.update(self._update_stream, task_seq_lengths)

    def _replay_draft_graph(
        self,
        entry: MTPDecodeGraphEntry,
        draft_seq_lengths: list[list[int]],
        *,
        use_mtp_index_share: bool = False,
    ) -> None:
        if len(entry.draft_tasks) not in (0, len(draft_seq_lengths)):
            raise RuntimeError(
                "MTP draft graph task count changed before replay: "
                f"tasks={len(entry.draft_tasks)}, "
                f"steps={len(draft_seq_lengths)}."
            )
        if use_mtp_index_share:
            entry.stage_mtp_actual_seq_lengths(draft_seq_lengths)
        entry.draft_graph.replay()
        with torch.npu.stream(self._update_stream):
            for task, seq_lengths in zip(
                entry.draft_tasks, draft_seq_lengths
            ):
                task.update(self._update_stream, seq_lengths)

    def _record_complete_replay(self, entry: MTPDecodeGraphEntry) -> None:
        entry.replay_count += 1
        self.replay_count += 1
        self.target_replay_count += 1
        self.draft_replay_count += 1

    def _call_draft_forward(
        self,
        entry: MTPDecodeGraphEntry,
        draft_seq_lengths: list[list[int]],
        *,
        use_mtp_index_share: bool,
    ) -> torch.Tensor:
        args = (
            entry.next_token_ids,
            entry.selected_positions,
            entry.selected_hidden_states,
            entry.mtp_block_tables,
            draft_seq_lengths,
        )
        if use_mtp_index_share:
            return self.draft_forward(*args, entry)
        return self.draft_forward(*args)

    def _call_draft_warmup(
        self,
        entry: MTPDecodeGraphEntry,
        draft_seq_lengths: list[list[int]],
        *,
        use_mtp_index_share: bool,
    ) -> int:
        if self.draft_warmup is None:
            return 0
        args = (
            entry.next_token_ids,
            entry.selected_positions,
            entry.selected_hidden_states,
            entry.mtp_block_tables,
            draft_seq_lengths,
        )
        if use_mtp_index_share:
            return self.draft_warmup(*args, entry)
        return self.draft_warmup(*args)

    def _capture(
        self,
        entry: MTPDecodeGraphEntry,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        draft_token_ids: torch.Tensor,
        runtime_context: Context,
        base_seq_lengths: list[int],
        mtp_block_tables: torch.Tensor | None = None,
        mtp_index_share: Any | None = None,
    ) -> None:
        self._ensure_capture_resources()
        distributed = dist.is_initialized() and dist.get_world_size() > 1
        torch.npu.current_stream().synchronize()
        if distributed:
            dist.barrier()
        try:
            with preserve_context(), torch.inference_mode():
                entry.copy_runtime_inputs(
                    input_ids,
                    positions,
                    draft_token_ids,
                    runtime_context,
                    mtp_block_tables,
                    mtp_index_share,
                    offload_mode=self.offload_mode,
                )
                if self.stateful_offload:
                    entry.stage_mtp_target_actual_seq_lengths()
                target_seq_lengths = self._target_seq_lengths(
                    base_seq_lengths, self.speculative_tokens
                )
                self._set_target_context(entry, target_seq_lengths)
                if self.target_warmup is not None:
                    self.target_warmup(entry.input_ids, entry.positions)
                    entry.copy_runtime_inputs(
                        input_ids,
                        positions,
                        draft_token_ids,
                        runtime_context,
                        mtp_block_tables,
                        mtp_index_share,
                        offload_mode=self.offload_mode,
                    )
                    if self.stateful_offload:
                        entry.stage_mtp_target_actual_seq_lengths()
                    self._set_target_context(entry, target_seq_lengths)

                # Allocate FIA workspaces and output buffers before capture.
                self.target_forward(
                    entry.input_ids,
                    entry.positions,
                    entry.draft_token_ids,
                )
                torch.npu.synchronize()
                gc.collect()
                torch.npu.empty_cache()

                entry.target_tasks.clear()
                entry.target_graph = torch.npu.NPUGraph()
                self._set_target_context(entry, target_seq_lengths)
                with _record_graph_tasks(entry.target_tasks):
                    with torch.npu.graph(
                        entry.target_graph, pool=self._graph_pool
                    ):
                        target_outputs = self.target_forward(
                            entry.input_ids,
                            entry.positions,
                            entry.draft_token_ids,
                        )
                torch.npu.synchronize()
                self._validate_target_outputs(entry, target_outputs)
                if len(entry.target_tasks) != self.expected_target_tasks:
                    raise RuntimeError(
                        "MTP target graph captured an unexpected number of "
                        f"FIA tasks: expected={self.expected_target_tasks}, "
                        f"actual={len(entry.target_tasks)}."
                    )

                # External FIA tasks do not guarantee usable capture-time
                # outputs. Replay target once before consuming acceptance.
                self._replay_target_graph(
                    entry,
                    self._target_task_seq_lengths(base_seq_lengths),
                )
                accepted_counts = entry.accepted_counts.cpu().tolist()
                draft_seq_lengths = self._draft_seq_lengths(
                    base_seq_lengths,
                    accepted_counts,
                    self.speculative_tokens,
                )
                if mtp_index_share is not None:
                    entry.stage_mtp_actual_seq_lengths(draft_seq_lengths)
                if self.draft_warmup is not None:
                    self._call_draft_warmup(
                        entry,
                        draft_seq_lengths,
                        use_mtp_index_share=mtp_index_share is not None,
                    )
                else:
                    self._call_draft_forward(
                        entry,
                        draft_seq_lengths,
                        use_mtp_index_share=mtp_index_share is not None,
                    )
                torch.npu.synchronize()
                gc.collect()
                torch.npu.empty_cache()

                entry.draft_tasks.clear()
                entry.draft_graph = torch.npu.NPUGraph()
                with _record_graph_tasks(entry.draft_tasks):
                    with torch.npu.graph(
                        entry.draft_graph, pool=self._graph_pool
                    ):
                        entry.next_drafts = self._call_draft_forward(
                            entry,
                            draft_seq_lengths,
                            use_mtp_index_share=mtp_index_share is not None,
                        )
                torch.npu.synchronize()
                expected_draft_tasks = (
                    0 if mtp_index_share is not None else self.speculative_tokens
                )
                if len(entry.draft_tasks) != expected_draft_tasks:
                    raise RuntimeError(
                        "MTP draft graph captured an unexpected number of FIA "
                        f"tasks: expected={expected_draft_tasks}, "
                        f"actual={len(entry.draft_tasks)}."
                    )
                if tuple(entry.next_drafts.shape) != (
                    entry.batch_size,
                    self.speculative_tokens,
                ):
                    raise RuntimeError(
                        "MTP draft graph returned an unexpected shape: "
                        f"{tuple(entry.next_drafts.shape)}."
                    )
                # For the same reason, the capture step returns only the first
                # real replay of the draft graph, never capture-time storage.
                self._replay_draft_graph(
                    entry,
                    draft_seq_lengths,
                    use_mtp_index_share=mtp_index_share is not None,
                )
                torch.npu.synchronize()
                self._record_complete_replay(entry)
                self.capture_count += 1
                self.eager_capture_count += 1
                if self.log_enabled:
                    logger.info(
                        "FULL_DECODE_ONLY MTP: captured exact batch_size=%d "
                        "as target(%d FIA tasks) + draft(%d FIA tasks).",
                        entry.batch_size,
                        len(entry.target_tasks),
                        len(entry.draft_tasks),
                    )
        finally:
            if distributed:
                dist.barrier()

    def _replay(
        self,
        entry: MTPDecodeGraphEntry,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        draft_token_ids: torch.Tensor,
        runtime_context: Context,
        base_seq_lengths: list[int],
        mtp_block_tables: torch.Tensor | None = None,
        mtp_index_share: Any | None = None,
    ) -> None:
        entry.copy_runtime_inputs(
            input_ids,
            positions,
            draft_token_ids,
            runtime_context,
            mtp_block_tables,
            mtp_index_share,
            offload_mode=self.offload_mode,
        )
        if self.stateful_offload:
            entry.stage_mtp_target_actual_seq_lengths()
        self._replay_target_graph(
            entry,
            self._target_task_seq_lengths(base_seq_lengths),
        )
        accepted_counts = entry.accepted_counts.cpu().tolist()

        draft_seq_lengths = self._draft_seq_lengths(
            base_seq_lengths,
            accepted_counts,
            self.speculative_tokens,
        )
        self._replay_draft_graph(
            entry,
            draft_seq_lengths,
            use_mtp_index_share=mtp_index_share is not None,
        )
        self._record_complete_replay(entry)
        if self.log_enabled and self.replay_count == 1:
            logger.info(
                "FULL_DECODE_ONLY MTP: first target+draft replay entered for "
                "exact batch_size=%d.",
                entry.batch_size,
            )

    def run(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        draft_token_ids: torch.Tensor,
        runtime_context: Context,
        base_seq_lengths: list[int],
        mtp_block_tables: torch.Tensor | None = None,
        mtp_index_share: Any | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size = len(base_seq_lengths)
        if batch_size not in self.capture_sizes:
            raise ValueError(
                "MTP graph run requires an exact configured batch size: "
                f"runtime={batch_size}, capture_sizes={self.capture_sizes}."
            )
        entry = self._entries.get(batch_size)
        if entry is None:
            entry = self._allocate_entry(batch_size)
        if entry.target_graph is None or entry.draft_graph is None:
            self._capture(
                entry,
                input_ids,
                positions,
                draft_token_ids,
                runtime_context,
                base_seq_lengths,
                mtp_block_tables,
                mtp_index_share,
            )
        else:
            self._replay(
                entry,
                input_ids,
                positions,
                draft_token_ids,
                runtime_context,
                base_seq_lengths,
                mtp_block_tables,
                mtp_index_share,
            )
        return entry.target_tokens, entry.accepted_counts, entry.next_drafts

    def is_stable_replay_ready(self, runtime_batch_size: int) -> bool:
        entry = self._entries.get(int(runtime_batch_size))
        return bool(entry is not None and entry.replay_count > 0)

    def stats(self) -> dict[str, Any]:
        return {
            "enabled": True,
            "mode": FULL_DECODE_ONLY,
            "capture_sizes": list(self.capture_sizes),
            "offload_mode": self.offload_mode,
            "serial_target_verification": self.serial_target_verification,
            "exact_size_only": True,
            "captures": self.capture_count,
            "replays": self.replay_count,
            "mtp_target_captures": self.capture_count,
            "mtp_draft_captures": self.capture_count,
            "mtp_target_replays": self.target_replay_count,
            "mtp_draft_replays": self.draft_replay_count,
            "eager_prefill": 0,
            "eager_first_decode": self.eager_first_decode_count,
            "eager_no_dsa": 0,
            "eager_lidu_uninitialized": self.eager_lidu_uninitialized_count,
            "eager_lidu_capture": self.eager_capture_count,
            "eager_uncaptured_batch": self.eager_uncaptured_batch_count,
            "eager_mtp_capture": self.eager_capture_count,
            "metadata_refreshes": sum(
                entry.metadata_refresh_count for entry in self._entries.values()
            ),
            "metadata_reuses": sum(
                entry.metadata_reuse_count for entry in self._entries.values()
            ),
        }
