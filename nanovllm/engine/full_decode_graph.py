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
    DSA_SELECTION_TOPK_TOKENS,
    OFFLOAD_GS,
    OFFLOAD_LIDU,
    OFFLOAD_NONE,
    max_lidu_cache_tokens,
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
                input_layout="BNSD_NBSD",
                atten_mask=None,
                sparse_mode=0,
                softmax_scale=self.softmax_scale,
                block_table=self.block_table,
                block_size=self.block_size,
                actual_seq_qlen=None,
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
    selection_block_columns: int
    input_ids: torch.Tensor
    positions: torch.Tensor
    flat_slot_mapping_i32: torch.Tensor
    block_tables: torch.Tensor
    index_block_tables: torch.Tensor
    dram_block_tables: torch.Tensor
    selection_block_tables: torch.Tensor
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
        selection_block_columns: int,
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
            selection_block_columns=selection_block_columns,
            input_ids=torch.zeros(batch_size, dtype=torch.int64, device=device),
            positions=torch.zeros(batch_size, dtype=torch.int64, device=device),
            flat_slot_mapping_i32=flat_slot_mapping_i32,
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
            selection_block_tables=torch.zeros(
                batch_size,
                selection_block_columns,
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
                DSA_SELECTION_TOPK_TOKENS,
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
                "would mutate another request's persistent gather status: "
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
                selection_block_tables=context.selection_block_tables,
                req_pool_entries=context.req_pool_entries,
                candidate_lens=context.candidate_lens,
                candidate_query_lens=context.candidate_query_lens,
            )
            if offload_mode == OFFLOAD_LIDU:
                required_metadata["lidu_cache_tokens"] = (
                    context.lidu_cache_tokens
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
                if offload_mode == OFFLOAD_LIDU:
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
                self._copy_table(
                    self.selection_block_tables,
                    context.selection_block_tables,
                    "selection_block_tables",
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
        if stateful_offload:
            return seq_lens
        return seq_lens + [0] * (self.batch_size - runtime_batch_size)


class FullDecodeOnlyGraphManager:
    """Capture/replay the complete steady-state decode model.

    DSA offload uses exact-size captures because gather status is persistent.
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
        enable_npugraph_ex: bool = True,
        log_enabled: bool = True,
    ) -> None:
        self.model = model
        self.capture_sizes = tuple(int(size) for size in capture_sizes)
        self.block_size = int(block_size)
        self.offload_mode = normalize_offload_mode(offload_mode)
        self.stateful_offload = self.offload_mode != OFFLOAD_NONE
        if (
            self.offload_mode == OFFLOAD_GS
            and DSA_SELECTION_TOPK_TOKENS % self.block_size != 0
        ):
            raise ValueError(
                "FULL_DECODE_ONLY requires the DSA sparse budget to be exactly "
                f"divisible by block_size: budget={DSA_SELECTION_TOPK_TOKENS}, "
                f"block_size={self.block_size}."
            )
        self.max_block_columns = (
            int(max_model_len) + self.block_size - 1
        ) // self.block_size
        self.selection_block_columns = (
            (
                DSA_SELECTION_TOPK_TOKENS
                if self.offload_mode == OFFLOAD_GS
                else max_lidu_cache_tokens(max_model_len)
            )
            // self.block_size
            if self.stateful_offload
            else 1
        )
        self.selection_block_columns = max(1, self.selection_block_columns)
        if (
            self.stateful_offload
            and self.max_block_columns < self.selection_block_columns
        ):
            sparse_budget = (
                DSA_SELECTION_TOPK_TOKENS
                if self.offload_mode == OFFLOAD_GS
                else max_lidu_cache_tokens(max_model_len)
            )
            raise ValueError(
                "DSA FULL_DECODE_ONLY requires max_model_len to cover the "
                f"{sparse_budget}-token sparse budget."
            )
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
        self.enable_npugraph_ex = bool(enable_npugraph_ex)
        self.log_enabled = bool(log_enabled)
        self._entries: dict[int, FullDecodeGraphEntry] = {}
        self._graph_pool = None
        self._update_stream = None
        self.capture_count = 0
        self.replay_count = 0
        self.eager_prefill_count = 0
        self.eager_first_decode_count = 0
        self.eager_no_dsa_count = 0
        self.eager_mixed_batch_count = 0
        self.eager_lidu_uninitialized_count = 0
        self.eager_lidu_capture_count = 0
        self.eager_uncaptured_batch_count = 0

        self._validate_runtime()
        self._callable = self._build_decode_callable(model)

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

    def _build_decode_callable(self, model: Callable) -> Callable:
        torch.npu.set_compile_mode(jit_compile=False)
        if not getattr(self, "enable_npugraph_ex", True):
            # GLM-5.1 W4A8 + EP16 is not reliable in TorchAir's compiled
            # warmup on CANN 8.5.1.  The raw callable is still captured by the
            # outer NPUGraph below, including Indexer/Gather/FIA/MoE, so the
            # steady-state boundary remains FULL_DECODE_ONLY.
            if self.log_enabled:
                logger.info(
                    "FULL_DECODE_ONLY: npugraph_ex FX optimization is "
                    "disabled for this model; one raw outer ACLGraph still "
                    "captures the complete decode forward."
                )
            return model

        if hasattr(torch, "_dynamo"):
            torch._dynamo.config.cache_size_limit = max(
                int(torch._dynamo.config.cache_size_limit),
                2048,
            )
            if hasattr(torch._dynamo.config, "accumulated_cache_size_limit"):
                torch._dynamo.config.accumulated_cache_size_limit = max(
                    int(torch._dynamo.config.accumulated_cache_size_limit),
                    8192,
                )

        try:
            import torchair  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "FULL_DECODE_ONLY requires TorchAir's npugraph_ex backend."
            ) from exc

        compiler_config = torchair.CompilerConfig()
        compiler_config.mode = "reduce-overhead"
        compiler_config.debug.run_eagerly = True
        compiler_config.debug.aclgraph.disable_reinplace_inplaceable_ops_pass = True
        backend = torchair.get_npu_backend(compiler_config=compiler_config)
        if self.log_enabled:
            logger.info(
                "FULL_DECODE_ONLY: enabling npugraph_ex FX optimization "
                "with one outer ACLGraph per capture size."
            )
        return torch.compile(
            model,
            backend=backend,
            fullgraph=False,
            dynamic=False,
        )

    def _allocate_entry(self, batch_size: int) -> FullDecodeGraphEntry:
        entry = FullDecodeGraphEntry.allocate(
            batch_size,
            self.max_block_columns,
            self.selection_block_columns,
            self.device,
        )
        self._entries[batch_size] = entry
        return entry

    def _set_capture_context(
        self,
        entry: FullDecodeGraphEntry,
        actual_seq_kvlen: list[int],
    ) -> None:
        kwargs = {}
        if self.stateful_offload:
            kwargs.update(
                index_block_tables=entry.index_block_tables,
                dram_block_tables=entry.dram_block_tables,
                selection_block_tables=entry.selection_block_tables,
                req_pool_entries=entry.req_pool_entries,
                candidate_lens=entry.candidate_lens,
                candidate_query_lens=entry.candidate_query_lens,
                needs_dsa_update=True,
                dsa_offload_all_rows=True,
                full_decode_graph=True,
            )
            if self.offload_mode == OFFLOAD_LIDU:
                kwargs.update(
                    lidu_cache_tokens=entry.lidu_cache_tokens,
                    lidu_all_rows_ready=True,
                )
        set_context(
            False,
            flat_slot_mapping_i32=entry.flat_slot_mapping_i32,
            actual_seq_lengths_kv=actual_seq_kvlen,
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
            callable_description = (
                "npugraph_ex-optimized decode model"
                if self.enable_npugraph_ex
                else "raw decode model in one outer ACLGraph"
            )
            logger.info(
                "FULL_DECODE_ONLY: pre-capturing %s for sizes=%s, "
                "offload_mode=%s",
                callable_description,
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
            dummy_kv_len = (
                DSA_SELECTION_TOPK_TOKENS
                if self.offload_mode == OFFLOAD_GS
                else 1
            )
            actual_seq_lens = [dummy_kv_len] * entry.batch_size
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
        if self.offload_mode == OFFLOAD_GS:
            if not runtime_context.needs_dsa_update:
                self.eager_no_dsa_count += 1
                return self._run_eager(input_ids, positions)
            if not runtime_context.dsa_offload_all_rows:
                self.eager_mixed_batch_count += 1
                return self._run_eager(input_ids, positions)
            graph_batch_size = (
                batch_size if batch_size in self.capture_sizes else None
            )
        elif self.offload_mode == OFFLOAD_LIDU:
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
            if self.offload_mode == OFFLOAD_LIDU:
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
            "npugraph_ex": self.enable_npugraph_ex,
            "capture_sizes": list(self.capture_sizes),
            "offload_mode": self.offload_mode,
            "exact_size_only": self.stateful_offload,
            "captures": self.capture_count,
            "replays": self.replay_count,
            "eager_prefill": self.eager_prefill_count,
            "eager_first_decode": self.eager_first_decode_count,
            "eager_no_dsa": self.eager_no_dsa_count,
            "eager_mixed_batch": self.eager_mixed_batch_count,
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
