from __future__ import annotations

from collections.abc import Callable, Iterable
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
import gc
from typing import Any

import torch

from nanovllm.engine.dsa_offload import DSA_SELECTION_TOPK_TOKENS
from nanovllm.utils.context import Context, get_context, reset_context, set_context
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
            "FULL_DECODE_ONLY requires at least one exact decode graph capture "
            "size. Pass decode_graph_capture_sizes, for example (16,)."
        )
    if sizes[0] <= 0:
        raise ValueError(f"Decode graph capture sizes must be positive, got {sizes}.")
    return sizes


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
    """Static inputs for one exact-size, all-long-request DSA decode graph."""

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
    graph: Any | None = None
    output: torch.Tensor | None = None
    mla_tasks: list[MLAGraphTask] = field(default_factory=list)

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
    ) -> list[int]:
        if input_ids.ndim != 1 or positions.ndim != 1:
            raise ValueError(
                "FULL_DECODE_ONLY expects one token and one position per request: "
                f"input_ids.shape={tuple(input_ids.shape)}, "
                f"positions.shape={tuple(positions.shape)}."
            )
        runtime_batch_size = int(input_ids.shape[0])
        if runtime_batch_size != self.batch_size:
            raise ValueError(
                "DSA FULL_DECODE_ONLY uses exact capture sizes because padding "
                "would mutate another request's persistent gather status: "
                f"runtime={runtime_batch_size}, graph={self.batch_size}."
            )
        if int(positions.shape[0]) != self.batch_size:
            raise ValueError(
                "Decode input_ids and positions must have the same batch size: "
                f"{self.batch_size} != {positions.shape[0]}."
            )
        required_metadata = {
            "flat_slot_mapping_i32": context.flat_slot_mapping_i32,
            "block_tables": context.block_tables,
            "index_block_tables": context.index_block_tables,
            "dram_block_tables": context.dram_block_tables,
            "selection_block_tables": context.selection_block_tables,
            "req_pool_entries": context.req_pool_entries,
            "candidate_lens": context.candidate_lens,
            "candidate_query_lens": context.candidate_query_lens,
        }
        missing = [name for name, value in required_metadata.items() if value is None]
        if missing:
            raise RuntimeError(
                "DSA FULL_DECODE_ONLY is missing decode metadata tensors: "
                + ", ".join(missing)
            )

        self.input_ids.copy_(input_ids)
        self.positions.copy_(positions)
        self.flat_slot_mapping_i32.copy_(context.flat_slot_mapping_i32)
        self.req_pool_entries.copy_(context.req_pool_entries)
        self.candidate_lens.copy_(context.candidate_lens)
        self.candidate_query_lens.copy_(context.candidate_query_lens)
        self._copy_table(self.block_tables, context.block_tables, "block_tables")
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

        seq_lens = list(context.actual_seq_lengths_kv or ())
        if len(seq_lens) != self.batch_size:
            raise ValueError(
                "Decode actual_seq_lengths_kv must have one value per request: "
                f"got {len(seq_lens)} for batch {self.batch_size}."
            )
        return seq_lens


class FullDecodeOnlyGraphManager:
    """Capture/replay the complete steady-state DSA decode model.

    Only exact-size batches in which every row uses DSA offload are replayed.
    Prefill, first decode, short requests, mixed batches, and unconfigured batch
    sizes stay eager. This keeps padding from corrupting persistent gather
    status while preserving normal scheduler behavior.
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
        log_enabled: bool = True,
    ) -> None:
        self.model = model
        self.capture_sizes = tuple(int(size) for size in capture_sizes)
        self.block_size = int(block_size)
        if DSA_SELECTION_TOPK_TOKENS % self.block_size != 0:
            raise ValueError(
                "FULL_DECODE_ONLY requires the DSA sparse budget to be exactly "
                f"divisible by block_size: budget={DSA_SELECTION_TOPK_TOKENS}, "
                f"block_size={self.block_size}."
            )
        self.max_block_columns = (
            int(max_model_len) + self.block_size - 1
        ) // self.block_size
        self.selection_block_columns = DSA_SELECTION_TOPK_TOKENS // self.block_size
        if self.max_block_columns < self.selection_block_columns:
            raise ValueError(
                "DSA FULL_DECODE_ONLY requires max_model_len to cover the "
                f"{DSA_SELECTION_TOPK_TOKENS}-token sparse budget."
            )
        self.device = torch.device(device)
        self.expected_mla_tasks = int(expected_mla_tasks)
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

        torch.npu.set_compile_mode(jit_compile=False)
        compiler_config = torchair.CompilerConfig()
        compiler_config.mode = "reduce-overhead"
        compiler_config.debug.run_eagerly = True
        compiler_config.debug.aclgraph.disable_reinplace_inplaceable_ops_pass = True
        backend = torchair.get_npu_backend(compiler_config=compiler_config)
        if self.log_enabled:
            logger.info(
                "DSA FULL_DECODE_ONLY: enabling npugraph_ex FX optimization "
                "with one outer ACLGraph per exact batch size."
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
        set_context(
            False,
            flat_slot_mapping_i32=entry.flat_slot_mapping_i32,
            actual_seq_lengths_kv=actual_seq_kvlen,
            block_tables=entry.block_tables,
            index_block_tables=entry.index_block_tables,
            dram_block_tables=entry.dram_block_tables,
            selection_block_tables=entry.selection_block_tables,
            req_pool_entries=entry.req_pool_entries,
            candidate_lens=entry.candidate_lens,
            candidate_query_lens=entry.candidate_query_lens,
            needs_dsa_update=True,
            dsa_offload_rows=None,
            dsa_offload_all_rows=True,
            has_first_decode=False,
            full_decode_graph=True,
        )

    def capture_all(self) -> None:
        if self._graph_pool is None:
            self._graph_pool = torch.npu.graph_pool_handle()
        if self._update_stream is None:
            self._update_stream = torch.npu.Stream()

        if self.log_enabled:
            logger.info(
                "DSA FULL_DECODE_ONLY: pre-capturing exact decode sizes=%s",
                self.capture_sizes,
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

    def _capture(self, entry: FullDecodeGraphEntry) -> None:
        dummy_seq_lens = [DSA_SELECTION_TOPK_TOKENS] * entry.batch_size
        self._set_capture_context(entry, dummy_seq_lens)
        capture_context = get_context()
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
                "DSA FULL_DECODE_ONLY: captured complete decode graph for "
                "batch_size=%d with %d refreshable MLA tasks.",
                entry.batch_size,
                len(entry.mla_tasks),
            )

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
        if not runtime_context.needs_dsa_update:
            self.eager_no_dsa_count += 1
            return self._run_eager(input_ids, positions)
        if not runtime_context.dsa_offload_all_rows:
            self.eager_mixed_batch_count += 1
            return self._run_eager(input_ids, positions)

        batch_size = int(input_ids.shape[0])
        if batch_size not in self.capture_sizes:
            self.eager_uncaptured_batch_count += 1
            return self._run_eager(input_ids, positions)
        entry = self._entries.get(batch_size)
        if entry is None or entry.graph is None or entry.output is None:
            raise RuntimeError(
                "FULL_DECODE_ONLY graph was not pre-captured for exact batch "
                f"size {batch_size}."
            )

        seq_lens = entry.copy_runtime_inputs(input_ids, positions, runtime_context)

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
                "DSA FULL_DECODE_ONLY: first complete graph replay entered "
                "for exact batch_size=%d.",
                batch_size,
            )
        self.replay_count += 1
        return entry.output

    def stats(self) -> dict[str, Any]:
        return {
            "enabled": True,
            "mode": FULL_DECODE_ONLY,
            "capture_sizes": list(self.capture_sizes),
            "exact_size_only": True,
            "captures": self.capture_count,
            "replays": self.replay_count,
            "eager_prefill": self.eager_prefill_count,
            "eager_first_decode": self.eager_first_decode_count,
            "eager_no_dsa": self.eager_no_dsa_count,
            "eager_mixed_batch": self.eager_mixed_batch_count,
            "eager_uncaptured_batch": self.eager_uncaptured_batch_count,
        }
