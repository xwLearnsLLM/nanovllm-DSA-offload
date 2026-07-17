from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import torch


DSA_SELECTION_TOPK_TOKENS = 2048
DSA_DEBUG_RETAINED_PREFIX_TOKENS = 128
DSA_DEBUG_SELECTION_MODES = frozenset(
    {
        "native",
        "native_interleave_stats",
        "native_half_stats",
        "retained_skip_gs",
        "retained_gs",
        "last2048_gs",
    }
)
DSA_DEBUG_NATIVE_SELECTION_MODES = frozenset(
    {
        "native",
        "native_interleave_stats",
        "native_half_stats",
    }
)
DSA_BOUNDARY_PROBE_MODES = frozenset(
    {
        "none",
        "project_sync",
        "li_clone",
        "li_sync",
        "gs_sync",
        "all_sync",
    }
)


def default_dsa_native_stats_layers(
    num_hidden_layers: int,
) -> frozenset[int]:
    """Layers sampled by the eager native-Indexer diagnostic by default."""

    num_hidden_layers = int(num_hidden_layers)
    if num_hidden_layers <= 0:
        raise ValueError(
            "num_hidden_layers must be positive, got "
            f"{num_hidden_layers}."
        )
    probes = (0, 1, 2, 4, 8, 16, 24, 32, 39, 48, 64)
    return frozenset(
        {layer for layer in probes if layer < num_hidden_layers}
        | {num_hidden_layers - 1}
    )


@dataclass(frozen=True)
class DSANativeSelectionStats:
    row: int
    candidate_len: int
    valid_count: int
    unique_count: int
    invalid_count: int
    duplicate_count: int
    min_index: int
    max_index: int
    retained_overlap: int
    last2048_overlap: int
    tail128_count: int
    quartile_counts: tuple[int, int, int, int]


@dataclass(frozen=True)
class DSANumericTensorStats:
    """Small eager-diagnostic summary for an Indexer input tensor."""

    numel: int
    finite_count: int
    nonzero_count: int
    abs_max: float
    l2_norm: float


def summarize_dsa_numeric_tensor(tensor: torch.Tensor) -> DSANumericTensorStats:
    """Reduce a tensor to scalar health statistics without copying it whole.

    NaN/Inf entries are reported by ``finite_count`` and excluded from the
    nonzero, max and norm reductions.  On NPU this performs one small D2H copy;
    callers must therefore keep it behind an explicit eager diagnostic mode.
    """

    numel = int(tensor.numel())
    if numel == 0:
        return DSANumericTensorStats(0, 0, 0, 0.0, 0.0)

    values = tensor.detach().to(dtype=torch.float32)
    finite = torch.isfinite(values)
    safe_values = torch.where(finite, values, torch.zeros_like(values))
    metrics = torch.stack(
        (
            finite.sum(dtype=torch.float32),
            torch.count_nonzero(finite & (safe_values != 0)).to(torch.float32),
            safe_values.abs().amax(),
            torch.sqrt(torch.sum(safe_values * safe_values)),
        )
    ).cpu().tolist()
    return DSANumericTensorStats(
        numel=numel,
        finite_count=int(metrics[0]),
        nonzero_count=int(metrics[1]),
        abs_max=float(metrics[2]),
        l2_norm=float(metrics[3]),
    )


def dsa_effective_index_cache_row(
    index_cache: torch.Tensor,
    block_table_row: torch.Tensor,
    candidate_len: int,
    block_size: int,
) -> torch.Tensor:
    """Resolve the logical candidate prefix consumed by LightningIndexer."""

    candidate_len = int(candidate_len)
    block_size = int(block_size)
    if candidate_len < 0:
        raise ValueError(f"candidate_len must be non-negative, got {candidate_len}.")
    if block_size <= 0:
        raise ValueError(f"block_size must be positive, got {block_size}.")
    if index_cache.ndim < 2 or int(index_cache.shape[1]) != block_size:
        raise ValueError(
            "index_cache must have shape [num_blocks, block_size, ...], "
            f"got {tuple(index_cache.shape)} for block_size={block_size}."
        )
    if block_table_row.ndim != 1:
        raise ValueError(
            "block_table_row must be one-dimensional, got shape="
            f"{tuple(block_table_row.shape)}."
        )

    num_blocks = (candidate_len + block_size - 1) // block_size
    if num_blocks > int(block_table_row.numel()):
        raise ValueError(
            "block table is too short for the candidate prefix: "
            f"need={num_blocks}, have={block_table_row.numel()}."
        )
    tail_shape = tuple(index_cache.shape[2:])
    if num_blocks == 0:
        return index_cache.new_empty((0, *tail_shape))
    physical_blocks = block_table_row[:num_blocks].to(
        device=index_cache.device,
        dtype=torch.int64,
    )
    cache = index_cache.index_select(0, physical_blocks)
    return cache.reshape(num_blocks * block_size, *tail_shape)[:candidate_len]


def dsa_paged_cache_tokens(
    cache: torch.Tensor,
    block_table_row: torch.Tensor,
    logical_token_ids: torch.Tensor,
    block_size: int,
) -> torch.Tensor:
    """Resolve arbitrary logical token IDs from one paged-cache row.

    This helper is intentionally used only by eager diagnostics.  It makes the
    GatherSelection copy contract explicit: destination resident slot ``i``
    must equal the full-cache token named by status slot ``i``.
    """

    block_size = int(block_size)
    if block_size <= 0:
        raise ValueError(f"block_size must be positive, got {block_size}.")
    if cache.ndim < 2 or int(cache.shape[1]) != block_size:
        raise ValueError(
            "cache must have shape [num_blocks, block_size, ...], "
            f"got {tuple(cache.shape)} for block_size={block_size}."
        )
    if block_table_row.ndim != 1:
        raise ValueError(
            "block_table_row must be one-dimensional, got shape="
            f"{tuple(block_table_row.shape)}."
        )

    token_ids = logical_token_ids.detach().reshape(-1).to(
        device=cache.device,
        dtype=torch.int64,
    )
    if token_ids.numel() == 0:
        return cache.new_empty((0, *tuple(cache.shape[2:])))
    min_id = int(token_ids.amin().item())
    max_id = int(token_ids.amax().item())
    if min_id < 0:
        raise ValueError(f"logical token IDs must be non-negative, got {min_id}.")
    max_logical_tokens = int(block_table_row.numel()) * block_size
    if max_id >= max_logical_tokens:
        raise ValueError(
            "logical token ID exceeds the block table: "
            f"max_id={max_id}, capacity={max_logical_tokens}."
        )

    logical_blocks = torch.div(token_ids, block_size, rounding_mode="floor")
    offsets = torch.remainder(token_ids, block_size)
    physical_blocks = block_table_row.to(
        device=cache.device,
        dtype=torch.int64,
    ).index_select(0, logical_blocks)
    physical_slots = physical_blocks * block_size + offsets
    flat_cache = cache.reshape(
        int(cache.shape[0]) * block_size,
        *tuple(cache.shape[2:]),
    )
    return flat_cache.index_select(0, physical_slots)


def parse_dsa_debug_selection(value: str | None) -> str:
    """Parse the eager-only DSA selection diagnostic mode."""

    mode = "native" if value is None or not value.strip() else value.strip()
    if mode not in DSA_DEBUG_SELECTION_MODES:
        choices = ", ".join(sorted(DSA_DEBUG_SELECTION_MODES))
        raise ValueError(
            "NANOVLLM_DSA_DEBUG_SELECTION must be one of "
            f"{choices}, got {mode!r}."
        )
    return mode


def validate_dsa_boundary_probe(
    value: str | None,
    *,
    enforce_eager: bool,
) -> str:
    """Validate a temporary eager-only DSA producer/consumer probe."""

    mode = "none" if value is None or not value.strip() else value.strip()
    if mode not in DSA_BOUNDARY_PROBE_MODES:
        choices = ", ".join(sorted(DSA_BOUNDARY_PROBE_MODES))
        raise ValueError(
            "NANOVLLM_DSA_BOUNDARY_PROBE must be one of "
            f"{choices}, got {mode!r}."
        )
    if mode != "none" and not enforce_eager:
        raise ValueError(
            "NANOVLLM_DSA_BOUNDARY_PROBE is eager-only; set "
            "NANOVLLM_ENFORCE_EAGER=1."
        )
    return mode


def dsa_debug_uses_native_selection(mode: str) -> bool:
    """Whether ``mode`` consumes the LightningIndexer top-k output."""

    return parse_dsa_debug_selection(mode) in DSA_DEBUG_NATIVE_SELECTION_MODES


def dsa_debug_rotary_mode(mode: str, configured_mode: str) -> str:
    """Resolve the Indexer RoPE variant selected by an eager diagnostic."""

    mode = parse_dsa_debug_selection(mode)
    if configured_mode not in ("half", "interleave"):
        raise ValueError(
            "configured Indexer rotary mode must be 'half' or 'interleave', "
            f"got {configured_mode!r}."
        )
    if mode == "native_half_stats":
        return "half"
    if mode == "native_interleave_stats":
        return "interleave"
    return configured_mode


def dsa_debug_prints_native_stats(mode: str) -> bool:
    mode = parse_dsa_debug_selection(mode)
    return mode in ("native_interleave_stats", "native_half_stats")


def validate_dsa_debug_selection(
    value: str | None,
    *,
    enforce_eager: bool,
    block_size: int,
) -> str:
    mode = parse_dsa_debug_selection(value)
    if mode == "native":
        return mode
    if not enforce_eager:
        raise ValueError(
            "NANOVLLM_DSA_DEBUG_SELECTION is eager-only when it is not "
            "'native'; set NANOVLLM_ENFORCE_EAGER=1."
        )
    if int(block_size) != DSA_DEBUG_RETAINED_PREFIX_TOKENS:
        raise ValueError(
            "Non-native NANOVLLM_DSA_DEBUG_SELECTION modes require "
            "NANOVLLM_KVCACHE_BLOCK_SIZE=128 so the retained selection is "
            "exactly the first 128 tokens plus the last 1920 tokens."
        )
    return mode


def build_dsa_debug_selection(
    candidate_lens: torch.Tensor,
    mode: str,
) -> torch.Tensor | None:
    """Build deterministic logical token IDs for DSA selection diagnostics.

    ``retained_*`` matches the exact logical order already present in the
    compact HBM selection blocks after prefill finalization: the first
    128-token block followed by the final 1920 candidate tokens.
    """

    mode = parse_dsa_debug_selection(mode)
    if dsa_debug_uses_native_selection(mode):
        return None
    if candidate_lens.ndim != 1:
        raise ValueError(
            "candidate_lens must be one-dimensional, got shape="
            f"{tuple(candidate_lens.shape)}."
        )
    if candidate_lens.dtype not in (torch.int32, torch.int64):
        raise TypeError(
            "candidate_lens must use int32 or int64, got "
            f"{candidate_lens.dtype}."
        )
    if candidate_lens.numel() == 0:
        return candidate_lens.new_empty((0, 1, DSA_SELECTION_TOPK_TOKENS))
    if bool(torch.any(candidate_lens <= DSA_SELECTION_TOPK_TOKENS).item()):
        raise ValueError(
            "DSA debug selection requires every candidate length to exceed "
            f"{DSA_SELECTION_TOPK_TOKENS}."
        )

    batch_size = int(candidate_lens.numel())
    candidate_lens = candidate_lens.reshape(batch_size, 1)
    if mode in ("retained_skip_gs", "retained_gs"):
        prefix = torch.arange(
            DSA_DEBUG_RETAINED_PREFIX_TOKENS,
            dtype=candidate_lens.dtype,
            device=candidate_lens.device,
        ).view(1, -1).expand(batch_size, -1)
        suffix_tokens = (
            DSA_SELECTION_TOPK_TOKENS - DSA_DEBUG_RETAINED_PREFIX_TOKENS
        )
        suffix_offsets = torch.arange(
            suffix_tokens,
            dtype=candidate_lens.dtype,
            device=candidate_lens.device,
        ).view(1, -1)
        suffix = candidate_lens - suffix_tokens + suffix_offsets
        selected = torch.cat((prefix, suffix), dim=1)
    else:
        offsets = torch.arange(
            DSA_SELECTION_TOPK_TOKENS,
            dtype=candidate_lens.dtype,
            device=candidate_lens.device,
        ).view(1, -1)
        selected = candidate_lens - DSA_SELECTION_TOPK_TOKENS + offsets
    return selected.unsqueeze(1).contiguous()


def summarize_dsa_native_selection(
    topk_indices: torch.Tensor,
    candidate_lens: torch.Tensor,
) -> list[DSANativeSelectionStats]:
    """Summarize native logical top-k IDs for the eager GLM diagnostic.

    The retained set is the selection present after prefill finalization:
    logical tokens ``[0, 128)`` plus the final 1920 candidate tokens.  Counts
    use unique valid IDs so a malformed duplicate cannot inflate overlap.
    Calling this helper synchronizes/copies its inputs to CPU; production
    ``native`` mode never calls it.
    """

    if candidate_lens.ndim != 1:
        raise ValueError(
            "candidate_lens must be one-dimensional, got shape="
            f"{tuple(candidate_lens.shape)}."
        )
    batch_size = int(candidate_lens.numel())
    if batch_size == 0:
        if topk_indices.numel() != 0:
            raise ValueError("topk_indices must be empty for an empty batch.")
        return []
    if topk_indices.numel() % batch_size:
        raise ValueError(
            "topk_indices cannot be reshaped into candidate_lens rows: "
            f"topk_numel={topk_indices.numel()}, batch={batch_size}."
        )

    topk_rows = topk_indices.detach().reshape(batch_size, -1).cpu().tolist()
    lens = candidate_lens.detach().reshape(-1).cpu().tolist()
    summaries: list[DSANativeSelectionStats] = []
    for row, (raw_ids, raw_candidate_len) in enumerate(zip(topk_rows, lens)):
        candidate_len = int(raw_candidate_len)
        if candidate_len <= DSA_SELECTION_TOPK_TOKENS:
            raise ValueError(
                "native selection stats require every candidate length to "
                f"exceed {DSA_SELECTION_TOPK_TOKENS}, got {candidate_len}."
            )
        valid_ids = [
            int(token_id)
            for token_id in raw_ids
            if 0 <= int(token_id) < candidate_len
        ]
        unique_ids = set(valid_ids)
        retained_suffix_start = candidate_len - (
            DSA_SELECTION_TOPK_TOKENS - DSA_DEBUG_RETAINED_PREFIX_TOKENS
        )
        last2048_start = candidate_len - DSA_SELECTION_TOPK_TOKENS
        retained_overlap = sum(
            token_id < DSA_DEBUG_RETAINED_PREFIX_TOKENS
            or token_id >= retained_suffix_start
            for token_id in unique_ids
        )
        last2048_overlap = sum(
            token_id >= last2048_start for token_id in unique_ids
        )
        tail128_count = sum(
            token_id >= candidate_len - DSA_DEBUG_RETAINED_PREFIX_TOKENS
            for token_id in unique_ids
        )
        quartiles = [0, 0, 0, 0]
        for token_id in unique_ids:
            quartiles[min((token_id * 4) // candidate_len, 3)] += 1
        summaries.append(
            DSANativeSelectionStats(
                row=row,
                candidate_len=candidate_len,
                valid_count=len(valid_ids),
                unique_count=len(unique_ids),
                invalid_count=len(raw_ids) - len(valid_ids),
                duplicate_count=len(valid_ids) - len(unique_ids),
                min_index=min(unique_ids) if unique_ids else -1,
                max_index=max(unique_ids) if unique_ids else -1,
                retained_overlap=retained_overlap,
                last2048_overlap=last2048_overlap,
                tail128_count=tail128_count,
                quartile_counts=tuple(quartiles),
            )
        )
    return summaries


def parse_gs_miss_rate_layers(
    value: str | None,
    num_hidden_layers: int,
) -> frozenset[int]:
    if value is None or not value.strip():
        return frozenset()
    parts = value.split(",")
    if any(not part.strip() for part in parts):
        raise ValueError(
            "NANOVLLM_GS_MISS_RATE_ON_LAYERS must be a comma-separated "
            "list such as 0,30,60."
        )
    try:
        layers = frozenset(int(part.strip()) for part in parts)
    except ValueError as exc:
        raise ValueError(
            "NANOVLLM_GS_MISS_RATE_ON_LAYERS must contain integers."
        ) from exc
    invalid = sorted(
        layer for layer in layers if layer < 0 or layer >= num_hidden_layers
    )
    if invalid:
        raise ValueError(
            "NANOVLLM_GS_MISS_RATE_ON_LAYERS contains out-of-range layers "
            f"{invalid}; valid range is [0, {num_hidden_layers - 1}]."
        )
    return layers


def compute_gs_miss_counts(
    topk_rows: list[list[int]],
    selection_rows: list[list[int]],
) -> list[int]:
    """Return |topk - selection| for each request, ignoring invalid -1 IDs."""
    if len(topk_rows) != len(selection_rows):
        raise ValueError("topk and selection row counts must match.")
    miss_counts = []
    for topk, selection in zip(topk_rows, selection_rows):
        topk_set = {int(token_id) for token_id in topk if int(token_id) >= 0}
        selection_set = {
            int(token_id) for token_id in selection if int(token_id) >= 0
        }
        miss_counts.append(len(topk_set - selection_set))
    return miss_counts


def compute_sparse_blocks(num_prefill_full_blocks: int, block_size: int = 128) -> int:
    """Only sparse-offload full prefill blocks when they exceed the 2048-token budget."""
    n = int(num_prefill_full_blocks)
    block_size = int(block_size)
    if n <= 0:
        return 0
    if n * block_size <= DSA_SELECTION_TOPK_TOKENS:
        return n
    budget_blocks = (DSA_SELECTION_TOPK_TOKENS + block_size - 1) // block_size
    return min(n, budget_blocks)


class SimpleBlockManager:
    """Non-prefix block manager for HBM KV and DRAM KV physical blocks."""

    def __init__(
        self,
        num_blocks: int,
        *,
        reserve_null_block: bool = True,
    ) -> None:
        self.num_blocks = int(num_blocks)
        self.free_block_ids: deque[int] = deque(range(self.num_blocks))
        self.used_block_ids: set[int] = set()
        self.null_block_id: int | None = None
        if reserve_null_block:
            if self.num_blocks <= 1:
                raise ValueError("Need at least two blocks when reserving null block.")
            self.null_block_id = self.free_block_ids.popleft()

    def can_allocate_blocks(self, num_blocks: int) -> bool:
        return len(self.free_block_ids) >= int(num_blocks)

    def allocate_blocks(self, num_blocks: int) -> list[int]:
        num_blocks = int(num_blocks)
        if not self.can_allocate_blocks(num_blocks):
            raise RuntimeError(
                f"Insufficient free blocks: need={num_blocks}, "
                f"free={len(self.free_block_ids)}."
            )
        block_ids: list[int] = []
        for _ in range(num_blocks):
            block_id = self.free_block_ids.popleft()
            self.used_block_ids.add(block_id)
            block_ids.append(block_id)
        return block_ids

    def free_blocks(self, block_ids: list[int]) -> None:
        for block_id in block_ids:
            block_id = int(block_id)
            if block_id == self.null_block_id:
                continue
            if block_id not in self.used_block_ids:
                continue
            self.used_block_ids.remove(block_id)
            self.free_block_ids.append(block_id)


class PoolEntryManager:
    def __init__(self, capacity: int) -> None:
        self.capacity = int(capacity)
        self.free_entries: deque[int] = deque(range(self.capacity))
        self.used_entries: set[int] = set()

    def can_allocate(self) -> bool:
        return bool(self.free_entries)

    def allocate(self) -> int:
        if not self.free_entries:
            raise RuntimeError("No free DSA offload pool entry.")
        entry = self.free_entries.popleft()
        self.used_entries.add(entry)
        return entry

    def free(self, entry: int | None) -> None:
        if entry is None:
            return
        entry = int(entry)
        if entry not in self.used_entries:
            return
        self.used_entries.remove(entry)
        self.free_entries.append(entry)
