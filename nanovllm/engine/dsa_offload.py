from __future__ import annotations

from collections import deque


DSA_SELECTION_TOPK_TOKENS = 2048


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
