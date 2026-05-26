from __future__ import annotations

from collections import deque


def compute_sparse_blocks(num_prefill_full_blocks: int) -> int:
    """Default sparse budget piecewise function from the DSA offload design."""
    n = int(num_prefill_full_blocks)
    if n < 64:
        return n
    if n < 128:
        return (30 * n + 99) // 100
    if n < 256:
        return (25 * n + 99) // 100
    if n < 512:
        return (22 * n + 99) // 100
    return (20 * n + 99) // 100


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
