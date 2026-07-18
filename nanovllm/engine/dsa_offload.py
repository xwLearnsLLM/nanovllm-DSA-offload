from __future__ import annotations

from collections import deque
from typing import Final


DSA_SELECTION_TOPK_TOKENS = 2048
OFFLOAD_NONE: Final = "none"
OFFLOAD_GS: Final = "gs"
OFFLOAD_LIDU: Final = "lidu"
OFFLOAD_MODES: Final = (OFFLOAD_NONE, OFFLOAD_GS, OFFLOAD_LIDU)
LIDU_CACHE_TOKEN_BUDGETS: Final = (2048, 3072, 5120, 8192, 12288)
LIDU_MAX_SOURCE_TOKENS: Final = (1 << 18) - 1


def normalize_offload_mode(value: str) -> str:
    if not isinstance(value, str):
        raise TypeError("offload_mode must be a string.")
    mode = value.strip().lower()
    if mode not in OFFLOAD_MODES:
        raise ValueError(
            "offload_mode must be one of "
            f"{OFFLOAD_MODES}, got {value!r}."
        )
    return mode


def lidu_cache_tokens(prompt_len: int) -> int:
    """Return the fixed per-request LIDU HBM cache budget C."""

    prompt_len = int(prompt_len)
    if prompt_len <= DSA_SELECTION_TOPK_TOKENS:
        return 0
    if prompt_len <= 8192:
        return 2048
    if prompt_len <= 16384:
        return 3072
    if prompt_len <= 32768:
        return 5120
    if prompt_len <= 65536:
        return 8192
    return 12288


def max_lidu_cache_tokens(max_model_len: int) -> int:
    return lidu_cache_tokens(int(max_model_len))


def compute_sparse_blocks(
    num_prefill_full_blocks: int,
    block_size: int = 128,
) -> int:
    """Return the HBM blocks retained for the 2048-token DSA budget."""

    num_blocks = int(num_prefill_full_blocks)
    block_size = int(block_size)
    if num_blocks <= 0:
        return 0
    if num_blocks * block_size <= DSA_SELECTION_TOPK_TOKENS:
        return num_blocks
    budget_blocks = (
        DSA_SELECTION_TOPK_TOKENS + block_size - 1
    ) // block_size
    return min(num_blocks, budget_blocks)


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
                raise ValueError(
                    "Need at least two blocks when reserving null block."
                )
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
