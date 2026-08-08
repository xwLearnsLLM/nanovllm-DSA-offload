from __future__ import annotations

from collections import deque
from typing import TYPE_CHECKING, Final

if TYPE_CHECKING:
    from nanovllm.engine.sequence import Sequence


DSA_SELECTION_TOPK_TOKENS = 2048
OFFLOAD_NONE: Final = "none"
OFFLOAD_SPLIT: Final = "offload_split"
OFFLOAD_FUSE: Final = "offload_fuse"
LIDU_OFFLOAD_MODES: Final = (OFFLOAD_SPLIT, OFFLOAD_FUSE)
OFFLOAD_MODES: Final = (OFFLOAD_NONE, *LIDU_OFFLOAD_MODES)
# User-tunable LIDU budgets for prompt ranges 8193-16384,
# 16385-32768, 32769-65536, and >=65537 respectively.  The <=8192 tiers
# remain fixed at C=0/2048.  Edit only this tuple when comparing cache sizes.
LIDU_CACHE_TOKEN_BUDGETS: Final = (3072, 6144, 8192, 12288)
# Each entry is bounded by the complete-block source length at the lower edge
# of its prompt range.  Once this repository's operators have been compiled,
# any block-aligned values within these limits can be selected in Python.
_LIDU_CACHE_TOKEN_BUDGET_LIMITS: Final = (8192, 16384, 32768, 65536)
LIDU_MAX_SOURCE_TOKENS: Final = (1 << 18) - 1


def parse_lidu_miss_count_layers(
    value: str | None,
    num_hidden_layers: int,
) -> frozenset[int]:
    """Parse the eager-only LIDU miss-count layer switch."""

    if value is None or not value.strip():
        return frozenset()
    parts = value.split(",")
    if any(not part.strip() for part in parts):
        raise ValueError(
            "NANOVLLM_LIDU_MISS_COUNT_ON_LAYERS must be a comma-separated "
            "list such as 0,30,60."
        )
    try:
        layers = frozenset(int(part.strip()) for part in parts)
    except ValueError as exc:
        raise ValueError(
            "NANOVLLM_LIDU_MISS_COUNT_ON_LAYERS must contain integers."
        ) from exc
    invalid = sorted(
        layer for layer in layers
        if layer < 0 or layer >= int(num_hidden_layers)
    )
    if invalid:
        raise ValueError(
            "NANOVLLM_LIDU_MISS_COUNT_ON_LAYERS contains out-of-range layers "
            f"{invalid}; valid range is [0, {int(num_hidden_layers) - 1}]."
        )
    return layers


def format_lidu_miss_count_report(
    per_layer_values: list[list[int]],
    selected_layers: frozenset[int],
    decode_step: int,
) -> tuple[str, ...]:
    """Format selected-layer details and one all-layer mean."""

    if not per_layer_values or not per_layer_values[0]:
        raise ValueError("LIDU miss-count report requires non-empty values.")
    batch_size = len(per_layer_values[0])
    if any(len(values) != batch_size for values in per_layer_values):
        raise ValueError("LIDU miss-count layer batch sizes must match.")

    lines = []
    for layer in sorted(selected_layers):
        values = per_layer_values[layer]
        lines.append(
            "LIDU_MISS_COUNT "
            f"decode_step={decode_step} layer={layer} "
            f"batch_size={batch_size} request_miss_tokens={values} "
            f"mean_miss_tokens={sum(values) / batch_size:.2f}"
        )
    total = sum(sum(values) for values in per_layer_values)
    lines.append(
        "LIDU_MISS_COUNT_ALL_LAYERS "
        f"decode_step={decode_step} batch_size={batch_size} "
        f"num_layers={len(per_layer_values)} "
        "mean_miss_tokens_of_all_layers="
        f"{total / (len(per_layer_values) * batch_size):.2f}"
    )
    return tuple(lines)


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
    tier_16k, tier_32k, tier_64k, tier_larger = (
        LIDU_CACHE_TOKEN_BUDGETS
    )
    if prompt_len <= DSA_SELECTION_TOPK_TOKENS:
        return 0
    if prompt_len <= 8192:
        return DSA_SELECTION_TOPK_TOKENS
    if prompt_len <= 16384:
        return tier_16k
    if prompt_len <= 32768:
        return tier_32k
    if prompt_len <= 65536:
        return tier_64k
    return tier_larger


def mtp_lidu_cache_tokens(prompt_len: int, block_size: int = 128) -> int:
    """Return an MTP3-safe LIDU cache budget.

    MTP verification protects the union of four independent top-2048 sets.
    Therefore an offloaded request needs room for up to 8192 source tokens.
    When the complete-block source is shorter than that, cache the complete
    source; otherwise retain the ordinary (possibly larger) tuned budget but
    raise it to 8192.
    """

    prompt_len = int(prompt_len)
    block_size = int(block_size)
    base_budget = lidu_cache_tokens(prompt_len)
    if base_budget == 0:
        return 0
    candidate_len = (prompt_len // block_size) * block_size
    return min(candidate_len, max(base_budget, 4 * DSA_SELECTION_TOPK_TOKENS))


def validate_lidu_cache_token_budgets(
    block_size: int,
) -> tuple[int, int, int, int]:
    """Validate the centralized Python budgets against the current kernels."""

    budgets = LIDU_CACHE_TOKEN_BUDGETS
    if len(budgets) != 4 or any(
        isinstance(budget, bool) or not isinstance(budget, int)
        for budget in budgets
    ):
        raise ValueError(
            "LIDU_CACHE_TOKEN_BUDGETS must contain exactly four integers."
        )
    if tuple(sorted(budgets)) != budgets:
        raise ValueError(
            "LIDU_CACHE_TOKEN_BUDGETS must be nondecreasing."
        )
    if any(budget < DSA_SELECTION_TOPK_TOKENS for budget in budgets):
        raise ValueError(
            "Every nonzero LIDU cache budget must be at least 2048 tokens."
        )
    oversized = [
        (budget, limit)
        for budget, limit in zip(
            budgets,
            _LIDU_CACHE_TOKEN_BUDGET_LIMITS,
        )
        if budget > limit
    ]
    if oversized:
        raise ValueError(
            "A LIDU cache budget exceeds the complete source available at "
            f"its prompt-range boundary: {oversized}."
        )
    if any(budget % int(block_size) for budget in budgets):
        raise ValueError(
            "Every LIDU cache budget must be divisible by the KV block size."
        )
    return budgets


def max_lidu_cache_tokens(max_model_len: int) -> int:
    return lidu_cache_tokens(int(max_model_len))


def finalize_prefill_hbm_layout(
    seq: "Sequence",
    offload_mode: str,
) -> None:
    """Commit the post-prefill HBM layout after KV has reached DRAM.

    LIDU does not need its C-token destination arena until first decode.  For
    a genuinely offloaded request, release every complete-prompt HBM block now
    and leave only the dense tail resident. The scheduler installs a fresh C
    block arena immediately before first decode.
    """

    if offload_mode not in LIDU_OFFLOAD_MODES:
        raise ValueError(
            "finalize_prefill_hbm_layout requires an LIDU offload mode, "
            f"got {offload_mode!r}."
        )

    old_hbm_block_table = list(seq.hbm_block_table)
    num_full_blocks = int(seq.num_prefill_full_blocks)
    num_prefill_blocks = int(seq.num_prefill_blocks)
    num_sparse_blocks = int(seq.num_sparse_blocks)
    if len(old_hbm_block_table) < num_prefill_blocks:
        raise RuntimeError(
            "Prefill HBM block table is shorter than its finalized layout: "
            f"table={len(old_hbm_block_table)}, "
            f"prefill_blocks={num_prefill_blocks}."
        )

    seq.lidu_decode_hbm_pending = False
    if num_sparse_blocks >= num_full_blocks:
        keep_sparse = old_hbm_block_table[:num_full_blocks]
        release_blocks: list[int] = []
        # The complete source fits in HBM and already has an identity
        # source-to-slot mapping, so no first-decode initialization is needed.
        seq.lidu_cache_initialized = True
    else:
        # Full-prompt KV was persisted to DRAM by every attention layer before
        # this host-side transition.  Borrow the future C-token arena for later
        # prefills, then allocate it atomically just before first decode.
        keep_sparse = []
        release_blocks = old_hbm_block_table[:num_full_blocks]
        seq.lidu_decode_hbm_pending = num_sparse_blocks > 0

    keep_tail = old_hbm_block_table[num_full_blocks:num_prefill_blocks]
    seq.hbm_block_table = keep_sparse + keep_tail
    seq.block_table = seq.hbm_block_table
    seq.hbm_blocks_to_release = release_blocks
    seq.offload_finalized = True
    seq.bump_decode_metadata_version()


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
