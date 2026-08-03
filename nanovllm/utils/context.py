from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field

import torch


@dataclass
class Context:
    is_prefill: bool = False
    is_spec_decode: bool = False
    cu_seqlens_q: torch.Tensor | None = None
    actual_seq_lengths_q: list[int] | None = None
    flat_slot_mapping: torch.Tensor | None = None
    flat_slot_mapping_i32: torch.Tensor | None = None
    flat_index_slot_mapping: torch.Tensor | None = None
    actual_seq_lengths_kv: list[int] | None = None
    actual_seq_lengths_kv_tensor: torch.Tensor | None = None
    block_tables: torch.Tensor | None = None
    index_block_tables: torch.Tensor | None = None
    dram_block_tables: torch.Tensor | None = None
    req_pool_entries: torch.Tensor | None = None
    candidate_lens: torch.Tensor | None = None
    candidate_query_lens: torch.Tensor | None = None
    lidu_cache_tokens: torch.Tensor | None = None
    needs_dsa_update: bool = False
    lidu_init_rows: torch.Tensor | None = None
    lidu_all_rows_ready: bool = False
    has_first_decode: bool = False
    decode_metadata_key: tuple[tuple[int, int], ...] | None = None
    full_decode_graph: bool = False
    scratch: dict = field(default_factory=dict)


_CONTEXT = Context()


def get_context() -> Context:
    return _CONTEXT


def set_context(
    is_prefill: bool,
    *,
    is_spec_decode=False,
    cu_seqlens_q=None,
    actual_seq_lengths_q=None,
    flat_slot_mapping=None,
    flat_slot_mapping_i32=None,
    flat_index_slot_mapping=None,
    actual_seq_lengths_kv=None,
    actual_seq_lengths_kv_tensor=None,
    block_tables=None,
    index_block_tables=None,
    dram_block_tables=None,
    req_pool_entries=None,
    candidate_lens=None,
    candidate_query_lens=None,
    lidu_cache_tokens=None,
    needs_dsa_update=False,
    lidu_init_rows=None,
    lidu_all_rows_ready=False,
    has_first_decode=False,
    decode_metadata_key=None,
    full_decode_graph=False,
) -> None:
    global _CONTEXT
    _CONTEXT = Context(
        is_prefill=is_prefill,
        is_spec_decode=bool(is_spec_decode),
        cu_seqlens_q=cu_seqlens_q,
        actual_seq_lengths_q=actual_seq_lengths_q,
        flat_slot_mapping=flat_slot_mapping,
        flat_slot_mapping_i32=flat_slot_mapping_i32,
        flat_index_slot_mapping=flat_index_slot_mapping,
        actual_seq_lengths_kv=actual_seq_lengths_kv,
        actual_seq_lengths_kv_tensor=actual_seq_lengths_kv_tensor,
        block_tables=block_tables,
        index_block_tables=index_block_tables,
        dram_block_tables=dram_block_tables,
        req_pool_entries=req_pool_entries,
        candidate_lens=candidate_lens,
        candidate_query_lens=candidate_query_lens,
        lidu_cache_tokens=lidu_cache_tokens,
        needs_dsa_update=bool(needs_dsa_update),
        lidu_init_rows=lidu_init_rows,
        lidu_all_rows_ready=bool(lidu_all_rows_ready),
        has_first_decode=bool(has_first_decode),
        decode_metadata_key=decode_metadata_key,
        full_decode_graph=bool(full_decode_graph),
    )


def reset_context() -> None:
    global _CONTEXT
    _CONTEXT = Context()


@contextmanager
def preserve_context():
    """Restore the active model context after temporary graph capture."""

    global _CONTEXT
    previous = _CONTEXT
    try:
        yield
    finally:
        _CONTEXT = previous
