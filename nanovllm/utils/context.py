from __future__ import annotations

from dataclasses import dataclass, field
import torch


@dataclass
class Context:
    is_prefill: bool = False
    cu_seqlens_q: torch.Tensor | None = None
    cu_seqlens_k: torch.Tensor | None = None
    max_seqlen_q: int = 0
    max_seqlen_k: int = 0
    slot_mapping: torch.Tensor | None = None
    flat_slot_mapping: torch.Tensor | None = None
    context_lens: torch.Tensor | None = None
    actual_seq_lengths_query: list[int] | None = None
    actual_seq_lengths_kv: list[int] | None = None
    block_tables: torch.Tensor | None = None
    index_block_tables: torch.Tensor | None = None
    hbm_block_tables: torch.Tensor | None = None
    dram_block_tables: torch.Tensor | None = None
    selection_block_tables: torch.Tensor | None = None
    index_slot_mapping: torch.Tensor | None = None
    flat_index_slot_mapping: torch.Tensor | None = None
    req_pool_entries: torch.Tensor | None = None
    candidate_lens: torch.Tensor | None = None
    candidate_query_lens: torch.Tensor | None = None
    needs_dsa_update: bool = False
    dsa_offload_rows: torch.Tensor | None = None
    dsa_offload_all_rows: bool = False
    has_first_decode: bool = False
    block_size: int = 256
    flat_slot_mapping_i32: torch.Tensor | None = None
    scratch: dict = field(default_factory=dict)


_CONTEXT = Context()


def get_context():
    return _CONTEXT


def set_context(is_prefill, cu_seqlens_q=None, cu_seqlens_k=None, max_seqlen_q=0, max_seqlen_k=0, slot_mapping=None,
                flat_slot_mapping=None, context_lens=None, actual_seq_lengths_query=None, actual_seq_lengths_kv=None,
                block_tables=None, index_block_tables=None, hbm_block_tables=None, dram_block_tables=None,
                selection_block_tables=None,
                index_slot_mapping=None, flat_index_slot_mapping=None, req_pool_entries=None, candidate_lens=None,
                candidate_query_lens=None,
                needs_dsa_update=None,
                dsa_offload_rows=None, dsa_offload_all_rows=None,
                has_first_decode=None, block_size=None, flat_slot_mapping_i32=None):
    global _CONTEXT
    if block_size is None:
        block_size = 256
    if needs_dsa_update is None:
        needs_dsa_update = False
    if dsa_offload_all_rows is None:
        dsa_offload_all_rows = False
    if has_first_decode is None:
        has_first_decode = False
    _CONTEXT = Context(
        is_prefill=is_prefill,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        slot_mapping=slot_mapping,
        flat_slot_mapping=flat_slot_mapping,
        context_lens=context_lens,
        actual_seq_lengths_query=actual_seq_lengths_query,
        actual_seq_lengths_kv=actual_seq_lengths_kv,
        block_tables=block_tables,
        index_block_tables=index_block_tables,
        hbm_block_tables=hbm_block_tables,
        dram_block_tables=dram_block_tables,
        selection_block_tables=selection_block_tables,
        index_slot_mapping=index_slot_mapping,
        flat_index_slot_mapping=flat_index_slot_mapping,
        req_pool_entries=req_pool_entries,
        candidate_lens=candidate_lens,
        candidate_query_lens=candidate_query_lens,
        needs_dsa_update=needs_dsa_update,
        dsa_offload_rows=dsa_offload_rows,
        dsa_offload_all_rows=bool(dsa_offload_all_rows),
        has_first_decode=has_first_decode,
        block_size=block_size,
        flat_slot_mapping_i32=flat_slot_mapping_i32,
    )
    # print(f"_CONTEXT is set to {_CONTEXT}")


def reset_context():
    global _CONTEXT
    _CONTEXT = Context()
