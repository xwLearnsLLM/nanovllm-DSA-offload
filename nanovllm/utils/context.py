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
    index_slot_mapping: torch.Tensor | None = None
    flat_index_slot_mapping: torch.Tensor | None = None
    req_pool_entries: torch.Tensor | None = None
    candidate_lens: torch.Tensor | None = None
    sparse_selected_lens: torch.Tensor | None = None
    prefill_tail_lens: torch.Tensor | None = None
    decode_lens: torch.Tensor | None = None
    sparse_kv_lens: torch.Tensor | None = None
    is_enforce_eager: bool = True
    real_bs: int = -1
    block_size: int = 256
    flat_slot_mapping_i32: torch.Tensor | None = None
    scratch: dict = field(default_factory=dict)


_CONTEXT = Context()


def get_context():
    return _CONTEXT


def set_context(is_prefill, cu_seqlens_q=None, cu_seqlens_k=None, max_seqlen_q=0, max_seqlen_k=0, slot_mapping=None,
                flat_slot_mapping=None, context_lens=None, actual_seq_lengths_query=None, actual_seq_lengths_kv=None,
                block_tables=None, index_block_tables=None, hbm_block_tables=None, dram_block_tables=None,
                index_slot_mapping=None, flat_index_slot_mapping=None, req_pool_entries=None, candidate_lens=None,
                sparse_selected_lens=None, prefill_tail_lens=None, decode_lens=None, sparse_kv_lens=None,
                is_enforce_eager=None, real_bs=None, block_size=None, flat_slot_mapping_i32=None):
    global _CONTEXT
    if is_enforce_eager is None:
        is_enforce_eager = True
    if real_bs is None:
        real_bs = -1
    if block_size is None:
        block_size = 256
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
        index_slot_mapping=index_slot_mapping,
        flat_index_slot_mapping=flat_index_slot_mapping,
        req_pool_entries=req_pool_entries,
        candidate_lens=candidate_lens,
        sparse_selected_lens=sparse_selected_lens,
        prefill_tail_lens=prefill_tail_lens,
        decode_lens=decode_lens,
        sparse_kv_lens=sparse_kv_lens,
        is_enforce_eager=is_enforce_eager,
        real_bs=real_bs,
        block_size=block_size,
        flat_slot_mapping_i32=flat_slot_mapping_i32,
    )
    # print(f"_CONTEXT is set to {_CONTEXT}")


def reset_context():
    global _CONTEXT
    _CONTEXT = Context()
