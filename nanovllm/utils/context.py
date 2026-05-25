from __future__ import annotations

from dataclasses import dataclass
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
    is_enforce_eager: bool = True
    real_bs: int = -1
    block_size: int = 256


_CONTEXT = Context()


def get_context():
    return _CONTEXT


def set_context(is_prefill, cu_seqlens_q=None, cu_seqlens_k=None, max_seqlen_q=0, max_seqlen_k=0, slot_mapping=None,
                flat_slot_mapping=None, context_lens=None, actual_seq_lengths_query=None, actual_seq_lengths_kv=None,
                block_tables=None, is_enforce_eager=None, real_bs=None, block_size=None):
    global _CONTEXT
    _CONTEXT = Context(is_prefill, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, slot_mapping,
                       flat_slot_mapping, context_lens, actual_seq_lengths_query, actual_seq_lengths_kv, block_tables,
                       is_enforce_eager, real_bs, block_size)
    # print(f"_CONTEXT is set to {_CONTEXT}")


def reset_context():
    global _CONTEXT
    _CONTEXT = Context()
