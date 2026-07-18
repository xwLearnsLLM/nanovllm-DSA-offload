from __future__ import annotations

from collections import deque
from typing import TYPE_CHECKING

from nanovllm.engine.dsa_offload import (
    OFFLOAD_LIDU,
    OFFLOAD_NONE,
    PoolEntryManager,
    SimpleBlockManager,
    compute_sparse_blocks,
    lidu_cache_tokens,
)
from nanovllm.engine.sequence import FinishReason, Sequence, SequenceStatus

if TYPE_CHECKING:
    from nanovllm.config import Config


class Scheduler:
    def __init__(self, config: Config):
        self.max_num_prefill_seqs_per_step = config.max_num_prefill_seqs_per_step
        self.prefill_chunk_size = config.prefill_chunk_size
        self.max_num_decode_seqs_per_step = config.max_num_decode_seqs_per_step
        self.block_size = config.kvcache_block_size
        self.offload_mode = config.offload_mode
        self.uses_offload = self.offload_mode != OFFLOAD_NONE
        eos = config.eos
        if isinstance(eos, int):
            eos = (eos,)
        self.eos = frozenset(int(token_id) for token_id in eos)
        self.hbm_block_manager = SimpleBlockManager(
            config.num_hbm_kvcache_blocks - 1,
            reserve_null_block=True,
        )
        self.index_block_manager = None
        self.dram_block_manager = None
        self.pool_entry_manager = None
        if self.uses_offload:
            self.index_block_manager = SimpleBlockManager(
                config.num_dram_kvcache_blocks - 1,
                reserve_null_block=True,
            )
            self.dram_block_manager = SimpleBlockManager(
                config.num_dram_kvcache_blocks,
                reserve_null_block=True,
            )
            self.pool_entry_manager = PoolEntryManager(
                config.max_num_decode_seqs_per_step
            )
        self.waiting: deque[Sequence] = deque()
        self.running: deque[Sequence] = deque()
        self.prefilling: Sequence | None = None
        self.max_model_len = config.max_model_len
        self.num_index_blocks = (
            config.num_dram_kvcache_blocks if self.uses_offload else 0
        )
        self.num_hbm_blocks = config.num_hbm_kvcache_blocks
        self.num_dram_blocks = (
            config.num_dram_kvcache_blocks if self.uses_offload else 0
        )

    def is_finished(self):
        return not self.waiting and not self.running and self.prefilling is None

    def cache_block_usage(self) -> tuple[tuple[int, int], tuple[int, int], tuple[int, int]]:
        # CPU-side counters only; printing them cannot disturb async NPU execution.
        hbm_kv = (len(self.hbm_block_manager.used_block_ids), self.num_hbm_blocks)
        dram_kv = (
            len(self.dram_block_manager.used_block_ids)
            if self.dram_block_manager is not None
            else 0,
            self.num_dram_blocks,
        )
        hbm_index = (
            len(self.index_block_manager.used_block_ids)
            if self.index_block_manager is not None
            else 0,
            self.num_index_blocks,
        )
        return hbm_kv, dram_kv, hbm_index

    def add(self, seq: Sequence):
        self.waiting.append(seq)

    def _prepare_prefill_metadata(self, seq: Sequence) -> None:
        num_prefill_blocks = seq.num_blocks
        # The offload source is permanently bounded by the original prompt's
        # complete blocks.  If decode preemption triggers recomputation, the
        # already-generated tokens stay in the dense tail and never become
        # LIDU/GS candidates.
        num_prefill_full_blocks = seq.num_prompt_tokens // seq.block_size
        prefill_tail_len = len(seq) - num_prefill_full_blocks * seq.block_size
        num_prefill_tail_blocks = num_prefill_blocks - num_prefill_full_blocks
        lidu_tokens = 0
        if self.offload_mode == OFFLOAD_LIDU:
            lidu_tokens = lidu_cache_tokens(seq.num_prompt_tokens)
            num_sparse_blocks = (
                lidu_tokens // seq.block_size
                if lidu_tokens
                else num_prefill_full_blocks
            )
        elif self.uses_offload:
            num_sparse_blocks = compute_sparse_blocks(
                num_prefill_full_blocks, seq.block_size
            )
        else:
            num_sparse_blocks = num_prefill_full_blocks

        seq.num_prefill_blocks = num_prefill_blocks
        seq.num_prefill_full_blocks = num_prefill_full_blocks
        seq.num_prefill_tail_blocks = num_prefill_tail_blocks
        seq.prefill_tail_len = prefill_tail_len
        seq.num_sparse_blocks = num_sparse_blocks
        seq.num_sparse_tokens = num_sparse_blocks * seq.block_size
        seq.lidu_cache_tokens = lidu_tokens
        seq.lidu_cache_initialized = not lidu_tokens
        seq.num_prefix_cached_blocks = 0
        seq.offload_finalized = False

    def _can_allocate_prefill(self, seq: Sequence) -> bool:
        self._prepare_prefill_metadata(seq)
        if not self.hbm_block_manager.can_allocate_blocks(
            seq.num_prefill_blocks
        ):
            return False
        if not self.uses_offload:
            return True
        if self.offload_mode == OFFLOAD_LIDU and seq.lidu_cache_tokens == 0:
            return self.pool_entry_manager.can_allocate()
        return (
            self.index_block_manager.can_allocate_blocks(seq.num_prefill_blocks)
            and self.dram_block_manager.can_allocate_blocks(
                seq.num_prefill_full_blocks
            )
            and self.pool_entry_manager.can_allocate()
        )

    def _allocate_prefill(self, seq: Sequence) -> None:
        assert not seq.index_block_table
        assert not seq.hbm_block_table
        assert not seq.dram_block_table
        seq.hbm_block_table = self.hbm_block_manager.allocate_blocks(
            seq.num_prefill_blocks,
        )
        seq.block_table = seq.hbm_block_table
        if self.uses_offload:
            seq.offload_pool_entry = self.pool_entry_manager.allocate()
            if not (
                self.offload_mode == OFFLOAD_LIDU
                and seq.lidu_cache_tokens == 0
            ):
                seq.index_block_table = self.index_block_manager.allocate_blocks(
                    seq.num_prefill_blocks,
                )
                seq.dram_block_table = self.dram_block_manager.allocate_blocks(
                    seq.num_prefill_full_blocks,
                )
        seq.bump_decode_metadata_version()

    def schedule(self) -> tuple[list[Sequence], bool]:
        if self.prefill_chunk_size and (
            self.prefilling is not None or self.waiting
        ):
            chunked_prefill = self._schedule_chunked_prefill()
            if chunked_prefill:
                return chunked_prefill, True

        scheduled_seqs = []
        # running is the decode concurrency set. Keep it capped so DSA pool entries,
        # HBM sparse budget, and decode batch size all share one clear upper bound.
        decode_room = self.max_num_decode_seqs_per_step - len(self.running)
        prefill_slots = 0
        if not self.prefill_chunk_size:
            prefill_slots = max(
                0,
                min(self.max_num_prefill_seqs_per_step, decode_room),
            )
        while self.waiting and len(scheduled_seqs) < prefill_slots:
            seq = self.waiting[0]
            if not self._can_allocate_prefill(seq):
                break
            self.waiting.popleft()
            self._allocate_prefill(seq)
            seq.reset_prefill_progress()
            seq.status = SequenceStatus.RUNNING
            seq.finish_reason = None
            self.running.append(seq)
            scheduled_seqs.append(seq)
        if scheduled_seqs:
            return scheduled_seqs, True

        while self.running and len(scheduled_seqs) < self.max_num_decode_seqs_per_step:
            seq = self.running.popleft()
            while not self.can_append(seq):
                if self.running:
                    self.preempt(self.running.pop())
                else:
                    self.preempt(seq)
                    seq = None
                    break
            if seq:
                self.may_append(seq)
                scheduled_seqs.append(seq)

        if scheduled_seqs:
            self.running.extendleft(reversed(scheduled_seqs))
        return scheduled_seqs, False

    def _schedule_chunked_prefill(self) -> list[Sequence]:
        if self.prefilling is None:
            decode_room = self.max_num_decode_seqs_per_step - len(self.running)
            if decode_room <= 0 or not self.waiting:
                return []
            seq = self.waiting[0]
            if not self._can_allocate_prefill(seq):
                return []
            self.waiting.popleft()
            self._allocate_prefill(seq)
            seq.reset_prefill_progress()
            seq.status = SequenceStatus.RUNNING
            seq.finish_reason = None
            self.prefilling = seq

        seq = self.prefilling
        remaining_tokens = len(seq) - seq.num_prefill_tokens_processed
        if remaining_tokens <= 0:
            raise RuntimeError(
                "Chunk-prefill sequence has no remaining tokens to schedule."
            )
        seq.num_scheduled_tokens = min(self.prefill_chunk_size, remaining_tokens)
        return [seq]

    def can_append(self, seq: Sequence) -> bool:
        need_new_block = len(seq) % self.block_size == 1
        if not need_new_block:
            return True
        if not self.hbm_block_manager.can_allocate_blocks(1):
            return False
        return True

    def may_append(self, seq: Sequence) -> None:
        if len(seq) % self.block_size != 1:
            return
        # Decode tokens never participate in sparse selection, so growing a
        # request must not consume IndexCache blocks.
        seq.hbm_block_table.extend(
            self.hbm_block_manager.allocate_blocks(1),
        )
        seq.block_table = seq.hbm_block_table
        seq.bump_decode_metadata_version()

    def release_prefill_hbm_blocks(self, seqs: list[Sequence]) -> None:
        if not self.uses_offload:
            return
        for seq in seqs:
            if not seq.hbm_blocks_to_release:
                continue
            self.hbm_block_manager.free_blocks(seq.hbm_blocks_to_release)
            seq.hbm_blocks_to_release.clear()

    def deallocate(self, seq: Sequence):
        if self.index_block_manager is not None:
            self.index_block_manager.free_blocks(seq.index_block_table)
        self.hbm_block_manager.free_blocks(seq.hbm_block_table)
        self.hbm_block_manager.free_blocks(seq.hbm_blocks_to_release)
        if self.dram_block_manager is not None:
            self.dram_block_manager.free_blocks(seq.dram_block_table)
        if self.pool_entry_manager is not None:
            self.pool_entry_manager.free(seq.offload_pool_entry)

        seq.index_block_table.clear()
        seq.hbm_block_table.clear()
        seq.block_table = seq.hbm_block_table
        seq.dram_block_table.clear()
        seq.hbm_blocks_to_release.clear()
        seq.offload_pool_entry = -1
        seq.offload_finalized = False
        seq.lidu_cache_initialized = not seq.lidu_cache_tokens
        seq.bump_decode_metadata_version()

    def preempt(self, seq: Sequence):
        seq.status = SequenceStatus.WAITING
        seq.finish_reason = FinishReason.PREEMPTED
        self.deallocate(seq)
        seq.reset_prefill_progress()
        self.waiting.appendleft(seq)

    def abort_seq_group(self, request_id: str) -> None:
        if self.prefilling is not None and self.prefilling.request_id == request_id:
            seq = self.prefilling
            self.prefilling = None
            self.free_seq(seq, FinishReason.ABORTED)
        for state_queue in [self.waiting, self.running]:
            matched = [s for s in state_queue if s.request_id == request_id]
            for seq in matched:
                state_queue.remove(seq)
                self.free_seq(seq, FinishReason.ABORTED)

    def free_seq(self, seq: Sequence, reason: FinishReason) -> None:
        seq.status = SequenceStatus.FINISHED
        seq.finish_reason = reason
        self.deallocate(seq)

    def postprocess(
        self,
        seqs: list[Sequence],
        token_ids: list[int] | None,
        is_prefill: bool,
    ) -> None:
        if self.prefill_chunk_size and is_prefill:
            self._postprocess_chunked_prefill(seqs, token_ids)
            return

        if token_ids is None:
            raise RuntimeError("Model runner returned no sampled tokens.")
        for seq, token_id in zip(seqs, token_ids):
            if is_prefill:
                seq.num_prefill_tokens_processed = len(seq)
            if self._append_sampled_token(seq, token_id):
                self.running.remove(seq)

    def _postprocess_chunked_prefill(
        self,
        seqs: list[Sequence],
        token_ids: list[int] | None,
    ) -> None:
        if len(seqs) != 1 or seqs[0] is not self.prefilling:
            raise RuntimeError("Chunk prefill must postprocess its active sequence.")
        seq = seqs[0]
        chunk_end = seq.num_prefill_tokens_processed + seq.num_scheduled_tokens
        is_last_chunk = chunk_end == len(seq)
        if is_last_chunk:
            if token_ids is None or len(token_ids) != 1:
                raise RuntimeError(
                    "The final prefill chunk must sample exactly one token."
                )
        elif token_ids is not None:
            raise RuntimeError("An intermediate prefill chunk must not sample.")

        seq.num_prefill_tokens_processed = chunk_end
        seq.num_scheduled_tokens = 0
        if not is_last_chunk:
            return

        self.prefilling = None
        if not self._append_sampled_token(seq, token_ids[0]):
            self.running.append(seq)

    def _append_sampled_token(self, seq: Sequence, token_id: int) -> bool:
        seq.append_token(token_id)

        is_max_model_len = (
            seq.num_prompt_tokens + seq.num_completion_tokens
            >= self.max_model_len
        )
        is_max_tokens = seq.num_completion_tokens >= seq.max_tokens
        is_eos = not seq.ignore_eos and token_id in self.eos

        if is_eos:
            self.free_seq(seq, FinishReason.EOS)
            return True
        if is_max_tokens or is_max_model_len:
            self.free_seq(seq, FinishReason.LENGTH)
            return True
        return False
