from collections import deque

from nanovllm.config import Config
from nanovllm.engine.dsa_offload import (
    PoolEntryManager,
    SimpleBlockManager,
    compute_sparse_blocks,
)
from nanovllm.engine.sequence import FinishReason, Sequence, SequenceStatus


class Scheduler:
    def __init__(self, config: Config):
        self.max_num_prefill_seqs_per_step = config.max_num_prefill_seqs_per_step
        self.max_num_decode_seqs_per_step = config.max_num_decode_seqs_per_step
        self.block_size = config.kvcache_block_size
        self.eos = config.eos
        self.index_block_manager = SimpleBlockManager(
            config.num_dram_kvcache_blocks - 1,
            reserve_null_block=True,
        )
        self.hbm_block_manager = SimpleBlockManager(
            config.num_hbm_kvcache_blocks - 1,
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
        self.max_model_len = config.max_model_len
        self.num_index_blocks = config.num_dram_kvcache_blocks
        self.num_hbm_blocks = config.num_hbm_kvcache_blocks
        self.num_dram_blocks = config.num_dram_kvcache_blocks

    def is_finished(self):
        return not self.waiting and not self.running

    def dsa_block_usage(self) -> tuple[tuple[int, int], tuple[int, int], tuple[int, int]]:
        # CPU-side counters only; printing them cannot disturb async NPU execution.
        hbm_kv = (len(self.hbm_block_manager.used_block_ids), self.num_hbm_blocks)
        dram_kv = (len(self.dram_block_manager.used_block_ids), self.num_dram_blocks)
        hbm_index = (len(self.index_block_manager.used_block_ids), self.num_index_blocks)
        return hbm_kv, dram_kv, hbm_index

    def add(self, seq: Sequence):
        self.waiting.append(seq)

    def _prepare_prefill_metadata(self, seq: Sequence) -> None:
        num_prefill_blocks = seq.num_blocks
        num_prefill_full_blocks = len(seq) // seq.block_size
        prefill_tail_len = len(seq) - num_prefill_full_blocks * seq.block_size
        num_prefill_tail_blocks = num_prefill_blocks - num_prefill_full_blocks
        num_sparse_blocks = compute_sparse_blocks(num_prefill_full_blocks, seq.block_size)

        seq.num_prefill_blocks = num_prefill_blocks
        seq.num_prefill_full_blocks = num_prefill_full_blocks
        seq.num_prefill_tail_blocks = num_prefill_tail_blocks
        seq.prefill_tail_len = prefill_tail_len
        seq.num_sparse_blocks = num_sparse_blocks
        seq.num_sparse_tokens = num_sparse_blocks * seq.block_size
        seq.num_prefix_cached_blocks = 0
        seq.offload_finalized = False

    def _can_allocate_prefill(self, seq: Sequence) -> bool:
        self._prepare_prefill_metadata(seq)
        return (
            self.index_block_manager.can_allocate_blocks(seq.num_prefill_blocks)
            and self.hbm_block_manager.can_allocate_blocks(seq.num_prefill_blocks)
            and self.dram_block_manager.can_allocate_blocks(seq.num_prefill_full_blocks)
            and self.pool_entry_manager.can_allocate()
        )

    def _allocate_prefill(self, seq: Sequence) -> None:
        assert not seq.index_block_table
        assert not seq.hbm_block_table
        assert not seq.dram_block_table
        seq.index_block_table = self.index_block_manager.allocate_blocks(
            seq.num_prefill_blocks,
        )
        seq.hbm_block_table = self.hbm_block_manager.allocate_blocks(
            seq.num_prefill_blocks,
        )
        seq.block_table = seq.hbm_block_table
        seq.dram_block_table = self.dram_block_manager.allocate_blocks(
            seq.num_prefill_full_blocks,
        )
        seq.hbm_cached_tokens_pool_entry = self.pool_entry_manager.allocate()

    def schedule(self) -> tuple[list[Sequence], bool]:
        scheduled_seqs = []
        # running is the decode concurrency set. Keep it capped so DSA pool entries,
        # HBM sparse budget, and decode batch size all share one clear upper bound.
        decode_room = self.max_num_decode_seqs_per_step - len(self.running)
        prefill_slots = max(0, min(self.max_num_prefill_seqs_per_step, decode_room))
        while self.waiting and len(scheduled_seqs) < prefill_slots:
            seq = self.waiting[0]
            if not self._can_allocate_prefill(seq):
                break
            self.waiting.popleft()
            self._allocate_prefill(seq)
            seq.status = SequenceStatus.RUNNING
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

    def can_append(self, seq: Sequence) -> bool:
        need_new_block = len(seq) % self.block_size == 1
        if not need_new_block:
            return True
        return (
            self.index_block_manager.can_allocate_blocks(1)
            and self.hbm_block_manager.can_allocate_blocks(1)
        )

    def may_append(self, seq: Sequence) -> None:
        if len(seq) % self.block_size != 1:
            return
        seq.index_block_table.extend(
            self.index_block_manager.allocate_blocks(1),
        )
        seq.hbm_block_table.extend(
            self.hbm_block_manager.allocate_blocks(1),
        )
        seq.block_table = seq.hbm_block_table

    def release_prefill_hbm_blocks(self, seqs: list[Sequence]) -> None:
        for seq in seqs:
            if not seq.hbm_blocks_to_release:
                continue
            self.hbm_block_manager.free_blocks(seq.hbm_blocks_to_release)
            seq.hbm_blocks_to_release.clear()

    def deallocate(self, seq: Sequence):
        self.index_block_manager.free_blocks(seq.index_block_table)
        self.hbm_block_manager.free_blocks(seq.hbm_block_table)
        self.hbm_block_manager.free_blocks(seq.hbm_blocks_to_release)
        self.dram_block_manager.free_blocks(seq.dram_block_table)
        self.pool_entry_manager.free(seq.hbm_cached_tokens_pool_entry)

        seq.index_block_table.clear()
        seq.hbm_block_table.clear()
        seq.block_table = seq.hbm_block_table
        seq.dram_block_table.clear()
        seq.hbm_blocks_to_release.clear()
        seq.hbm_cached_tokens_pool_entry = -1
        seq.offload_finalized = False

    def preempt(self, seq: Sequence):
        seq.status = SequenceStatus.WAITING
        seq.finish_reason = FinishReason.PREEMPTED
        self.deallocate(seq)
        self.waiting.appendleft(seq)

    def abort_seq_group(self, request_id: str) -> None:
        for state_queue in [self.waiting, self.running]:
            matched = [s for s in state_queue if s.request_id == request_id]
            for seq in matched:
                state_queue.remove(seq)
                self.free_seq(seq, FinishReason.ABORTED)

    def free_seq(self, seq: Sequence, reason: FinishReason) -> None:
        seq.status = SequenceStatus.FINISHED
        seq.finish_reason = reason
        self.deallocate(seq)

    def postprocess(self, seqs: list[Sequence], token_ids: list[int]) -> None:
        for seq, token_id in zip(seqs, token_ids):
            seq.append_token(token_id)

            is_max_model_len = self.max_model_len == seq.num_prompt_tokens + seq.num_completion_tokens
            is_max_tokens = seq.num_completion_tokens == seq.max_tokens
            is_eos = (not seq.ignore_eos and token_id == self.eos)

            if is_eos:
                self.free_seq(seq, FinishReason.EOS)
                self.running.remove(seq)
            elif is_max_tokens or is_max_model_len:
                self.free_seq(seq, FinishReason.LENGTH)
                self.running.remove(seq)
