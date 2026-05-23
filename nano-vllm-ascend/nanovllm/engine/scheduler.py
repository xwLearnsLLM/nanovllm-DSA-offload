from collections import deque
import os

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence, SequenceStatus, FinishReason
from nanovllm.engine.block_manager import BlockManager


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.lower() in ("1", "true", "yes", "on")


class Scheduler:

    def __init__(self, config: Config):
        self.max_num_seqs = config.max_num_seqs
        self.max_num_batched_tokens = config.max_num_batched_tokens
        self.eos = config.eos
        # Keep the last physical block for padded dummy slots. When NPU
        # indexer/SFA is enabled, also keep block 0 as the null block to match
        # vLLM's paged-cache convention used by the Ascend kernels.
        non_cache_token_ids: list[int] = []
        self.block_manager = BlockManager(
            config.num_kvcache_blocks - 1,
            config.kvcache_block_size,
            non_cache_token_ids=non_cache_token_ids,
            reserve_null_block=(
                _env_flag("NANOVLLM_ENABLE_NPU_INDEXER", False)
                or _env_flag("NANOVLLM_ENABLE_NPU_SFA_PREFILL", False)
                or _env_flag("NANOVLLM_ENABLE_NPU_SFA_DECODE", False)
            ),
        )
        self.waiting: deque[Sequence] = deque()
        self.running: deque[Sequence] = deque()
        self.max_model_len = config.max_model_len

    def is_finished(self):
        return not self.waiting and not self.running

    def add(self, seq: Sequence):
        self.waiting.append(seq)

    def schedule(self) -> tuple[list[Sequence], bool]:
        scheduled_seqs = []
        num_seqs = 0
        num_batched_tokens = 0
        # prefill
        while self.waiting and num_seqs < self.max_num_seqs:
            seq = self.waiting[0]
            if num_batched_tokens + len(seq) > self.max_num_batched_tokens or not self.block_manager.can_allocate(seq):
                break
            self.waiting.popleft()
            self.block_manager.allocate(seq)
            seq.status = SequenceStatus.RUNNING
            self.running.append(seq)
            scheduled_seqs.append(seq)
            num_seqs += 1
            num_batched_tokens += len(seq) - seq.num_cached_tokens
        if scheduled_seqs:
            return scheduled_seqs, True
        # decode
        while self.running and num_seqs < self.max_num_seqs:
            seq = self.running.popleft()
            while not self.block_manager.can_append(seq):
                if self.running:
                    self.preempt(self.running.pop())
                else:
                    self.preempt(seq)
                    seq = None
                    break
            if seq:
                num_seqs += 1
                self.block_manager.may_append(seq)
                scheduled_seqs.append(seq)

        # 如果当前全都被抢占了，返回空列表
        if scheduled_seqs:
            self.running.extendleft(reversed(scheduled_seqs))
        return scheduled_seqs, False

    def preempt(self, seq: Sequence):
        seq.status = SequenceStatus.WAITING
        seq.finish_reason = FinishReason.PREEMPTED
        self.block_manager.deallocate(seq)
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
        self.block_manager.deallocate(seq)

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
