from __future__ import annotations

from collections import deque
from typing import TYPE_CHECKING

from nanovllm.engine.dsa_offload import (
    OFFLOAD_NONE,
    PoolEntryManager,
    SimpleBlockManager,
    lidu_cache_tokens,
    mtp_lidu_cache_tokens,
)
from nanovllm.engine.sequence import (
    FinishReason,
    Sequence,
    SequenceStatus,
    SpeculativeStepOutput,
)

if TYPE_CHECKING:
    from nanovllm.config import Config


class Scheduler:
    def __init__(self, config: Config):
        self.max_num_prefill_seqs_per_step = config.max_num_prefill_seqs_per_step
        self.prefill_chunk_size = config.prefill_chunk_size
        self.max_num_decode_seqs_per_step = config.max_num_decode_seqs_per_step
        self.block_size = config.kvcache_block_size
        self.offload_mode = config.offload_mode
        self.num_speculative_tokens = int(
            getattr(config, "num_speculative_tokens", 0)
        )
        self.uses_offload = self.offload_mode != OFFLOAD_NONE
        self.uses_mtp_index_share = bool(
            self.num_speculative_tokens
            and getattr(
                getattr(config, "hf_config", None),
                "index_share_for_mtp_iteration",
                False,
            )
        )
        self.uses_separate_mtp_cache = bool(
            self.num_speculative_tokens
            and (self.uses_offload or self.uses_mtp_index_share)
        )
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
        self.mtp_block_manager = None
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
        if self.uses_separate_mtp_cache:
            # One dense MTP layer is cheap enough to keep in HBM. With target
            # offload it follows the full-source DRAM pool; otherwise it
            # follows the ordinary dense target HBM pool.
            mtp_blocks = (
                config.num_dram_kvcache_blocks
                if self.uses_offload
                else config.num_hbm_kvcache_blocks
            )
            self.mtp_block_manager = SimpleBlockManager(
                mtp_blocks - 1,
                reserve_null_block=True,
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
        self.last_speculative_stats: dict[str, int] | None = None

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
        # LIDU candidates.
        num_prefill_full_blocks = seq.num_prompt_tokens // seq.block_size
        prefill_tail_len = len(seq) - num_prefill_full_blocks * seq.block_size
        num_prefill_tail_blocks = num_prefill_blocks - num_prefill_full_blocks
        lidu_tokens = 0
        if self.uses_offload:
            lidu_tokens = (
                mtp_lidu_cache_tokens(seq.num_prompt_tokens, self.block_size)
                if self.num_speculative_tokens
                else lidu_cache_tokens(seq.num_prompt_tokens)
            )
            num_sparse_blocks = (
                lidu_tokens // seq.block_size
                if lidu_tokens
                else num_prefill_full_blocks
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
        seq.lidu_decode_hbm_pending = False
        seq.num_prefix_cached_blocks = 0
        seq.offload_finalized = False

    def _lidu_decode_reservation_blocks(self, seq: Sequence) -> int:
        """Return this request's initial/active decode HBM footprint.

        A request that has not completed prefill yet will sample one token at
        the end of prefill, hence the extra token below.  The reservation is a
        logical admission-control budget only: pending C blocks remain free and
        may be borrowed by a later prefill.
        """

        source_tokens = seq.num_prefill_full_blocks * self.block_size
        tail_and_decode_tokens = max(0, len(seq) - source_tokens)
        if not seq.offload_finalized:
            tail_and_decode_tokens += 1
        if self.num_speculative_tokens:
            tail_and_decode_tokens += self.num_speculative_tokens
        tail_and_decode_blocks = (
            tail_and_decode_tokens + self.block_size - 1
        ) // self.block_size
        return seq.num_sparse_blocks + tail_and_decode_blocks

    def _can_reserve_lidu_decode_hbm(self, candidate: Sequence) -> bool:
        if not self.uses_offload:
            return True
        active = list(self.running)
        if self.prefilling is not None and self.prefilling not in active:
            active.append(self.prefilling)
        if candidate not in active:
            active.append(candidate)
        reserved_blocks = sum(
            self._lidu_decode_reservation_blocks(seq) for seq in active
        )
        usable_blocks = (
            len(self.hbm_block_manager.free_block_ids)
            + len(self.hbm_block_manager.used_block_ids)
        )
        return reserved_blocks <= usable_blocks

    def _can_allocate_prefill(self, seq: Sequence) -> bool:
        self._prepare_prefill_metadata(seq)
        if not self._can_reserve_lidu_decode_hbm(seq):
            return False
        if not self.hbm_block_manager.can_allocate_blocks(
            self._prefill_hbm_blocks(seq)
        ):
            return False
        if (
            self.mtp_block_manager is not None
            and not self.mtp_block_manager.can_allocate_blocks(
                self._mtp_prefill_blocks(seq)
            )
        ):
            return False
        if not self.uses_offload:
            return True
        if seq.lidu_cache_tokens == 0:
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
        assert not seq.mtp_block_table
        assert not seq.dram_block_table
        seq.hbm_block_table = self.hbm_block_manager.allocate_blocks(
            self._prefill_hbm_blocks(seq),
        )
        seq.block_table = seq.hbm_block_table
        if self.mtp_block_manager is not None:
            seq.mtp_block_table = self.mtp_block_manager.allocate_blocks(
                self._mtp_prefill_blocks(seq)
            )
        if self.uses_offload:
            seq.offload_pool_entry = self.pool_entry_manager.allocate()
            if seq.lidu_cache_tokens != 0:
                seq.index_block_table = self.index_block_manager.allocate_blocks(
                    seq.num_prefill_blocks,
                )
                seq.dram_block_table = self.dram_block_manager.allocate_blocks(
                    seq.num_prefill_full_blocks,
                )
        seq.bump_decode_metadata_version()

    def _prefill_hbm_blocks(self, seq: Sequence) -> int:
        if not self.num_speculative_tokens or self.uses_offload:
            return seq.num_prefill_blocks
        # Final prefill immediately runs the MTP recurrence. Reserve enough
        # slots for its K-token lookahead and the following target verify.
        required_tokens = len(seq) + self.num_speculative_tokens
        return max(
            seq.num_prefill_blocks,
            (required_tokens + self.block_size - 1) // self.block_size,
        )

    def _mtp_prefill_blocks(self, seq: Sequence) -> int:
        required_tokens = len(seq) + self.num_speculative_tokens
        return (required_tokens + self.block_size - 1) // self.block_size

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

        # Finish the short ordinary tail first. Otherwise a stable MTP row at
        # the front of ``running`` could indefinitely defer a tail row because
        # the two fixed query shapes intentionally cannot share one forward.
        kinds = [self._decode_kind(seq) for seq in self.running]
        decode_kind = "tail" if "tail" in kinds else None
        deferred: deque[Sequence] = deque()
        remaining_to_consider = len(self.running)
        while (
            self.running
            and remaining_to_consider > 0
            and len(scheduled_seqs) < self.max_num_decode_seqs_per_step
        ):
            seq = self.running.popleft()
            remaining_to_consider -= 1
            kind = self._decode_kind(seq)
            if decode_kind is None:
                decode_kind = kind
            if kind != decode_kind:
                deferred.append(seq)
                continue
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

        self.running.extend(deferred)
        if scheduled_seqs:
            self.running.extendleft(reversed(scheduled_seqs))
        return scheduled_seqs, False

    def _decode_kind(self, seq: Sequence) -> str:
        if not self.num_speculative_tokens:
            return "normal"
        draft_count = len(seq.draft_token_ids)
        if draft_count not in (0, self.num_speculative_tokens):
            raise RuntimeError(
                "MTP request has an invalid draft count: "
                f"expected 0 or {self.num_speculative_tokens}, got "
                f"{draft_count}."
            )
        if (
            draft_count == self.num_speculative_tokens
            and len(seq) + 2 * self.num_speculative_tokens - 1
            <= self.max_model_len
        ):
            return "mtp"
        # The final K positions use ordinary one-token eager decode. Keep
        # these requests in their own batch so other requests' drafts do not
        # become stale.
        seq.draft_token_ids.clear()
        return "tail"

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
        pending_sparse_blocks = (
            seq.num_sparse_blocks
            if (
                self.uses_offload and seq.lidu_decode_hbm_pending
            )
            else 0
        )
        growth_blocks = self._decode_growth_blocks(seq)
        target_ready = self.hbm_block_manager.can_allocate_blocks(
            pending_sparse_blocks + growth_blocks
        )
        mtp_growth = self._mtp_decode_growth_blocks(seq)
        mtp_ready = (
            self.mtp_block_manager is None
            or self.mtp_block_manager.can_allocate_blocks(mtp_growth)
        )
        return target_ready and mtp_ready

    def may_append(self, seq: Sequence) -> None:
        pending_sparse_blocks = (
            seq.num_sparse_blocks
            if (
                self.uses_offload and seq.lidu_decode_hbm_pending
            )
            else 0
        )
        growth_blocks = self._decode_growth_blocks(seq)
        num_new_blocks = pending_sparse_blocks + growth_blocks
        mtp_growth = self._mtp_decode_growth_blocks(seq)
        if not num_new_blocks and not mtp_growth:
            return
        new_blocks = (
            self.hbm_block_manager.allocate_blocks(num_new_blocks)
            if num_new_blocks
            else []
        )
        if pending_sparse_blocks:
            seq.hbm_block_table = (
                new_blocks[:pending_sparse_blocks]
                + seq.hbm_block_table
            )
            seq.lidu_decode_hbm_pending = False
        # Decode tokens never participate in sparse selection, so growing a
        # request must not consume IndexCache blocks.
        if growth_blocks:
            seq.hbm_block_table.extend(new_blocks[pending_sparse_blocks:])
        if mtp_growth:
            if self.mtp_block_manager is None:
                raise RuntimeError("MTP decode growth requires an MTP block pool.")
            seq.mtp_block_table.extend(
                self.mtp_block_manager.allocate_blocks(mtp_growth)
            )
        seq.block_table = seq.hbm_block_table
        if num_new_blocks or mtp_growth:
            seq.bump_decode_metadata_version()

    def _decode_growth_blocks(self, seq: Sequence) -> int:
        if not self.num_speculative_tokens:
            return int(len(seq) % self.block_size == 1)
        if self.uses_offload:
            lookahead = (
                self.num_speculative_tokens
                if len(seq.draft_token_ids) == self.num_speculative_tokens
                else 0
            )
            source_tokens = seq.num_prefill_full_blocks * self.block_size
            required_tail_tokens = max(
                0,
                len(seq) + lookahead - source_tokens,
            )
            required_tail_blocks = (
                required_tail_tokens + self.block_size - 1
            ) // self.block_size
            current_tail_blocks = max(
                0, len(seq.hbm_block_table) - seq.num_sparse_blocks
            )
            return max(0, required_tail_blocks - current_tail_blocks)
        lookahead = (
            self.num_speculative_tokens
            if len(seq.draft_token_ids) == self.num_speculative_tokens
            else 0
        )
        # Target verification writes through L+K-1. If all K drafts are
        # accepted, the K-step MTP recurrence can additionally write through
        # L+2K-2 before the scheduler sees the accepted count. Reserve that
        # exact worst-case range up front; at K<=3 it costs at most one block.
        required_tokens = len(seq) + (2 * lookahead - 1 if lookahead else 0)
        required_blocks = (
            required_tokens + self.block_size - 1
        ) // self.block_size
        return max(0, required_blocks - len(seq.hbm_block_table))

    def _mtp_decode_growth_blocks(self, seq: Sequence) -> int:
        if self.mtp_block_manager is None:
            return 0
        lookahead = (
            self.num_speculative_tokens
            if len(seq.draft_token_ids) == self.num_speculative_tokens
            else 0
        )
        required_tokens = len(seq) + (2 * lookahead - 1 if lookahead else 0)
        required_blocks = (
            required_tokens + self.block_size - 1
        ) // self.block_size
        return max(0, required_blocks - len(seq.mtp_block_table))

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
        if self.mtp_block_manager is not None:
            self.mtp_block_manager.free_blocks(seq.mtp_block_table)
        if self.dram_block_manager is not None:
            self.dram_block_manager.free_blocks(seq.dram_block_table)
        if self.pool_entry_manager is not None:
            self.pool_entry_manager.free(seq.offload_pool_entry)
        seq.index_block_table.clear()
        seq.hbm_block_table.clear()
        seq.mtp_block_table.clear()
        seq.block_table = seq.hbm_block_table
        seq.dram_block_table.clear()
        seq.hbm_blocks_to_release.clear()
        seq.offload_pool_entry = -1
        seq.offload_finalized = False
        seq.lidu_cache_initialized = not seq.lidu_cache_tokens
        seq.lidu_decode_hbm_pending = False
        seq.draft_token_ids.clear()
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
        token_ids: list[int] | SpeculativeStepOutput | None,
        is_prefill: bool,
    ) -> None:
        self.last_speculative_stats = None
        if self.prefill_chunk_size and is_prefill:
            self._postprocess_chunked_prefill(seqs, token_ids)
            return

        if isinstance(token_ids, SpeculativeStepOutput):
            self._postprocess_speculative(seqs, token_ids, is_prefill)
            return

        if token_ids is None:
            raise RuntimeError("Model runner returned no sampled tokens.")
        for seq, token_id in zip(seqs, token_ids):
            if is_prefill:
                seq.num_prefill_tokens_processed = len(seq)
            else:
                seq.num_decode_steps += 1
            finished = self._append_sampled_token(seq, token_id)
            if (
                not finished
                and not is_prefill
                and self._finish_at_step_limit(seq)
            ):
                finished = True
            if finished:
                self.running.remove(seq)

    def _postprocess_chunked_prefill(
        self,
        seqs: list[Sequence],
        token_ids: list[int] | SpeculativeStepOutput | None,
    ) -> None:
        if len(seqs) != 1 or seqs[0] is not self.prefilling:
            raise RuntimeError("Chunk prefill must postprocess its active sequence.")
        seq = seqs[0]
        chunk_end = seq.num_prefill_tokens_processed + seq.num_scheduled_tokens
        is_last_chunk = chunk_end == len(seq)
        if is_last_chunk:
            if isinstance(token_ids, SpeculativeStepOutput):
                valid_output = (
                    len(token_ids.token_ids) == 1
                    and len(token_ids.token_ids[0]) == 1
                    and len(token_ids.draft_token_ids) == 1
                )
            else:
                valid_output = token_ids is not None and len(token_ids) == 1
            if not valid_output:
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
        if isinstance(token_ids, SpeculativeStepOutput):
            sampled_token = token_ids.token_ids[0][0]
            next_drafts = token_ids.draft_token_ids[0]
        else:
            sampled_token = token_ids[0]
            next_drafts = []
        if not self._append_sampled_token(seq, sampled_token):
            seq.draft_token_ids = list(next_drafts)
            self.running.append(seq)

    def _postprocess_speculative(
        self,
        seqs: list[Sequence],
        output: SpeculativeStepOutput,
        is_prefill: bool,
    ) -> None:
        batch_size = len(seqs)
        if not (
            len(output.token_ids)
            == len(output.draft_token_ids)
            == len(output.accepted_draft_counts)
            == batch_size
        ):
            raise RuntimeError(
                "MTP output batch size does not match scheduled sequences."
            )

        emitted = 0
        for seq, accepted, drafts in zip(
            seqs, output.token_ids, output.draft_token_ids
        ):
            if is_prefill:
                seq.num_prefill_tokens_processed = len(seq)
            else:
                seq.num_decode_steps += 1
            if not accepted:
                raise RuntimeError("MTP must commit at least one token.")
            finished = False
            for token_id in accepted:
                emitted += 1
                if self._append_sampled_token(seq, int(token_id)):
                    finished = True
                    break
            if (
                not finished
                and not is_prefill
                and self._finish_at_step_limit(seq)
            ):
                finished = True
            if not finished:
                seq.draft_token_ids = [int(token_id) for token_id in drafts]
            else:
                seq.draft_token_ids.clear()
            if finished and seq in self.running:
                self.running.remove(seq)

        proposed = (
            0 if is_prefill else batch_size * self.num_speculative_tokens
        )
        accepted_drafts = (
            0 if is_prefill else sum(output.accepted_draft_counts)
        )
        self.last_speculative_stats = {
            "batch_size": batch_size,
            "emitted_tokens": emitted,
            "accepted_drafts": accepted_drafts,
            "proposed_drafts": proposed,
        }

    def _finish_at_step_limit(self, seq: Sequence) -> bool:
        if (
            seq.max_steps is not None
            and seq.num_decode_steps >= seq.max_steps
        ):
            self.free_seq(seq, FinishReason.LENGTH)
            return True
        return False

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
