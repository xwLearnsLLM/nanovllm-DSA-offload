from copy import copy
from dataclasses import dataclass
from enum import Enum, auto
from itertools import count

from nanovllm.sampling_params import SamplingParams


class SequenceStatus(Enum):
    WAITING = auto()
    RUNNING = auto()
    FINISHED = auto()


class FinishReason(Enum):
    EOS = auto()  # 模型生成了停止符
    LENGTH = auto()  # 达到 max_tokens 或 max_model_len
    ABORTED = auto()  # 外部取消
    PREEMPTED = auto()  # 被调度器抢占（虽然通常会回到 WAITING，但在某些统计中也算结束）


class Sequence:
    counter = count()

    def __init__(self,
                 token_ids: list[int],
                 sampling_params=SamplingParams(),
                 request_id: str = None,
                 block_size: int = 256
                 ):
        self.block_size = block_size
        self.seq_id = next(Sequence.counter)
        self.request_id = request_id
        self.status = SequenceStatus.WAITING
        self.token_ids = copy(token_ids)
        self.last_token = token_ids[-1]
        self.num_tokens = len(self.token_ids)
        self.num_prompt_tokens = len(token_ids)
        self.num_cached_tokens = 0
        self.num_prefill_tokens_processed = 0
        self.num_scheduled_tokens = 0
        self.block_table = []
        self.index_block_table = []
        self.hbm_block_table = self.block_table
        self.dram_block_table = []
        self.offload_pool_entry = -1
        self.offload_finalized = False
        self.lidu_cache_tokens = 0
        self.lidu_cache_initialized = False
        self.num_prefill_blocks = 0
        self.num_prefill_full_blocks = 0
        self.num_prefill_tail_blocks = 0
        self.num_prefix_cached_blocks = 0
        self.num_sparse_blocks = 0
        self.num_sparse_tokens = 0
        self.prefill_tail_len = 0
        self.hbm_blocks_to_release = []
        # Incremented only when decode metadata changes, not for every sampled
        # token.  ModelRunner uses this revision to keep the large DSA block
        # tables resident across steady-state decode steps.
        self.decode_metadata_version = 0
        self.temperature = sampling_params.temperature
        self.max_tokens = sampling_params.max_tokens
        self.ignore_eos = sampling_params.ignore_eos
        self.finish_reason = None

    def __len__(self):
        return self.num_tokens

    def __getitem__(self, key):
        return self.token_ids[key]

    @property
    def is_finished(self):
        return self.status == SequenceStatus.FINISHED

    @property
    def num_completion_tokens(self):
        return self.num_tokens - self.num_prompt_tokens

    @property
    def is_first_decode_after_prefill(self):
        return (
            self.num_prefill_tokens_processed > 0
            and self.num_decode_tokens_since_prefill == 1
        )

    @property
    def num_decode_tokens_since_prefill(self):
        return len(self) - self.num_prefill_tokens_processed

    @property
    def prompt_token_ids(self):
        return self.token_ids[:self.num_prompt_tokens]

    @property
    def completion_token_ids(self):
        return self.token_ids[self.num_prompt_tokens:]

    @property
    def num_cached_blocks(self):
        return self.num_cached_tokens // self.block_size

    @property
    def num_blocks(self):
        return (self.num_tokens + self.block_size - 1) // self.block_size

    @property
    def last_block_num_tokens(self):
        return self.num_tokens - (self.num_blocks - 1) * self.block_size

    def block(self, i):
        assert 0 <= i < self.num_blocks
        return self.token_ids[i * self.block_size: (i + 1) * self.block_size]

    def append_token(self, token_id: int):
        self.token_ids.append(token_id)
        self.last_token = token_id
        self.num_tokens += 1

    def bump_decode_metadata_version(self) -> None:
        self.decode_metadata_version += 1

    def reset_prefill_progress(self):
        self.num_prefill_tokens_processed = 0
        self.num_scheduled_tokens = 0

    def scheduled_prefill_chunk(
        self,
    ) -> tuple[
        list[int],
        list[int],
        list[int],
        list[int],
        int,
        bool,
    ]:
        start = self.num_prefill_tokens_processed
        end = start + self.num_scheduled_tokens
        if self.num_scheduled_tokens <= 0 or not (0 <= start < end <= len(self)):
            raise RuntimeError(
                "Invalid scheduled prefill range: "
                f"start={start}, end={end}, sequence_length={len(self)}."
            )

        positions = list(range(start, end))

        def slot_mapping(block_table: list[int]) -> list[int]:
            if end > len(block_table) * self.block_size:
                raise RuntimeError(
                    "Scheduled prefill range exceeds its cache block table."
                )
            return [
                block_table[position // self.block_size] * self.block_size
                + position % self.block_size
                for position in positions
            ]

        return (
            self.token_ids[start:end],
            positions,
            slot_mapping(self.hbm_block_table),
            slot_mapping(self.index_block_table)
            if self.index_block_table
            else [],
            end,
            end == len(self),
        )

    def __getstate__(self):
        return {
            "seq_id": self.seq_id,
            "request_id": self.request_id,
            "num_tokens": self.num_tokens,
            "num_prompt_tokens": self.num_prompt_tokens,
            "num_cached_tokens": self.num_cached_tokens,
            "num_prefill_tokens_processed": self.num_prefill_tokens_processed,
            "num_scheduled_tokens": self.num_scheduled_tokens,
            "block_table": self.block_table,
            "index_block_table": self.index_block_table,
            "hbm_block_table": self.hbm_block_table,
            "dram_block_table": self.dram_block_table,
            "offload_pool_entry": self.offload_pool_entry,
            "offload_finalized": self.offload_finalized,
            "lidu_cache_tokens": self.lidu_cache_tokens,
            "lidu_cache_initialized": self.lidu_cache_initialized,
            "num_prefill_blocks": self.num_prefill_blocks,
            "num_prefill_full_blocks": self.num_prefill_full_blocks,
            "num_prefill_tail_blocks": self.num_prefill_tail_blocks,
            "num_prefix_cached_blocks": self.num_prefix_cached_blocks,
            "num_sparse_blocks": self.num_sparse_blocks,
            "num_sparse_tokens": self.num_sparse_tokens,
            "prefill_tail_len": self.prefill_tail_len,
            "hbm_blocks_to_release": self.hbm_blocks_to_release,
            "decode_metadata_version": self.decode_metadata_version,
            "token_ids": self.token_ids,
            "status": self.status,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "ignore_eos": self.ignore_eos,
            "block_size": self.block_size,
            "finish_reason": self.finish_reason,
        }

    def __setstate__(self, state):
        self.seq_id = state["seq_id"]
        self.request_id = state["request_id"]
        self.num_tokens = state["num_tokens"]
        self.num_prompt_tokens = state["num_prompt_tokens"]
        self.num_cached_tokens = state["num_cached_tokens"]
        self.num_prefill_tokens_processed = state.get(
            "num_prefill_tokens_processed", 0
        )
        self.num_scheduled_tokens = state.get("num_scheduled_tokens", 0)
        self.block_table = state["block_table"]
        self.index_block_table = state.get("index_block_table", self.block_table)
        self.hbm_block_table = state.get("hbm_block_table", self.block_table)
        self.dram_block_table = state.get("dram_block_table", [])
        self.offload_pool_entry = state.get("offload_pool_entry", -1)
        self.offload_finalized = state.get("offload_finalized", False)
        self.lidu_cache_tokens = state.get("lidu_cache_tokens", 0)
        self.lidu_cache_initialized = state.get(
            "lidu_cache_initialized", False
        )
        self.num_prefill_blocks = state.get("num_prefill_blocks", 0)
        self.num_prefill_full_blocks = state.get("num_prefill_full_blocks", 0)
        self.num_prefill_tail_blocks = state.get("num_prefill_tail_blocks", 0)
        self.num_prefix_cached_blocks = state.get("num_prefix_cached_blocks", 0)
        self.num_sparse_blocks = state.get("num_sparse_blocks", 0)
        self.num_sparse_tokens = state.get("num_sparse_tokens", 0)
        self.prefill_tail_len = state.get("prefill_tail_len", 0)
        self.hbm_blocks_to_release = state.get("hbm_blocks_to_release", [])
        self.decode_metadata_version = state.get("decode_metadata_version", 0)
        self.token_ids = state["token_ids"]
        self.status = state["status"]
        self.temperature = state["temperature"]
        self.max_tokens = state["max_tokens"]
        self.ignore_eos = state["ignore_eos"]
        self.block_size = state["block_size"]
        self.finish_reason = state["finish_reason"]
        if self.token_ids:
            self.last_token = self.token_ids[-1]

    def __repr__(self):
        return f"Seq(id={self.seq_id}, status={self.status.name}, reason={self.finish_reason.name if self.finish_reason else 'None'})"


@dataclass
class DecodeSequenceMetadata:
    """Small worker-side view of a sequence for a decode-only forward.

    Full ``Sequence`` serialization includes the complete prompt and generated
    token history.  Stable decode only needs the last token plus cache layout
    metadata, so TP workers receive this compact snapshot instead.
    """

    seq_id: int
    num_tokens: int
    last_token: int
    num_prefill_tokens_processed: int
    block_size: int
    hbm_block_table: list[int]
    index_block_table: list[int]
    dram_block_table: list[int]
    offload_pool_entry: int
    lidu_cache_tokens: int
    lidu_cache_initialized: bool
    num_prefill_full_blocks: int
    num_sparse_blocks: int
    num_sparse_tokens: int
    prefill_tail_len: int
    temperature: float
    decode_metadata_version: int

    @classmethod
    def from_sequence(cls, seq: Sequence) -> "DecodeSequenceMetadata":
        return cls(
            seq_id=seq.seq_id,
            num_tokens=len(seq),
            last_token=seq.last_token,
            num_prefill_tokens_processed=seq.num_prefill_tokens_processed,
            block_size=seq.block_size,
            hbm_block_table=list(seq.hbm_block_table),
            index_block_table=list(seq.index_block_table),
            dram_block_table=list(seq.dram_block_table),
            offload_pool_entry=seq.offload_pool_entry,
            lidu_cache_tokens=seq.lidu_cache_tokens,
            lidu_cache_initialized=seq.lidu_cache_initialized,
            num_prefill_full_blocks=seq.num_prefill_full_blocks,
            num_sparse_blocks=seq.num_sparse_blocks,
            num_sparse_tokens=seq.num_sparse_tokens,
            prefill_tail_len=seq.prefill_tail_len,
            temperature=seq.temperature,
            decode_metadata_version=seq.decode_metadata_version,
        )

    def __len__(self) -> int:
        return self.num_tokens

    @property
    def num_decode_tokens_since_prefill(self) -> int:
        return self.num_tokens - self.num_prefill_tokens_processed

    @property
    def is_first_decode_after_prefill(self) -> bool:
        return (
            self.num_prefill_tokens_processed > 0
            and self.num_decode_tokens_since_prefill == 1
        )
