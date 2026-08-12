# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the Nano-vLLM project

from __future__ import annotations

import pickle
from dataclasses import dataclass
from multiprocessing.shared_memory import SharedMemory
from multiprocessing.synchronize import Event
from typing import Any

import torch
import torch.distributed as dist
import torch_npu  # noqa: F401

torch.npu.config.allow_internal_format = True

from nanovllm.config import Config
from nanovllm.engine.dsa_offload import (
    OFFLOAD_FUSE,
    OFFLOAD_NONE,
    finalize_prefill_hbm_layout,
    max_lidu_cache_tokens,
    mtp_lidu_cache_tokens,
)
from nanovllm.engine.full_decode_graph import (
    FullDecodeOnlyGraphManager,
    MTPDecodeOnlyGraphManager,
    is_full_decode_graph_capturing,
    select_capture_size,
)
from nanovllm.engine.sequence import (
    DecodeBatchDelta,
    DecodeBatchPacket,
    DecodeBatchSnapshot,
    DecodeMetadataKey,
    DecodeSequenceMetadata,
    Sequence,
    SpeculativeStepOutput,
    apply_decode_batch_packet,
    build_decode_batch_packet,
    decode_metadata_key,
)
from nanovllm.engine.speculative import (
    greedy_prefix_accept,
    materialize_accepted_tokens,
    shifted_mtp_prefill_tokens,
)
from nanovllm.layers.sampler import Sampler
from nanovllm.models.glm_moe_dsa import GlmMoeDsaForCausalLM
from nanovllm.utils.context import get_context, set_context, reset_context
from nanovllm.utils.glm_quant import (
    GLM_BALANCED_MOE_EXPERT_IDS_KEY,
    balanced_moe_expert_ids,
)
from nanovllm.utils.loader import load_model
from nanovllm.utils.logger import init_logger

logger = init_logger(__name__)


@dataclass
class _HostDeviceVector:
    host: torch.Tensor
    device: torch.Tensor
    host_array: Any

    @classmethod
    def allocate(
        cls,
        size: int,
        dtype: torch.dtype,
        device: str,
    ) -> "_HostDeviceVector":
        host = torch.empty(
            size,
            dtype=dtype,
            device="cpu",
            pin_memory=True,
        )
        return cls(
            host=host,
            device=torch.empty(size, dtype=dtype, device=device),
            host_array=host.numpy(),
        )

    def stage(self, values: list[int] | list[float]) -> torch.Tensor:
        if len(values) != self.host.numel():
            raise ValueError(
                f"Decode vector size changed: runtime={len(values)}, "
                f"buffer={self.host.numel()}."
            )
        self.host_array[...] = values
        self.device.copy_(self.host, non_blocking=True)
        return self.device


@dataclass
class _DecodeDynamicBuffers:
    input_ids: _HostDeviceVector
    positions: _HostDeviceVector
    slot_mapping_i32: _HostDeviceVector
    actual_seq_lengths_kv: _HostDeviceVector

    @classmethod
    def allocate(cls, batch_size: int, device: str) -> "_DecodeDynamicBuffers":
        return cls(
            input_ids=_HostDeviceVector.allocate(
                batch_size,
                torch.int64,
                device,
            ),
            positions=_HostDeviceVector.allocate(
                batch_size,
                torch.int64,
                device,
            ),
            slot_mapping_i32=_HostDeviceVector.allocate(
                batch_size,
                torch.int32,
                device,
            ),
            actual_seq_lengths_kv=_HostDeviceVector.allocate(
                batch_size,
                torch.int32,
                device,
            ),
        )


@dataclass
class _DecodeStaticMetadata:
    key: tuple[tuple[int, int], ...]
    query_len: int
    block_tables: torch.Tensor
    index_block_tables: torch.Tensor
    dram_block_tables: torch.Tensor
    req_pool_entries: torch.Tensor
    candidate_lens: torch.Tensor
    candidate_query_lens: torch.Tensor
    lidu_cache_tokens: torch.Tensor
    temperatures: torch.Tensor | None


@dataclass
class _MtpIndexShareMetadata:
    block_tables: torch.Tensor
    req_pool_entries: torch.Tensor
    candidate_lens: torch.Tensor
    lidu_cache_tokens: torch.Tensor

    def select(self, rows: torch.Tensor) -> "_MtpIndexShareMetadata":
        return _MtpIndexShareMetadata(
            block_tables=self.block_tables.index_select(0, rows),
            req_pool_entries=self.req_pool_entries.index_select(0, rows),
            candidate_lens=self.candidate_lens.index_select(0, rows),
            lidu_cache_tokens=self.lidu_cache_tokens.index_select(0, rows),
        )


class ModelRunner:

    def __init__(self, config: Config, rank: int, event: Event | list[Event]):
        self.config = config
        self.hf_config = config.hf_config
        self.block_size = config.kvcache_block_size
        self.offload_mode = config.offload_mode
        self.num_speculative_tokens = config.num_speculative_tokens
        self.uses_offload = self.offload_mode != OFFLOAD_NONE
        self.uses_mtp_index_share = bool(
            self.num_speculative_tokens
            and getattr(
                self.hf_config, "index_share_for_mtp_iteration", False
            )
        )
        self.uses_separate_mtp_cache = bool(
            self.num_speculative_tokens
            and (self.uses_offload or self.uses_mtp_index_share)
        )
        self.world_size = config.tensor_parallel_size
        self.rank = rank
        self.event = event
        self.device = config.device

        torch.npu.set_device(rank)
        dist.init_process_group("hccl", f"tcp://localhost:{config.hccl_port}", world_size=self.world_size, rank=rank)
        default_dtype = torch.get_default_dtype()

        torch.set_default_dtype(torch.bfloat16)
        torch.set_default_device(self.device)

        self.model = self._load_model()
        self.uses_sparse_tail_attention = self.uses_offload

        self.sampler = Sampler()
        self._decode_dynamic_buffers: dict[int, _DecodeDynamicBuffers] = {}
        self._decode_static_metadata: _DecodeStaticMetadata | None = None
        self._decode_metadata_cache_hits = 0
        self._decode_metadata_cache_misses = 0
        self._compact_decode_ipc_steps = 0
        self._compact_decode_ipc_bytes = 0
        self._decode_ipc_snapshot_steps = 0
        self._decode_ipc_snapshot_bytes = 0
        self._decode_ipc_delta_steps = 0
        self._decode_ipc_delta_bytes = 0
        self._worker_decode_metadata_key: DecodeMetadataKey | None = None
        self._worker_decode_sequences: list[DecodeSequenceMetadata] | None = None
        self._mtp_decode_cu_seqlens: dict[int, torch.Tensor] = {}
        torch.npu.empty_cache()
        self._allocate_mla_cache()
        self._setup_index_share_owners()
        self.decode_graph_manager = None
        if not config.enforce_eager:
            text_config = getattr(config.hf_config, "text_config", config.hf_config)
            if self.num_speculative_tokens:
                self.decode_graph_manager = MTPDecodeOnlyGraphManager(
                    target_forward=self._mtp_target_graph_forward,
                    draft_forward=self._mtp_draft_graph_forward,
                    target_warmup=self.model.full_decode_graph_eager_warmup,
                    draft_warmup=self._mtp_draft_graph_eager_warmup,
                    capture_sizes=config.decode_graph_capture_sizes,
                    max_model_len=config.max_model_len,
                    block_size=config.kvcache_block_size,
                    device=self.device,
                    speculative_tokens=self.num_speculative_tokens,
                    expected_target_tasks=(
                        0
                        if self.uses_offload
                        else int(text_config.num_hidden_layers)
                        * (
                            self.num_speculative_tokens + 1
                            if config.glm_version == "5.2"
                            else 1
                        )
                    ),
                    serial_target_verification=(config.glm_version == "5.2"),
                    offload_mode=self.offload_mode,
                    log_enabled=self.rank == 0,
                )
                if self.rank == 0:
                    logger.info(
                        "FULL_DECODE_ONLY MTP: first decode and uninitialized "
                        "LIDU rows stay eager; exact-size target and draft "
                        "graphs are captured lazily."
                    )
                if self.world_size > 1:
                    dist.barrier()
            else:
                self.decode_graph_manager = FullDecodeOnlyGraphManager(
                    self.model,
                    capture_sizes=config.decode_graph_capture_sizes,
                    max_model_len=config.max_model_len,
                    block_size=config.kvcache_block_size,
                    device=self.device,
                    expected_mla_tasks=(
                        0
                        if self.uses_sparse_tail_attention
                        else int(text_config.num_hidden_layers)
                    ),
                    offload_mode=self.offload_mode,
                    uses_tensor_mla_lengths=self.uses_sparse_tail_attention,
                    log_enabled=self.rank == 0,
                )
                if self.uses_offload:
                    if self.rank == 0:
                        logger.info(
                            "FULL_DECODE_ONLY: deferring LIDU graph capture "
                            "until the first initialized stable decode batch."
                        )
                    if self.world_size > 1:
                        dist.barrier()
                else:
                    if self.world_size > 1:
                        dist.barrier()
                    self.decode_graph_manager.capture_all()
                    if self.world_size > 1:
                        dist.barrier()

        torch.set_default_device("cpu")
        torch.set_default_dtype(default_dtype)
        self._share_memory(rank)

    def _load_model(self):
        model = GlmMoeDsaForCausalLM(self.hf_config)
        if self.rank == 0:
            quant_metadata = getattr(
                self.hf_config, "nanovllm_quant_metadata", {}
            )
            attention = "dense MLA (all KV)"
            if self.offload_mode == OFFLOAD_FUSE:
                attention = "LIM + COPYSFA sparse-and-tail MLA"
            elif self.uses_offload:
                attention = "LIM + SCATTER + sparse-and-tail MLA"
            logger.info(
                "%s W4A8: %s decode, attention=%s, max_model_len=%d, "
                "EP%d (%d local experts/rank), ModelSlim version=%s "
                "group_size=%s; MTP K=%d.",
                getattr(self.hf_config, "nanovllm_model_name", "GLM"),
                (
                    "eager"
                    if self.config.enforce_eager
                    else "FULL_DECODE_ONLY raw ACLGraph"
                ),
                attention,
                self.config.max_model_len,
                self.world_size,
                int(self.hf_config.n_routed_experts) // self.world_size,
                quant_metadata.get("version"),
                quant_metadata.get("group_size"),
                self.num_speculative_tokens,
            )
        loaded_parameters = load_model(
            model,
            self.config.model,
            name_mapping=getattr(model, "weight_name_mapping", None),
        )
        validate_loaded_weights = getattr(
            model, "validate_loaded_weights", None
        )
        if callable(validate_loaded_weights):
            validate_loaded_weights(loaded_parameters)
        if hasattr(model, "post_load_prepare"):
            model.post_load_prepare()
        return model

    def _share_memory(self, rank):
        share_free_name = "nano_vllm_ascend"
        if self.world_size > 1:
            if rank == 0:
                try:
                    self.shm = SharedMemory(name=share_free_name, create=True, size=2 ** 20)
                except FileExistsError:
                    # A previous crash can leave /dev/shm/nano_vllm_ascend behind.
                    existing_shm = SharedMemory(name=share_free_name)
                    existing_shm.close()
                    existing_shm.unlink()
                    self.shm = SharedMemory(name=share_free_name, create=True, size=2 ** 20)
                dist.barrier()
            else:
                dist.barrier()
                self.shm = SharedMemory(name=share_free_name)
                self.loop()

    def exit(self):
        if self.rank == 0 and self.decode_graph_manager is not None:
            logger.info(
                "FULL_DECODE_ONLY final stats: %s",
                self.decode_graph_manager.stats(),
            )
        if self.rank == 0:
            ipc_stats = self._decode_ipc_stats()
            logger.info(
                "Decode hot-path stats: compact_ipc_steps=%d, "
                "average_ipc_bytes=%d, ipc_snapshot_steps=%d, "
                "ipc_snapshot_avg_bytes=%d, ipc_delta_steps=%d, "
                "ipc_delta_avg_bytes=%d, metadata_cache_hits=%d, "
                "metadata_cache_misses=%d",
                ipc_stats["compact_ipc_steps"],
                ipc_stats["average_ipc_bytes"],
                ipc_stats["ipc_snapshot_steps"],
                ipc_stats["ipc_snapshot_avg_bytes"],
                ipc_stats["ipc_delta_steps"],
                ipc_stats["ipc_delta_avg_bytes"],
                self._decode_metadata_cache_hits,
                self._decode_metadata_cache_misses,
            )
        if self.world_size > 1:
            self.shm.close()
            dist.barrier()
            if self.rank == 0:
                self.shm.unlink()
        torch.npu.synchronize()
        dist.destroy_process_group()

    def get_decode_graph_stats(self):
        if self.decode_graph_manager is None:
            return {"enabled": False, "mode": "eager"}
        stats = self.decode_graph_manager.stats()
        stats.update(
            **self._decode_ipc_stats(),
            metadata_cache_hits=self._decode_metadata_cache_hits,
            metadata_cache_misses=self._decode_metadata_cache_misses,
        )
        return stats

    def _decode_ipc_stats(self) -> dict[str, int]:
        return {
            "compact_ipc_steps": self._compact_decode_ipc_steps,
            "average_ipc_bytes": (
                self._compact_decode_ipc_bytes
                // max(self._compact_decode_ipc_steps, 1)
            ),
            "ipc_snapshot_steps": self._decode_ipc_snapshot_steps,
            "ipc_snapshot_avg_bytes": (
                self._decode_ipc_snapshot_bytes
                // max(self._decode_ipc_snapshot_steps, 1)
            ),
            "ipc_delta_steps": self._decode_ipc_delta_steps,
            "ipc_delta_avg_bytes": (
                self._decode_ipc_delta_bytes
                // max(self._decode_ipc_delta_steps, 1)
            ),
        }

    def loop(self):
        while True:
            method_name, args = self.read_shm()
            args = self._expand_worker_args(method_name, args)
            self.call(method_name, *args)
            if method_name == "exit":
                break

    def read_shm(self):
        assert self.world_size > 1 and self.rank > 0
        self.event.wait()
        n = int.from_bytes(self.shm.buf[0:4], "little")
        if n <= 0 or n + 4 > len(self.shm.buf):
            raise RuntimeError(
                f"Invalid TP worker IPC payload size {n}; "
                f"shared memory capacity is {len(self.shm.buf) - 4}."
            )
        method_name, *args = pickle.loads(self.shm.buf[4:n + 4])
        self.event.clear()
        return method_name, args

    def write_shm(self, method_name, *args):
        assert self.world_size > 1 and self.rank == 0
        data = pickle.dumps(
            (method_name, *args),
            protocol=pickle.HIGHEST_PROTOCOL,
        )
        n = len(data)
        if n + 4 > len(self.shm.buf):
            raise RuntimeError(
                "TP worker IPC payload exceeds the shared-memory buffer: "
                f"payload={n}, capacity={len(self.shm.buf) - 4}. "
                "Enable chunk prefill or reduce the prefill batch size."
            )
        self.shm.buf[0:4] = n.to_bytes(4, "little")
        self.shm.buf[4:n + 4] = data
        for event in self.event:
            event.set()
        return n

    def _compact_worker_args(self, method_name: str, args: tuple) -> tuple:
        if (
            method_name == "run"
            and len(args) == 2
            and args[1] is False
        ):
            packet, key = build_decode_batch_packet(
                args[0],
                self._worker_decode_metadata_key,
            )
            self._worker_decode_metadata_key = key
            return packet, False
        return args

    def _expand_worker_args(self, method_name: str, args: list) -> tuple:
        if (
            method_name != "run"
            or len(args) != 2
            or args[1] is not False
        ):
            return tuple(args)
        packet = args[0]
        if not isinstance(packet, (DecodeBatchSnapshot, DecodeBatchDelta)):
            raise RuntimeError(
                "TP worker received a decode run without a metadata packet."
            )
        sequences, key = apply_decode_batch_packet(
            packet,
            self._worker_decode_sequences,
        )
        self._worker_decode_sequences = sequences
        self._worker_decode_metadata_key = key
        return sequences, False

    def call(self, method_name, *args):
        if self.world_size > 1 and self.rank == 0:
            worker_args = self._compact_worker_args(method_name, args)
            payload_bytes = self.write_shm(method_name, *worker_args)
            if worker_args is not args:
                self._compact_decode_ipc_steps += 1
                self._compact_decode_ipc_bytes += payload_bytes
                packet: DecodeBatchPacket = worker_args[0]
                if isinstance(packet, DecodeBatchSnapshot):
                    self._decode_ipc_snapshot_steps += 1
                    self._decode_ipc_snapshot_bytes += payload_bytes
                else:
                    self._decode_ipc_delta_steps += 1
                    self._decode_ipc_delta_bytes += payload_bytes
        method = getattr(self, method_name, None)
        return method(*args)

    def _allocate_mla_cache(self):
        config = self.config
        hf_config = config.hf_config
        text_config = getattr(hf_config, "text_config", hf_config)
        cache_dtype = torch.bfloat16
        num_target_layers = int(text_config.num_hidden_layers)
        num_layers = num_target_layers + int(
            self.num_speculative_tokens > 0 and not self.uses_offload
        )
        kv_lora_rank = int(text_config.kv_lora_rank)
        rope_dim = int(text_config.qk_rope_head_dim)
        hbm_kv_block_bytes = num_layers * self.block_size * (kv_lora_rank + rope_dim) * torch.empty((), dtype=cache_dtype).element_size()
        ckv_shape = (
            num_layers,
            config.num_hbm_kvcache_blocks,
            self.block_size,
            1,
            kv_lora_rank,
        )
        kpe_shape = (
            num_layers,
            config.num_hbm_kvcache_blocks,
            self.block_size,
            1,
            rope_dim,
        )
        if not self.uses_offload:
            if self.rank == 0:
                logger.info(
                    "Using dense MLA cache: hbm_blocks=%d, "
                    "single_block=%.2f MB",
                    config.num_hbm_kvcache_blocks,
                    hbm_kv_block_bytes / 1024 ** 2,
                )
                logger.info("Dense MLA CKV cache shape: %s", ckv_shape)
                logger.info("Dense MLA KPE cache shape: %s", kpe_shape)
            mtp_lidu_slots = None
            if self.uses_mtp_index_share:
                max_source_tokens = (
                    self.config.max_model_len // self.block_size
                ) * self.block_size
                mtp_lidu_slots = torch.full(
                    (
                        config.max_num_decode_seqs_per_step,
                        max_source_tokens,
                    ),
                    -1,
                    dtype=torch.int32,
                    device=self.device,
                )
            for module in self.model.modules():
                if not hasattr(module, "assign_mla_cache"):
                    continue
                ckv_cache = torch.empty(
                    ckv_shape[1:], dtype=cache_dtype, device=self.device
                )
                kpe_cache = torch.empty(
                    kpe_shape[1:], dtype=cache_dtype, device=self.device
                )
                ckv_cache.zero_()
                kpe_cache.zero_()
                module.assign_mla_cache(ckv_cache, kpe_cache)
                if (
                    self.uses_mtp_index_share
                    and getattr(module, "uses_mtp_index_share", False)
                ):
                    index_cache = torch.zeros(
                        (
                            ckv_shape[1],
                            self.block_size,
                            1,
                            int(text_config.index_head_dim),
                        ),
                        dtype=cache_dtype,
                        device=self.device,
                    )
                    if mtp_lidu_slots is None:
                        raise RuntimeError(
                            "MTP IndexShare slots were not allocated."
                        )
                    module.assign_mtp_index_share_state(
                        index_cache, mtp_lidu_slots
                    )
            return

        index_dim = int(text_config.index_head_dim)
        if self.rank == 0:
            logger.info(
                "Using LIDU cache blocks: hbm=%d, dram=%d, "
                "index=%d, max_sparse_tokens=%d",
                config.num_hbm_kvcache_blocks,
                config.num_dram_kvcache_blocks,
                config.num_dram_kvcache_blocks,
                (
                    mtp_lidu_cache_tokens(
                        config.max_model_len, self.block_size
                    )
                    if self.num_speculative_tokens
                    else max_lidu_cache_tokens(config.max_model_len)
                ),
            )

        index_shape = (num_target_layers, config.num_dram_kvcache_blocks, self.block_size, 1, index_dim)
        dram_ckv_shape = (num_target_layers, config.num_dram_kvcache_blocks, self.block_size, 1, kv_lora_rank)
        dram_kpe_shape = (num_target_layers, config.num_dram_kvcache_blocks, self.block_size, 1, rope_dim)
        max_source_tokens = max(
            self.block_size,
            (
                (config.max_model_len + self.block_size - 1)
                // self.block_size
                * self.block_size
            ),
        )
        lidu_slots_shape = (
            config.max_num_decode_seqs_per_step,
            max_source_tokens,
        )
        if self.rank == 0:
            logger.info(f"Single HBM KV Block Size: {hbm_kv_block_bytes / 1024 ** 2:.2f} MB")
            for name, shape in [
                ("DSA CKV cache", ckv_shape),
                ("DSA KPE cache", kpe_shape),
                ("DSA index cache", index_shape),
                ("DSA DRAM CKV cache", dram_ckv_shape),
                ("DSA DRAM KPE cache", dram_kpe_shape),
                (
                    "DSA request state",
                    lidu_slots_shape,
                ),
            ]:
                logger.info(f"{name} shape: {shape}")
        layer_shapes = (
            ckv_shape[1:],
            kpe_shape[1:],
            index_shape[1:],
            dram_ckv_shape[1:],
            dram_kpe_shape[1:],
        )
        index_share_groups = getattr(
            self.hf_config, "nanovllm_index_share_groups", None
        )
        # GLM-5.2 IndexShare: one cache_slots_pool per group, shared by all
        # member layers.  GLM-5.1 (no indexer_types) gives every layer its
        # own pool, preserving the original per-layer behaviour.
        group_lidu_slots: dict[int, torch.Tensor] = {}
        if index_share_groups is not None:
            for group in index_share_groups.groups():
                group_lidu_slots[group.group_id] = torch.full(
                    lidu_slots_shape,
                    -1,
                    dtype=torch.int32,
                    device=self.device,
                )
        for module in self.model.modules():
            if not hasattr(module, "assign_mla_cache"):
                continue
            if getattr(module, "uses_offload", False):
                ckv_cache = torch.empty(layer_shapes[0], dtype=cache_dtype, device=self.device)
                kpe_cache = torch.empty(layer_shapes[1], dtype=cache_dtype, device=self.device)
                # GLM-5.2 IndexShare: only owner (full) layers need an index
                # key cache.  Shared layers consume the owner's selection
                # metadata and never access index_cache at runtime.
                if getattr(module, "is_index_share_owner", True):
                    index_cache = torch.empty(layer_shapes[2], dtype=cache_dtype, device=self.device)
                else:
                    index_cache = torch.empty(0, dtype=cache_dtype, device=self.device)
                dram_ckv_cache = torch_npu.empty_with_swapped_memory(layer_shapes[3], dtype=cache_dtype, device=self.device)
                dram_kpe_cache = torch_npu.empty_with_swapped_memory(layer_shapes[4], dtype=cache_dtype, device=self.device)
                if index_share_groups is not None:
                    gid = index_share_groups.group_of(module.layer_idx)
                    lidu_cache_slots = group_lidu_slots[gid]
                else:
                    lidu_cache_slots = torch.full(
                        lidu_slots_shape,
                        -1,
                        dtype=torch.int32,
                        device=self.device,
                    )
                ckv_cache.zero_()
                kpe_cache.zero_()
                if index_cache.numel() > 0:
                    index_cache.zero_()
                module.assign_dsa_cache(
                    ckv_cache,
                    kpe_cache,
                    index_cache,
                    dram_ckv_cache,
                    dram_kpe_cache,
                    lidu_cache_slots,
                )
                continue

            # The recursively reused MTP layer keeps its complete history in a
            # separate dense HBM cache. Use the DRAM-source block capacity so
            # long/multi-request offload runs are not capped by the target
            # layers' deliberately small sparse HBM pool.
            mtp_blocks = config.num_dram_kvcache_blocks
            dense_ckv = torch.empty(
                (mtp_blocks, self.block_size, 1, kv_lora_rank),
                dtype=cache_dtype,
                device=self.device,
            )
            dense_kpe = torch.empty(
                (mtp_blocks, self.block_size, 1, rope_dim),
                dtype=cache_dtype,
                device=self.device,
            )
            dense_ckv.zero_()
            dense_kpe.zero_()
            module.assign_mla_cache(dense_ckv, dense_kpe)
            if getattr(module, "uses_mtp_index_share", False):
                mtp_index_cache = torch.zeros(
                    (
                        mtp_blocks,
                        self.block_size,
                        1,
                        index_dim,
                    ),
                    dtype=cache_dtype,
                    device=self.device,
                )
                module.assign_mtp_index_share_state(
                    mtp_index_cache,
                    torch.full(
                        lidu_slots_shape,
                        -1,
                        dtype=torch.int32,
                        device=self.device,
                    ),
                )
            if self.rank == 0:
                logger.info(
                    "MTP dense MLA cache: blocks=%d, CKV=%s, KPE=%s",
                    mtp_blocks,
                    tuple(dense_ckv.shape),
                    tuple(dense_kpe.shape),
                )

    def _setup_index_share_owners(self) -> None:
        """Wire shared layers to their owner's LIM output buffers.

        After cache allocation, each shared (non-owner) target layer gets
        a direct reference to its group's owner ``GlmMLAAttention``.
        This lets the shared layer read the owner's
        ``_fused_li_manage_outputs`` without duplicating the LIM call.
        """

        index_share_groups = getattr(
            self.hf_config, "nanovllm_index_share_groups", None
        )
        if index_share_groups is None:
            return
        target_layers = self.model.model.layers
        for group in index_share_groups.groups():
            owner_attn = target_layers[group.owner_layer_idx].self_attn
            for member_idx in group.member_layer_idxs:
                if member_idx == group.owner_layer_idx:
                    continue
                target_layers[member_idx].self_attn._index_share_owner = (
                    owner_attn
                )
        if self.rank == 0 and index_share_groups.num_groups != index_share_groups.num_hidden_layers:
            logger.info(
                "IndexShare: %d groups, %d owner layers, %d shared layers; "
                "shared layers reference their owner's LIM output.",
                index_share_groups.num_groups,
                len(index_share_groups.owner_layer_idxs),
                len(index_share_groups.shared_layer_idxs),
            )

    def prepare_block_tables(self, seqs: list[Sequence], table_name: str = "hbm_block_table"):
        static_max_block_cols = (self.config.max_model_len + self.config.kvcache_block_size - 1) // self.config.kvcache_block_size
        tables = [getattr(seq, table_name) for seq in seqs]
        max_len = max(len(table) for table in tables)
        num_cols = max(max_len, static_max_block_cols)
        return torch.tensor([table + [0] * (num_cols - len(table)) for table in tables], dtype=torch.int32, pin_memory=True).to(device=self.device, non_blocking=True)

    def _get_decode_dynamic_buffers(
        self,
        batch_size: int,
    ) -> _DecodeDynamicBuffers:
        buffers = self._decode_dynamic_buffers.get(batch_size)
        if buffers is None:
            buffers = _DecodeDynamicBuffers.allocate(batch_size, self.device)
            self._decode_dynamic_buffers[batch_size] = buffers
        return buffers

    def _get_decode_static_metadata(
        self,
        seqs: list[Sequence | DecodeSequenceMetadata],
        candidate_lens: list[int],
        req_pool_entries: list[int],
        lidu_cache_tokens: list[int],
        query_len: int = 1,
    ) -> _DecodeStaticMetadata:
        key = decode_metadata_key(seqs)
        cached = self._decode_static_metadata
        if (
            cached is not None
            and cached.key == key
            and cached.query_len == int(query_len)
        ):
            self._decode_metadata_cache_hits += 1
            return cached

        self._decode_metadata_cache_misses += 1
        temperatures = None
        if self.rank == 0:
            temperatures = torch.tensor(
                [seq.temperature for seq in seqs],
                dtype=torch.float32,
                pin_memory=True,
            ).to(self.device, non_blocking=True)
        cached = _DecodeStaticMetadata(
            key=key,
            query_len=int(query_len),
            block_tables=self.prepare_block_tables(seqs, "hbm_block_table"),
            index_block_tables=self.prepare_block_tables(
                seqs,
                "index_block_table",
            ),
            dram_block_tables=self.prepare_block_tables(
                seqs,
                "dram_block_table",
            ),
            req_pool_entries=torch.tensor(
                req_pool_entries,
                dtype=torch.int32,
                pin_memory=True,
            ).to(self.device, non_blocking=True),
            candidate_lens=torch.tensor(
                candidate_lens,
                dtype=torch.int32,
                pin_memory=True,
            ).to(self.device, non_blocking=True),
            candidate_query_lens=torch.arange(
                int(query_len),
                (len(seqs) + 1) * int(query_len),
                int(query_len),
                dtype=torch.int32,
                pin_memory=True,
            ).to(self.device, non_blocking=True),
            lidu_cache_tokens=torch.tensor(
                lidu_cache_tokens,
                dtype=torch.int32,
                pin_memory=True,
            ).to(self.device, non_blocking=True),
            temperatures=temperatures,
        )
        self._decode_static_metadata = cached
        return cached

    def _mtp_index_share_metadata(
        self,
        seqs: list[Sequence | DecodeSequenceMetadata],
    ) -> _MtpIndexShareMetadata | None:
        if not self.uses_mtp_index_share:
            return None
        if any(int(seq.mtp_index_pool_entry) < 0 for seq in seqs):
            raise RuntimeError("MTP IndexShare sequence is missing a pool row.")
        return _MtpIndexShareMetadata(
            block_tables=self.prepare_block_tables(seqs, "mtp_block_table"),
            req_pool_entries=torch.tensor(
                [seq.mtp_index_pool_entry for seq in seqs],
                dtype=torch.int32,
                device=self.device,
            ),
            candidate_lens=torch.tensor(
                [
                    seq.num_prefill_full_blocks * self.block_size
                    for seq in seqs
                ],
                dtype=torch.int32,
                device=self.device,
            ),
            lidu_cache_tokens=torch.tensor(
                [seq.mtp_lidu_cache_tokens for seq in seqs],
                dtype=torch.int32,
                device=self.device,
            ),
        )

    def _sequence_slots(self, block_table: list[int], seq_len: int) -> list[int]:
        slots: list[int] = []
        remaining = int(seq_len)
        for block_id in block_table:
            if remaining <= 0:
                break
            take = min(self.block_size, remaining)
            start = int(block_id) * self.block_size
            slots.extend(range(start, start + take))
            remaining -= take
        return slots

    def prepare_prefill(self, seqs: list[Sequence]):
        input_ids = []
        positions = []
        cu_seqlens_q = [0]
        flat_slot_mapping = []
        flat_index_slot_mapping = []
        needs_dsa_update = False
        actual_seq_lengths_kv = None
        is_last_chunk = True

        if self.config.prefill_chunk_size:
            if len(seqs) != 1:
                raise RuntimeError("Chunk prefill requires exactly one sequence.")
            (
                chunk_input_ids,
                chunk_positions,
                chunk_slot_mapping,
                chunk_index_slot_mapping,
                chunk_end,
                is_last_chunk,
            ) = seqs[0].scheduled_prefill_chunk()
            input_ids.extend(chunk_input_ids)
            positions.extend(chunk_positions)
            flat_slot_mapping.extend(chunk_slot_mapping)
            if self.uses_offload:
                flat_index_slot_mapping.extend(chunk_index_slot_mapping)
            cu_seqlens_q.append(len(chunk_input_ids))
            actual_seq_lengths_kv = [chunk_end]
            candidate_len = seqs[0].num_prefill_full_blocks * self.block_size
            needs_dsa_update = self.uses_offload and (
                candidate_len > seqs[0].num_sparse_tokens > 0
            )
        else:
            for seq in seqs:
                input_ids.extend(seq[:])
                seqlen = len(seq)
                positions.extend(range(seqlen))
                cu_seqlens_q.append(cu_seqlens_q[-1] + seqlen)
                if seq.hbm_block_table:
                    flat_slot_mapping.extend(
                        self._sequence_slots(seq.hbm_block_table, seqlen)
                    )
                    if self.uses_offload:
                        flat_index_slot_mapping.extend(
                            self._sequence_slots(
                                seq.index_block_table, seqlen
                            )
                        )
                candidate_len = seq.num_prefill_full_blocks * self.block_size
                needs_dsa_update = needs_dsa_update or (
                    self.uses_offload
                    and candidate_len > seq.num_sparse_tokens > 0
                )

        input_ids = torch.tensor(input_ids, dtype=torch.int64).to(self.device)
        positions = torch.tensor(positions, dtype=torch.int64).to(self.device)
        cu_seqlens_q = torch.tensor(cu_seqlens_q, dtype=torch.int32).to(self.device)
        flat_slot_mapping = torch.tensor(
            flat_slot_mapping, dtype=torch.int64
        ).to(self.device)
        flat_index_slot_mapping = torch.tensor(
            flat_index_slot_mapping, dtype=torch.int64
        ).to(self.device)

        set_context(
            True,
            cu_seqlens_q=cu_seqlens_q,
            flat_slot_mapping=flat_slot_mapping,
            flat_index_slot_mapping=flat_index_slot_mapping,
            actual_seq_lengths_kv=actual_seq_lengths_kv,
            block_tables=self.prepare_block_tables(seqs, "hbm_block_table"),
            needs_dsa_update=needs_dsa_update,
        )

        return input_ids, positions, is_last_chunk

    def _set_mtp_prefill_context(self, seqs: list[Sequence]) -> None:
        """Point the MTP layer at its independent dense cache during prefill."""

        if not (self.uses_separate_mtp_cache or self.uses_mtp_index_share):
            return
        target_context = get_context()
        if target_context.cu_seqlens_q is None:
            raise RuntimeError("MTP prefill is missing target query lengths.")
        slots: list[int] = []
        for seq in seqs:
            block_table = (
                seq.mtp_block_table
                if self.uses_separate_mtp_cache
                else seq.hbm_block_table
            )
            if not block_table:
                raise RuntimeError("MTP prefill has no dense MTP block table.")
            if self.config.prefill_chunk_size:
                start = seq.num_prefill_tokens_processed
                end = start + seq.num_scheduled_tokens
            else:
                start, end = 0, len(seq)
            slots.extend(
                block_table[position // self.block_size]
                * self.block_size
                + position % self.block_size
                for position in range(start, end)
            )
        flat_slots = torch.tensor(
            slots, dtype=torch.int64, device=self.device
        )
        set_context(
            True,
            cu_seqlens_q=target_context.cu_seqlens_q,
            flat_slot_mapping=flat_slots,
            flat_slot_mapping_i32=flat_slots.to(torch.int32),
            flat_index_slot_mapping=(
                flat_slots if self.uses_mtp_index_share else None
            ),
            actual_seq_lengths_kv=target_context.actual_seq_lengths_kv,
            block_tables=self.prepare_block_tables(
                seqs,
                (
                    "mtp_block_table"
                    if self.uses_separate_mtp_cache
                    else "hbm_block_table"
                ),
            ),
            mtp_index_cache_write=self.uses_mtp_index_share,
        )

    def prepare_decode(
        self,
        seqs: list[Sequence | DecodeSequenceMetadata],
    ):
        input_ids = []
        positions = []
        flat_slot_mapping = []
        candidate_lens = []
        sparse_kv_lens = []
        req_pool_entries = []
        lidu_cache_tokens = []
        lidu_init_rows = []
        needs_dsa_update = False
        has_first_decode = False
        for row, seq in enumerate(seqs):
            input_ids.append(seq.last_token)
            position = len(seq) - 1
            positions.append(position)
            decode_len = seq.num_decode_tokens_since_prefill
            has_first_decode = (
                has_first_decode or seq.is_first_decode_after_prefill
            )
            tail_decode_offset = position - seq.num_prefill_full_blocks * self.block_size
            hbm_logical_block = seq.num_sparse_blocks + tail_decode_offset // self.block_size
            hbm_offset = tail_decode_offset % self.block_size
            hbm_block_id = seq.hbm_block_table[hbm_logical_block]
            flat_slot_mapping.append(hbm_block_id * self.block_size + hbm_offset)
            candidate_len = seq.num_prefill_full_blocks * self.block_size
            sparse_selected_len = seq.num_sparse_tokens
            sparse_kv_len = sparse_selected_len + seq.prefill_tail_len + decode_len
            if self.uses_offload:
                cache_tokens = int(seq.lidu_cache_tokens)
                row_needs_offload = cache_tokens > 0
                if row_needs_offload and not seq.lidu_cache_initialized:
                    lidu_init_rows.append(row)
                # Keep one static decode graph for C=0/C>0 mixed batches.  The
                # LIDU kernel no-ops rows whose cache budget is zero.
                needs_dsa_update = needs_dsa_update or row_needs_offload
            else:
                cache_tokens = 0

            candidate_lens.append(candidate_len)
            sparse_kv_lens.append(sparse_kv_len)
            req_pool_entries.append(seq.offload_pool_entry)
            lidu_cache_tokens.append(cache_tokens)
        lidu_all_rows_ready = not lidu_init_rows
        use_persistent_decode_buffers = (
            self.decode_graph_manager is not None
            and not self.num_speculative_tokens
            and not has_first_decode
            and (
                (
                    needs_dsa_update
                    and lidu_all_rows_ready
                    and len(seqs) in self.decode_graph_manager.capture_sizes
                )
                if self.uses_offload
                else select_capture_size(
                    len(seqs), self.decode_graph_manager.capture_sizes
                )
                is not None
            )
        )
        if use_persistent_decode_buffers:
            # FULL_DECODE_ONLY synchronizes the current stream before replay,
            # so these pinned buffers are safe to reuse on the next step.
            dynamic_buffers = self._get_decode_dynamic_buffers(len(seqs))
            input_ids = dynamic_buffers.input_ids.stage(input_ids)
            positions = dynamic_buffers.positions.stage(positions)
            flat_slot_mapping_i32 = dynamic_buffers.slot_mapping_i32.stage(
                flat_slot_mapping
            )
            actual_seq_lengths_kv_tensor = (
                dynamic_buffers.actual_seq_lengths_kv.stage(sparse_kv_lens)
                if self.uses_sparse_tail_attention
                else None
            )
            flat_slot_mapping_i64 = None
        else:
            # Preserve the allocation-based eager path.  Worker ranks can
            # return before their stream finishes, so a single pinned staging
            # buffer must not be overwritten by the following eager step.
            input_ids = torch.tensor(
                input_ids,
                dtype=torch.int64,
                pin_memory=True,
            ).to(self.device, non_blocking=True)
            positions = torch.tensor(
                positions,
                dtype=torch.int64,
                pin_memory=True,
            ).to(self.device, non_blocking=True)
            flat_slot_mapping_i32 = torch.tensor(
                flat_slot_mapping,
                dtype=torch.int32,
                pin_memory=True,
            ).to(self.device, non_blocking=True)
            flat_slot_mapping_i64 = None
            actual_seq_lengths_kv_tensor = None
            if self.uses_sparse_tail_attention:
                actual_seq_lengths_kv_tensor = torch.tensor(
                    sparse_kv_lens,
                    dtype=torch.int32,
                    pin_memory=True,
                ).to(self.device, non_blocking=True)
            if has_first_decode:
                flat_slot_mapping_i64 = torch.tensor(
                    flat_slot_mapping,
                    dtype=torch.int64,
                    pin_memory=True,
                ).to(self.device, non_blocking=True)
        static_metadata = self._get_decode_static_metadata(
            seqs,
            candidate_lens,
            req_pool_entries,
            lidu_cache_tokens,
        )
        lidu_init_rows_tensor = None
        if lidu_init_rows:
            lidu_init_rows_tensor = torch.tensor(
                lidu_init_rows,
                dtype=torch.long,
                pin_memory=True,
            ).to(self.device, non_blocking=True)
        set_context(
            False,
            flat_slot_mapping=flat_slot_mapping_i64,
            flat_slot_mapping_i32=flat_slot_mapping_i32,
            actual_seq_lengths_kv=sparse_kv_lens,
            actual_seq_lengths_kv_tensor=actual_seq_lengths_kv_tensor,
            block_tables=static_metadata.block_tables,
            index_block_tables=static_metadata.index_block_tables,
            dram_block_tables=static_metadata.dram_block_tables,
            req_pool_entries=static_metadata.req_pool_entries,
            candidate_lens=static_metadata.candidate_lens,
            candidate_query_lens=static_metadata.candidate_query_lens,
            lidu_cache_tokens=static_metadata.lidu_cache_tokens,
            needs_dsa_update=needs_dsa_update,
            lidu_init_rows=lidu_init_rows_tensor,
            lidu_all_rows_ready=lidu_all_rows_ready,
            has_first_decode=has_first_decode,
            decode_metadata_key=static_metadata.key,
        )
        return input_ids, positions

    def _prepare_mtp_prefill_input_ids(
        self,
        seqs: list[Sequence],
        sampled_token_ids: torch.Tensor | None,
    ) -> torch.Tensor:
        sampled = (
            sampled_token_ids.detach().cpu().tolist()
            if sampled_token_ids is not None
            else None
        )
        shifted: list[int] = []
        for row, seq in enumerate(seqs):
            if self.config.prefill_chunk_size:
                start = seq.num_prefill_tokens_processed
                end = start + seq.num_scheduled_tokens
            else:
                start = 0
                end = len(seq)
            shifted.extend(
                shifted_mtp_prefill_tokens(
                    seq.token_ids,
                    start,
                    end,
                    sampled_token_id=(
                        None if sampled is None else int(sampled[row])
                    ),
                )
            )
        return torch.tensor(
            shifted, dtype=torch.int64, device=self.device
        )

    @staticmethod
    def _greedy_sample(logits: torch.Tensor) -> torch.Tensor:
        return logits.float().argmax(dim=-1)

    def _slots_from_positions(
        self,
        block_tables: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        block_indices = torch.div(
            positions, self.block_size, rounding_mode="floor"
        ).to(torch.long)
        block_ids = block_tables.gather(
            1, block_indices.unsqueeze(1)
        ).squeeze(1)
        return (
            block_ids.to(torch.long) * self.block_size
            + torch.remainder(positions, self.block_size)
        )

    def _get_mtp_decode_cu_seqlens(self, batch_size: int) -> torch.Tensor:
        cu_seqlens = self._mtp_decode_cu_seqlens.get(batch_size)
        if cu_seqlens is None:
            cu_seqlens = torch.arange(
                batch_size + 1,
                dtype=torch.int32,
                device=self.device,
            )
            self._mtp_decode_cu_seqlens[batch_size] = cu_seqlens
        return cu_seqlens

    def _set_mtp_decode_context(
        self,
        block_tables: torch.Tensor,
        positions: torch.Tensor,
        actual_seq_lengths_kv: list[int] | None = None,
        *,
        has_first_decode: bool = False,
        index_share: _MtpIndexShareMetadata | None = None,
        actual_seq_lengths_kv_tensor: torch.Tensor | None = None,
        dram_block_tables: torch.Tensor | None = None,
        lidu_init_rows: torch.Tensor | None = None,
        flat_slot_mapping: torch.Tensor | None = None,
    ) -> None:
        batch_size = int(positions.numel())
        slots = (
            flat_slot_mapping
            if flat_slot_mapping is not None
            else self._slots_from_positions(block_tables, positions)
        )
        cu_seqlens_q = self._get_mtp_decode_cu_seqlens(batch_size)
        if actual_seq_lengths_kv is None:
            actual_seq_lengths_kv = positions.add(1).detach().cpu().tolist()
        if len(actual_seq_lengths_kv) != batch_size:
            raise ValueError(
                "MTP recurrence KV-length batch changed: "
                f"expected={batch_size}, actual={len(actual_seq_lengths_kv)}."
            )
        if index_share is not None and actual_seq_lengths_kv_tensor is None:
            if int(index_share.block_tables.shape[0]) != batch_size:
                raise ValueError("MTP IndexShare metadata batch changed.")
            actual_seq_lengths_kv_tensor = torch.tensor(
                actual_seq_lengths_kv,
                dtype=torch.int32,
                device=self.device,
            )
        elif index_share is None:
            actual_seq_lengths_kv_tensor = None
        set_context(
            False,
            cu_seqlens_q=cu_seqlens_q,
            flat_slot_mapping=slots,
            flat_slot_mapping_i32=slots.to(torch.int32),
            actual_seq_lengths_kv=actual_seq_lengths_kv,
            actual_seq_lengths_kv_tensor=actual_seq_lengths_kv_tensor,
            block_tables=block_tables,
            index_block_tables=(
                index_share.block_tables if index_share is not None else None
            ),
            # MTP draft recurrence keeps source KV in its dense HBM cache.
            # Target verification supplies its target-layer DRAM table.
            dram_block_tables=(
                dram_block_tables
                if dram_block_tables is not None
                else (
                    index_share.block_tables
                    if index_share is not None
                    else None
                )
            ),
            req_pool_entries=(
                index_share.req_pool_entries if index_share is not None else None
            ),
            candidate_lens=(
                index_share.candidate_lens if index_share is not None else None
            ),
            candidate_query_lens=(
                cu_seqlens_q[1:] if index_share is not None else None
            ),
            lidu_cache_tokens=(
                index_share.lidu_cache_tokens
                if index_share is not None
                else None
            ),
            needs_dsa_update=index_share is not None,
            lidu_init_rows=lidu_init_rows,
            lidu_all_rows_ready=(
                index_share is not None and lidu_init_rows is None
            ),
            has_first_decode=has_first_decode,
            full_decode_graph=is_full_decode_graph_capturing(),
        )

    def _run_mtp_recurrence(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        previous_hidden_states: torch.Tensor,
        block_tables: torch.Tensor,
        steps: int,
        actual_seq_lengths_by_step: list[list[int]] | None = None,
        balanced_route_offset: int | None = None,
        index_share: _MtpIndexShareMetadata | None = None,
        actual_seq_lengths_tensors_by_step: list[torch.Tensor] | None = None,
    ) -> torch.Tensor:
        mtp = getattr(self.model, "mtp", None)
        if mtp is None:
            raise RuntimeError("MTP recurrence requested without an MTP model.")
        if (
            actual_seq_lengths_by_step is not None
            and len(actual_seq_lengths_by_step) != steps
        ):
            raise ValueError(
                "MTP recurrence requires one KV-length row per step: "
                f"steps={steps}, rows={len(actual_seq_lengths_by_step)}."
            )
        if (
            actual_seq_lengths_tensors_by_step is not None
            and len(actual_seq_lengths_tensors_by_step) != steps
        ):
            raise ValueError(
                "MTP recurrence requires one graph KV tensor per step."
            )
        drafts: list[torch.Tensor] = []
        try:
            for step in range(steps):
                seq_lengths = (
                    None
                    if actual_seq_lengths_by_step is None
                    else actual_seq_lengths_by_step[step]
                )
                mtp.set_skip_topk(index_share is not None and step > 0)
                self._set_mtp_decode_context(
                    block_tables,
                    positions,
                    actual_seq_lengths_kv=seq_lengths,
                    index_share=index_share,
                    actual_seq_lengths_kv_tensor=(
                        None
                        if actual_seq_lengths_tensors_by_step is None
                        else actual_seq_lengths_tensors_by_step[step]
                    ),
                )
                if balanced_route_offset is not None:
                    moe = mtp.mtp_block.mlp
                    routes_per_step = int(input_ids.shape[0]) * int(moe.top_k)
                    get_context().scratch[GLM_BALANCED_MOE_EXPERT_IDS_KEY] = (
                        balanced_moe_expert_ids(
                            rows=int(input_ids.shape[0]),
                            top_k=int(moe.top_k),
                            num_experts=int(moe.num_experts),
                            ep_size=int(moe.ep_size),
                            route_offset=(
                                int(balanced_route_offset)
                                + step * routes_per_step
                            ),
                            device=input_ids.device,
                            dtype=torch.int32,
                        )
                    )
                previous_hidden_states = mtp(
                    input_ids, positions, previous_hidden_states
                )
                input_ids = self._greedy_sample(
                    mtp.compute_logits(previous_hidden_states)
                )
                drafts.append(input_ids)
                positions = positions + 1
        finally:
            mtp.set_skip_topk(False)
        if not drafts:
            return torch.empty(
                (input_ids.shape[0], 0),
                dtype=torch.long,
                device=input_ids.device,
            )
        return torch.stack(drafts, dim=1)

    def _run_mtp_prefill(
        self,
        seqs: list[Sequence],
        positions: torch.Tensor,
        target_hidden_states: torch.Tensor,
        sampled_token_ids: torch.Tensor | None,
        is_last_chunk: bool,
    ) -> tuple[torch.Tensor, list[bool]] | None:
        mtp = getattr(self.model, "mtp", None)
        if mtp is None:
            raise RuntimeError("MTP prefill requested without an MTP model.")
        mtp_input_ids = self._prepare_mtp_prefill_input_ids(
            seqs, sampled_token_ids
        )
        self._set_mtp_prefill_context(seqs)
        mtp_hidden_states = mtp(
            mtp_input_ids, positions, target_hidden_states
        )
        if not is_last_chunk:
            return None

        mtp_index_share_metadata = self._mtp_index_share_metadata(seqs)
        if mtp_index_share_metadata is not None:
            mtp.mtp_block.self_attn.initialize_mtp_index_share_rows(
                [seq.mtp_index_pool_entry for seq in seqs],
                [seq.mtp_lidu_cache_tokens for seq in seqs],
            )
            for seq in seqs:
                seq.mtp_lidu_cache_initialized = True
        # Prefill is eager even when stable decode uses a graph, so it can
        # directly consume the just-built MTP IndexShare metadata.
        mtp_index_share = mtp_index_share_metadata

        draft_eligible = [
            len(seq) + 2 * self.num_speculative_tokens
            <= self.config.max_model_len
            for seq in seqs
        ]

        context = get_context()
        last_indices = context.cu_seqlens_q[1:] - 1
        last_hidden_states = mtp_hidden_states.index_select(
            0, last_indices.to(torch.long)
        )
        first_draft = self._greedy_sample(
            mtp.compute_logits(mtp_hidden_states)
        )
        if first_draft.shape != (len(seqs),):
            raise RuntimeError(
                "MTP prefill shared head must produce one draft per request, "
                f"got shape={tuple(first_draft.shape)} for batch={len(seqs)}."
            )
        drafts = torch.zeros(
            (len(seqs), self.num_speculative_tokens),
            dtype=torch.long,
            device=self.device,
        )
        drafts[:, 0] = first_draft
        eligible_rows = torch.tensor(
            [row for row, eligible in enumerate(draft_eligible) if eligible],
            dtype=torch.long,
            device=self.device,
        )
        if self.num_speculative_tokens > 1 and eligible_rows.numel():
            last_positions = positions.index_select(
                0, last_indices.to(torch.long)
            )
            remaining = self._run_mtp_recurrence(
                first_draft.index_select(0, eligible_rows),
                last_positions.index_select(0, eligible_rows) + 1,
                last_hidden_states.index_select(0, eligible_rows),
                (
                    mtp_index_share.block_tables.index_select(
                        0, eligible_rows
                    )
                    if mtp_index_share is not None
                    else context.block_tables.index_select(0, eligible_rows)
                ),
                self.num_speculative_tokens - 1,
                index_share=(
                    mtp_index_share.select(eligible_rows)
                    if mtp_index_share is not None
                    else None
                ),
            )
            eligible_drafts = torch.cat(
                (
                    first_draft.index_select(0, eligible_rows).unsqueeze(1),
                    remaining,
                ),
                dim=1,
            )
            drafts.index_copy_(0, eligible_rows, eligible_drafts)
        return drafts, draft_eligible

    def _prepare_mtp_verify(
        self,
        seqs: list[Sequence | DecodeSequenceMetadata],
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        k = self.num_speculative_tokens
        query_len = k + 1
        input_ids: list[int] = []
        positions: list[int] = []
        slots: list[int] = []
        cu_seqlens_q = [0]
        candidate_lens: list[int] = []
        sparse_kv_lens: list[int] = []
        req_pool_entries: list[int] = []
        cache_tokens_by_row: list[int] = []
        lidu_init_rows: list[int] = []
        needs_dsa_update = False
        has_first_decode = False
        for row, seq in enumerate(seqs):
            if len(seq.draft_token_ids) != k:
                raise RuntimeError(
                    "MTP verification requires exactly "
                    f"{k} draft tokens per request."
                )
            row_ids = [seq.last_token, *seq.draft_token_ids]
            row_positions = list(range(len(seq) - 1, len(seq) + k))
            input_ids.extend(row_ids)
            positions.extend(row_positions)
            if self.uses_offload:
                source_tokens = seq.num_prefill_full_blocks * self.block_size
                for position in row_positions:
                    tail_offset = position - source_tokens
                    logical_block = (
                        seq.num_sparse_blocks + tail_offset // self.block_size
                    )
                    block_id = seq.hbm_block_table[logical_block]
                    slots.append(
                        block_id * self.block_size
                        + tail_offset % self.block_size
                    )
            else:
                slots.extend(
                    seq.hbm_block_table[position // self.block_size]
                    * self.block_size
                    + position % self.block_size
                    for position in row_positions
                )
            cu_seqlens_q.append(cu_seqlens_q[-1] + query_len)

            candidate_len = seq.num_prefill_full_blocks * self.block_size
            cache_tokens = int(seq.lidu_cache_tokens) if self.uses_offload else 0
            if cache_tokens and not seq.lidu_cache_initialized:
                lidu_init_rows.append(row)
            needs_dsa_update = needs_dsa_update or cache_tokens > 0
            has_first_decode = (
                has_first_decode or seq.is_first_decode_after_prefill
            )
            candidate_lens.append(candidate_len)
            sparse_kv_lens.append(
                (
                    seq.num_sparse_tokens
                    + seq.prefill_tail_len
                    + seq.num_decode_tokens_since_prefill
                    + k
                )
                if self.uses_offload
                else len(seq) + k
            )
            req_pool_entries.append(seq.offload_pool_entry)
            cache_tokens_by_row.append(cache_tokens)

        input_ids_tensor = torch.tensor(
            input_ids, dtype=torch.int64, device=self.device
        )
        positions_tensor = torch.tensor(
            positions, dtype=torch.int64, device=self.device
        )
        slots_tensor = torch.tensor(
            slots, dtype=torch.int64, device=self.device
        )
        cu_seqlens_tensor = torch.tensor(
            cu_seqlens_q, dtype=torch.int32, device=self.device
        )
        static_metadata = self._get_decode_static_metadata(
            seqs,
            candidate_lens,
            req_pool_entries,
            cache_tokens_by_row,
            query_len=query_len,
        )
        lidu_init_rows_tensor = None
        if lidu_init_rows:
            lidu_init_rows_tensor = torch.tensor(
                lidu_init_rows, dtype=torch.long, device=self.device
            )
        actual_seq_lengths_kv_tensor = torch.tensor(
            sparse_kv_lens, dtype=torch.int32, device=self.device
        )
        set_context(
            False,
            is_spec_decode=True,
            cu_seqlens_q=cu_seqlens_tensor,
            actual_seq_lengths_q=cu_seqlens_q[1:],
            flat_slot_mapping=slots_tensor,
            flat_slot_mapping_i32=slots_tensor.to(torch.int32),
            actual_seq_lengths_kv=sparse_kv_lens,
            actual_seq_lengths_kv_tensor=actual_seq_lengths_kv_tensor,
            block_tables=static_metadata.block_tables,
            index_block_tables=static_metadata.index_block_tables,
            dram_block_tables=static_metadata.dram_block_tables,
            req_pool_entries=static_metadata.req_pool_entries,
            candidate_lens=static_metadata.candidate_lens,
            candidate_query_lens=static_metadata.candidate_query_lens,
            lidu_cache_tokens=static_metadata.lidu_cache_tokens,
            needs_dsa_update=needs_dsa_update,
            lidu_init_rows=lidu_init_rows_tensor,
            lidu_all_rows_ready=not lidu_init_rows,
            has_first_decode=has_first_decode,
            decode_metadata_key=static_metadata.key,
        )
        drafts = torch.tensor(
            [seq.draft_token_ids for seq in seqs],
            dtype=torch.long,
            device=self.device,
        )
        mtp_block_tables = (
            self.prepare_block_tables(seqs, "mtp_block_table")
            if self.uses_separate_mtp_cache
            else static_metadata.block_tables
        )
        return (
            input_ids_tensor,
            positions_tensor,
            drafts,
            static_metadata.block_tables,
            mtp_block_tables,
        )

    def _mtp_target_forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        draft_token_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        batch_size = int(draft_token_ids.shape[0])
        k = int(draft_token_ids.shape[1])
        if k != self.num_speculative_tokens:
            raise RuntimeError(
                "MTP target draft width changed: "
                f"expected={self.num_speculative_tokens}, actual={k}."
            )
        target_hidden_states = self.model(input_ids, positions)
        target_tokens = self._greedy_sample(
            self.model.compute_logits(target_hidden_states)
        ).view(batch_size, k + 1)
        accepted_counts, next_token_ids = greedy_prefix_accept(
            target_tokens, draft_token_ids
        )
        rows = torch.arange(
            batch_size, dtype=torch.long, device=input_ids.device
        )
        hidden_by_request = target_hidden_states.view(
            batch_size, k + 1, -1
        )
        selected_hidden_states = hidden_by_request[rows, accepted_counts]
        base_positions = positions.view(batch_size, k + 1)[:, 0]
        selected_positions = base_positions + accepted_counts
        return (
            target_tokens,
            accepted_counts,
            next_token_ids,
            selected_hidden_states,
            selected_positions,
        )

    def _mtp_target_forward_serial(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        draft_token_ids: torch.Tensor,
        block_tables: torch.Tensor,
        *,
        has_first_decode: bool,
    ) -> tuple[torch.Tensor, ...]:
        """Verify GLM-5.2 MTP drafts through the ordinary target decode path.

        The TND K+1 verification kernel is faster, but GLM-5.2's first
        bring-up must preserve exact K=0 greedy semantics across partial
        draft rejection.  One-token target forwards provide the same cache
        and causal-attention contract as ordinary decode.  The MTP layer still
        produces all drafts; only target verification is serialized.
        """

        batch_size, k = draft_token_ids.shape
        if k != self.num_speculative_tokens:
            raise RuntimeError(
                "MTP target draft width changed: "
                f"expected={self.num_speculative_tokens}, actual={k}."
            )
        query_len = k + 1
        if input_ids.numel() != batch_size * query_len:
            raise ValueError(
                "Serialized MTP target input shape does not match draft "
                f"batch: tokens={input_ids.numel()}, batch={batch_size}, "
                f"query_len={query_len}."
            )
        if positions.shape != input_ids.shape:
            raise ValueError(
                "Serialized MTP target positions must match input_ids: "
                f"positions={tuple(positions.shape)}, "
                f"input_ids={tuple(input_ids.shape)}."
            )

        token_rows = input_ids.view(batch_size, query_len)
        position_rows = positions.view(batch_size, query_len)
        capture_base_seq_lengths: list[int] | None = None
        target_index_share: _MtpIndexShareMetadata | None = None
        target_dram_block_tables: torch.Tensor | None = None
        target_lidu_init_rows: torch.Tensor | None = None
        target_final_seq_lengths: list[int] | None = None
        target_slot_rows: torch.Tensor | None = None
        if self.uses_offload:
            target_context = get_context()
            required = {
                "index_block_tables": target_context.index_block_tables,
                "dram_block_tables": target_context.dram_block_tables,
                "req_pool_entries": target_context.req_pool_entries,
                "candidate_lens": target_context.candidate_lens,
                "lidu_cache_tokens": target_context.lidu_cache_tokens,
                "actual_seq_lengths_kv": target_context.actual_seq_lengths_kv,
                "flat_slot_mapping": target_context.flat_slot_mapping,
            }
            missing = [
                name for name, value in required.items() if value is None
            ]
            if missing:
                raise RuntimeError(
                    "GLM-5.2 MTP offload target is missing: "
                    + ", ".join(missing)
                )
            target_index_share = _MtpIndexShareMetadata(
                block_tables=target_context.index_block_tables,
                req_pool_entries=target_context.req_pool_entries,
                candidate_lens=target_context.candidate_lens,
                lidu_cache_tokens=target_context.lidu_cache_tokens,
            )
            target_dram_block_tables = target_context.dram_block_tables
            target_lidu_init_rows = target_context.lidu_init_rows
            target_final_seq_lengths = [
                int(length) for length in target_context.actual_seq_lengths_kv
            ]
            target_slot_rows = target_context.flat_slot_mapping.view(
                batch_size, query_len
            )
        if is_full_decode_graph_capturing():
            target_seq_lengths = (
                target_final_seq_lengths
                if target_final_seq_lengths is not None
                else get_context().actual_seq_lengths_kv
            )
            if target_seq_lengths is None or len(target_seq_lengths) != batch_size:
                raise RuntimeError(
                    "GLM-5.2 MTP graph target is missing captured KV "
                    "lengths."
                )
            capture_base_seq_lengths = [
                int(length) - self.num_speculative_tokens
                for length in target_seq_lengths
            ]
        target_tokens: list[torch.Tensor] = []
        target_hidden_states: list[torch.Tensor] = []
        for step in range(query_len):
            step_positions = position_rows[:, step]
            if capture_base_seq_lengths is None:
                if target_final_seq_lengths is None:
                    actual_seq_lengths_kv = (
                        step_positions.add(1).cpu().tolist()
                    )
                else:
                    actual_seq_lengths_kv = [
                        length - self.num_speculative_tokens + step
                        for length in target_final_seq_lengths
                    ]
            else:
                actual_seq_lengths_kv = [
                    length + step for length in capture_base_seq_lengths
                ]
            self._set_mtp_decode_context(
                block_tables,
                step_positions,
                actual_seq_lengths_kv=actual_seq_lengths_kv,
                has_first_decode=has_first_decode and step == 0,
                index_share=target_index_share,
                dram_block_tables=target_dram_block_tables,
                lidu_init_rows=(
                    target_lidu_init_rows if step == 0 else None
                ),
                flat_slot_mapping=(
                    None
                    if target_slot_rows is None
                    else target_slot_rows[:, step]
                ),
            )
            hidden_states = self.model(
                token_rows[:, step], step_positions
            )
            target_hidden_states.append(hidden_states)
            target_tokens.append(
                self._greedy_sample(
                    self.model.compute_logits(hidden_states)
                )
            )

        targets = torch.stack(target_tokens, dim=1)
        accepted_counts, next_token_ids = greedy_prefix_accept(
            targets, draft_token_ids
        )
        hidden_by_request = torch.stack(target_hidden_states, dim=1)
        rows = torch.arange(
            batch_size, dtype=torch.long, device=input_ids.device
        )
        selected_hidden_states = hidden_by_request[rows, accepted_counts]
        selected_positions = (
            position_rows[:, 0] + accepted_counts
        )
        return (
            targets,
            accepted_counts,
            next_token_ids,
            selected_hidden_states,
            selected_positions,
        )

    def _mtp_target_graph_forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        draft_token_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        """Select the graph-safe target verification implementation."""

        if self.config.glm_version != "5.2":
            return self._mtp_target_forward(
                input_ids, positions, draft_token_ids
            )
        context = get_context()
        if context.block_tables is None:
            raise RuntimeError(
                "GLM-5.2 MTP graph target is missing block tables."
            )
        return self._mtp_target_forward_serial(
            input_ids,
            positions,
            draft_token_ids,
            context.block_tables,
            has_first_decode=False,
        )

    def _mtp_draft_graph_forward(
        self,
        next_token_ids: torch.Tensor,
        selected_positions: torch.Tensor,
        selected_hidden_states: torch.Tensor,
        block_tables: torch.Tensor,
        actual_seq_lengths_by_step: list[list[int]],
        graph_index_share: Any | None = None,
    ) -> torch.Tensor:
        index_share = None
        actual_seq_lengths_tensors_by_step = None
        if graph_index_share is not None:
            index_share = _MtpIndexShareMetadata(
                block_tables=graph_index_share.mtp_index_block_tables,
                req_pool_entries=graph_index_share.mtp_req_pool_entries,
                candidate_lens=graph_index_share.mtp_candidate_lens,
                lidu_cache_tokens=graph_index_share.mtp_lidu_cache_tokens,
            )
            actual_seq_lengths_tensors_by_step = (
                graph_index_share.mtp_actual_seq_lengths_by_step
            )
        return self._run_mtp_recurrence(
            next_token_ids,
            selected_positions,
            selected_hidden_states,
            block_tables,
            self.num_speculative_tokens,
            actual_seq_lengths_by_step=actual_seq_lengths_by_step,
            index_share=index_share,
            actual_seq_lengths_tensors_by_step=(
                actual_seq_lengths_tensors_by_step
            ),
        )

    def _mtp_draft_graph_eager_warmup(
        self,
        next_token_ids: torch.Tensor,
        selected_positions: torch.Tensor,
        selected_hidden_states: torch.Tensor,
        block_tables: torch.Tensor,
        actual_seq_lengths_by_step: list[list[int]],
        graph_index_share: Any | None = None,
    ) -> int:
        mtp = getattr(self.model, "mtp", None)
        if mtp is None:
            raise RuntimeError("MTP graph warmup requested without MTP weights.")
        moe = mtp.mtp_block.mlp
        routes_per_pass = int(next_token_ids.shape[0]) * int(moe.top_k)
        warmup_passes = max(
            1,
            (int(moe.ep_size) + routes_per_pass - 1) // routes_per_pass,
        )
        for pass_index in range(warmup_passes):
            self._run_mtp_recurrence(
                next_token_ids,
                selected_positions,
                selected_hidden_states,
                block_tables,
                self.num_speculative_tokens,
                actual_seq_lengths_by_step=actual_seq_lengths_by_step,
                balanced_route_offset=pass_index * routes_per_pass,
                index_share=(
                    None
                    if graph_index_share is None
                    else _MtpIndexShareMetadata(
                        block_tables=graph_index_share.mtp_index_block_tables,
                        req_pool_entries=(
                            graph_index_share.mtp_req_pool_entries
                        ),
                        candidate_lens=(
                            graph_index_share.mtp_candidate_lens
                        ),
                        lidu_cache_tokens=(
                            graph_index_share.mtp_lidu_cache_tokens
                        ),
                    )
                ),
                actual_seq_lengths_tensors_by_step=(
                    None
                    if graph_index_share is None
                    else graph_index_share.mtp_actual_seq_lengths_by_step
                ),
            )
            torch.npu.synchronize()
        reset_context()
        return warmup_passes

    @staticmethod
    def _mtp_draft_seq_lengths(
        base_seq_lengths: list[int],
        accepted_counts: list[int],
        steps: int,
    ) -> list[list[int]]:
        return [
            [
                int(length) + int(accepted) + step
                for length, accepted in zip(
                    base_seq_lengths, accepted_counts
                )
            ]
            for step in range(steps)
        ]

    def _run_mtp_verify(
        self,
        seqs: list[Sequence | DecodeSequenceMetadata],
    ) -> SpeculativeStepOutput | None:
        (
            input_ids,
            positions,
            draft_token_ids,
            block_tables,
            mtp_block_tables,
        ) = self._prepare_mtp_verify(seqs)
        init_rows = get_context().lidu_init_rows
        init_rows_host = (
            []
            if init_rows is None
            else [int(row) for row in init_rows.detach().cpu().tolist()]
        )
        base_seq_lengths = [len(seq) for seq in seqs]
        mtp_index_share = self._mtp_index_share_metadata(seqs)
        graph_manager = self.decode_graph_manager
        if (
            isinstance(graph_manager, MTPDecodeOnlyGraphManager)
            and graph_manager.should_use_graph(len(seqs), get_context())
        ):
            target_tokens, accepted_counts, next_drafts = graph_manager.run(
                input_ids,
                positions,
                draft_token_ids,
                get_context(),
                base_seq_lengths,
                mtp_block_tables,
                mtp_index_share,
            )
        else:
            if self.config.glm_version == "5.2":
                (
                    target_tokens,
                    accepted_counts,
                    next_token_ids,
                    selected_hidden_states,
                    selected_positions,
                ) = self._mtp_target_forward_serial(
                    input_ids,
                    positions,
                    draft_token_ids,
                    block_tables,
                    has_first_decode=get_context().has_first_decode,
                )
            else:
                (
                    target_tokens,
                    accepted_counts,
                    next_token_ids,
                    selected_hidden_states,
                    selected_positions,
                ) = self._mtp_target_forward(
                    input_ids, positions, draft_token_ids
                )
            accepted_counts_host = accepted_counts.cpu().tolist()
            next_drafts = self._run_mtp_recurrence(
                next_token_ids,
                selected_positions,
                selected_hidden_states,
                mtp_block_tables,
                self.num_speculative_tokens,
                actual_seq_lengths_by_step=self._mtp_draft_seq_lengths(
                    base_seq_lengths,
                    accepted_counts_host,
                    self.num_speculative_tokens,
                ),
                index_share=mtp_index_share,
            )

        if init_rows_host:
            # Every target layer has now completed cache initialization. The
            # scheduler-visible state is published only after device work is
            # complete; subsequent steps can use the stable MTP-LIDU path.
            torch.npu.synchronize()
            for row in init_rows_host:
                seq = seqs[row]
                if isinstance(seq, Sequence):
                    seq.lidu_cache_initialized = True
                    seq.bump_decode_metadata_version()

        if self.rank != 0:
            return None
        accepted_counts_cpu = accepted_counts.cpu().tolist()
        target_tokens_cpu = target_tokens.cpu().tolist()
        next_drafts_cpu = next_drafts.cpu().tolist()
        accepted_token_ids = materialize_accepted_tokens(
            [list(seq.draft_token_ids) for seq in seqs],
            target_tokens_cpu,
            accepted_counts_cpu,
        )
        return SpeculativeStepOutput(
            token_ids=accepted_token_ids,
            draft_token_ids=next_drafts_cpu,
            accepted_draft_counts=[int(value) for value in accepted_counts_cpu],
        )

    @torch.inference_mode()
    def finalize_prefill_offload(self, seqs: list[Sequence]) -> None:
        if not self.uses_offload:
            return
        for seq in seqs:
            if seq.offload_finalized:
                continue
            old_hbm_block_table = list(seq.hbm_block_table)
            for module in self.model.modules():
                if not getattr(module, "uses_offload", False):
                    continue
                finalize = getattr(module, "finalize_prefill_offload", None)
                if finalize is not None:
                    finalize(seq, old_hbm_block_table)

            finalize_prefill_hbm_layout(seq, self.offload_mode)

    @torch.inference_mode()
    def run(
        self,
        seqs: list[Sequence | DecodeSequenceMetadata],
        is_prefill: bool,
    ) -> list[int] | SpeculativeStepOutput | None:
        try:
            if (
                self.num_speculative_tokens
                and not is_prefill
                and all(
                    len(seq.draft_token_ids)
                    == self.num_speculative_tokens
                    for seq in seqs
                )
            ):
                return self._run_mtp_verify(seqs)

            if is_prefill:
                input_ids, positions, should_sample = self.prepare_prefill(seqs)
            else:
                input_ids, positions = self.prepare_decode(seqs)
                should_sample = True
            temperatures = None
            if should_sample and self.rank == 0 and is_prefill:
                temperatures = torch.tensor(
                    [seq.temperature for seq in seqs],
                    dtype=torch.float32,
                    pin_memory=True,
                ).to(self.device, non_blocking=True)
            elif should_sample and self.rank == 0:
                static_metadata = self._decode_static_metadata
                if static_metadata is None:
                    raise RuntimeError("Decode sampling metadata was not prepared.")
                temperatures = static_metadata.temperatures
            if (
                is_prefill
                or self.decode_graph_manager is None
                or self.num_speculative_tokens
            ):
                hidden_states = self.model(input_ids, positions)
            else:
                hidden_states = self.decode_graph_manager.run(input_ids, positions)
            if self.num_speculative_tokens and is_prefill:
                sampled_token_ids = None
                if should_sample:
                    sampled_token_ids = self._greedy_sample(
                        self.model.compute_logits(hidden_states)
                    )
                draft_result = self._run_mtp_prefill(
                    seqs,
                    positions,
                    hidden_states,
                    sampled_token_ids,
                    should_sample,
                )
                if not should_sample:
                    return None
                self.finalize_prefill_offload(seqs)
                if self.rank != 0:
                    return None
                if draft_result is None:
                    raise RuntimeError(
                        "Final MTP prefill did not produce draft state."
                    )
                drafts, draft_eligible = draft_result
                draft_rows = drafts.cpu().tolist()
                return SpeculativeStepOutput(
                    token_ids=[
                        [int(token_id)]
                        for token_id in sampled_token_ids.cpu().tolist()
                    ],
                    draft_token_ids=[
                        row if eligible else []
                        for row, eligible in zip(
                            draft_rows, draft_eligible
                        )
                    ],
                    accepted_draft_counts=[0] * len(seqs),
                )
            if not is_prefill and self.uses_offload:
                init_rows = get_context().lidu_init_rows
                if init_rows is not None and init_rows.numel() > 0:
                    # Initialization is deliberately outside the stable graph.
                    # Finish every layer before publishing the request state to
                    # subsequent decode metadata.
                    torch.npu.synchronize()
                    for row in init_rows.detach().cpu().tolist():
                        seq = seqs[int(row)]
                        if isinstance(seq, Sequence):
                            seq.lidu_cache_initialized = True
                            seq.bump_decode_metadata_version()
            if not should_sample:
                return None
            logits = self.model.compute_logits(hidden_states)
            if is_prefill:
                self.finalize_prefill_offload(seqs)
            return (
                self.sampler(logits, temperatures).tolist()
                if self.rank == 0
                else None
            )
        finally:
            reset_context()
