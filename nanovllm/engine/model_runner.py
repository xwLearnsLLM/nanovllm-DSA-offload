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
    DSA_SELECTION_TOPK_TOKENS,
    OFFLOAD_GS,
    OFFLOAD_LIDU,
    OFFLOAD_NONE,
    max_lidu_cache_tokens,
)
from nanovllm.engine.full_decode_graph import (
    FullDecodeOnlyGraphManager,
    select_capture_size,
)
from nanovllm.engine.sequence import (
    DecodeBatchDelta,
    DecodeBatchPacket,
    DecodeBatchSnapshot,
    DecodeMetadataKey,
    DecodeSequenceMetadata,
    Sequence,
    apply_decode_batch_packet,
    build_decode_batch_packet,
    decode_metadata_key,
)
from nanovllm.layers.sampler import Sampler
from nanovllm.models.deepseek_v32 import DeepseekV32ForCausalLM
from nanovllm.models.glm_moe_dsa import GlmMoeDsaForCausalLM
from nanovllm.utils.context import get_context, set_context, reset_context
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
        )


@dataclass
class _DecodeStaticMetadata:
    key: tuple[tuple[int, int], ...]
    block_tables: torch.Tensor
    index_block_tables: torch.Tensor
    dram_block_tables: torch.Tensor
    selection_block_tables: torch.Tensor
    req_pool_entries: torch.Tensor
    candidate_lens: torch.Tensor
    candidate_query_lens: torch.Tensor
    lidu_cache_tokens: torch.Tensor
    temperatures: torch.Tensor | None


class ModelRunner:

    def __init__(self, config: Config, rank: int, event: Event | list[Event]):
        self.config = config
        self.hf_config = config.hf_config
        self.block_size = config.kvcache_block_size
        self.offload_mode = config.offload_mode
        self.uses_offload = self.offload_mode != OFFLOAD_NONE
        self.world_size = config.tensor_parallel_size
        self.rank = rank
        self.event = event
        self.device = config.device

        torch.npu.set_device(rank)
        dist.init_process_group("hccl", f"tcp://localhost:{config.hccl_port}", world_size=self.world_size, rank=rank)
        default_dtype = torch.get_default_dtype()

        torch_dtype = self._set_torch_dtype(self.hf_config)
        torch.set_default_dtype(torch_dtype)
        torch.set_default_device(self.device)

        self.model = self._load_default_strategy()

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
        torch.npu.empty_cache()
        self._allocate_mla_cache()
        self.decode_graph_manager = None
        if not config.enforce_eager:
            text_config = getattr(config.hf_config, "text_config", config.hf_config)
            self.decode_graph_manager = FullDecodeOnlyGraphManager(
                self.model,
                capture_sizes=config.decode_graph_capture_sizes,
                max_model_len=config.max_model_len,
                block_size=config.kvcache_block_size,
                device=self.device,
                expected_mla_tasks=int(text_config.num_hidden_layers),
                offload_mode=self.offload_mode,
                # GLM W4A8 + EP16 is captured as one raw outer ACLGraph.
                # TorchAir's optional FX lowering is left enabled for
                # DeepSeek, where it is already validated and faster.
                enable_npugraph_ex=not isinstance(
                    self.model, GlmMoeDsaForCausalLM
                ),
                log_enabled=self.rank == 0,
            )
            if self.offload_mode == OFFLOAD_LIDU:
                if self.rank == 0:
                    logger.info(
                        "FULL_DECODE_ONLY: deferring LIDU graph capture until "
                        "the first initialized stable decode batch."
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

    def _load_default_strategy(self):
        arch = (getattr(self.hf_config, "architectures", None) or [""])[0]
        model_type = getattr(self.hf_config, "model_type", "")
        if model_type == "glm_moe_dsa" or arch == "GlmMoeDsaForCausalLM":
            model = GlmMoeDsaForCausalLM(self.hf_config)
            if self.rank == 0:
                quant_metadata = getattr(
                    self.hf_config, "nanovllm_quant_metadata", {}
                )
                logger.info(
                    "GLM-5.1 W4A8: %s decode, attention=%s, "
                    "max_model_len=%d, "
                    "EP%d (%d local experts/rank), "
                    "ModelSlim version=%s group_size=%s; MTP is disabled.",
                    (
                        "eager"
                        if self.config.enforce_eager
                        else "FULL_DECODE_ONLY raw ACLGraph"
                    ),
                    (
                        f"{self.offload_mode} decode offload (topk=2048)"
                        if self.uses_offload
                        else "dense MLA (all KV)"
                    ),
                    self.config.max_model_len,
                    self.world_size,
                    int(self.hf_config.n_routed_experts) // self.world_size,
                    quant_metadata.get("version"),
                    quant_metadata.get("group_size"),
                )
        elif arch in (
            "DeepseekV32ForCausalLM",
            "DeepseekV3ForCausalLM",
            "",
        ):
            model = DeepseekV32ForCausalLM(self.hf_config)
        else:
            raise ValueError(
                f"Unsupported architecture {arch!r}; expected DeepSeek-V3.2 "
                "or GlmMoeDsaForCausalLM."
            )
        load_model(
            model,
            self.config.model,
            name_mapping=getattr(model, "weight_name_mapping", None),
        )
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

    @staticmethod
    def _set_torch_dtype(hf_config):
        torch_dtype = getattr(hf_config, "torch_dtype", None)
        if torch_dtype is None:
            torch_dtype = getattr(hf_config, "dtype", None)
        if torch_dtype is None and hasattr(hf_config, "text_config"):
            torch_dtype = getattr(hf_config.text_config, "torch_dtype", None)
            if torch_dtype is None:
                torch_dtype = getattr(hf_config.text_config, "dtype", None)
        if isinstance(torch_dtype, str):
            resolved_dtype = getattr(torch, torch_dtype, None)
            if resolved_dtype is None:
                alias_map = {
                    "bf16": torch.bfloat16,
                    "bfloat16": torch.bfloat16,
                    "fp16": torch.float16,
                    "float16": torch.float16,
                    "fp32": torch.float32,
                    "float32": torch.float32,
                }
                resolved_dtype = alias_map.get(torch_dtype.lower())
            torch_dtype = resolved_dtype
        if torch_dtype is None:
            torch_dtype = torch.float16
        return torch_dtype

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
        cache_dtype = self._set_torch_dtype(text_config)
        num_layers = int(text_config.num_hidden_layers)
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
            return

        index_dim = int(text_config.index_head_dim)
        if (
            self.offload_mode == OFFLOAD_GS
            and DSA_SELECTION_TOPK_TOKENS % self.block_size != 0
        ):
            raise ValueError(
                "DSA gather_selection path expects 2048 to be divisible by "
                f"kvcache block_size, got block_size={self.block_size}."
            )
        if self.rank == 0:
            logger.info(
                "Using explicit DSA cache blocks: hbm=%d, dram=%d, "
                "index=%d, max_sparse_tokens=%d",
                config.num_hbm_kvcache_blocks,
                config.num_dram_kvcache_blocks,
                config.num_dram_kvcache_blocks,
                (
                    DSA_SELECTION_TOPK_TOKENS
                    if self.offload_mode == OFFLOAD_GS
                    else max_lidu_cache_tokens(config.max_model_len)
                ),
            )

        index_shape = (num_layers, config.num_dram_kvcache_blocks, self.block_size, 1, index_dim)
        dram_ckv_shape = (num_layers, config.num_dram_kvcache_blocks, self.block_size, 1, kv_lora_rank)
        dram_kpe_shape = (num_layers, config.num_dram_kvcache_blocks, self.block_size, 1, rope_dim)
        gather_status_shape = None
        lidu_slots_shape = None
        if self.offload_mode == OFFLOAD_GS:
            gather_status_shape = (
                config.max_num_decode_seqs_per_step,
                1,
                1,
                DSA_SELECTION_TOPK_TOKENS + 1,
            )
        else:
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
                    gather_status_shape or lidu_slots_shape,
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
        for module in self.model.modules():
            if hasattr(module, "assign_dsa_cache"):
                ckv_cache = torch.empty(layer_shapes[0], dtype=cache_dtype, device=self.device)
                kpe_cache = torch.empty(layer_shapes[1], dtype=cache_dtype, device=self.device)
                index_cache = torch.empty(layer_shapes[2], dtype=cache_dtype, device=self.device)
                dram_ckv_cache = torch_npu.empty_with_swapped_memory(layer_shapes[3], dtype=cache_dtype, device=self.device)
                dram_kpe_cache = torch_npu.empty_with_swapped_memory(layer_shapes[4], dtype=cache_dtype, device=self.device)
                gather_status = None
                lidu_cache_slots = None
                if self.offload_mode == OFFLOAD_GS:
                    gather_status = torch.full(
                        gather_status_shape,
                        -1,
                        dtype=torch.int32,
                        device=self.device,
                    )
                else:
                    lidu_cache_slots = torch.full(
                        lidu_slots_shape,
                        -1,
                        dtype=torch.int32,
                        device=self.device,
                    )
                ckv_cache.zero_()
                kpe_cache.zero_()
                index_cache.zero_()
                module.assign_dsa_cache(
                    ckv_cache,
                    kpe_cache,
                    index_cache,
                    dram_ckv_cache,
                    dram_kpe_cache,
                    gather_status,
                    lidu_cache_slots,
                )

    def prepare_block_tables(self, seqs: list[Sequence], table_name: str = "hbm_block_table"):
        static_max_block_cols = (self.config.max_model_len + self.config.kvcache_block_size - 1) // self.config.kvcache_block_size
        tables = [getattr(seq, table_name) for seq in seqs]
        max_len = max(len(table) for table in tables)
        num_cols = max(max_len, static_max_block_cols)
        return torch.tensor([table + [0] * (num_cols - len(table)) for table in tables], dtype=torch.int32, pin_memory=True).to(device=self.device, non_blocking=True)

    def prepare_selection_block_tables(self, seqs: list[Sequence]) -> torch.Tensor:
        max_sparse_tokens = (
            DSA_SELECTION_TOPK_TOKENS
            if self.offload_mode == OFFLOAD_GS
            else max_lidu_cache_tokens(self.config.max_model_len)
        )
        max_sparse_blocks = max(1, max_sparse_tokens // self.block_size)
        tables = []
        for seq in seqs:
            sparse_blocks = min(int(seq.num_sparse_blocks), max_sparse_blocks)
            table = list(seq.hbm_block_table[:sparse_blocks])
            tables.append(table + [0] * (max_sparse_blocks - len(table)))
        return torch.tensor(tables, dtype=torch.int32, pin_memory=True).to(self.device, non_blocking=True)

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
    ) -> _DecodeStaticMetadata:
        key = decode_metadata_key(seqs)
        cached = self._decode_static_metadata
        if cached is not None and cached.key == key:
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
            block_tables=self.prepare_block_tables(seqs, "hbm_block_table"),
            index_block_tables=self.prepare_block_tables(
                seqs,
                "index_block_table",
            ),
            dram_block_tables=self.prepare_block_tables(
                seqs,
                "dram_block_table",
            ),
            selection_block_tables=self.prepare_selection_block_tables(seqs),
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
                1,
                len(seqs) + 1,
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
        offload_rows = []
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
            if self.offload_mode == OFFLOAD_LIDU:
                cache_tokens = int(seq.lidu_cache_tokens)
                row_needs_offload = cache_tokens > 0
                if row_needs_offload and not seq.lidu_cache_initialized:
                    lidu_init_rows.append(row)
                # Keep one static decode graph for C=0/C>0 mixed batches.  The
                # LIDU kernel no-ops rows whose cache budget is zero.
                needs_dsa_update = needs_dsa_update or row_needs_offload
            else:
                cache_tokens = 0
                row_needs_offload = (
                    self.offload_mode == OFFLOAD_GS
                    and candidate_len > sparse_selected_len > 0
                )
                needs_dsa_update = needs_dsa_update or row_needs_offload
            if row_needs_offload:
                offload_rows.append(row)

            candidate_lens.append(candidate_len)
            sparse_kv_lens.append(sparse_kv_len)
            req_pool_entries.append(seq.offload_pool_entry)
            lidu_cache_tokens.append(cache_tokens)
        dsa_offload_all_rows = (
            (self.offload_mode == OFFLOAD_LIDU and needs_dsa_update)
            or bool(offload_rows and len(offload_rows) == len(seqs))
        )
        lidu_all_rows_ready = not lidu_init_rows
        use_persistent_decode_buffers = (
            self.decode_graph_manager is not None
            and not has_first_decode
            and (
                (
                    needs_dsa_update
                    and lidu_all_rows_ready
                    and len(seqs) in self.decode_graph_manager.capture_sizes
                )
                if self.offload_mode == OFFLOAD_LIDU
                else (
                    dsa_offload_all_rows
                    and len(seqs) in self.decode_graph_manager.capture_sizes
                )
                if self.offload_mode == OFFLOAD_GS
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
        dsa_offload_rows = None
        if needs_dsa_update and not dsa_offload_all_rows:
            dsa_offload_rows = torch.tensor(offload_rows, dtype=torch.long, pin_memory=True).to(self.device, non_blocking=True)
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
            block_tables=static_metadata.block_tables,
            index_block_tables=static_metadata.index_block_tables,
            dram_block_tables=static_metadata.dram_block_tables,
            selection_block_tables=static_metadata.selection_block_tables,
            req_pool_entries=static_metadata.req_pool_entries,
            candidate_lens=static_metadata.candidate_lens,
            candidate_query_lens=static_metadata.candidate_query_lens,
            lidu_cache_tokens=static_metadata.lidu_cache_tokens,
            needs_dsa_update=needs_dsa_update,
            dsa_offload_rows=dsa_offload_rows,
            dsa_offload_all_rows=dsa_offload_all_rows,
            lidu_init_rows=lidu_init_rows_tensor,
            lidu_all_rows_ready=lidu_all_rows_ready,
            has_first_decode=has_first_decode,
            decode_metadata_key=static_metadata.key,
        )
        return input_ids, positions

    @torch.inference_mode()
    def finalize_prefill_offload(self, seqs: list[Sequence]) -> None:
        if not self.uses_offload:
            return
        for seq in seqs:
            if seq.offload_finalized:
                continue
            old_hbm_block_table = list(seq.hbm_block_table)
            for module in self.model.modules():
                finalize = getattr(module, "finalize_prefill_offload", None)
                if finalize is not None:
                    finalize(seq, old_hbm_block_table)

            if seq.num_sparse_blocks >= seq.num_prefill_full_blocks:
                keep_sparse = old_hbm_block_table[: seq.num_prefill_full_blocks]
                release_blocks = []
                if self.offload_mode == OFFLOAD_LIDU:
                    # The complete source already fits the tier budget.  Every
                    # layer installed an identity source->slot map above, so no
                    # first-decode top-C initialization or DRAM copy is needed.
                    seq.lidu_cache_initialized = True
            elif self.offload_mode == OFFLOAD_LIDU:
                # LIDU owns a dense destination arena [0, C).  The first
                # decode eagerly replaces these initial tokens with top-C;
                # prompt tail and future decode blocks remain after the arena.
                keep_sparse = old_hbm_block_table[: seq.num_sparse_blocks]
                release_blocks = old_hbm_block_table[
                    seq.num_sparse_blocks : seq.num_prefill_full_blocks
                ]
            else:
                # GS preserves its prefix/suffix layout and mutable status map.
                prefix_blocks = 1 if seq.num_sparse_blocks > 0 else 0
                suffix_blocks = seq.num_sparse_blocks - prefix_blocks
                suffix_start = seq.num_prefill_full_blocks - suffix_blocks
                keep_sparse = (
                    old_hbm_block_table[:prefix_blocks]
                    + old_hbm_block_table[suffix_start : seq.num_prefill_full_blocks]
                )
                release_blocks = old_hbm_block_table[prefix_blocks:suffix_start]
            keep_tail = old_hbm_block_table[seq.num_prefill_full_blocks : seq.num_prefill_blocks]
            seq.hbm_block_table = keep_sparse + keep_tail
            seq.block_table = seq.hbm_block_table
            seq.hbm_blocks_to_release = release_blocks
            seq.offload_finalized = True
            seq.bump_decode_metadata_version()

    @torch.inference_mode()
    def run(
        self,
        seqs: list[Sequence | DecodeSequenceMetadata],
        is_prefill: bool,
    ) -> list[int] | None:
        try:
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
            if is_prefill or self.decode_graph_manager is None:
                hidden_states = self.model(input_ids, positions)
            else:
                hidden_states = self.decode_graph_manager.run(input_ids, positions)
            if not is_prefill and self.offload_mode == OFFLOAD_LIDU:
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
