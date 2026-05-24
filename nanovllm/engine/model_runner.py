# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the Nano-vLLM project


import os
import pickle
from multiprocessing.shared_memory import SharedMemory
from multiprocessing.synchronize import Event

os.environ.setdefault("VLLM_ASCEND_ENABLE_NZ", "0")

import torch
import torch.distributed as dist
import torch_npu  # noqa: F401

torch.npu.config.allow_internal_format = True

import vllm  # noqa: F401
import vllm_ascend  # noqa: F401
from vllm_ascend import vllm_ascend_C  # noqa: F401
from vllm_ascend.ops.layer_shard_linear import (  # noqa: F401
    is_hidden_layer,
    post_process_after_loading_for_shard_weight_series,
    reach_layer_for_shard_weight_series,
    register_all_layers_to_shard_weight_series,
)

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence
from nanovllm.layers.sampler import Sampler
from nanovllm.models.deepseek_v32 import DeepseekV32ForCausalLM
from nanovllm.utils.context import set_context, reset_context
from nanovllm.utils.loader import load_model
from nanovllm.utils.logger import init_logger

logger = init_logger(__name__)


class ModelRunner:

    def __init__(self, config: Config, rank: int, event: Event | list[Event]):
        self.config = config
        self.hf_config = config.hf_config
        self.block_size = config.kvcache_block_size
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
        torch.npu.empty_cache()
        self.allocate_kv_cache()

        torch.set_default_device("cpu")
        torch.set_default_dtype(default_dtype)
        logger.info(f"config: {config}")
        self._share_memory(rank)

    def _load_default_strategy(self):
        arch = (getattr(self.hf_config, "architectures", None) or [""])[0]
        if arch not in ("DeepseekV32ForCausalLM", "DeepseekV3ForCausalLM", ""):
            raise ValueError(
                f"Unsupported architecture {arch!r}; nano-vllm-ascend now "
                "only loads DeepSeek-V3.2 style models."
            )
        model = DeepseekV32ForCausalLM(self.hf_config)
        load_model(
            model,
            self.config.model,
            name_mapping=getattr(model, "weight_name_mapping", None),
        )
        if hasattr(model, "post_load_prepare"):
            model.post_load_prepare()
        return model

    def _share_memory(self, rank):
        # todo  /dev/shm/nanovllm 残留
        share_free_name = "nano_vllm_ascend"
        if self.world_size > 1:
            if rank == 0:
                try:
                    self.shm = SharedMemory(name=share_free_name, create=True, size=2 ** 20)
                except FileExistsError:
                    # 发现残留，强制回收
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
        if self.world_size > 1:
            self.shm.close()
            dist.barrier()
            if self.rank == 0:
                self.shm.unlink()
        torch.npu.synchronize()
        dist.destroy_process_group()

    def loop(self):
        while True:
            method_name, args = self.read_shm()
            self.call(method_name, *args)
            if method_name == "exit":
                break

    def read_shm(self):
        assert self.world_size > 1 and self.rank > 0
        self.event.wait()
        n = int.from_bytes(self.shm.buf[0:4], "little")
        method_name, *args = pickle.loads(self.shm.buf[4:n + 4])
        self.event.clear()
        return method_name, args

    def write_shm(self, method_name, *args):
        assert self.world_size > 1 and self.rank == 0
        data = pickle.dumps([method_name, *args])
        n = len(data)
        self.shm.buf[0:4] = n.to_bytes(4, "little")
        self.shm.buf[4:n + 4] = data
        for event in self.event:
            event.set()

    def call(self, method_name, *args):
        if self.world_size > 1 and self.rank == 0:
            self.write_shm(method_name, *args)
        method = getattr(self, method_name, None)
        return method(*args)

    def allocate_kv_cache(self):
        self._allocate_deepseek_dsa_cache()

    def _get_available_cache_mem(self):
        config = self.config
        free, total = torch.npu.mem_get_info()
        used = total - free
        peak = torch.npu.memory_stats()["allocated_bytes.all.peak"]
        current = torch.npu.memory_stats()["allocated_bytes.all.current"]
        available_mem = total * config.gpu_memory_utilization - used - peak + current
        return available_mem, used, total

    @staticmethod
    def _dtype_itemsize(dtype: torch.dtype) -> int:
        return torch.empty((), dtype=dtype).element_size()

    def _log_cache_allocation(
        self,
        *,
        total: int,
        used: int,
        block_bytes: int,
        num_blocks: int,
        shapes: list[tuple[str, tuple[int, ...]]],
    ) -> None:
        logger.info(f"Total NPU Mem: {total / 1024 ** 2:.2f} MB")
        logger.info(f"Used NPU Mem (Weights): {used / 1024 ** 2:.2f} MB")
        logger.info(f"Single Block Size: {block_bytes / 1024 ** 2:.2f} MB")
        logger.info(f"Allocating {num_blocks} blocks.")
        for name, shape in shapes:
            logger.info(f"{name} allocated successfully shape: {shape}")

    def _sync_kvcache_blocks_across_tp(self, local_num_blocks: int) -> int:
        if self.world_size <= 1 or not dist.is_initialized():
            return local_num_blocks
        blocks = torch.tensor(
            [local_num_blocks],
            dtype=torch.int32,
            device=self.device,
        )
        dist.all_reduce(blocks, op=dist.ReduceOp.MIN)
        synced_num_blocks = int(blocks.item())
        if synced_num_blocks != local_num_blocks:
            logger.info(
                "Using TP-wide minimum KV cache blocks: rank=%d local=%d synced=%d",
                self.rank,
                local_num_blocks,
                synced_num_blocks,
            )
        return synced_num_blocks

    def _allocate_deepseek_dsa_cache(self):
        config = self.config
        hf_config = config.hf_config
        text_config = getattr(hf_config, "text_config", hf_config)
        available_mem, used, total = self._get_available_cache_mem()
        cache_dtype = self._set_torch_dtype(text_config)
        num_layers = int(text_config.num_hidden_layers)
        kv_lora_rank = int(text_config.kv_lora_rank)
        rope_dim = int(text_config.qk_rope_head_dim)
        index_dim = int(text_config.index_head_dim)
        block_bytes = (
            num_layers
            * self.block_size
            * (kv_lora_rank + rope_dim + index_dim)
            * self._dtype_itemsize(cache_dtype)
        )
        local_num_blocks = int(available_mem) // block_bytes
        config.num_kvcache_blocks = self._sync_kvcache_blocks_across_tp(
            local_num_blocks,
        )
        assert config.num_kvcache_blocks > 0, (
            "Failed to allocate any DeepSeek DSA cache blocks due to "
            "insufficient memory."
        )
        ckv_shape = (
            num_layers,
            config.num_kvcache_blocks,
            self.block_size,
            1,
            kv_lora_rank,
        )
        kpe_shape = (
            num_layers,
            config.num_kvcache_blocks,
            self.block_size,
            1,
            rope_dim,
        )
        index_shape = (
            num_layers,
            config.num_kvcache_blocks,
            self.block_size,
            1,
            index_dim,
        )
        self._log_cache_allocation(
            total=total,
            used=used,
            block_bytes=block_bytes,
            num_blocks=config.num_kvcache_blocks,
            shapes=[
                ("DeepSeek CKV cache", ckv_shape),
                ("DeepSeek KPE cache", kpe_shape),
                ("DeepSeek index cache", index_shape),
            ],
        )
        layer_shapes = (ckv_shape[1:], kpe_shape[1:], index_shape[1:])
        for module in self.model.modules():
            if hasattr(module, "assign_dsa_cache") and hasattr(module, "layer_id"):
                ckv_cache = torch.empty(
                    layer_shapes[0], dtype=cache_dtype, device=self.device
                )
                kpe_cache = torch.empty(
                    layer_shapes[1], dtype=cache_dtype, device=self.device
                )
                index_cache = torch.empty(
                    layer_shapes[2], dtype=cache_dtype, device=self.device
                )
                ckv_cache.zero_()
                kpe_cache.zero_()
                index_cache.zero_()
                module.assign_dsa_cache(ckv_cache, kpe_cache, index_cache)

    def prepare_block_tables(self, seqs: list[Sequence]):
        max_len = max(len(seq.block_table) for seq in seqs)
        block_tables = [seq.block_table + [-1] * (max_len - len(seq.block_table)) for seq in seqs]
        block_tables = torch.tensor(block_tables, dtype=torch.int32, pin_memory=True).to(device=self.device,
                                                                                         non_blocking=True)
        return block_tables

    def _pad_block_tables_to_static_max(self, block_tables: torch.Tensor) -> torch.Tensor:
        block_tables = torch.where(
            block_tables >= 0,
            block_tables,
            torch.zeros_like(block_tables),
        )
        static_max_block_cols = (
            self.config.max_model_len + self.config.kvcache_block_size - 1
        ) // self.config.kvcache_block_size
        if block_tables.shape[1] >= static_max_block_cols:
            return block_tables
        padded_block_tables = torch.zeros(
            (block_tables.shape[0], static_max_block_cols),
            dtype=block_tables.dtype,
            device=block_tables.device,
        )
        padded_block_tables[:, : block_tables.shape[1]] = block_tables
        return padded_block_tables

    def prepare_prefill(self, seqs: list[Sequence]):
        input_ids = []
        positions = []
        cu_seqlens_q = [0]
        cu_seqlens_k = [0]
        max_seqlen_q = 0
        max_seqlen_k = 0
        slot_mapping = []

        for seq in seqs:
            actual_tokens = seq[:]
            input_ids.extend(actual_tokens)

            seqlen = len(seq)
            positions.extend(list(range(0, seqlen)))

            seqlen_q = seqlen
            seqlen_k = seqlen

            cu_seqlens_q.append(cu_seqlens_q[-1] + seqlen_q)
            cu_seqlens_k.append(cu_seqlens_k[-1] + seqlen_k)

            max_seqlen_q = max(seqlen_q, max_seqlen_q)
            max_seqlen_k = max(seqlen_k, max_seqlen_k)

            if seq.block_table:
                for i in range(seq.num_blocks):
                    start = seq.block_table[i] * self.block_size
                    if i != seq.num_blocks - 1:
                        end = start + self.block_size
                    else:
                        end = start + seq.last_block_num_tokens
                    slot_mapping.extend(list(range(start, end)))

        block_tables = self.prepare_block_tables(seqs)
        block_tables = self._pad_block_tables_to_static_max(block_tables)

        input_ids = torch.tensor(input_ids, dtype=torch.int64).to(self.device)
        positions = torch.tensor(positions, dtype=torch.int64).to(self.device)
        cu_seqlens_q = torch.tensor(cu_seqlens_q, dtype=torch.int32).to(self.device)
        cu_seqlens_k = torch.tensor(cu_seqlens_k, dtype=torch.int32).to(self.device)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32).to(self.device)

        set_context(True,
                    cu_seqlens_q=cu_seqlens_q,
                    cu_seqlens_k=cu_seqlens_k,
                    max_seqlen_q=max_seqlen_q,
                    max_seqlen_k=max_seqlen_k,
                    slot_mapping=slot_mapping,
                    context_lens=None,
                    block_tables=block_tables,
                    block_size=self.config.kvcache_block_size)

        return input_ids, positions

    def prepare_decode(self, seqs: list[Sequence]):
        input_ids = []
        positions = []
        slot_mapping = []
        context_lens = []
        for seq in seqs:
            input_ids.append(seq.last_token)
            positions.append(len(seq) - 1)
            context_lens.append(len(seq))
            slot_mapping.append([seq.block_table[-1], seq.last_block_num_tokens - 1])
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).to(self.device, non_blocking=True)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).to(self.device, non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).to(self.device, non_blocking=True)
        context_lens = torch.tensor(context_lens, dtype=torch.int32, pin_memory=True).to(self.device, non_blocking=True)
        block_tables = self.prepare_block_tables(seqs)
        block_tables = self._pad_block_tables_to_static_max(block_tables)
        set_context(False,
                    slot_mapping=slot_mapping,
                    context_lens=context_lens,
                    block_tables=block_tables,
                    is_enforce_eager=True,
                    real_bs=len(seqs),
                    block_size=self.config.kvcache_block_size)
        return input_ids, positions

    def prepare_sample(self, seqs: list[Sequence]):
        temperatures = []
        for seq in seqs:
            temperatures.append(seq.temperature)
        temperatures = torch.tensor(temperatures, dtype=torch.float32, pin_memory=True).to(self.device,
                                                                                           non_blocking=True)
        return temperatures

    @torch.inference_mode()
    def run_model(self,
                  input_ids: torch.Tensor,
                  positions: torch.Tensor,
                  is_prefill: bool,
                  ):
        hidden_states = self.model(input_ids, positions)
        logits = self.model.compute_logits(hidden_states)
        return logits

    def run(self, seqs: list[Sequence], is_prefill: bool) -> list[int]:
        input_ids, positions = (
            self.prepare_prefill(seqs)
            if is_prefill
            else self.prepare_decode(seqs)
        )
        temperatures = self.prepare_sample(seqs) if self.rank == 0 else None
        logits = self.run_model(input_ids, positions, is_prefill)
        token_ids = self.sampler(logits, temperatures).tolist() if self.rank == 0 else None
        reset_context()
        return token_ids
