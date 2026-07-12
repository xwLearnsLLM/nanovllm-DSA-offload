# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the Nano-vLLM project

import pickle
from multiprocessing.shared_memory import SharedMemory
from multiprocessing.synchronize import Event

import torch
import torch.distributed as dist
import torch_npu  # noqa: F401

torch.npu.config.allow_internal_format = True

from nanovllm.config import Config
from nanovllm.engine.dsa_offload import DSA_SELECTION_TOPK_TOKENS
from nanovllm.engine.full_decode_graph import FullDecodeOnlyGraphManager
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
        self._allocate_deepseek_dsa_cache()
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
                log_enabled=self.rank == 0,
            )
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
                "DSA FULL_DECODE_ONLY final stats: %s",
                self.decode_graph_manager.stats(),
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
        return self.decode_graph_manager.stats()

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

    def _allocate_deepseek_dsa_cache(self):
        config = self.config
        hf_config = config.hf_config
        text_config = getattr(hf_config, "text_config", hf_config)
        cache_dtype = self._set_torch_dtype(text_config)
        num_layers = int(text_config.num_hidden_layers)
        kv_lora_rank = int(text_config.kv_lora_rank)
        rope_dim = int(text_config.qk_rope_head_dim)
        index_dim = int(text_config.index_head_dim)
        hbm_kv_block_bytes = num_layers * self.block_size * (kv_lora_rank + rope_dim) * torch.empty((), dtype=cache_dtype).element_size()
        if DSA_SELECTION_TOPK_TOKENS % self.block_size != 0:
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
                DSA_SELECTION_TOPK_TOKENS,
            )

        ckv_shape = (num_layers, config.num_hbm_kvcache_blocks, self.block_size, 1, kv_lora_rank)
        kpe_shape = (num_layers, config.num_hbm_kvcache_blocks, self.block_size, 1, rope_dim)
        index_shape = (num_layers, config.num_dram_kvcache_blocks, self.block_size, 1, index_dim)
        dram_ckv_shape = (num_layers, config.num_dram_kvcache_blocks, self.block_size, 1, kv_lora_rank)
        dram_kpe_shape = (num_layers, config.num_dram_kvcache_blocks, self.block_size, 1, rope_dim)
        gather_status_shape = (config.max_num_decode_seqs_per_step, 1, 1, DSA_SELECTION_TOPK_TOKENS + 1)
        if self.rank == 0:
            logger.info(f"Single HBM KV Block Size: {hbm_kv_block_bytes / 1024 ** 2:.2f} MB")
            for name, shape in [
                ("DeepSeek CKV cache", ckv_shape),
                ("DeepSeek KPE cache", kpe_shape),
                ("DeepSeek index cache", index_shape),
                ("DeepSeek DRAM CKV cache", dram_ckv_shape),
                ("DeepSeek DRAM KPE cache", dram_kpe_shape),
                ("DeepSeek gather selection status", gather_status_shape),
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
                gather_status = torch.full(gather_status_shape, -1, dtype=torch.int32, device=self.device)
                ckv_cache.zero_()
                kpe_cache.zero_()
                index_cache.zero_()
                module.assign_dsa_cache(ckv_cache, kpe_cache, index_cache, dram_ckv_cache, dram_kpe_cache, gather_status)

    def prepare_block_tables(self, seqs: list[Sequence], table_name: str = "hbm_block_table"):
        static_max_block_cols = (self.config.max_model_len + self.config.kvcache_block_size - 1) // self.config.kvcache_block_size
        tables = [getattr(seq, table_name) for seq in seqs]
        max_len = max(len(table) for table in tables)
        num_cols = max(max_len, static_max_block_cols)
        return torch.tensor([table + [0] * (num_cols - len(table)) for table in tables], dtype=torch.int32, pin_memory=True).to(device=self.device, non_blocking=True)

    def prepare_selection_block_tables(self, seqs: list[Sequence]) -> torch.Tensor:
        max_sparse_blocks = DSA_SELECTION_TOPK_TOKENS // self.block_size
        tables = []
        for seq in seqs:
            sparse_blocks = min(int(seq.num_sparse_blocks), max_sparse_blocks)
            table = list(seq.hbm_block_table[:sparse_blocks])
            tables.append(table + [0] * (max_sparse_blocks - len(table)))
        return torch.tensor(tables, dtype=torch.int32, pin_memory=True).to(self.device, non_blocking=True)

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
            flat_index_slot_mapping.extend(chunk_index_slot_mapping)
            cu_seqlens_q.append(len(chunk_input_ids))
            actual_seq_lengths_kv = [chunk_end]
            candidate_len = seqs[0].num_prefill_full_blocks * self.block_size
            needs_dsa_update = candidate_len > seqs[0].num_sparse_tokens > 0
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
                    flat_index_slot_mapping.extend(
                        self._sequence_slots(seq.index_block_table, seqlen)
                    )
                candidate_len = seq.num_prefill_full_blocks * self.block_size
                needs_dsa_update = needs_dsa_update or (
                    candidate_len > seq.num_sparse_tokens > 0
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

    def prepare_decode(self, seqs: list[Sequence]):
        input_ids = []
        positions = []
        flat_slot_mapping = []
        candidate_lens = []
        sparse_kv_lens = []
        req_pool_entries = []
        offload_rows = []
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
            row_needs_offload = candidate_len > sparse_selected_len > 0
            needs_dsa_update = needs_dsa_update or row_needs_offload
            if row_needs_offload:
                offload_rows.append(row)

            candidate_lens.append(candidate_len)
            sparse_kv_lens.append(sparse_kv_len)
            req_pool_entries.append(seq.hbm_cached_tokens_pool_entry)
        dsa_offload_all_rows = bool(offload_rows and len(offload_rows) == len(seqs))
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).to(self.device, non_blocking=True)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).to(self.device, non_blocking=True)
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
        hbm_block_tables = self.prepare_block_tables(seqs, "hbm_block_table")
        index_block_tables = self.prepare_block_tables(seqs, "index_block_table")
        dram_block_tables = self.prepare_block_tables(seqs, "dram_block_table")
        selection_block_tables = self.prepare_selection_block_tables(seqs)
        req_pool_entries = torch.tensor(req_pool_entries, dtype=torch.int32, pin_memory=True).to(self.device, non_blocking=True)
        candidate_lens = torch.tensor(candidate_lens, dtype=torch.int32, pin_memory=True).to(self.device, non_blocking=True)
        candidate_query_lens = torch.arange(1, len(seqs) + 1, dtype=torch.int32, pin_memory=True).to(self.device, non_blocking=True)
        dsa_offload_rows = None
        if needs_dsa_update and not dsa_offload_all_rows:
            dsa_offload_rows = torch.tensor(offload_rows, dtype=torch.long, pin_memory=True).to(self.device, non_blocking=True)
        set_context(
            False,
            flat_slot_mapping=flat_slot_mapping_i64,
            flat_slot_mapping_i32=flat_slot_mapping_i32,
            actual_seq_lengths_kv=sparse_kv_lens,
            block_tables=hbm_block_tables,
            index_block_tables=index_block_tables,
            dram_block_tables=dram_block_tables,
            selection_block_tables=selection_block_tables,
            req_pool_entries=req_pool_entries,
            candidate_lens=candidate_lens,
            candidate_query_lens=candidate_query_lens,
            needs_dsa_update=needs_dsa_update,
            dsa_offload_rows=dsa_offload_rows,
            dsa_offload_all_rows=dsa_offload_all_rows,
            has_first_decode=has_first_decode,
        )
        return input_ids, positions

    @torch.inference_mode()
    def finalize_prefill_offload(self, seqs: list[Sequence]) -> None:
        for seq in seqs:
            if seq.offload_finalized:
                continue
            old_hbm_block_table = list(seq.hbm_block_table)
            for module in self.model.modules():
                finalize = getattr(module, "finalize_prefill_offload", None)
                if finalize is not None:
                    finalize(seq, old_hbm_block_table)

            # Sparse decode is logically packed, but HBM KV stays in its original
            # physical blocks: first prefix block, suffix blocks, then tail/decode.
            if seq.num_sparse_blocks >= seq.num_prefill_full_blocks:
                keep_sparse = old_hbm_block_table[: seq.num_prefill_full_blocks]
                release_blocks = []
            else:
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

    @torch.inference_mode()
    def run(self, seqs: list[Sequence], is_prefill: bool) -> list[int] | None:
        try:
            if is_prefill:
                input_ids, positions, should_sample = self.prepare_prefill(seqs)
            else:
                input_ids, positions = self.prepare_decode(seqs)
                should_sample = True
            temperatures = None
            if should_sample and self.rank == 0:
                temperatures = torch.tensor(
                    [seq.temperature for seq in seqs],
                    dtype=torch.float32,
                    pin_memory=True,
                ).to(self.device, non_blocking=True)
            if is_prefill or self.decode_graph_manager is None:
                hidden_states = self.model(input_ids, positions)
            else:
                hidden_states = self.decode_graph_manager.run(input_ids, positions)
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
