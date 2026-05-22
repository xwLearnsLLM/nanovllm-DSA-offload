# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the Nano-vLLM project


import os
import pickle
from multiprocessing.shared_memory import SharedMemory
from multiprocessing.synchronize import Event
from time import perf_counter

import torch
import torch.distributed as dist

from nanovllm.config import Config, GraphMode
from nanovllm.engine.sequence import Sequence
from nanovllm.layers.sampler import Sampler
from nanovllm.models.models_map import model_dict
from nanovllm.models.qwen3_vl import load_qwen3_vl_model
from nanovllm.utils.context import set_context, reset_context, get_context
from nanovllm.utils.loader import load_model
from nanovllm.utils.logger import init_logger

logger = init_logger(__name__)


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.lower() in ("1", "true", "yes", "on")


def _import_torchair():
    try:
        import torchair  # type: ignore
    except Exception as exc:
        raise RuntimeError(
            "torchair is only required for the non-eager graph/compile path. "
            "Install torchair, or run with `enforce_eager=True`."
        ) from exc
    return torchair


def _register_vllm_ascend_custom_ops() -> bool:
    os.environ.setdefault("VLLM_ASCEND_ENABLE_NZ", "0")
    try:
        import torch_npu  # noqa: F401  # type: ignore
        import vllm  # noqa: F401  # type: ignore
        import vllm_ascend  # noqa: F401  # type: ignore
        from vllm_ascend import vllm_ascend_C  # noqa: F401  # type: ignore
        from vllm_ascend.ops.layer_shard_linear import (  # noqa: F401
            is_hidden_layer,
            post_process_after_loading_for_shard_weight_series,
            reach_layer_for_shard_weight_series,
            register_all_layers_to_shard_weight_series,
        )
    except Exception as exc:
        logger.warning("Failed to register vLLM-Ascend custom ops early: %r", exc)
        return False
    logger.info("Registered vLLM-Ascend custom ops early for Nano NPU indexer.")
    return True


class ModelRunner:

    def __init__(self, config: Config, rank: int, event: Event | list[Event]):
        self.config = config
        self.hf_config = config.hf_config
        self.block_size = config.kvcache_block_size
        self.enforce_eager = config.enforce_eager
        self.world_size = config.tensor_parallel_size
        self.rank = rank
        self.event = event
        self.device = config.device
        self.model_type = self.hf_config.model_type

        torch.npu.set_device(rank)
        if _env_flag("NANOVLLM_ENABLE_NPU_INDEXER", False):
            _register_vllm_ascend_custom_ops()
        dist.init_process_group("hccl", f"tcp://localhost:{config.hccl_port}", world_size=self.world_size, rank=rank)
        default_dtype = torch.get_default_dtype()

        torch_dtype = self._set_torch_dtype(self.hf_config)
        torch.set_default_dtype(torch_dtype)
        torch.set_default_device(self.device)

        self.is_multimodal = False
        self._load_strategies = {
            "qwen3_vl": self._load_qwen3_vl_strategy,
        }
        loader = self._load_strategies.get(self.model_type, self._load_default_strategy)
        self.model = loader()
        self.config.is_multimodal = self.is_multimodal

        embed_module = self._get_embed_module()
        # Keep a reference dtype so that cached vision embeddings can be copied
        # back to the GPU without hitting dtype mismatches.
        self.model_dtype = embed_module.embed_tokens.weight.dtype

        self.sampler = Sampler()
        self.profile_decode = _env_flag("NANOVLLM_PROFILE_DECODE", False)
        self.profile_decode_every = max(
            1,
            int(os.environ.get("NANOVLLM_PROFILE_DECODE_EVERY", "31")),
        )
        self.profile_decode_steps = 0
        self.profile_decode_tokens = 0
        self.profile_prepare_time = 0.0
        self.profile_model_time = 0.0
        self.profile_forward_time = 0.0
        self.profile_logits_time = 0.0
        self.profile_sample_time = 0.0
        self._last_forward_time = 0.0
        self._last_logits_time = 0.0
        torch.npu.empty_cache()
        self.allocate_kv_cache()
        if not self.enforce_eager:
            self.compiler_config = None
            self.compile_decode = None
            self.decode_compile(config)

        torch.set_default_device("cpu")
        torch.set_default_dtype(default_dtype)
        logger.info(f"config: {config}")
        self._share_memory(rank)

    def _get_embed_module(self):
        # Multimodal support is optional; fall back to text-only runner when
        # the extended Qwen3-VL stack is not available.
        embed_module = getattr(self.model, "language_model", self.model)
        if hasattr(embed_module, "model"):
            embed_module = embed_module.model
        return embed_module

    def _load_qwen3_vl_strategy(self):
        self.is_multimodal = True
        return load_qwen3_vl_model(self.config.model, self.config)

    def _load_default_strategy(self):
        arch = self.hf_config.architectures[0]
        model = model_dict[arch](self.hf_config)
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

    def decode_compile(self, config):
        torchair = _import_torchair()
        """
        max-autotune模式（Ascend IR）：将PyTorch的FX计算图转换为昇腾中间表示（IR，Intermediate Representation），
        即Ascend IR计算图，并通过GE（Graph Engine，图引擎）实现计算图的编译和执行。
        """
        logger.info(f"graph mode: {config.graph_mode}")
        if config.graph_mode == GraphMode.MAX_AUTOTUNE.value:
            if config.use_graph_cache:
                self.compile_decode = torchair.inference.cache_compile(self.model.forward,
                                                                       config=self.compiler_config,
                                                                       dynamic=False,
                                                                       fullgraph=True,
                                                                       ge_cache=True)
            else:
                npu_backend = torchair.get_npu_backend(compiler_config=self.compiler_config)
                self.compile_decode = torch.compile(self.model.forward, dynamic=False, fullgraph=True,
                                                    backend=npu_backend)
        """
        reduce-overhead模式（aclgraph）：采用Capture&Replay方式实现任务一次捕获多次执行，Capture阶段捕获Stream任务到Device侧，暂不执行；
        Replay阶段从Host侧发出执行指令，Device侧再执行已捕获的任务，从而减少Host调度开销，提升性能。
        """
        if config.graph_mode == GraphMode.REDUCE_OVERHEAD.value:
            from torchair.configs.compiler_config import CompilerConfig

            compiler_config = CompilerConfig()
            compiler_config.mode = GraphMode.REDUCE_OVERHEAD.value
            npu_backend = torchair.get_npu_backend(compiler_config=compiler_config)
            self.compile_decode = torch.compile(self.model.forward, backend=npu_backend)

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
        if self.model_type == "deepseek_v32":
            self._allocate_deepseek_dsa_cache()
            return
        self._allocate_dense_kv_cache()

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

    def _allocate_dense_kv_cache(self):
        config = self.config
        hf_config = config.hf_config
        text_config = getattr(hf_config, "text_config", hf_config)
        available_mem, used, total = self._get_available_cache_mem()
        cache_dtype = self._set_torch_dtype(text_config)
        num_kv_heads = text_config.num_key_value_heads // self.world_size
        head_dim = (
            text_config.head_dim
            if hasattr(text_config, "head_dim")
            else text_config.hidden_size // text_config.num_attention_heads
        )
        num_layers = text_config.num_hidden_layers
        block_bytes = (
            2
            * num_layers
            * self.block_size
            * num_kv_heads
            * head_dim
            * self._dtype_itemsize(cache_dtype)
        )
        config.num_kvcache_blocks = int(available_mem) // block_bytes
        assert config.num_kvcache_blocks > 0, "Failed to allocate any KV cache blocks due to insufficient memory."
        cache_shape = (2, num_layers, config.num_kvcache_blocks, self.block_size, num_kv_heads * head_dim)
        kv_cache = torch.empty(cache_shape, dtype=cache_dtype, device=self.device)
        kv_cache.zero_()
        self._log_cache_allocation(
            total=total,
            used=used,
            block_bytes=block_bytes,
            num_blocks=config.num_kvcache_blocks,
            shapes=[("KV Cache", cache_shape)],
        )
        layer_id = 0
        for module in self.model.modules():
            if hasattr(module, "k_cache") and hasattr(module, "v_cache"):
                module.k_cache = kv_cache[0, layer_id]
                module.v_cache = kv_cache[1, layer_id]
                if hasattr(module, "layer_id"):
                    module.layer_id = layer_id
                layer_id += 1

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
        config.num_kvcache_blocks = int(available_mem) // block_bytes
        assert config.num_kvcache_blocks > 0, (
            "Failed to allocate any DeepSeek DSA cache blocks due to "
            "insufficient memory."
        )
        use_indexer_cache_layout = (
            _env_flag("NANOVLLM_ENABLE_NPU_INDEXER", False)
            or _env_flag("NANOVLLM_ENABLE_NPU_SFA_PREFILL", False)
            or _env_flag("NANOVLLM_ENABLE_NPU_SFA_DECODE", False)
        )

        if use_indexer_cache_layout:
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
        else:
            ckv_shape = (
                num_layers,
                config.num_kvcache_blocks,
                self.block_size,
                kv_lora_rank,
            )
            kpe_shape = (
                num_layers,
                config.num_kvcache_blocks,
                self.block_size,
                rope_dim,
            )
            index_shape = (
                num_layers,
                config.num_kvcache_blocks,
                self.block_size,
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
        if use_indexer_cache_layout:
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
        else:
            ckv_cache = torch.empty(ckv_shape, dtype=cache_dtype, device=self.device)
            kpe_cache = torch.empty(kpe_shape, dtype=cache_dtype, device=self.device)
            index_cache = torch.empty(index_shape, dtype=cache_dtype, device=self.device)
            ckv_cache.zero_()
            kpe_cache.zero_()
            index_cache.zero_()
            for module in self.model.modules():
                if hasattr(module, "assign_dsa_cache") and hasattr(module, "layer_id"):
                    module.assign_dsa_cache(
                        ckv_cache[module.layer_id],
                        kpe_cache[module.layer_id],
                        index_cache[module.layer_id],
                    )

    def prepare_block_tables(self, seqs: list[Sequence]):
        max_len = max(len(seq.block_table) for seq in seqs)
        block_tables = [seq.block_table + [-1] * (max_len - len(seq.block_table)) for seq in seqs]
        block_tables = torch.tensor(block_tables, dtype=torch.int32, pin_memory=True).to(device=self.device,
                                                                                         non_blocking=True)
        return block_tables

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

    def prepare_decode_padding(self, seqs: list[Sequence]):
        input_ids, positions, slot_mapping, context_lens = [], [], [], []
        max_compile_bs = self.config.max_num_seqs
        real_bs = len(seqs)

        for seq in seqs:
            input_ids.append(seq.last_token)
            positions.append(len(seq) - 1)
            context_lens.append(len(seq))
            slot_mapping.append([seq.block_table[-1], seq.last_block_num_tokens - 1])

        # padding
        padding_size = max_compile_bs - real_bs
        if padding_size > 0:
            input_ids.extend([0] * padding_size)
            positions.extend([0] * padding_size)
            context_lens.extend([0] * padding_size)
            # dummy_slot block_num=1891 max_num_seq=4 seq=3 [[0,10],[1,11],[2,19],[1890,0]]
            dummy_slot = [self.config.num_kvcache_blocks - 1, 0]
            slot_mapping.extend([dummy_slot] * padding_size)

        input_ids = torch.tensor(input_ids, dtype=torch.int64).to(self.device).contiguous()
        positions = torch.tensor(positions, dtype=torch.int64).to(self.device).contiguous()
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32).to(self.device).contiguous()
        context_lens = torch.tensor(context_lens, dtype=torch.int32).to(self.device).contiguous()

        # 构造静态形状的 block_tables (max_compile_bs, static_max_block_cols) 避免图编译报错
        static_max_block_cols = self.config.max_model_len // self.config.kvcache_block_size
        raw_block_tables = self.prepare_block_tables(seqs)

        block_tables = torch.full(
            (max_compile_bs, static_max_block_cols),
            fill_value=-1,
            dtype=torch.int32,
            device=self.device
        )

        curr_real_bs, curr_cols = raw_block_tables.shape
        actual_cols = min(curr_cols, static_max_block_cols)
        block_tables[:curr_real_bs, :actual_cols] = raw_block_tables[:, :actual_cols]

        set_context(False,
                    slot_mapping=slot_mapping,
                    context_lens=context_lens,
                    block_tables=block_tables,
                    is_enforce_eager=self.enforce_eager,
                    real_bs=curr_real_bs,
                    block_size=self.config.kvcache_block_size
                    )

        return input_ids, positions

    def prepare_decode(self, seqs: list[Sequence]):
        input_ids = []
        positions = []
        slot_mapping = []
        context_lens = []
        decode_slot_rows = []
        decode_mask_rows = []
        max_context_len = max(len(seq) for seq in seqs)
        for seq in seqs:
            input_ids.append(seq.last_token)
            positions.append(len(seq) - 1)
            context_lens.append(len(seq))
            slot_mapping.append([seq.block_table[-1], seq.last_block_num_tokens - 1])
            seq_slots: list[int] = []
            remaining = len(seq)
            for block in seq.block_table[:seq.num_blocks]:
                take = min(self.block_size, remaining)
                start = block * self.block_size
                seq_slots.extend(range(start, start + take))
                remaining -= take
                if remaining <= 0:
                    break
            padding = max_context_len - len(seq_slots)
            decode_slot_rows.append(seq_slots + [0] * padding)
            decode_mask_rows.append([True] * len(seq_slots) + [False] * padding)
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).to(self.device, non_blocking=True)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).to(self.device, non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).to(self.device, non_blocking=True)
        context_lens = torch.tensor(context_lens, dtype=torch.int32, pin_memory=True).to(self.device, non_blocking=True)
        decode_slots = torch.tensor(
            decode_slot_rows,
            dtype=torch.int64,
            pin_memory=True,
        ).to(self.device, non_blocking=True)
        decode_mask = torch.tensor(
            decode_mask_rows,
            dtype=torch.bool,
            pin_memory=True,
        ).to(self.device, non_blocking=True)
        block_tables = self.prepare_block_tables(seqs)
        if _env_flag("NANOVLLM_ENABLE_NPU_INDEXER", False):
            static_max_block_cols = (
                self.config.max_model_len + self.config.kvcache_block_size - 1
            ) // self.config.kvcache_block_size
            if block_tables.shape[1] < static_max_block_cols:
                padded_block_tables = torch.full(
                    (len(seqs), static_max_block_cols),
                    fill_value=-1,
                    dtype=torch.int32,
                    device=self.device,
                )
                padded_block_tables[:, : block_tables.shape[1]] = block_tables
                block_tables = padded_block_tables
        set_context(False,
                    slot_mapping=slot_mapping,
                    context_lens=context_lens,
                    block_tables=block_tables,
                    is_enforce_eager=self.enforce_eager,
                    real_bs=len(seqs),
                    block_size=self.config.kvcache_block_size,
                    decode_slots=decode_slots,
                    decode_mask=decode_mask)
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
                  sequence_lengths: list[int] | None = None,
                  vision_slices_per_seq: list[list[dict]] | None = None,
                  ):
        model_kwargs = {}
        if self.is_multimodal:
            # Prefill can stream only part of the visual tokens. Pass
            # slice metadata so the forward pass knows which cached chunks
            # to use.
            model_kwargs["sequence_lengths"] = sequence_lengths
            model_kwargs["vision_slices_per_seq"] = vision_slices_per_seq
        execute_tokens = len(input_ids) if is_prefill else get_context().real_bs
        if _env_flag("NANOVLLM_LOG_EXECUTE_TOKENS", False):
            logger.info(
                f"{'prefill' if is_prefill else 'decode'} execute tokens: "
                f"{execute_tokens}"
            )
        should_profile = self.profile_decode and not is_prefill
        if should_profile:
            torch.npu.synchronize()
            forward_start = perf_counter()
        if is_prefill or self.enforce_eager or input_ids.size(0) > 512:
            hidden_states = self.model(input_ids, positions, **model_kwargs)
        else:
            hidden_states = self.compile_decode(input_ids, positions)
        if should_profile:
            torch.npu.synchronize()
            self._last_forward_time = perf_counter() - forward_start
            logits_start = perf_counter()
        logits = self.model.compute_logits(hidden_states)
        if should_profile:
            torch.npu.synchronize()
            self._last_logits_time = perf_counter() - logits_start
        return logits

    def run(self, seqs: list[Sequence], is_prefill: bool) -> list[int]:
        def _sync_if_profile():
            if self.profile_decode and not is_prefill:
                torch.npu.synchronize()

        _sync_if_profile()
        prepare_start = perf_counter()
        input_ids, positions = self.prepare_prefill(seqs) if is_prefill else (
            self.prepare_decode(seqs) if self.enforce_eager else self.prepare_decode_padding(seqs))
        _sync_if_profile()
        prepare_time = perf_counter() - prepare_start

        # Track how many freshly decoded tokens each sequence contributes; the
        # model uses these lengths to align partial vision slices with text.
        sequence_lengths = ([len(seq) - seq.num_cached_tokens for seq in seqs] if is_prefill else None)
        vision_slices_per_seq = None

        vision_slices_per_seq = self._get_vision_slices_per_seq(is_prefill, seqs, vision_slices_per_seq)

        def _advance_vision_offsets():
            if not is_prefill or not self.is_multimodal:
                return
            if vision_slices_per_seq is None:
                return
            for seq, slices in zip(seqs, vision_slices_per_seq):
                for slice_info in slices:
                    length = slice_info["length"]
                    placeholder_idx = slice_info["placeholder_idx"]
                    if placeholder_idx < len(seq.vision_consumed):
                        span = seq.vision_placeholders[placeholder_idx][1]
                        seq.vision_consumed[placeholder_idx] += length
                        seq.vision_consumed[placeholder_idx] = min(
                            seq.vision_consumed[placeholder_idx],
                            span,
                        )
                if seq.vision_placeholders:
                    # Once every placeholder has been consumed we can drop the
                    # cached tensors to release CPU memory.
                    all_consumed = all(
                        seq.vision_consumed[idx] >= span
                        for idx, (_, span) in enumerate(
                            seq.vision_placeholders
                        )
                    )
                else:
                    all_consumed = True
                if all_consumed:
                    seq.cached_vision_tokens = None
                    seq.cached_deepstack_tokens = None

        temperatures = self.prepare_sample(seqs) if self.rank == 0 else None
        _sync_if_profile()
        model_start = perf_counter()
        logits = self.run_model(
            input_ids,
            positions,
            is_prefill,
            sequence_lengths=sequence_lengths,
            vision_slices_per_seq=vision_slices_per_seq,
        )
        _sync_if_profile()
        model_time = perf_counter() - model_start
        _advance_vision_offsets()

        sample_start = perf_counter()
        token_ids = self.sampler(logits, temperatures).tolist() if self.rank == 0 else None
        sample_time = perf_counter() - sample_start
        if self.profile_decode and not is_prefill and self.rank == 0:
            real_bs = get_context().real_bs
            self.profile_decode_steps += 1
            self.profile_decode_tokens += int(real_bs)
            self.profile_prepare_time += prepare_time
            self.profile_model_time += model_time
            self.profile_forward_time += self._last_forward_time
            self.profile_logits_time += self._last_logits_time
            self.profile_sample_time += sample_time
            if self.profile_decode_steps % self.profile_decode_every == 0:
                steps = self.profile_decode_steps
                tokens = max(self.profile_decode_tokens, 1)
                total_time = (
                    self.profile_prepare_time
                    + self.profile_model_time
                    + self.profile_sample_time
                )
                logger.info(
                    "Decode profile avg: steps=%d tokens=%d "
                    "prepare=%.4fs/step model_forward=%.4fs/step "
                    "lm_head=%.4fs/step model_lm=%.4fs/step "
                    "sample=%.4fs/step total=%.4fs/step %.2f toks/s",
                    steps,
                    tokens,
                    self.profile_prepare_time / steps,
                    self.profile_forward_time / steps,
                    self.profile_logits_time / steps,
                    self.profile_model_time / steps,
                    self.profile_sample_time / steps,
                    total_time / steps,
                    tokens / max(total_time, 1e-9),
                )
        reset_context()
        return token_ids

    def _get_vision_slices_per_seq(self, is_prefill, seqs, vision_slices_per_seq):
        if is_prefill and self.is_multimodal:
            vision_slices_per_seq = []
            has_slices = False
            for seq in seqs:
                # Cache the full vision tower output once; subsequent prefill
                # steps only read the portions still needed for this sequence.
                self._ensure_vision_cache(seq)
                slices_for_seq: list[dict] = []
                window_start = seq.num_cached_tokens
                window_end = len(seq)

                for placeholder_idx, (offset, length) in enumerate(seq.vision_placeholders):
                    if placeholder_idx >= len(seq.vision_counts):
                        continue
                    consumed = seq.vision_consumed[placeholder_idx]
                    total_len = length
                    if consumed >= total_len:
                        continue
                    range_start = offset
                    range_end = offset + total_len

                    overlap_start = max(range_start, window_start)
                    overlap_end = min(range_end, window_end)
                    if overlap_end <= overlap_start:
                        continue
                    slice_offset = max(consumed, overlap_start - range_start)
                    remaining = total_len - slice_offset
                    overlap_available = overlap_end - overlap_start
                    take = min(remaining, overlap_available)
                    if take <= 0:
                        continue
                    target_offset = overlap_start - window_start
                    token_slice = self._get_token_slice(placeholder_idx, seq, slice_offset, take)
                    deepstack_slice = self._get_deepstack_slice(placeholder_idx, seq, slice_offset, take)
                    self._get_slices_for_seq(deepstack_slice, placeholder_idx, slices_for_seq, take, target_offset,
                                             token_slice)
                    has_slices = True
                vision_slices_per_seq.append(slices_for_seq)
            if not has_slices:
                vision_slices_per_seq = None
        return vision_slices_per_seq

    @staticmethod
    def _get_slices_for_seq(deepstack_slice, placeholder_idx, slices_for_seq, take, target_offset, token_slice):
        slices_for_seq.append(
            {
                "tokens": token_slice,
                "deepstack": deepstack_slice,
                "length": take,
                "target_offset": target_offset,
                "placeholder_idx": placeholder_idx,
            }
        )

    def _get_deepstack_slice(self, placeholder_idx, seq, slice_offset, take):
        deepstack_slice: list[torch.Tensor] | None = None
        if seq.cached_deepstack_tokens:
            deepstack_slice = []
            for layer_tokens in seq.cached_deepstack_tokens:
                if placeholder_idx >= len(layer_tokens):
                    deepstack_slice.append(None)
                    continue
                layer_slice = layer_tokens[placeholder_idx][slice_offset:slice_offset + take].to(
                    device=self.device,
                    dtype=self.model_dtype,
                    non_blocking=True,
                ).contiguous()
                deepstack_slice.append(layer_slice)
        return deepstack_slice

    def _get_token_slice(self, placeholder_idx, seq, slice_offset, take):
        chunk_tokens = seq.cached_vision_tokens[placeholder_idx]
        token_slice = chunk_tokens[slice_offset:slice_offset + take].to(
            device=self.device,
            dtype=self.model_dtype,
            non_blocking=True,
        ).contiguous()
        return token_slice

    def _ensure_vision_cache(self, seq: Sequence):
        if seq.cached_vision_tokens is not None:
            return
        if seq.pixel_values is None or seq.image_grid_thw is None:
            seq.cached_vision_tokens = []
            seq.cached_deepstack_tokens = []
            return

        # Run the vision encoder once on the GPU and stash the outputs on CPU.
        # Later prefill iterations reuse these tensors without recomputing the
        # expensive 3D convolutions.
        pixel = seq.pixel_values.to(device=self.device, dtype=self.model_dtype, non_blocking=True).contiguous()
        grid = seq.image_grid_thw.to(device=self.device, dtype=torch.int32, non_blocking=True).contiguous()

        image_embeds, deepstack_features = self.model.visual(pixel, grid)
        seq.cached_vision_tokens = [emb.detach().cpu() for emb in image_embeds]
        if deepstack_features:
            cached_deepstack = []
            for layer_tokens in deepstack_features:
                cached_layer = [feat.detach().cpu() for feat in layer_tokens]
                cached_deepstack.append(cached_layer)
            seq.cached_deepstack_tokens = cached_deepstack
        else:
            seq.cached_deepstack_tokens = []
        seq.pixel_values = None
        seq.image_grid_thw = None
