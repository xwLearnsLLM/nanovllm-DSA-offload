# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the Nano-vLLM project


import atexit
import os
from dataclasses import fields
from time import monotonic, perf_counter

import torch
import torch.multiprocessing as mp

from nanovllm.config import Config, merge_eos_token_ids
from nanovllm.sampling_params import SamplingParams
from nanovllm.engine.sequence import Sequence
from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.model_runner import ModelRunner
from nanovllm.engine.dsa_offload import OFFLOAD_NONE
from nanovllm.utils.glm_tokenizer import (
    load_glm_tokenizer,
    normalize_token_ids,
)
from nanovllm.utils.logger import init_logger

logger = init_logger(__name__)


class LLMEngine:

    def __init__(self, model, **kwargs):
        config_fields = {field.name for field in fields(Config) if field.init}
        unknown = sorted(set(kwargs) - config_fields)
        if unknown:
            raise TypeError(
                "Unknown LLM configuration argument(s): " + ", ".join(unknown)
            )
        config = Config(model, **kwargs)
        self.block_size = config.kvcache_block_size
        self.config = config
        logger.info(f"config: {config}")
        logger.info(
            "execution mode: prefill=eager, first_decode=eager, "
            "stable_decode=%s, attention=%s, mtp_k=%d",
            "eager" if config.enforce_eager else "full_decode_only",
            config.offload_mode,
            config.num_speculative_tokens,
        )
        # Fail fast on tokenizer/version problems before all TP ranks load the
        # 400+ GB GLM checkpoint.
        self.tokenizer = self._load_tokenizer(config)
        config.eos = merge_eos_token_ids(
            config.eos, self.tokenizer.eos_token_id
        )
        self.ps = []
        self.events = []
        try:
            ctx = mp.get_context("spawn")
            for i in range(1, config.tensor_parallel_size):
                event = ctx.Event()
                process = ctx.Process(
                    target=ModelRunner,
                    args=(config, i, event),
                )
                process.start()
                self.ps.append(process)
                self.events.append(event)
            self.model_runner = ModelRunner(config, 0, self.events)
            self.scheduler = Scheduler(config)
            self._last_prefill_chunk_progress = None
            self._last_speculative_stats = None
            # Profiling is created lazily in the rank-0 engine. Eager mode
            # starts at its first decode; graph mode waits until lazy capture
            # and the first replay have completed, so profiling cannot skew
            # one TP rank during graph construction.
            self._decode_profile_output = os.environ.get(
                "NANOVLLM_PROFILE_DECODE_OUTPUT",
                "",
            ).strip()
            self._decode_profiler = None
        except BaseException:
            self._cleanup_failed_initialization()
            raise
        atexit.register(self.exit)

    def _cleanup_failed_initialization(self) -> None:
        """Best-effort cleanup for errors after TP workers have started."""

        logger.error(
            "LLM initialization failed; terminating %d TP worker(s).",
            len(self.ps),
        )
        for process in self.ps:
            try:
                if process.is_alive():
                    process.terminate()
            except Exception:
                logger.exception("Failed to terminate a TP worker.")
        terminate_deadline = monotonic() + 5.0
        for process in self.ps:
            try:
                process.join(
                    timeout=max(0.0, terminate_deadline - monotonic())
                )
            except Exception:
                logger.exception("Failed to join a TP worker.")
        for process in self.ps:
            try:
                if process.is_alive():
                    process.kill()
            except Exception:
                logger.exception("Failed to kill a TP worker.")
        kill_deadline = monotonic() + 2.0
        for process in self.ps:
            try:
                process.join(timeout=max(0.0, kill_deadline - monotonic()))
            except Exception:
                logger.exception("Failed to reap a TP worker.")

        runner = getattr(self, "model_runner", None)
        shm = getattr(runner, "shm", None)
        if shm is not None:
            try:
                shm.close()
            except Exception:
                logger.exception("Failed to close TP shared memory.")
            try:
                shm.unlink()
            except FileNotFoundError:
                pass
            except Exception:
                logger.exception("Failed to unlink TP shared memory.")

    @staticmethod
    def _is_true_env(name: str, default: bool = False) -> bool:
        value = os.environ.get(name)
        if value is None:
            return default
        return value.lower() in ("1", "true", "yes", "on")

    def _encode_string_prompt(self, prompt: str) -> list[int]:
        token_ids = normalize_token_ids(
            self.tokenizer.encode(
                prompt,
                add_special_tokens=False,
            )
        )
        add_bos = self._is_true_env(
            "NANOVLLM_ADD_BOS",
            False,
        )
        bos_token_id = self.tokenizer.bos_token_id
        if add_bos and bos_token_id is not None:
            token_ids = [bos_token_id] + token_ids
        return token_ids

    @staticmethod
    def _load_tokenizer(config: Config):
        tokenizer = load_glm_tokenizer(
            config.model,
            trust_remote_code=config.trust_remote_code,
        )
        if not getattr(tokenizer, "chat_template", None):
            template_path = os.path.join(
                config.model, "chat_template.jinja"
            )
            if not os.path.isfile(template_path):
                raise ValueError(
                    "GLM tokenizer has no embedded chat template and "
                    "chat_template.jinja is missing from the model directory."
                )
            with open(template_path, "r", encoding="utf-8") as file:
                tokenizer.chat_template = file.read()
        return tokenizer

    def _decode_token_ids(self, token_ids: list[int]) -> str:
        text = self.tokenizer.decode(token_ids)
        if "Ġ" not in text and "Ċ" not in text:
            return text

        try:
            tokens = self.tokenizer.convert_ids_to_tokens(token_ids)
            decoded = self.tokenizer.convert_tokens_to_string(tokens)
            if "Ġ" not in decoded and "Ċ" not in decoded:
                return decoded
        except Exception:
            pass

        backend = getattr(self.tokenizer, "backend_tokenizer", None)
        decoder = getattr(backend, "decoder", None)
        if decoder is not None:
            try:
                tokens = self.tokenizer.convert_ids_to_tokens(token_ids)
                decoded = decoder.decode(tokens)
                if "Ġ" not in decoded and "Ċ" not in decoded:
                    return decoded
            except Exception:
                pass

        return (
            text.replace("Ċ", "\n")
            .replace("ĉ", "\n")
            .replace("Ġ", " ")
        )

    def _start_decode_profiler(self) -> None:
        if not self._decode_profile_output or self._decode_profiler is not None:
            return

        import torch_npu

        output_dir = os.path.abspath(self._decode_profile_output)
        experimental_config = torch_npu.profiler._ExperimentalConfig(
            profiler_level=torch_npu.profiler.ProfilerLevel.Level1,
            aic_metrics=torch_npu.profiler.AiCMetrics.AiCoreNone,
            data_simplification=False,
        )
        self._decode_profiler = torch_npu.profiler.profile(
            activities=[
                torch_npu.profiler.ProfilerActivity.CPU,
                torch_npu.profiler.ProfilerActivity.NPU,
            ],
            on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(
                output_dir
            ),
            record_shapes=False,
            profile_memory=False,
            with_stack=False,
            with_modules=False,
            with_flops=False,
            experimental_config=experimental_config,
        )
        logger.info(
            "Starting TP rank-0 profiler before %s: %s",
            (
                "the first eager decode step"
                if self.config.enforce_eager
                else "a stable FULL_DECODE_ONLY replay"
            ),
            output_dir,
        )
        self._decode_profiler.start()

    def _decode_profile_ready(self, batch_size: int) -> bool:
        if not self._decode_profile_output or self._decode_profiler is not None:
            return False
        if self.config.enforce_eager:
            return True
        manager = self.model_runner.decode_graph_manager
        return bool(
            manager is not None
            and manager.is_stable_replay_ready(batch_size)
        )

    def _stop_decode_profiler(self) -> None:
        profiler = self._decode_profiler
        if profiler is None:
            return
        self._decode_profiler = None
        torch.npu.synchronize()
        profiler.stop()
        logger.info(
            "Stopped TP rank-0 decode profiler; profile data: %s",
            os.path.abspath(self._decode_profile_output),
        )

    def exit(self):
        self._stop_decode_profiler()
        self.model_runner.call("exit")
        del self.model_runner
        for p in self.ps:
            p.join()

    def add_request(self,
                    prompt: str | list[int],
                    sampling_params: SamplingParams,
                    request_id: str = None,
                    ):
        if (
            self.config.num_speculative_tokens
            and sampling_params.temperature > 1e-10
        ):
            raise ValueError(
                "GLM MTP phase 1 supports greedy sampling only; set "
                "temperature=0."
            )
        if isinstance(prompt, str):
            prompt = self._encode_string_prompt(prompt)
        seq = Sequence(
            prompt,
            sampling_params,
            request_id=request_id,
            block_size=self.block_size
        )
        self.scheduler.add(seq)

    def step(self, return_stats: bool = False):
        seqs, is_prefill = self.scheduler.schedule()
        self._last_prefill_chunk_progress = None
        is_final_prefill_step = is_prefill
        if is_prefill and self.config.prefill_chunk_size:
            num_tokens = sum(seq.num_scheduled_tokens for seq in seqs)
            seq = seqs[0]
            progress = (
                seq.num_prefill_tokens_processed + seq.num_scheduled_tokens
            )
            self._last_prefill_chunk_progress = (progress, len(seq))
            is_final_prefill_step = progress == len(seq)
        else:
            num_tokens = sum(len(seq) for seq in seqs) if is_prefill else -len(seqs)
        if not is_prefill and self._decode_profile_ready(len(seqs)):
            self._start_decode_profiler()
        token_ids = self.model_runner.call("run", seqs, is_prefill)
        if is_final_prefill_step:
            self.scheduler.release_prefill_hbm_blocks(seqs)
        self.scheduler.postprocess(seqs, token_ids, is_prefill)
        self._last_speculative_stats = self.scheduler.last_speculative_stats
        if not is_prefill and self._last_speculative_stats is not None:
            num_tokens = -self._last_speculative_stats["emitted_tokens"]
        outputs = [(seq.seq_id, seq.completion_token_ids, seq.num_prompt_tokens, seq.num_cached_tokens) for seq in seqs
                   if seq.is_finished]
        if return_stats:
            return outputs, num_tokens, len(seqs), is_prefill
        return outputs, num_tokens

    def abort_request(self, request_id: str) -> None:
        """Aborts a request with the given ID.

        Args:
            request_id: The ID of the request to abort.
        """
        self.scheduler.abort_seq_group(request_id)

    def is_finished(self):
        return self.scheduler.is_finished()

    def generate(
            self,
            prompts: list[str] | list[list[int]],
            sampling_params: SamplingParams | list[SamplingParams],
    ) -> list[str]:
        total_prompts = len(prompts)
        if not isinstance(sampling_params, list):
            sampling_params = [sampling_params] * len(prompts)
        for prompt, sp in zip(prompts, sampling_params):
            self.add_request(prompt, sp)

        print(f"llm.generate {total_prompts} start ...")
        outputs = {}
        total_input_tokens = 0
        total_output_tokens = 0
        total_start = perf_counter()
        total_prefill_time = 0.0
        total_decode_time = 0.0
        total_decode_request_time = 0.0
        total_prefill_tokens = 0
        total_decode_tokens = 0
        prefill_steps = 0
        decode_steps = 0
        total_accepted_drafts = 0
        total_proposed_drafts = 0
        last_is_prefill = False
        i_step = 0
        while not self.is_finished():
            step_start = perf_counter()
            output, num_tokens, batch_size, is_prefill = self.step(return_stats=True)
            step_elapsed = perf_counter() - step_start
            i_step += 1

            step_tokens = num_tokens if is_prefill else -num_tokens
            step_tps = step_tokens / max(step_elapsed, 1e-9)
            if is_prefill:
                prefill_steps += 1
                total_prefill_tokens += step_tokens
                total_prefill_time += step_elapsed
            else:
                decode_steps += 1
                total_decode_tokens += step_tokens
                total_decode_time += step_elapsed
                total_decode_request_time += step_elapsed * batch_size
                spec_stats = self._last_speculative_stats
                if spec_stats is not None:
                    total_accepted_drafts += spec_stats["accepted_drafts"]
                    total_proposed_drafts += spec_stats["proposed_drafts"]

            if (
                is_prefill
                or last_is_prefill
                or i_step % 1 == 0
                or self.is_finished()
            ):
                progress_text = ""
                if is_prefill and self._last_prefill_chunk_progress is not None:
                    latency_name = "PREFILL_STEP"
                    progress, prompt_length = self._last_prefill_chunk_progress
                    progress_text = f", progress={progress}/{prompt_length}"
                else:
                    latency_name = "TTFT" if is_prefill else "TPOT"
                (hbm_used, hbm_total), (dram_used, dram_total), (
                    index_used,
                    index_total,
                ) = self.scheduler.cache_block_usage()
                cache_text = f"HBM_KV={hbm_used}/{hbm_total}, "
                if self.config.offload_mode != OFFLOAD_NONE:
                    cache_text += (
                        f"DRAM_KV={dram_used}/{dram_total}, "
                        f"HBM_INDEX={index_used}/{index_total}, "
                    )
                spec_text = ""
                step_latency_text = ""
                latency_value = step_elapsed
                if not is_prefill and self._last_speculative_stats is not None:
                    emitted = self._last_speculative_stats["emitted_tokens"]
                    mean_emitted = emitted / max(batch_size, 1)
                    latency_value = step_elapsed / max(mean_emitted, 1e-9)
                    proposed = self._last_speculative_stats["proposed_drafts"]
                    accepted = self._last_speculative_stats["accepted_drafts"]
                    step_latency_text = (
                        f"step_latency={step_elapsed:.4f} sec, "
                    )
                    spec_text = f", accepted_drafts={accepted}/{proposed}"
                print(
                    f"[step{i_step:4d} {'Prefill' if is_prefill else ' Decode'}] "
                    f"bsz={batch_size}, num_tokens={step_tokens}{progress_text}, "
                    f"{cache_text}"
                    f"{step_latency_text}"
                    f"{latency_name}={latency_value:.4f} sec, "
                    f"TPS={step_tps:.2f} tok/s{spec_text}"
                )
            last_is_prefill = is_prefill

            for seq_id, token_ids, prompt_len, cache_tokens in output:
                outputs[seq_id] = (token_ids, prompt_len, cache_tokens)
                total_input_tokens += prompt_len
                total_output_tokens += len(token_ids)

        self._stop_decode_profiler()
        outputs = [outputs[seq_id] for seq_id in sorted(outputs.keys())]
        outputs = [{"text": self._decode_token_ids(token_ids), "token_ids": token_ids, "prompt_len": prompt_len,
                    "cache_tokens": cache_tokens} for
                   token_ids, prompt_len, cache_tokens in outputs]
        elapsed = perf_counter() - total_start
        prefill_tps = (
            total_prefill_tokens / total_prefill_time
            if total_prefill_time > 0
            else 0.0
        )
        decode_tps = (
            total_decode_tokens / total_decode_time
            if total_decode_time > 0
            else 0.0
        )
        if self.config.num_speculative_tokens:
            decode_mean_tpot = (
                total_decode_request_time / total_decode_tokens
                if total_decode_tokens > 0
                else 0.0
            )
        else:
            # Preserve the historical K=0 statistic exactly.
            decode_mean_tpot = (
                total_decode_time / decode_steps if decode_steps else 0.0
            )
        acceptance_rate = (
            total_accepted_drafts / total_proposed_drafts
            if total_proposed_drafts
            else 0.0
        )
        e2e_input_tps = total_input_tokens / elapsed if elapsed > 0 else 0.0
        e2e_output_tps = total_output_tokens / elapsed if elapsed > 0 else 0.0
        graph_stats = self.model_runner.call("get_decode_graph_stats")
        print(
            f"llm.generate {total_prompts} requests in {i_step} steps, "
            f"e2e latency = {elapsed:.2f} sec\n"
            f"    prefill steps = {prefill_steps}, "
            f"prefill tokens = {total_prefill_tokens}, "
            f"total prefill time = {total_prefill_time:.4f} sec, "
            f"prefill TPS = {prefill_tps:.2f} tok/s\n"
            f"    decode steps = {decode_steps}, "
            f"decode tokens = {total_decode_tokens}, "
            f"total decode time = {total_decode_time:.4f} sec, "
            f"decode mean TPOT = {decode_mean_tpot:.4f} sec, "
            f"decode TPS = {decode_tps:.2f} tok/s\n"
            + (
                f"    MTP accepted drafts = {total_accepted_drafts}/"
                f"{total_proposed_drafts}, acceptance rate = "
                f"{acceptance_rate:.4f}\n"
                if self.config.num_speculative_tokens
                else ""
            )
            + f"    e2e input TPS = {e2e_input_tps:.2f} tok/s, "
            f"e2e output TPS = {e2e_output_tps:.2f} tok/s\n"
            + (
                "    PREFILL_STEP is one chunk latency; TPOT is the per-step "
                "decode latency printed above."
                if self.config.prefill_chunk_size
                else "    TTFT/TPOT are per-step request latencies printed above."
            )
        )
        if graph_stats.get("enabled"):
            print(
                "    FULL_DECODE_ONLY proof: "
                f"offload_mode={graph_stats['offload_mode']}, "
                f"capture_sizes={graph_stats['capture_sizes']}, "
                f"captures={graph_stats['captures']}, "
                f"replays={graph_stats['replays']}, "
                f"eager_first_decode={graph_stats['eager_first_decode']}, "
                f"eager_no_dsa={graph_stats['eager_no_dsa']}, "
                f"eager_lidu_uninitialized="
                f"{graph_stats['eager_lidu_uninitialized']}, "
                f"eager_uncaptured_batch={graph_stats['eager_uncaptured_batch']}"
            )
            print(
                "    Decode hot path: "
                f"compact_ipc_steps={graph_stats['compact_ipc_steps']}, "
                f"average_ipc_bytes={graph_stats['average_ipc_bytes']}, "
                f"ipc_snapshots={graph_stats['ipc_snapshot_steps']}, "
                f"snapshot_avg_bytes="
                f"{graph_stats['ipc_snapshot_avg_bytes']}, "
                f"ipc_deltas={graph_stats['ipc_delta_steps']}, "
                f"delta_avg_bytes={graph_stats['ipc_delta_avg_bytes']}, "
                f"metadata_cache_hits={graph_stats['metadata_cache_hits']}, "
                f"metadata_cache_misses={graph_stats['metadata_cache_misses']}, "
                f"graph_metadata_refreshes={graph_stats['metadata_refreshes']}, "
                f"graph_metadata_reuses={graph_stats['metadata_reuses']}"
            )
            if "mtp_target_replays" in graph_stats:
                print(
                    "    MTP graph proof: "
                    f"target_captures={graph_stats['mtp_target_captures']}, "
                    f"draft_captures={graph_stats['mtp_draft_captures']}, "
                    f"target_replays={graph_stats['mtp_target_replays']}, "
                    f"draft_replays={graph_stats['mtp_draft_replays']}, "
                    f"eager_capture={graph_stats['eager_mtp_capture']}"
                )
        return outputs
