# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the Nano-vLLM project


import atexit
import os
from dataclasses import fields
from random import randint
from time import perf_counter

import torch
import torch_npu
from transformers import LlamaTokenizerFast, PreTrainedTokenizerFast
import torch.multiprocessing as mp

from nanovllm.config import Config
from nanovllm.sampling_params import SamplingParams
from nanovllm.engine.sequence import Sequence
from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.model_runner import ModelRunner
from nanovllm.utils.logger import init_logger

logger = init_logger(__name__)


DEEPSEEK_V32_CHAT_TEMPLATE = """{% if not add_generation_prompt is defined %}{% set add_generation_prompt = false %}{% endif %}
{% set ns = namespace(system_prompt='') %}
{%- for message in messages %}
    {%- if message['role'] == 'system' %}
        {%- if ns.system_prompt %}
            {% set ns.system_prompt = ns.system_prompt + '\\n\\n' + message['content'] %}
        {%- else %}
            {% set ns.system_prompt = message['content'] %}
        {%- endif %}
    {%- endif %}
{%- endfor %}
{{ bos_token }}{{ ns.system_prompt }}
{%- for message in messages %}
    {%- if message['role'] == 'user' %}
        {{ '<\uFF5CUser\uFF5C>' + message['content'] }}
    {%- elif message['role'] == 'assistant' %}
        {{ '<\uFF5CAssistant\uFF5C>' + message['content'] + eos_token }}
    {%- endif %}
{%- endfor %}
{%- if add_generation_prompt %}
    {{ '<\uFF5CAssistant\uFF5C>' }}
{%- endif %}"""


class LLMEngine:

    def __init__(self, model, **kwargs):
        config_fields = {field.name for field in fields(Config)}
        config_kwargs = {k: v for k, v in kwargs.items() if k in config_fields}
        config = Config(model, **config_kwargs)
        self.block_size = config.kvcache_block_size
        self.config = config
        logger.info(f"config: {config}")
        self.ps = []
        self.events = []
        ctx = mp.get_context("spawn")
        for i in range(1, config.tensor_parallel_size):
            event = ctx.Event()
            process = ctx.Process(target=ModelRunner, args=(config, i, event))
            process.start()
            self.ps.append(process)
            self.events.append(event)
        self.model_runner = ModelRunner(config, 0, self.events)
        self.tokenizer = self._load_tokenizer(config)
        config.eos = self.tokenizer.eos_token_id
        self.scheduler = Scheduler(config)
        if config.skip_warmup:
            logger.info("skip warmup enabled, skipping model warmup.")
        else:
            self.warmup_model()
        atexit.register(self.exit)

    @staticmethod
    def _is_true_env(name: str, default: bool = False) -> bool:
        value = os.environ.get(name)
        if value is None:
            return default
        return value.lower() in ("1", "true", "yes", "on")

    @staticmethod
    def _format_deepseek_prompt(prompt: str, use_chat_template: bool) -> str:
        if not use_chat_template:
            return prompt
        return f"<\uFF5CUser\uFF5C>{prompt}<\uFF5CAssistant\uFF5C>"

    def _encode_string_prompt(self, prompt: str) -> list[int]:
        use_chat_template = self._is_true_env(
            "NANOVLLM_USE_DEEPSEEK_CHAT",
            False,
        )
        formatted_prompt = self._format_deepseek_prompt(prompt, use_chat_template)
        token_ids = self.tokenizer.encode(
            formatted_prompt,
            add_special_tokens=False,
        )
        add_bos = self._is_true_env(
            "NANOVLLM_ADD_BOS",
            use_chat_template,
        )
        bos_token_id = self.tokenizer.bos_token_id
        if add_bos and bos_token_id is not None:
            token_ids = [bos_token_id] + token_ids
        return token_ids

    @staticmethod
    def _load_tokenizer(config: Config):
        try:
            tokenizer = PreTrainedTokenizerFast.from_pretrained(
                config.model,
                trust_remote_code=config.trust_remote_code,
                fix_mistral_regex=False,
            )
        except Exception:
            logger.warning(
                "Falling back to LlamaTokenizerFast for deepseek_v32 export."
            )
            tokenizer = LlamaTokenizerFast.from_pretrained(
                config.model,
                legacy=True,
                fix_mistral_regex=False,
            )
        if not getattr(tokenizer, "chat_template", None):
            tokenizer.chat_template = DEEPSEEK_V32_CHAT_TEMPLATE
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

    def prefill_warmup(self):
        max_num_batched_tokens, max_model_len = self.config.max_num_batched_tokens, self.config.max_model_len
        num_seqs = min(max_num_batched_tokens // max_model_len, self.config.max_num_seqs)
        prompt_token_ids = [[randint(0, 10000) for _ in range(max_model_len)] for _ in range(num_seqs)]
        sampling_params = [SamplingParams(temperature=0.6, ignore_eos=True, max_tokens=1) for
                           _ in range(num_seqs)]
        # prefill max_num_batched_tokens
        self.generate(prompt_token_ids, sampling_params)

    def decode_warmup(self):
        max_num_batched_tokens, max_model_len = self.config.max_num_batched_tokens, self.config.max_model_len
        num_seqs = min(max_num_batched_tokens // max_model_len, self.config.max_num_seqs)
        # decode max_num_seqs
        prompt_token_ids = [[randint(0, 10000) for _ in range(randint(10, 50))] for _ in
                            range(self.config.max_num_seqs)]
        sampling_params = [SamplingParams(temperature=0.6, ignore_eos=True, max_tokens=2) for
                           _ in range(num_seqs)]
        self.generate(prompt_token_ids, sampling_params)

    def warmup_model(self):
        logger.info(f"warmup start !!!!!!")
        start_time = perf_counter()
        self.prefill_warmup()
        self.decode_warmup()
        end_time = perf_counter()
        duration = end_time - start_time
        logger.info(f"warmup end !!!!!!")
        logger.info(f"Successfully finished model warmup in {duration:.2f} seconds.")

    def exit(self):
        self.model_runner.call("exit")
        del self.model_runner
        for p in self.ps:
            p.join()

    def add_request(self,
                    prompt: str | list[int],
                    sampling_params: SamplingParams,
                    request_id: str = None,
                    ):
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
        token_ids = self.model_runner.call("run", seqs, is_prefill)
        self.scheduler.postprocess(seqs, token_ids)
        outputs = [(seq.seq_id, seq.completion_token_ids, seq.num_prompt_tokens, seq.num_cached_tokens) for seq in seqs
                   if seq.is_finished]
        num_tokens = sum(len(seq) for seq in seqs) if is_prefill else -len(seqs)
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
        total_prefill_tokens = 0
        total_decode_tokens = 0
        prefill_steps = 0
        decode_steps = 0
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

            if (
                is_prefill
                or last_is_prefill
                or i_step % 64 == 0
                or self.is_finished()
            ):
                latency_name = "TTFT" if is_prefill else "TPOT"
                print(
                    f"[step{i_step:4d} {'Prefill' if is_prefill else ' Decode'}] "
                    f"bsz={batch_size}, num_tokens={step_tokens}, "
                    f"{latency_name}={step_elapsed:.4f} sec, "
                    f"TPS={step_tps:.2f} tok/s"
                )
            last_is_prefill = is_prefill

            for seq_id, token_ids, prompt_len, cache_tokens in output:
                outputs[seq_id] = (token_ids, prompt_len, cache_tokens)
                total_input_tokens += prompt_len
                total_output_tokens += len(token_ids)

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
        e2e_input_tps = total_input_tokens / elapsed if elapsed > 0 else 0.0
        e2e_output_tps = total_output_tokens / elapsed if elapsed > 0 else 0.0
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
            f"decode TPS = {decode_tps:.2f} tok/s\n"
            f"    e2e input TPS = {e2e_input_tps:.2f} tok/s, "
            f"e2e output TPS = {e2e_output_tps:.2f} tok/s\n"
            f"    TTFT/TPOT are per-step request latencies printed above."
        )
        return outputs
