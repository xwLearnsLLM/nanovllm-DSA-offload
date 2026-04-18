# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the Nano-vLLM project


import atexit
import os
from dataclasses import fields
from random import randint
from time import perf_counter

import torch
import torch_npu
from torch import Tensor
from tqdm.auto import tqdm
from transformers import AutoTokenizer, LlamaTokenizerFast
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
        {{ '<｜User｜>' + message['content'] }}
    {%- elif message['role'] == 'assistant' %}
        {{ '<｜Assistant｜>' + message['content'] + eos_token }}
    {%- endif %}
{%- endfor %}
{%- if add_generation_prompt %}
    {{ '<｜Assistant｜>' }}
{%- endif %}"""


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
    def _is_deepseek_v32(config: Config) -> bool:
        return getattr(config.hf_config, "model_type", None) == "deepseek_v32"

    @classmethod
    def _tokenizer_load_kwargs(cls, config: Config) -> dict:
        kwargs = {
            "trust_remote_code": config.trust_remote_code,
        }
        if cls._is_deepseek_v32(config):
            kwargs["fix_mistral_regex"] = True
        return kwargs

    @staticmethod
    def _format_deepseek_prompt(prompt: str, use_chat_template: bool) -> str:
        if not use_chat_template:
            return prompt
        return f"<｜User｜>{prompt}<｜Assistant｜>"

    def _encode_string_prompt(self, prompt: str) -> list[int]:
        if not self._is_deepseek_v32(self.config):
            return self.tokenizer.encode(prompt)

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

    @classmethod
    def _load_tokenizer(cls, config: Config):
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                config.model,
                use_fast=True,
                **cls._tokenizer_load_kwargs(config),
            )
        except Exception:
            if getattr(config.hf_config, "model_type", None) != "deepseek_v32":
                raise
            logger.warning(
                "Falling back to LlamaTokenizerFast for deepseek_v32 export."
            )
            tokenizer = LlamaTokenizerFast.from_pretrained(
                config.model,
                legacy=True,
                fix_mistral_regex=True,
            )
        if (
            getattr(config.hf_config, "model_type", None) == "deepseek_v32"
            and not getattr(tokenizer, "chat_template", None)
        ):
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
        self.generate(prompt_token_ids, sampling_params, False)

    def decode_warmup(self):
        max_num_batched_tokens, max_model_len = self.config.max_num_batched_tokens, self.config.max_model_len
        num_seqs = min(max_num_batched_tokens // max_model_len, self.config.max_num_seqs)
        # decode max_num_seqs
        prompt_token_ids = [[randint(0, 10000) for _ in range(randint(10, 50))] for _ in
                            range(self.config.max_num_seqs)]
        sampling_params = [SamplingParams(temperature=0.6, ignore_eos=True, max_tokens=2) for
                           _ in range(num_seqs)]
        self.generate(prompt_token_ids, sampling_params, False)

    def warmup_model(self):
        logger.info(f"warmup start !!!!!!")
        start_time = perf_counter()
        self.prefill_warmup()
        if self.config.enforce_eager:
            self.decode_warmup()
        else:
            cache_dir = os.path.join(os.getcwd(), ".torchair_cache")
            has_cache = os.path.exists(cache_dir) and os.listdir(cache_dir)
            if has_cache:
                logger.info(f"Graph cache found at {cache_dir}.")
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
                    images=None,
                    pixel_values=None,
                    image_grid_thw=None,
                    vision_counts=None,
                    vision_placeholders=None,
                    ):
        if isinstance(prompt, str):
            prompt = self._encode_string_prompt(prompt)
        seq = Sequence(
            prompt,
            sampling_params,
            request_id=request_id,
            images=images,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            vision_counts=vision_counts,
            vision_placeholders=vision_placeholders,
            block_size=self.block_size
        )
        self.scheduler.add(seq)

    def step(self):
        seqs, is_prefill = self.scheduler.schedule()
        token_ids = self.model_runner.call("run", seqs, is_prefill)
        self.scheduler.postprocess(seqs, token_ids)
        outputs = [(seq.seq_id, seq.completion_token_ids, seq.num_prompt_tokens, seq.num_cached_tokens) for seq in seqs
                   if seq.is_finished]
        num_tokens = sum(len(seq) for seq in seqs) if is_prefill else -len(seqs)
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
            use_tqdm: bool = True,
    ) -> list[str]:
        total_prompts = len(prompts)
        if use_tqdm:
            pbar = tqdm(
                total=total_prompts,
                desc="Processed prompts",
                dynamic_ncols=True,
                postfix=(
                    f"est. speed input: {0:.2f} toks/s, output: {0:.2f} toks/s"
                ),
            )
        if not isinstance(sampling_params, list):
            sampling_params = [sampling_params] * len(prompts)
        for prompt, sp in zip(prompts, sampling_params):
            self.add_request(prompt, sp)
        outputs = {}
        total_input_tokens = 0
        total_output_tokens = 0
        total_start = perf_counter()
        engine_prefill_time = 0.0
        engine_decode_time = 0.0
        engine_prefill_tokens = 0
        engine_decode_tokens = 0
        prefill_steps = 0
        decode_steps = 0
        while not self.is_finished():
            step_start = perf_counter()
            output, num_tokens = self.step()
            step_elapsed = perf_counter() - step_start
            if num_tokens > 0:
                prefill_steps += 1
                engine_prefill_tokens += num_tokens
                engine_prefill_time += step_elapsed
            else:
                decode_steps += 1
                engine_decode_tokens += -num_tokens
                engine_decode_time += step_elapsed
            for seq_id, token_ids, prompt_len, cache_tokens in output:
                outputs[seq_id] = (token_ids, prompt_len, cache_tokens)
                if use_tqdm:
                    total_input_tokens += prompt_len
                    total_output_tokens += len(token_ids)
                    elapsed = pbar.format_dict.get("elapsed")
                    if not elapsed:
                        elapsed = perf_counter() - total_start
                    elapsed = max(elapsed, 1e-9)
                    pbar.set_postfix_str(
                        f"est. speed input: {total_input_tokens / elapsed:.2f} "
                        f"toks/s, output: {total_output_tokens / elapsed:.2f} toks/s"
                    )
                    pbar.update(1)
            if use_tqdm and pbar.n == total_prompts:
                pbar.refresh()
        outputs = [outputs[seq_id] for seq_id in sorted(outputs.keys())]
        outputs = [{"text": self._decode_token_ids(token_ids), "token_ids": token_ids, "prompt_len": prompt_len,
                    "cache_tokens": cache_tokens} for
                   token_ids, prompt_len, cache_tokens in outputs]
        if self._is_true_env("NANOVLLM_PERF_SUMMARY", False):
            elapsed = max(perf_counter() - total_start, 1e-9)
            logger.info(
                "Perf summary: requests=%d elapsed=%.2fs "
                "est_input=%.2f toks/s est_output=%.2f toks/s "
                "engine_prefill=%.2f toks/s engine_decode=%.2f toks/s "
                "prefill_steps=%d decode_steps=%d",
                total_prompts,
                elapsed,
                total_input_tokens / elapsed,
                total_output_tokens / elapsed,
                (
                    engine_prefill_tokens / engine_prefill_time
                    if engine_prefill_time > 0
                    else 0.0
                ),
                (
                    engine_decode_tokens / engine_decode_time
                    if engine_decode_time > 0
                    else 0.0
                ),
                prefill_steps,
                decode_steps,
            )
        if use_tqdm:
            pbar.close()
        return outputs

    def generate_multimodal(
            self,
            requests: list[dict],
            sampling_params: SamplingParams | list[SamplingParams],
            processor,
            use_tqdm: bool = True,
    ) -> list[str]:
        if use_tqdm:
            pbar = tqdm(
                total=len(requests),
                desc="Processed prompts",
                dynamic_ncols=True,
                postfix=(
                    f"est. speed input: {0:.2f} toks/s, output: {0:.2f} toks/s"
                ),
            )

        self._mm_add_request(processor, requests, sampling_params)

        outputs = {}
        total_input_tokens = 0
        total_output_tokens = 0
        total_start = perf_counter()
        while not self.is_finished():
            output, num_tokens = self.step()
            for seq_id, token_ids, _, _ in output:
                outputs[seq_id] = token_ids
                if use_tqdm:
                    total_output_tokens += len(token_ids)
                    elapsed = pbar.format_dict.get("elapsed")
                    if not elapsed:
                        elapsed = perf_counter() - total_start
                    elapsed = max(elapsed, 1e-9)
                    pbar.set_postfix_str(
                        f"est. speed input: {total_input_tokens / elapsed:.2f} "
                        f"toks/s, output: {total_output_tokens / elapsed:.2f} toks/s"
                    )
                    pbar.update(1)
            if use_tqdm and pbar.n == len(requests):
                pbar.refresh()

        outputs = [outputs[seq_id] for seq_id in sorted(outputs.keys())]
        results = self.get_mm_results(outputs)
        if use_tqdm:
            pbar.close()
        return results

    def _expand_vision_placeholders(
            self,
            input_ids: list[int],
            image_grid_thw: torch.Tensor,
    ) -> tuple[list[int], list[int], list[tuple[int, int]]]:
        """Expand vision placeholders according to the vision grid metadata."""
        hf_config = self.model_runner.config.hf_config
        vision_config = hf_config.vision_config
        merge_size = vision_config.spatial_merge_size

        image_token_id = getattr(hf_config, "image_token_id", None)
        vision_start_token_id = getattr(hf_config, "vision_start_token_id", None)
        vision_end_token_id = getattr(hf_config, "vision_end_token_id", None)

        if None in (image_token_id, vision_start_token_id, vision_end_token_id):
            raise ValueError("Missing vision placeholder token ids in the config")

        if image_grid_thw.dim() != 2 or image_grid_thw.size(-1) != 3:
            raise ValueError("image_grid_thw must have shape [num_images, 3]")

        grids = image_grid_thw.tolist()
        expected_counts = [int(t * h * w // (merge_size ** 2)) for t, h, w in grids]

        new_input_ids: list[int] = []
        i = 0
        image_idx = 0
        total_images = len(expected_counts)
        length = len(input_ids)

        placeholder_ranges: list[tuple[int, int]] = []

        while i < length:
            token = input_ids[i]
            if token == vision_start_token_id and image_idx < total_images:
                new_input_ids.append(token)
                i += 1
                # Skip original contents until matching vision_end_token_id
                while i < length and input_ids[i] != vision_end_token_id:
                    i += 1
                if i == length:
                    raise ValueError("vision_start_token does not have a matching vision_end_token")

                required = expected_counts[image_idx]
                start_offset = len(new_input_ids)
                new_input_ids.extend([image_token_id] * required)
                new_input_ids.append(vision_end_token_id)
                placeholder_ranges.append((start_offset, required))
                i += 1  # Skip the original vision_end token
                image_idx += 1
            else:
                new_input_ids.append(token)
                i += 1

        if image_idx != total_images:
            raise ValueError(f"{total_images - image_idx} images do not have matching placeholders")

        return new_input_ids, expected_counts, placeholder_ranges

    def get_mm_results(self, outputs):
        results = [
            {
                # Decode without special tokens so the response is clean.
                "text": self.tokenizer.decode(
                    token_ids,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                ),
                "token_ids": token_ids,
            }
            for token_ids in outputs
        ]
        return results

    def _mm_add_request(self, processor, requests, sampling_params):
        if not isinstance(sampling_params, list):
            sampling_params = [sampling_params] * len(requests)
        for request, sp in zip(requests, sampling_params):
            messages = request.get("messages")
            text = request.get("text")
            images = request.get("images")

            images, text = self._text_is_none(images, messages, processor, text)

            if images is not None and not isinstance(images, (list, tuple)):
                images = [images]

            processor_kwargs = {
                "text": [text],
                "return_tensors": "pt",
                "padding": True,
            }
            if images:
                # Let the processor handle image normalization + batching.
                processor_kwargs["images"] = images

            processor_outputs = processor(**processor_kwargs)

            input_ids = processor_outputs["input_ids"][0].tolist()
            pixel_values = processor_outputs.get("pixel_values")
            image_grid_thw = processor_outputs.get("image_grid_thw")

            vision_counts = []
            vision_placeholders = []
            if image_grid_thw is not None:
                image_grid_thw: Tensor = image_grid_thw.squeeze(0) if image_grid_thw.dim() == 3 else image_grid_thw
                (expanded_input_ids, vision_counts, vision_placeholders) \
                    = self._expand_vision_placeholders(input_ids, image_grid_thw)
                input_ids = expanded_input_ids

            if pixel_values is not None:
                # Move vision features to CPU; ModelRunner will re-upload.
                pixel_values = pixel_values.contiguous().cpu()

            if image_grid_thw is not None:
                image_grid_thw = image_grid_thw.contiguous().cpu()

            self.add_request(
                input_ids,
                sp,
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
                vision_counts=vision_counts,
                vision_placeholders=vision_placeholders,
            )

    @staticmethod
    def _text_is_none(images, messages, processor, text):
        if text is None:
            if messages is None:
                raise ValueError(
                    "multimodal request requires 'text' or 'messages'"
                )

            text = processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

            if images is None:
                extracted_images = []
                for message in messages:
                    for content in message.get("content", []):
                        is_image = content.get("type") == "image"
                        has_payload = "image" in content
                        if is_image and has_payload:
                            extracted_images.append(content["image"])
                images = extracted_images if extracted_images else None
        return images, text
