import os

import torch
from torch import nn

from nanovllm.utils.logger import init_logger


logger = init_logger(__name__)


class Sampler(nn.Module):
    def __init__(self):
        super().__init__()
        self.debug_first_step = (
            os.environ.get("NANOVLLM_DEBUG_FIRST_STEP", "0").lower()
            in ("1", "true", "yes", "on")
        )
        self.debug_topk = int(os.environ.get("NANOVLLM_DEBUG_TOPK", "10"))
        self.debug_num_seqs = int(
            os.environ.get("NANOVLLM_DEBUG_NUM_SEQS", "2")
        )
        self._debug_logged = False

    def _maybe_log_topk(
        self,
        logits: torch.Tensor,
        temperatures: torch.Tensor,
    ) -> None:
        if self._debug_logged or not self.debug_first_step:
            return

        num_seqs = min(int(logits.shape[0]), self.debug_num_seqs)
        topk = min(int(logits.shape[-1]), self.debug_topk)
        for seq_idx in range(num_seqs):
            temperature = float(temperatures[seq_idx].item())
            seq_logits = logits[seq_idx]
            if temperature <= 1e-10:
                score_view = seq_logits
                score_name = "logits"
            else:
                score_view = torch.softmax(seq_logits / temperature, dim=-1)
                score_name = "probs"
            values, indices = torch.topk(score_view, k=topk, dim=-1)
            logger.info(
                "debug first-step seq=%s temperature=%s topk_ids=%s topk_%s=%s",
                seq_idx,
                temperature,
                indices.tolist(),
                score_name,
                [round(float(value), 6) for value in values.tolist()],
            )
        self._debug_logged = True

    def forward(self, logits: torch.Tensor, temperatures: torch.Tensor):
        real_bs = temperatures.shape[0]
        if logits.shape[0] != real_bs:
            logits = logits[:real_bs, :]
        logits = logits.float()
        self._maybe_log_topk(logits, temperatures)
        greedy_mask = temperatures <= 1e-10
        sampled = torch.empty(real_bs, dtype=torch.long, device=logits.device)
        if greedy_mask.any():
            sampled[greedy_mask] = logits[greedy_mask].argmax(dim=-1)
        if (~greedy_mask).any():
            scaled_logits = logits[~greedy_mask] / temperatures[~greedy_mask].unsqueeze(-1)
            probs = torch.softmax(scaled_logits, dim=-1)
            sampled[~greedy_mask] = torch.multinomial(probs, num_samples=1).squeeze(-1)
        return sampled
