import torch
from torch import nn


class Sampler(nn.Module):
    def forward(self, logits: torch.Tensor, temperatures: torch.Tensor):
        real_bs = temperatures.shape[0]
        if logits.shape[0] != real_bs:
            logits = logits[:real_bs, :]
        logits = logits.float()
        greedy_mask = temperatures <= 1e-10
        sample_mask = ~greedy_mask
        sampled = torch.empty(real_bs, dtype=torch.long, device=logits.device)
        if greedy_mask.any():
            sampled[greedy_mask] = logits[greedy_mask].argmax(dim=-1)
        if sample_mask.any():
            scaled_logits = logits[sample_mask] / temperatures[sample_mask].unsqueeze(-1)
            # Gumbel-max samples from Categorical(logits=scaled_logits)
            # without materializing a full softmax distribution.
            gumbel = -torch.empty_like(scaled_logits).exponential_().log()
            sampled[sample_mask] = (scaled_logits + gumbel).argmax(dim=-1)
        return sampled
