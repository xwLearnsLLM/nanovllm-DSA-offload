import torch
from torch import nn


class Sampler(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, logits: torch.Tensor, temperatures: torch.Tensor):
        real_bs = temperatures.shape[0]
        if logits.shape[0] != real_bs:
            logits = logits[:real_bs, :]
        logits = logits.float()
        greedy_mask = temperatures <= 1e-10
        sampled = torch.empty(real_bs, dtype=torch.long, device=logits.device)
        if greedy_mask.any():
            sampled[greedy_mask] = logits[greedy_mask].argmax(dim=-1)
        if (~greedy_mask).any():
            scaled_logits = logits[~greedy_mask] / temperatures[~greedy_mask].unsqueeze(-1)
            probs = torch.softmax(scaled_logits, dim=-1)
            sampled[~greedy_mask] = torch.multinomial(probs, num_samples=1).squeeze(-1)
        return sampled
