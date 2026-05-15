import torch
from torch import nn


class Sampler(nn.Module):

    def __init__(self):
        super().__init__()

    @torch.compile
    def _sample(self, logits: torch.Tensor, temperatures: torch.Tensor):
        logits = logits.float()
        greedy_mask = temperatures <= 1e-10
        safe_temperatures = torch.where(greedy_mask, torch.ones_like(temperatures), temperatures)
        sampled_logits = logits.div(safe_temperatures.unsqueeze(dim=1))
        probs = torch.softmax(sampled_logits, dim=-1)
        sample_tokens = probs.div_(torch.empty_like(probs).exponential_(1).clamp_min_(1e-10)).argmax(dim=-1)
        greedy_tokens = logits.argmax(dim=-1)
        return torch.where(greedy_mask, greedy_tokens, sample_tokens)

    def forward(self, logits: torch.Tensor, temperatures: torch.Tensor | None, all_greedy: bool = False):
        if all_greedy:
            return logits.argmax(dim=-1)
        if temperatures is None:
            raise ValueError("temperatures must be provided when all_greedy=False")
        return self._sample(logits, temperatures)
