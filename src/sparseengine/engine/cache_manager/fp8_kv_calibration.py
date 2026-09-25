"""Device-side extrema collection at the actual KV cache write boundary."""

from __future__ import annotations

import torch


class FP8KVCalibrationObserver:
    def __init__(self, num_layers: int, device: torch.device):
        self.key_max = torch.zeros(num_layers, device=device, dtype=torch.float32)
        self.value_max = torch.zeros_like(self.key_max)
        self.token_counts = torch.zeros(num_layers, device=device, dtype=torch.int64)

    def reset(self) -> None:
        self.key_max.zero_()
        self.value_max.zero_()
        self.token_counts.zero_()

    def update(self, layer_idx: int, key: torch.Tensor, value: torch.Tensor,
               write_slots: torch.Tensor) -> None:
        if key.numel() == 0:
            return
        valid = write_slots.ge(0)
        mask = valid[:, None, None]
        key_max = torch.where(mask, key.float().abs(), 0.0).amax()
        value_max = torch.where(mask, value.float().abs(), 0.0).amax()
        self.key_max[layer_idx] = torch.maximum(self.key_max[layer_idx], key_max)
        self.value_max[layer_idx] = torch.maximum(self.value_max[layer_idx], value_max)
        self.token_counts[layer_idx] += valid.sum()
