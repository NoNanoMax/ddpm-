"""Sinusoidal time embedding (общий для всех моделей)."""
import torch
import torch.nn as nn


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int = 64):
        super().__init__()
        assert dim % 2 == 0
        self.half_dim = dim // 2

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.half_dim
        freqs = torch.exp(-torch.arange(half, device=t.device, dtype=t.dtype)
                          * (torch.log(torch.tensor(10000.0)) / (half - 1)))
        args = t.view(-1, 1).float() * freqs.view(1, -1)
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
