"""Базовый epsilon-predictor на MLP для toy-моделей (2D данные).
"""
from __future__ import annotations

import torch
import torch.nn as nn

from common.embedding import SinusoidalTimeEmbedding


class EpsMLP(nn.Module):

    def __init__(self, in_dim: int, hidden: int = 256, depth: int = 4, time_dim: int = 64):
        super().__init__()
        self.time_emb = SinusoidalTimeEmbedding(time_dim)
        layers = [nn.Sequential(nn.Linear(in_dim + time_dim, hidden), nn.SiLU())]
        for _ in range(depth - 1):
            layers.append(nn.Sequential(nn.Linear(hidden, hidden), nn.SiLU()))
        layers.append(nn.Linear(hidden, in_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        temb = self.time_emb(t)
        return self.net(torch.cat([x_t, temb], dim=-1))
