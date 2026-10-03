"""Базовый epsilon-predictor на MLP для toy-моделей (2D данные).

Почему MLP, а не U-Net: для 2D-вектора (смесь гауссов) пространственной
иерархии нет, MLP — честная минимальная архитектура, на которой видно
сам процесс. U-Net появится на MNIST.

Синусоидальный embedding времени — тот же, что в оригинальной статье DDPM.
Модель должна UЗНАВАТЬ, какой сейчас уровень шума, чтобы предсказать
именно тот шум, что был добавлен.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int = 64):
        super().__init__()
        assert dim % 2 == 0
        self.half_dim = dim // 2

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # t: (B,) → embedding: (B, dim)
        # Частоты логарифмически от 1 до 10000 — классический приём из Transformer'ов
        half = self.half_dim
        freqs = torch.exp(-torch.arange(half, device=t.device, dtype=t.dtype)
                          * (torch.log(torch.tensor(10000.0)) / (half - 1)))
        args = t.view(-1, 1).float() * freqs.view(1, -1)
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


class EpsMLP(nn.Module):
    """Простой U-Net-подобный (residual) MLP: x и embedding времени
    склеиваются, прогоняются через hidden-слои, предсказывается eps той же
    размерности, что и x."""

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
