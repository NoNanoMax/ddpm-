"""Compact DDPM U-Net for 64x64 images.

Классика из DDPM-статьи, упрощённая:
  - base channels 128, 4 уровня
  - timestep через embedding → добавляется в каждый block
  - attention на среднем уровне (один head)
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from common.embedding import SinusoidalTimeEmbedding


class SinEmb(nn.Module):
    def __init__(self, dim: int, project_dim: int | None = None):
        super().__init__()
        self.emb = SinusoidalTimeEmbedding(dim)
        self.proj = nn.Sequential(
            nn.Linear(dim, project_dim or dim), nn.SiLU(),
            nn.Linear(project_dim or dim, project_dim or dim),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return self.proj(self.emb(t))


class Block(nn.Module):
    """GN → SiLU → Conv → (timestep add) → GN → SiLU → Conv → residual."""

    def __init__(self, cin: int, cout: int, time_dim: int, dropout: float = 0.1):
        super().__init__()
        self.n1 = nn.GroupNorm(8, cin)
        self.c1 = nn.Conv2d(cin, cout, 3, padding=1)
        self.n2 = nn.GroupNorm(8, cout)
        self.c2 = nn.Conv2d(cout, cout, 3, padding=1)
        self.act = nn.SiLU()
        self.time = nn.Linear(time_dim, cout)
        self.drop = nn.Dropout(dropout)
        self.skip = nn.Conv2d(cin, cout, 1) if cin != cout else nn.Identity()

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        h = self.c1(self.act(self.n1(x)))
        h = h + self.time(t)[:, :, None, None]
        h = self.c2(self.act(self.n2(h)))
        return self.drop(h) + self.skip(x)


class Attention(nn.Module):
    """Single-head spatial attention: QKV на (H*W) токенах."""

    def __init__(self, dim: int):
        super().__init__()
        self.qkv = nn.Conv2d(dim, 3 * dim, 1)
        self.proj = nn.Conv2d(dim, dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=1)
        q = q.reshape(B, C, -1).permute(0, 2, 1)   # B, HW, C
        k = k.reshape(B, C, -1)                    # B, C, HW
        v = v.reshape(B, C, -1).permute(0, 2, 1)   # B, HW, C
        a = q @ k / math.sqrt(C)                   # B, HW, HW
        a = F.softmax(a, dim=-1)
        h = (a @ v).permute(0, 2, 1).reshape(B, C, H, W)
        return x + self.proj(h)


class Down(nn.Module):
    def __init__(self, cin: int, cout: int):
        super().__init__()
        self.net = nn.Sequential(nn.SiLU(), nn.Conv2d(cin, cout, 3, stride=2, padding=1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Up(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.SiLU(), nn.Conv2d(dim, dim, 3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class UNet(nn.Module):
    """in 1ch (grayscale) → out 1ch (epsilon). base=128, 4 уровня."""

    def __init__(self, base: int = 128, time_dim: int = 256, ch_in: int = 1):
        super().__init__()
        chs = [base // 2, base, base * 2, base * 2]
        self.time = SinEmb(64, time_dim)
        self.stem = nn.Conv2d(ch_in, chs[0], 3, padding=1)

        self.down = nn.ModuleList([
            nn.ModuleList([Block(chs[i], chs[i], time_dim) for _ in range(2)])
            for i in range(3)
        ])
        self.downs = nn.ModuleList([Down(chs[i], chs[i + 1]) for i in range(3)])
        self.mid = nn.ModuleList([
            Block(chs[3], chs[3], time_dim),
            Attention(chs[3]),
            Block(chs[3], chs[3], time_dim),
        ])
        self.up = nn.ModuleList([
            nn.ModuleList([Block(chs[i] + chs[i + 1], chs[i], time_dim),
                           Block(chs[i], chs[i], time_dim)])
            for i in reversed(range(3))
        ])
        self.ups = nn.ModuleList([Up(chs[i + 1]) for i in reversed(range(3))])
        self.head = nn.Sequential(
            nn.GroupNorm(8, chs[0]), nn.SiLU(), nn.Conv2d(chs[0], 1, 3, padding=1)
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t = self.time(t)
        h = self.stem(x)
        skips = [h]
        for i in range(3):
            h = self.down[i][0](h, t)
            h = self.down[i][1](h, t)
            skips.append(h)  # последний block этапа — skip для декодера
            h = self.downs[i](h)
        for b in self.mid:
            h = b(h, t) if not isinstance(b, Attention) else b(h)
        for i in range(3):
            h = self.ups[i](h)
            h = torch.cat([h, skips.pop()], dim=1)
            for b in self.up[i]:
                h = b(h, t)
        return self.head(h)
