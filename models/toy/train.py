"""Toy DDPM: обучение на 2D-смеси гауссов.

Данные — 4 гауссовых «пятна» + кольцо. Весь цикл:
  1. берём чистые x_0
  2. берём случайный t, добавляем шум за один прыжок → x_t
  3. модель предсказывает eps
  4. loss = MSE(pred, eps) — и весь loss функции DDPM (v1)

Запуск (быстро, CPU):
  python models/toy/train.py
  python models/toy/train.py --steps 5000 --device cuda
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.schedulers import LinearBetaSchedule  # noqa: E402
from model import EpsMLP  # noqa: E402


def make_dataset(n: int = 20_000, seed: int = 0) -> torch.Tensor:
    """4 гауссовых пятна в квадрате [-2, 2]."""
    g = torch.Generator().manual_seed(seed)
    centers = torch.tensor([[1.0, 1.0], [-1.0, 1.0], [1.0, -1.0], [-1.0, -1.0]])
    idx = torch.randint(0, 4, (n,), generator=g)
    x = centers[idx] + 0.15 * torch.randn(n, 2, generator=g)
    return x


@torch.no_grad()
def sample(model: nn.Module, sched: LinearBetaSchedule, n: int = 4096,
           device: str = "cpu") -> torch.Tensor:
    """Reverse process: x_T ~ N(0, I) → T шагов к x_0.

    Формула шага (epsilon-predictor):
      x_{t-1} = 1/sqrt(alpha_t) * (x_t - beta_t/sqrt(1-abar_t) * pred_eps) + sigma_t * z
    где z ~ N(0,I) — НОВЫЙ шум на каждом шаге (кроме t=0).
    """
    device = torch.device(device)
    model.eval()
    x = torch.randn(n, 2, device=device)
    betas, alphas, abar = sched.beta, sched.alpha, sched.alpha_bar
    for t in reversed(range(sched.num_steps)):
        pred = model(x, torch.full((n,), t, device=device))
        coef1 = 1.0 / alphas[t].sqrt()
        coef2 = betas[t] / (1 - abar[t]).sqrt()
        sigma = betas[t].sqrt()
        z = torch.randn_like(x) if t > 0 else torch.zeros_like(x)
        x = coef1 * (x - coef2 * pred) + sigma * z
    return x


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--bs", type=int, default=512)
    p.add_argument("--num-t", type=int, default=256, help="число T шагов диффузии")
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--device", default="cpu")
    p.add_argument("--save", default=None, help="путь для сохранения .pt")
    p.add_argument("--plot-every", type=int, default=1000)
    args = p.parse_args()

    device = torch.device(args.device)
    sched = LinearBetaSchedule(num_steps=args.num_t).to(device)
    model = EpsMLP(in_dim=2, hidden=256, depth=4).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    data = make_dataset().to(device)
    model.train()
    for step in range(args.steps):
        x0 = data[torch.randint(0, len(data), (args.bs,))]
        x_t, t, eps = sched.training_targets(x0)
        pred = model(x_t, t)
        loss = nn.functional.mse_loss(pred, eps)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % args.plot_every == 0 or step == args.steps - 1:
            print(f"step {step:5d}  loss {loss.item():.5f}", flush=True)

    out = sample(model, sched, n=4096, device=args.device)
    print(f"sampled {out.shape}, range [{out.min():.2f}, {out.max():.2f}]")

    if args.save:
        Path(args.save).parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "model_state": model.state_dict(),
            "config": {"in_dim": 2, "hidden": 256, "depth": 4, "num_t": args.num_t},
        }, args.save)
        print(f"saved → {args.save}")

    # Текстовое "ascii-plot": сразу видно, что точки собрались в 4 пятна
    grid = torch.zeros(32, 32)
    ix = ((out[:, 0] + 2) / 4 * 31).long().clamp(0, 31)
    iy = ((out[:, 1] + 2) / 4 * 31).long().clamp(0, 31)
    for a, b in zip(ix.tolist(), iy.tolist()):
        grid[31 - b, a] += 1
    chars = " .:-=+*#%@"
    for row in grid:
        print("".join(chars[min(int(v), len(chars) - 1)] if v else " " for v in row.numpy()))


if __name__ == "__main__":
    main()
