"""DDPM (UNet) на MNIST 64x64.

Запуск (GPU):
  python3 models/mnist/train.py --device cuda --steps 60000
Сэмпли + визуализация:
  python3 models/mnist/train.py --device cuda --sample --ckpt checkpoints/mnist_ddpm.pt
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.schedulers import LinearBetaSchedule  # noqa: E402
from unet import UNet  # noqa: E402


def get_loader(batch: int, data_dir: str) -> DataLoader:
    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: F.interpolate(
            x.unsqueeze(0), size=(64, 64), mode="bicubic", align_corners=False)[0]),
        transforms.Lambda(lambda x: 2 * x - 1),  # [0,1] → [-1,1]
    ])
    ds = datasets.MNIST(data_dir, train=True, download=True, transform=tf)
    return DataLoader(ds, batch_size=batch, shuffle=True, num_workers=2,
                      pin_memory=True, drop_last=True)


@torch.no_grad()
def ddim_sample(model: nn.Module, sched: LinearBetaSchedule, n: int = 256,
                steps: int = 50, device: str = "cpu") -> torch.Tensor:
    """DDIM: детерминированный (eta=0) субсэмплинг — 50 шагов вместо 256."""
    device = torch.device(device)
    model.eval()
    alphas, abar = sched.alpha, sched.alpha_bar
    ts = list(torch.linspace(0, sched.num_steps - 1, steps).long().flip(0).tolist())
    x = torch.randn(n, 1, 64, 64, device=device)
    for t in ts:
        tb = torch.full((n,), t, device=device)
        pred = model(x, tb)
        a_prev = abar[t - 1] if t > 0 else torch.tensor(1.0, device=device)
        # DDIM eta=0: x_t → x0, потом x0 → x_{t-1} без случайного шума
        x0 = (x - (1 - abar[t]).sqrt() * pred) / abar[t].sqrt().clamp(min=1e-8)
        x = a_prev.sqrt() * x0
    return x


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cuda")
    p.add_argument("--steps", type=int, default=60000)
    p.add_argument("--bs", type=int, default=128)
    p.add_argument("--num-t", type=int, default=256)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--save", default="checkpoints/mnist_ddpm.pt")
    p.add_argument("--ckpt", default=None)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--save-every", type=int, default=10000)
    p.add_argument("--sample", action="store_true")
    p.add_argument("--data", default="data/mnist")
    p.add_argument("--plot-every", type=int, default=1000)
    args = p.parse_args()

    device = torch.device(args.device)
    sched = LinearBetaSchedule(num_steps=args.num_t).to(device)
    model = UNet().to(device)
    print(f"UNet params: {sum(x.numel() for x in model.parameters())/1e6:.1f}M")

    if args.ckpt:
        ckpt = torch.load(args.ckpt, map_location=device, weights_only=True)
        model.load_state_dict(ckpt["model_state"])

    if args.sample:
        x = ddim_sample(model, sched, n=256, steps=50, device=args.device)
        Path(args.save).parent.mkdir(parents=True, exist_ok=True)
        grid = (x.clamp(-1, 1) + 1) / 2
        save_image(grid.cpu(), Path(args.save).parent / "mnist_samples.png")
        print(f"saved {Path(args.save).parent / 'mnist_samples.png'}")
        return

    Path(args.save).parent.mkdir(parents=True, exist_ok=True)
    loader = get_loader(args.bs, args.data)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scaler = torch.cuda.amp.GradScaler() if device.type == "cuda" else None
    start = 0

    def save(ckpt_path, step):
        torch.save({"model_state": model.state_dict(),
                    "opt_state": opt.state_dict(),
                    "step": step,
                    "config": {"base": 128, "num_t": args.num_t, "lr": args.lr}}, ckpt_path)
        print(f"saved → {ckpt_path} (step {step})", flush=True)

    if args.resume:
        ckpt = torch.load(args.save, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state"])
        opt.load_state_dict(ckpt["opt_state"])
        start = ckpt["step"] + 1
        print(f"resuming from step {start}", flush=True)

    model.train()
    it = iter(loader)
    for step in range(start, args.steps):
        try:
            x0, _ = next(it)
        except StopIteration:
            it = iter(loader)
            x0, _ = next(it)
        x0 = x0.to(device, non_blocking=True)
        x_t, t, eps = sched.training_targets(x0)
        with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
            loss = nn.functional.mse_loss(model(x_t, t), eps)
        if scaler:
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
        else:
            loss.backward()
            opt.step()
        opt.zero_grad()
        if step % args.plot_every == 0 or step == args.steps - 1:
            print(f"step {step:6d}  loss {loss.item():.5f}", flush=True)
        if (step + 1) % args.save_every == 0 or step == args.steps - 1:
            save(args.save, step + 1)

    save(args.save, args.steps)


if __name__ == "__main__":
    from torchvision.utils import save_image  # локально, не тянуть на старте
    main()
