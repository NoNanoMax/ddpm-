"""Визуализация toy-модели:
Запуск:
  python3 models/toy/visualize.py --ckpt checkpoints/toy.pt
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from common.schedulers import LinearBetaSchedule  # noqa: E402
from model import EpsMLP  # noqa: E402
from train import make_dataset, sample  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="checkpoints/toy.pt")
    p.add_argument("--out-dir", default="visuals")
    p.add_argument("--n", type=int, default=4096)
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ckpt = torch.load(args.ckpt, weights_only=True)
    cfg = ckpt["config"]
    model = EpsMLP(**{k: cfg[k] for k in ("in_dim", "hidden", "depth")}).eval()
    model.load_state_dict(ckpt["model_state"])
    sched = LinearBetaSchedule(num_steps=cfg["num_t"])

    data = make_dataset()
    samples = sample(model, sched, n=args.n)
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].scatter(data[::8, 0], data[::8, 1], s=1, alpha=0.4, color="tab:blue")
    ax[0].set_title("data (4 gaussian blobs)")
    ax[1].scatter(samples[::8, 0], samples[::8, 1], s=1, alpha=0.4, color="tab:red")
    ax[1].set_title("model samples")
    for a in ax:
        a.set_xlim(-2.5, 2.5); a.set_ylim(-2.5, 2.5); a.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(out_dir / "toy_samples.png", dpi=110)
    print(f"saved {out_dir/'toy_samples.png'}")

    @torch.no_grad()
    def denoise_trajectory(x_T: torch.Tensor, seed: int = 7):
        g = torch.Generator().manual_seed(seed)
        x = x_T.clone()
        betas, alphas, abar = sched.beta, sched.alpha, sched.alpha_bar
        frames = []
        for t in reversed(range(sched.num_steps)):
            pred = model(x, torch.full((x.shape[0],), t))
            c1 = 1.0 / alphas[t].sqrt()
            c2 = betas[t] / (1 - abar[t]).sqrt()
            sigma = betas[t].sqrt()
            z = torch.randn(x.shape, generator=g) if t > 0 else torch.zeros_like(x)
            x = c1 * (x - c2 * pred) + sigma * z
            if t % 16 == sched.num_steps - 1 or (t < sched.num_steps and (sched.num_steps - 1 - t) % 16 == 0):
                frames.append(x.clone())
        return frames

    torch.manual_seed(0)
    frames = denoise_trajectory(torch.randn(2048, 2))
    imgs = []
    for i, f in enumerate(frames):
        fig, ax = plt.subplots(figsize=(4, 4), facecolor="black")
        ax.set_facecolor("black")
        ax.scatter(f[::8, 0], f[::8, 1], s=1, alpha=0.5, color="cyan")
        ax.set_xlim(-2.5, 2.5); ax.set_ylim(-2.5, 2.5); ax.set_aspect("equal")
        ax.set_title(f"step {sched.num_steps - i}", color="white")
        for s in ax.get_xticklabels() + ax.get_yticklabels():
            s.set_color("white")
        for spine in ax.spines.values():
            spine.set_color("gray")
        fig.tight_layout()
        fig.canvas.draw()  
        w, h = fig.canvas.get_width_height()
        rgba = Image.frombuffer("RGBA", (w, h),
                                fig.canvas.buffer_rgba(), "raw", "RGBA", 0, 1)
        bg = Image.new("RGB", (w, h), "black")
        bg.paste(rgba, mask=rgba.split()[3])
        im = bg
        imgs.append(im)
        plt.close(fig)
    gif = out_dir / "denoise.gif"
    imgs[0].save(gif, save_all=True, append_images=imgs[1:],
                 duration=150, loop=0, optimize=True)
    print(f"saved {gif} ({len(imgs)} frames)")


if __name__ == "__main__":
    main()
