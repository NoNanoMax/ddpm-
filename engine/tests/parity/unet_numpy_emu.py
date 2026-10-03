"""NumPy-эмуляция графа U-Net (NHWC) — сверка с PyTorch до сравнения с C++."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "models"))
sys.path.insert(0, str(ROOT / "models" / "mnist"))

from unet import UNet  # noqa: E402


def conv(x, w, b, k=3, stride=1, pad=1):
    B, H, W, C = x.shape
    Ho, Wo = (H + 2 * pad - k) // stride + 1, (W + 2 * pad - k) // stride + 1
    xp = np.pad(x, ((0, 0), (pad, pad), (pad, pad), (0, 0)))
    out = np.empty((B, Ho, Wo, w.shape[0]), np.float32)
    for oh in range(Ho):
        for ow in range(Wo):
            acc = np.zeros((B, w.shape[0]), np.float64)
            for fh in range(k):
                for fw in range(k):
                    px = xp[:, oh * stride + fh, ow * stride + fw, :]
                    acc += px.astype(np.float64) @ w.reshape(w.shape[0], C, k * k)[:, :, fh * k + fw].T
            out[:, oh, ow] = (acc + b).astype(np.float32)
    return out


def gn(x, gamma, beta, g=8):
    B, H, W, C = x.shape
    x64 = x.reshape(B, H, W, g, C // g).astype(np.float64)
    mu = x64.mean(axis=(1, 2, 4), keepdims=True)
    va = x64.var(axis=(1, 2, 4), keepdims=True)
    return ((x64 - mu) / np.sqrt(va + 1e-5) * gamma.reshape(1, 1, 1, g, C // g)
            + beta.reshape(1, 1, 1, g, C // g)).reshape(B, H, W, C).astype(np.float32)


def silu(x):
    return (x * (1 / (1 + np.exp(-x.astype(np.float64))))).astype(np.float32)


def graph_forward(st, x, tval):
    freqs = np.power(10000.0, -np.arange(32) / 31)
    tt = np.concatenate([np.sin(freqs * tval)[:, None], np.cos(freqs * tval)[:, None]],
                        axis=0).astype(np.float32).reshape(1, 64)
    t2 = silu(tt @ st["time.proj.0.weight"].T + st["time.proj.0.bias"])
    temb = t2 @ st["time.proj.2.weight"].T + st["time.proj.2.bias"]

    def tadd(b, cout):
        return (temb @ st[f"{b}.time.weight"].T + st[f"{b}.time.bias"]).astype(np.float32)

    def resid(b, h_in, hh):
        if h_in.shape[3] != hh.shape[3]:
            return hh + conv(h_in, st[f"{b}.skip.weight"], st[f"{b}.skip.bias"], k=1, pad=0)
        return hh + h_in

    h = conv(x, st["stem.weight"], st["stem.bias"])
    skips = []
    for i in range(3):
        for j in range(2):
            b = f"down.{i}.{j}."
            hh = gn(h, st[b + "n1.weight"], st[b + "n1.bias"])
            hh = conv(silu(hh), st[b + "c1.weight"], st[b + "c1.bias"])
            hh = gn(hh + tadd(f"down.{i}.{j}", h.shape[3])[:, None, None, :],
                    st[b + "n2.weight"], st[b + "n2.bias"])
            hh = conv(silu(hh), st[b + "c2.weight"], st[b + "c2.bias"])
            h_in, h = h, resid(f"down.{i}.{j}", h, hh)
        skips.append(h)
        h = conv(silu(h), st[f"downs.{i}.net.1.weight"], st[f"downs.{i}.net.1.bias"], stride=2)

    for mid in ("mid.0", "mid.2"):
        b = f"{mid}."
        hh = gn(h, st[b + "n1.weight"], st[b + "n1.bias"])
        hh = conv(silu(hh), st[b + "c1.weight"], st[b + "c1.bias"])
        hh = gn(hh + tadd(mid, h.shape[3])[:, None, None, :],
                st[b + "n2.weight"], st[b + "n2.bias"])
        hh = conv(silu(hh), st[b + "c2.weight"], st[b + "c2.bias"])
        h_in, h = h, resid(mid, h, hh)
        if mid == "mid.0":
            C, HW = h.shape[3], h.shape[1] * h.shape[2]
            qkv = conv(h, st["mid.1.qkv.weight"], st["mid.1.qkv.bias"], k=1, pad=0)
            q = qkv[..., :C].reshape(1, HW, C)
            k_ = qkv[..., C:2 * C].reshape(1, HW, C)
            v = qkv[..., 2 * C:].reshape(1, HW, C)
            a = (q.astype(np.float64) @ np.swapaxes(k_.astype(np.float64), -1, -2)) / np.sqrt(C)
            a -= a.max(-1, keepdims=True)
            a = np.exp(a)
            a /= a.sum(-1, keepdims=True)
            at = (a @ v.astype(np.float64)).astype(np.float32).reshape(1, h.shape[1], h.shape[2], C)
            h = conv(at, st["mid.1.proj.weight"], st["mid.1.proj.bias"], k=1, pad=0) + h

    for i in range(3):
        h = np.repeat(np.repeat(h, 2, axis=1), 2, axis=2)
        h = conv(silu(h), st[f"ups.{i}.net.2.weight"], st[f"ups.{i}.net.2.bias"])
        h = np.concatenate([h, skips[2 - i]], axis=-1)
        for j in range(2):
            b = f"up.{i}.{j}."
            hh = gn(h, st[b + "n1.weight"], st[b + "n1.bias"])
            hh = conv(silu(hh), st[b + "c1.weight"], st[b + "c1.bias"])
            hh = gn(hh + tadd(f"up.{i}.{j}", h.shape[3])[:, None, None, :],
                    st[b + "n2.weight"], st[b + "n2.bias"])
            hh = conv(silu(hh), st[b + "c2.weight"], st[b + "c2.bias"])
            h_in, h = h, resid(f"up.{i}.{j}", h, hh)

    h = conv(silu(gn(h, st["head.0.weight"], st["head.0.bias"])),
             st["head.2.weight"], st["head.2.bias"])
    return h.transpose(0, 3, 1, 2)


def main():
    m = UNet().eval()
    st = {k: v.numpy().astype(np.float32) for k, v in m.state_dict().items()}
    x = torch.randn(1, 1, 64, 64)
    t = torch.tensor([137])
    with torch.no_grad():
        ref = m(x, t).numpy()
    emu = graph_forward(st, x.numpy().transpose(0, 2, 3, 1).astype(np.float32), 137)
    print("numpy vs pytorch max diff:", np.abs(emu - ref).max())


if __name__ == "__main__":
    main()
