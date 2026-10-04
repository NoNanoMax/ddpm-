"""Мини-parity: каждый новый C++ op отдельно, сверка с PyTorch."""
from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[3]
ENGINE = str(ROOT / "engine/build/ddpm-engine")
torch.manual_seed(0)


def run_engine(graph: str, x: np.ndarray, t: float = 0) -> np.ndarray:
    """graph: строка header-тела (без магической строки и weights-секции)."""
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        (td / "m.ddpm").write_bytes((graph + "\nweights\n").encode())
        (td / "in.bin").write_bytes(x.astype(np.float32).tobytes())
        subprocess.run([ENGINE, "run", str(td / "m.ddpm"), "--t", str(t),
                        "--in-shape", ",".join(map(str, x.shape)),
                        "--in", str(td / "in.bin"), "--out", str(td / "out.bin")], check=True)
        return np.fromfile(td / "out.bin", dtype=np.float32)


def check_conv2d():
    x = torch.randn(1, 2, 5, 4)  # NCHW
    conv = nn.Conv2d(2, 3, 3, stride=2, padding=1).eval()
    with torch.no_grad():
        ref = conv(x).numpy()
    W = conv.weight.detach().numpy().astype(np.float32)
    b = conv.bias.detach().numpy().astype(np.float32)
    xn = x.permute(0, 2, 3, 1).numpy()
    graph = ("DDPM-V0\ninput x_t 4 1 5 4 2\n"
             f"tensor w 4 {W.shape[0]} {W.shape[1]} {W.shape[2]} {W.shape[3]}\ntensor b 1 {b.shape[0]}\n"
             "op 0 conv2d x_t w b out {kh=3 kw=3 stride=2 pad=1}\noutput out")
    body = (W.ravel().astype(np.float32).tobytes() + b.ravel().astype(np.float32).tobytes())
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        f = td / "m.ddpm"
        f.write_bytes((graph + "\nweights\n").encode() + body)
        (td / "in.bin").write_bytes(xn.tobytes())
        subprocess.run([ENGINE, "run", str(td / "m.ddpm"), "--t", "0",
                        "--in-shape", "1,5,4,2", "--in", str(td / "in.bin"),
                        "--out", str(td / "out.bin")], check=True)
        out = np.fromfile(td / "out.bin", dtype=np.float32).reshape(1, 3, 2, 3)
    d = np.abs(out - ref.transpose(0, 2, 3, 1)).max()
    print(f"conv2d  diff={d:.2e}")
    return d < 1e-5


def check_groupnorm():
    x = torch.randn(1, 16, 4, 3)
    gn = nn.GroupNorm(8, 16).eval()
    with torch.no_grad():
        ref = gn(x).numpy()
    g, bt = gn.weight.detach().numpy().astype(np.float32), gn.bias.detach().numpy().astype(np.float32)
    xn = x.permute(0, 2, 3, 1).numpy()
    graph = ("DDPM-V0\ninput x_t 4 1 4 3 16\n"
             f"tensor w 1 16\ntensor b 1 16\n"
             "op 0 groupnorm x_t w b out {groups=8}\noutput out")
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        f = td / "m.ddpm"
        f.write_bytes((graph + "\nweights\n").encode()
                      + g.tobytes() + bt.tobytes())
        (td / "in.bin").write_bytes(xn.tobytes())
        subprocess.run([ENGINE, "run", str(td / "m.ddpm"), "--t", "0",
                        "--in-shape", "1,4,3,16", "--in", str(td / "in.bin"),
                        "--out", str(td / "out.bin")], check=True)
        out = np.fromfile(td / "out.bin", dtype=np.float32).reshape(1, 4, 3, 16)
    d = np.abs(out - ref.transpose(0, 2, 3, 1)).max()
    print(f"groupnorm diff={d:.2e}")
    return d < 1e-4


def check_attention():
    B, H, W, C = 1, 4, 3, 8
    x = torch.randn(B, 3 * C, H, W)  # NCHW, qkv склеен
    q, k, v = x.chunk(3, dim=1)
    q = q.reshape(B, C, -1).permute(0, 2, 1)
    k = k.reshape(B, C, -1)
    v = v.reshape(B, C, -1).permute(0, 2, 1)
    a = F.softmax(q @ k / (C ** 0.5), dim=-1)
    ref = (a @ v).permute(0, 2, 1).reshape(B, C, H, W).numpy()
    xn = x.permute(0, 2, 3, 1).numpy()
    graph = ("DDPM-V0\ninput x_t 4 1 4 3 24\n"
             "op 0 attention x_t out\noutput out")
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        (td / "m.ddpm").write_bytes((graph + "\nweights\n").encode())
        (td / "in.bin").write_bytes(xn.tobytes())
        subprocess.run([ENGINE, "run", str(td / "m.ddpm"), "--t", "0",
                        "--in-shape", "1,4,3,24", "--in", str(td / "in.bin"),
                        "--out", str(td / "out.bin")], check=True)
        out = np.fromfile(td / "out.bin", dtype=np.float32).reshape(B, H, W, C)
    d = np.abs(out - ref.transpose(0, 2, 3, 1)).max()
    print(f"attention diff={d:.2e}")
    return d < 1e-4


def check_upsample():
    x = torch.randn(1, 2, 3, 4)
    ref = F.interpolate(x, scale_factor=2, mode="nearest").numpy()
    xn = x.permute(0, 2, 3, 1).numpy()
    graph = ("DDPM-V0\ninput x_t 4 1 3 4 2\n"
             "op 0 upsample x_t out {scale=2}\noutput out")
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        (td / "m.ddpm").write_bytes((graph + "\nweights\n").encode())
        (td / "in.bin").write_bytes(xn.tobytes())
        subprocess.run([ENGINE, "run", str(td / "m.ddpm"), "--t", "0",
                        "--in-shape", "1,3,4,2", "--in", str(td / "in.bin"),
                        "--out", str(td / "out.bin")], check=True)
        out = np.fromfile(td / "out.bin", dtype=np.float32).reshape(1, 6, 8, 2)
    d = np.abs(out - ref.transpose(0, 2, 3, 1)).max()
    print(f"upsample  diff={d:.2e}")
    return d == 0


def main():
    ok = all([check_conv2d(), check_groupnorm(), check_attention(), check_upsample()])
    print("ALL OPS PARITY OK" if ok else "OPS PARITY FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
