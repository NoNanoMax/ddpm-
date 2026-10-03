"""Parity: UNet (PyTorch) vs C++ движок. Полный forward, 64x64, batch=1."""
from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "models"))
sys.path.insert(0, str(ROOT / "models" / "mnist"))

from unet import UNet  # noqa: E402


def main():
    engine = str(ROOT / "engine/build/ddpm-engine")
    ckpt_path = sys.argv[1] if len(sys.argv) > 1 else str(ROOT / "checkpoints/mnist_ddpm.pt")

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    model = UNet().eval()
    model.load_state_dict(ckpt["model_state"])

    torch.manual_seed(7)
    x = torch.randn(1, 1, 64, 64)
    t = torch.tensor([137])

    with torch.no_grad():
        expected = model(x, t).numpy()  # NCHW

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        subprocess.run([sys.executable, str(ROOT / "format/pytorch_to_ddpm.py"),
                        "--pt", ckpt_path, "--out", str(td / "m.ddpm")], check=True)

        x_nhwc = x.permute(0, 2, 3, 1).numpy().astype(np.float32)
        x_nhwc.tofile(td / "in.bin")

        subprocess.run([engine, "run", str(td / "m.ddpm"),
                        "--t", "137", "--in-shape", "1,64,64,1",
                        "--in", str(td / "in.bin"), "--out", str(td / "out.bin")],
                       check=True)
        out = np.fromfile(td / "out.bin", dtype=np.float32).reshape(1, 64, 64, 1)

    diff = np.abs(out - expected.transpose(0, 2, 3, 1))
    print(f"max abs diff: {diff.max():.3e}")
    if diff.max() < 5e-3:  # U-Net глубже, float32 накапливает
        print("PARITY OK")
        return 0
    print("PARITY FAILED")
    return 1


if __name__ == "__main__":
    sys.exit(main())
