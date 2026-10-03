"""Parity-тест: PyTorch vs C++ движок.

Сценарий:
  1. обучаем toy-модель (CPU, 300 шагов — только для получения весов)
  2. конвертируем в .ddpm
  3. фиксированный вход (seed) → PyTorch forward → expected.bin
  4. C++ ddpm-engine run → out.bin
  5. numpy: max |diff| < atol

Запуск:
  python3 tests/parity/run_parity.py [path_to_engine]
"""
from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "models"))
sys.path.insert(0, str(ROOT / "models" / "toy"))

from common.schedulers import LinearBetaSchedule  # noqa: E402
from model import EpsMLP  # noqa: E402


def main():
    engine = sys.argv[1] if len(sys.argv) > 1 else str(ROOT / "engine/build/ddpm-engine")
    torch.manual_seed(0)

    # 1. мини-обучение
    model = EpsMLP(in_dim=2, hidden=256, depth=4)
    sched = LinearBetaSchedule(num_steps=256)
    opt = torch.optim.Adam(model.parameters(), lr=2e-4)
    for _ in range(300):
        x0 = torch.randn(256, 2)
        x_t, t, eps = sched.training_targets(x0)
        loss = torch.nn.functional.mse_loss(model(x_t, t), eps)
        opt.zero_grad()
        loss.backward()
        opt.step()

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        pt_path = td / "toy.pt"
        torch.save({"model_state": model.state_dict(),
                    "config": {"in_dim": 2, "hidden": 256, "depth": 4, "num_t": 256}},
                   pt_path)

        # 2. конвертация
        subprocess.run(
            [sys.executable, str(ROOT / "format/pytorch_to_ddpm.py"),
             "--pt", str(pt_path), "--out", str(td / "toy.ddpm")],
            check=True)

        # 3. фиксированный вход
        torch.manual_seed(123)
        B, t_val = 8, 17
        x_t = torch.randn(B, 2)
        t = torch.full((B,), t_val)
        with torch.no_grad():
            expected = model(x_t, t).numpy()
        x_t.numpy().astype(np.float32).tofile(td / "in.bin")
        expected.astype(np.float32).tofile(td / "expected.bin")

        # 4. C++
        subprocess.run([engine, "run", str(td / "toy.ddpm"),
                        "--n", str(B), "--t", str(t_val),
                        "--in", str(td / "in.bin"), "--out", str(td / "out.bin")],
                       check=True)
        out = np.fromfile(td / "out.bin", dtype=np.float32)

    # 5. сравнение
    diff = np.abs(out - expected.ravel())
    print(f"max abs diff: {diff.max():.3e}")
    if diff.max() < 1e-4:
        print("PARITY OK")
        return 0
    print("PARITY FAILED")
    return 1


if __name__ == "__main__":
    sys.exit(main())
