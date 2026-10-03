"""Конвертер: PyTorch state_dict (EpsMLP) → наш бинарный формат .ddpm (v0).

Для v0 конвертер понимает одну архитектуру: EpsMLP
(Linear+SiLU стек). Это осознанно: формат графа общий, а конвертеры
будут наращиваться по мере усложнения моделей (U-Net и т.д.).

Файл результата:
  DDPM-V0
  input t 1
  input x_t 2 2
  tensor temb_w ...
  op ...
  weights
  <raw f32>
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from unet_converter import convert_unet
import torch


def convert(state: dict, cfg: dict, out_path: Path) -> None:
    in_dim = cfg["in_dim"]
    time_dim = 64  # размер эмбеддинга t (см. EpsMLP: SinusoidalTimeEmbedding(64))
    half_dim = time_dim // 2
    depth = cfg["depth"]

    tensors: dict[str, np.ndarray] = {}
    # извлекаем веса в порядок записи
    lines = [
        "DDPM-V0",
        "input t 1",
        f"input x_t 2 {in_dim}",
    ]

    # 1. time embedding → concat
    lines.append(f"op 0 sincos t temb {{half_dim={half_dim}}}")
    lines.append(f"op 1 concat x_t temb c")

    # 2. Linear+SiLU × (depth-1), финальный Linear
    prev = "c"
    prev_dim = in_dim + time_dim
    idx = 2
    for layer_i in range(depth + 1):  # depth блоков Linear+SiLU + финальный Linear
        last = layer_i == depth
        w_key = f"net.{layer_i}.weight" if last else f"net.{layer_i}.0.weight"
        b_key = f"net.{layer_i}.bias" if last else f"net.{layer_i}.0.bias"
        w = state[w_key]
        b = state[b_key]
        wname, bname = f"w{layer_i}", f"b{layer_i}"
        out_name = "h" + str(idx) if not last else "out"
        out_dim = w.shape[0]
        lines.append(f"tensor {wname} 2 {out_dim} {prev_dim}")
        lines.append(f"tensor {bname} 1 {out_dim}")
        lines.append(f"op {idx} linear {prev} {wname} {bname} {out_name}")
        tensors[wname] = w.numpy().astype(np.float32).ravel()
        tensors[bname] = b.numpy().astype(np.float32).ravel()
        if not last:
            lines.append(f"op {idx + 1} silu {out_name} {prev}_n")
            prev = prev + "_n"
            idx += 2
        else:
            idx += 1
        prev_dim = out_dim

    lines.append("output out")
    lines.append("weights")

    header = "\n".join(lines) + "\n"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("wb") as f:
        f.write(header.encode())
        for name, arr in tensors.items():
            f.write(arr.tobytes())
    print(f"converted → {out_path} ({out_path.stat().st_size} bytes, {len(tensors)} tensors)")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pt", required=True, help="torch.save'ed dict: model_state, config")
    p.add_argument("--out", required=True)
    args = p.parse_args()

    ckpt = torch.load(args.pt, weights_only=True)
    if "base" in ckpt["config"]:  # UNet
        convert_unet(ckpt["model_state"], ckpt["config"], Path(args.out))
    else:  # EpsMLP
        convert(ckpt["model_state"], ckpt["config"], Path(args.out))


if __name__ == "__main__":
    main()
