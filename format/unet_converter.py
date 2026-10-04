from __future__ import annotations

import numpy as np


def convert_unet(state: dict, cfg: dict, out_path) -> None:
    base = cfg["base"]
    chs = [base // 2, base, base * 2, base * 2]

    L: list[str] = ["DDPM-V0", "input x_t 4 1 64 64 1", "input t 1"]
    tensors: dict[str, np.ndarray] = {}

    def conv1x1(name, x, wkey, bkey, out, groups=0, stride=1):
        nonlocal op_i
        w = state[wkey]
        wn, bn = f"w_{name}", f"b_{name}"
        tensors[wn] = w.numpy().astype(np.float32).ravel()
        L.append(f"tensor {wn} 4 {w.shape[0]} {w.shape[1]} {w.shape[2]} {w.shape[3]}")
        tensors[bn] = state[bkey].numpy().astype(np.float32).ravel()
        L.append(f"tensor {bn} 1 {w.shape[0]}")
        L.append(f"op {op_i} conv2d {x} {wn} {bn} {out} {{kh=1 kw=1 stride={stride} pad=0}}")
        op_i += 1

    def gn(name, x, wkey, out):
        nonlocal op_i
        L.append(f"op {op_i} groupnorm {x} w_{name} b_{name} {out} {{groups=8}}")
        tensors[f"w_{name}"] = state[wkey + ".weight"].numpy().astype(np.float32).ravel()
        tensors[f"b_{name}"] = state[wkey + ".bias"].numpy().astype(np.float32).ravel()
        L.append(f"tensor w_{name} 1 {state[wkey + '.weight'].shape[0]}")
        L.append(f"tensor b_{name} 1 {state[wkey + '.bias'].shape[0]}")
        op_i += 1

    def block(name, x, wbase, skip):
        # GN → SiLU → Conv → +time(temb) → GN → SiLU → Conv → +skip
        nonlocal op_i
        o = []
        w = state[f"{wbase}.time.weight"]
        tensors[f"w_{name}.time"] = w.numpy().astype(np.float32).ravel()
        tensors[f"b_{name}.time"] = state[f"{wbase}.time.bias"].numpy().astype(np.float32).ravel()
        L.append(f"tensor w_{name}.time 2 {w.shape[0]} {w.shape[1]}")
        L.append(f"tensor b_{name}.time 1 {w.shape[0]}")
        o.append(f"op {op_i} linear temb w_{name}.time b_{name}.time {name}.ta"); op_i += 1
        o.append(f"op {op_i} groupnorm {x} w_{name}.n1 b_{name}.n1 {name}.g1 {{groups=8}}"); op_i += 1
        tensors[f"w_{name}.n1"] = state[wbase + ".n1.weight"].numpy().astype(np.float32).ravel()
        tensors[f"b_{name}.n1"] = state[wbase + ".n1.bias"].numpy().astype(np.float32).ravel()
        L.append(f"tensor w_{name}.n1 1 {state[wbase + '.n1.weight'].shape[0]}")
        L.append(f"tensor b_{name}.n1 1 {state[wbase + '.n1.bias'].shape[0]}")
        o.append(f"op {op_i} silu {name}.g1 {name}.s1"); op_i += 1
        for j, cn in enumerate(["c1", "c2"]):
            w = state[f"{wbase}.{cn}.weight"]
            wn, bn = f"w_{name}.{cn}", f"b_{name}.{cn}"
            tensors[wn] = w.numpy().astype(np.float32).ravel()
            tensors[bn] = state[f"{wbase}.{cn}.bias"].numpy().astype(np.float32).ravel()
            L.append(f"tensor {wn} 4 {w.shape[0]} {w.shape[1]} {w.shape[2]} {w.shape[3]}")
            L.append(f"tensor {bn} 1 {w.shape[0]}")
            o.append(f"op {op_i} conv2d {name}.s{1 + j} {wn} {bn} {name}.{cn} {{kh=3 kw=3 stride=1 pad=1}}")
            op_i += 1
            if cn == "c1":
                o.append(f"op {op_i} addb {name}.c1 {name}.ta {name}.t"); op_i += 1
                o.append(f"op {op_i} groupnorm {name}.t w_{name}.n2 b_{name}.n2 {name}.g2 {{groups=8}}"); op_i += 1
                tensors[f"w_{name}.n2"] = state[wbase + ".n2.weight"].numpy().astype(np.float32).ravel()
                tensors[f"b_{name}.n2"] = state[wbase + ".n2.bias"].numpy().astype(np.float32).ravel()
                L.append(f"tensor w_{name}.n2 1 {state[wbase + '.n2.weight'].shape[0]}")
                L.append(f"tensor b_{name}.n2 1 {state[wbase + '.n2.bias'].shape[0]}")
                o.append(f"op {op_i} silu {name}.g2 {name}.s2"); op_i += 1
        if skip is not None:  # Conv1x1 skip
            w = state[f"{wbase}.skip.weight"]
            wn, bn = f"w_{name}.skip", f"b_{name}.skip"
            tensors[wn] = w.numpy().astype(np.float32).ravel()
            tensors[bn] = state[f"{wbase}.skip.bias"].numpy().astype(np.float32).ravel()
            L.append(f"tensor {wn} 4 {w.shape[0]} {w.shape[1]} {w.shape[2]} {w.shape[3]}")
            L.append(f"tensor {bn} 1 {w.shape[0]}")
            o.append(f"op {op_i} conv2d {x} {wn} {bn} {name}.sk {{kh=1 kw=1 stride=1 pad=0}}")
            op_i += 1
            o.append(f"op {op_i} add {name}.c2 {name}.sk {name}.out"); op_i += 1
        else:
            o.append(f"op {op_i} add {name}.c2 {x} {name}.out"); op_i += 1
        L.extend(o)

    op_i = 0
    # time embedding: sincos → Linear → SiLU → Linear
    L.append(f"op {op_i} sincos t t1 {{half_dim=32}}"); op_i += 1
    for j, k in enumerate([0, 2]):
        w = state[f"time.proj.{k}.weight"]
        L.append(f"tensor w_tt{j} 2 {w.shape[0]} {w.shape[1]}")
        tensors[f"w_tt{j}"] = w.numpy().astype(np.float32).ravel()
        L.append(f"tensor b_tt{j} 1 {w.shape[0]}")
        tensors[f"b_tt{j}"] = state[f"time.proj.{k}.bias"].numpy().astype(np.float32).ravel()
        # цепочка: t1 → t2 → (silu) t2b → tt
        src = "t1" if j == 0 else "t2b"
        out = "t2" if j == 0 else "temb"
        L.append(f"op {op_i} linear {src} w_tt{j} b_tt{j} {out}"); op_i += 1
        if j == 0:
            L.append(f"op {op_i} silu t2 t2b"); op_i += 1

    # stem
    w = state["stem.weight"]
    L.append(f"tensor w_stem 4 {w.shape[0]} {w.shape[1]} {w.shape[2]} {w.shape[3]}")
    tensors["w_stem"] = w.numpy().astype(np.float32).ravel()
    L.append(f"tensor b_stem 1 {w.shape[0]}")
    tensors["b_stem"] = state["stem.bias"].numpy().astype(np.float32).ravel()
    L.append(f"op {op_i} conv2d x_t w_stem b_stem h0 {{kh=3 kw=3 stride=1 pad=1}}"); op_i += 1

    prev = "h0"
    skips = []
    for i in range(3):
        for j in range(2):
            has_skip = f"down.{i}.{j}.skip.weight" in state
            block(f"d{i}_{j}", prev, f"down.{i}.{j}", "yes" if has_skip else None)
            prev = f"d{i}_{j}.out"
        skips.append(prev)
        w = state[f"downs.{i}.net.1.weight"]
        wn, bn = f"w_down{i}", f"b_down{i}"
        L.append(f"tensor {wn} 4 {w.shape[0]} {w.shape[1]} {w.shape[2]} {w.shape[3]}")
        tensors[wn] = w.numpy().astype(np.float32).ravel()
        L.append(f"tensor {bn} 1 {w.shape[0]}")
        tensors[bn] = state[f"downs.{i}.net.1.bias"].numpy().astype(np.float32).ravel()
        L.append(f"op {op_i} silu {prev} us_{i}.s"); op_i += 1
        L.append(f"op {op_i} conv2d us_{i}.s {wn} {bn} h{i + 1} {{kh=3 kw=3 stride=2 pad=1}}"); op_i += 1
        prev = f"h{i + 1}"

    # mid: block, attention, block
    block("m0", prev, "mid.0", None)
    prev = "m0.out"
    w = state["mid.1.qkv.weight"]
    L.append(f"tensor w_attnqkv 4 {w.shape[0]} {w.shape[1]} 1 1")
    tensors["w_attnqkv"] = w.numpy().astype(np.float32).ravel()
    L.append(f"tensor b_attnqkv 1 {w.shape[0]}")
    tensors["b_attnqkv"] = state["mid.1.qkv.bias"].numpy().astype(np.float32).ravel()
    L.append(f"op {op_i} conv2d {prev} w_attnqkv b_attnqkv attn3x {{kh=1 kw=1 stride=1 pad=0}}"); op_i += 1
    L.append(f"op {op_i} attention attn3x attno"); op_i += 1
    w = state["mid.1.proj.weight"]
    L.append(f"tensor w_attnproj 4 {w.shape[0]} {w.shape[1]} 1 1")
    tensors["w_attnproj"] = w.numpy().astype(np.float32).ravel()
    L.append(f"tensor b_attnproj 1 {w.shape[0]}")
    tensors["b_attnproj"] = state["mid.1.proj.bias"].numpy().astype(np.float32).ravel()
    L.append(f"op {op_i} conv2d attno w_attnproj b_attnproj attnp {{kh=1 kw=1 stride=1 pad=0}}"); op_i += 1
    L.append(f"op {op_i} add attnp {prev} m1"); op_i += 1
    block("m2", "m1", "mid.2", None)
    prev = "m2.out"

    # decoder
    for i in range(3):
        stage = 2 - i
        w = state[f"ups.{i}.net.2.weight"]
        wn, bn = f"w_up{i}", f"b_up{i}"
        L.append(f"tensor {wn} 4 {w.shape[0]} {w.shape[1]} {w.shape[2]} {w.shape[3]}")
        tensors[wn] = w.numpy().astype(np.float32).ravel()
        L.append(f"tensor {bn} 1 {w.shape[0]}")
        tensors[bn] = state[f"ups.{i}.net.2.bias"].numpy().astype(np.float32).ravel()
        L.append(f"op {op_i} upsample {prev} upx{i} {{scale=2}}"); op_i += 1
        L.append(f"op {op_i} silu upx{i} ups{i}.s"); op_i += 1
        L.append(f"op {op_i} conv2d ups{i}.s {wn} {bn} upc{i} {{kh=3 kw=3 stride=1 pad=1}}"); op_i += 1
        L.append(f"op {op_i} concat upc{i} {skips[stage]} cat{i}"); op_i += 1
        for j in range(2):
            has_skip = f"up.{i}.{j}.skip.weight" in state
            block(f"u{i}_{j}", f"cat{i}" if j == 0 else f"u{i}_{j - 1}.out",
                  f"up.{i}.{j}", "yes" if has_skip else None)
        prev = f"u{i}_{1}.out"

    # head
    L.append(f"op {op_i} groupnorm {prev} w_head.b0 b_head.b0 hd.g {{groups=8}}"); op_i += 1
    L.append(f"tensor w_head.b0 1 {state['head.0.weight'].shape[0]}")
    tensors["w_head.b0"] = state["head.0.weight"].numpy().astype(np.float32).ravel()
    L.append(f"tensor b_head.b0 1 {state['head.0.bias'].shape[0]}")
    tensors["b_head.b0"] = state["head.0.bias"].numpy().astype(np.float32).ravel()
    L.append(f"op {op_i} silu hd.g hd.s"); op_i += 1
    w = state["head.2.weight"]
    L.append(f"tensor w_head 4 {w.shape[0]} {w.shape[1]} {w.shape[2]} {w.shape[3]}")
    tensors["w_head"] = w.numpy().astype(np.float32).ravel()
    L.append(f"tensor b_head 1 {w.shape[0]}")
    tensors["b_head"] = state["head.2.bias"].numpy().astype(np.float32).ravel()
    L.append(f"op {op_i} conv2d hd.s w_head b_head out {{kh=3 kw=3 stride=1 pad=1}}"); op_i += 1

    L.append("output out")
    L.append("weights")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("wb") as f:
        f.write(("\n".join(L) + "\n").encode())
        for arr in tensors.values():
            f.write(arr.tobytes())
    print(f"converted UNet → {out_path} ({out_path.stat().st_size / 1e6:.1f} MB, {len(tensors)} tensors)")
