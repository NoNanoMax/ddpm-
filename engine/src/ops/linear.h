#pragma once
// Linear: y = x · W^T + b  (как nn.Linear)
// x: (B, in)  W: (out, in)  b: (out)  →  y: (B, out)
// W хранится в layout PyTorch (out × in) — конвертер просто сливает state_dict.
#include <memory>

#include "op.h"

namespace ddpm {
std::unique_ptr<Op> make_linear();
}  // namespace ddpm
