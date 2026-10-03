#pragma once
#include <memory>
#include "op.h"

namespace ddpm {
// Single-head spatial attention.
// x: (B,H,W,3C) — qkv уже склеенные (conv1x1 C→3C)
// q=x[:, :, :, 0:C], k=x[:, :, :, C:2C], v=x[:, :, :, 2C:3C]
// y = softmax(q kᵀ/√C) v → (B,H,W,C)
std::unique_ptr<Op> make_attention();
}  // namespace ddpm
