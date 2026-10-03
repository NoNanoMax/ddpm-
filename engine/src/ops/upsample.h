#pragma once
#include <memory>
#include "op.h"

namespace ddpm {
// Nearest-upsample ×scale по H и W: x (B,H,W,C) → (B,sH,sW,C)
std::unique_ptr<Op> make_upsample(int scale);
}  // namespace ddpm
