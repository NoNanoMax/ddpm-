#pragma once
#include <memory>
#include "op.h"

namespace ddpm {
// Concat по последней оси: (B, A) и (B, C) → (B, A+C). 2D, v0.
std::unique_ptr<Op> make_concat();
}  // namespace ddpm
