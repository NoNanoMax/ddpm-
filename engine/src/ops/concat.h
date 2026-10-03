#pragma once
#include <memory>
#include "op.h"

namespace ddpm {
// Concat по ПОСЛЕДНЕЙ оси, N dim'ов: (B,H,W,A)+(B,H,W,C) → (B,H,W,A+C)
std::unique_ptr<Op> make_concat();
}  // namespace ddpm
