#pragma once
#include <memory>
#include "op.h"

namespace ddpm {
// SiLU: x * sigmoid(x). Pointwise, x — любой shape, out = x.shape.
std::unique_ptr<Op> make_silu();
}  // namespace ddpm
