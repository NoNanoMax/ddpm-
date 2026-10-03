#pragma once
#include <memory>
#include "op.h"

namespace ddpm {
// GroupNorm: x (B,H,W,C) → нормализация по (H,W,C/groups) на группу
std::unique_ptr<Op> make_groupnorm(int groups);
}  // namespace ddpm
