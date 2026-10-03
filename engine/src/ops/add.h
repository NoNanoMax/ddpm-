#pragma once
#include <memory>
#include "op.h"

namespace ddpm {
// add: x + y, same shape (elementwise)
// addb: x (B,H,W,C) + t (B,C) — broadcast timestep-эмбеддинга
std::unique_ptr<Op> make_add();
std::unique_ptr<Op> make_addb();
}  // namespace ddpm
