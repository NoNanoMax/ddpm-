#pragma once
#include <memory>

#include "op.h"

namespace ddpm {
std::unique_ptr<Op> make_identity();
}  // namespace ddpm
