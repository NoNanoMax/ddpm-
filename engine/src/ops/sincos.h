#pragma once
#include <memory>
#include "op.h"

namespace ddpm {
// Sinusoidal time embedding: t (B,) → cat(sin(t*f), cos(t*f)) (B, 2*half_dim)
// Атрибуты из графа: half_dim.
// Частоты: f_i = exp(-i * ln(10000)/(half_dim-1)), i = 0..half_dim-1.
// Те же формулы, что в models/toy/model.py::SinusoidalTimeEmbedding.
std::unique_ptr<Op> make_sincos(int half_dim);
}  // namespace ddpm
