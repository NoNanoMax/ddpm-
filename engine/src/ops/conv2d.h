#pragma once
#include <memory>
#include "op.h"

namespace ddpm {
// Conv2d, NHWC. x: (B,H,W,I)  W: (O,I,kh,kw)  b: (O)  →  (B,Ho,Wo,O)
// attrs: kh, kw, stride, pad
std::unique_ptr<Op> make_conv2d(int kh, int kw, int stride, int pad);
}  // namespace ddpm
