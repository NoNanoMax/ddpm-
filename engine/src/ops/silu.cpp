#include "ops/silu.h"

#include <cmath>

namespace ddpm {

class SiluOp : public Op {
  public:
    void forward(const std::vector<const Tensor*>& in,
                 std::vector<Tensor*>& out) const override {
        const Tensor& x = *in[0];
        Tensor& y = *out[0];
        for (size_t i = 0; i < x.numel(); ++i) {
            float v = x.at(i);
            y.at(i) = v / (1.0f + std::exp(-v));
        }
    }
};

std::unique_ptr<Op> make_silu() { return std::make_unique<SiluOp>(); }

}  // namespace ddpm
