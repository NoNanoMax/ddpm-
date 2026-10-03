#include "ops/concat.h"

#include <cstring>

namespace ddpm {

class ConcatOp : public Op {
  public:
    void forward(const std::vector<const Tensor*>& in,
                 std::vector<Tensor*>& out) const override {
        const Tensor& a = *in[0];
        const Tensor& b = *in[1];
        Tensor& y = *out[0];

        int64_t B = a.shape.values[0];
        int64_t A = a.shape.values[1];
        int64_t C = b.shape.values[1];
        for (int64_t i = 0; i < B; ++i) {
            float* dst = &y.data[i * (A + C)];
            std::memcpy(dst, &a.data[i * A], A * sizeof(float));
            std::memcpy(dst + A, &b.data[i * C], C * sizeof(float));
        }
    }
};

std::unique_ptr<Op> make_concat() { return std::make_unique<ConcatOp>(); }

}  // namespace ddpm
