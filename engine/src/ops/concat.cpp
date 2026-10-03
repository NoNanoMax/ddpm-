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

        int64_t outer = 1;
        for (size_t i = 0; i < a.shape.values.size() - 1; ++i)
            outer *= a.shape.values[i];
        int64_t A = a.shape.values.back();
        int64_t C = b.shape.values.back();

        for (int64_t r = 0; r < outer; ++r) {
            float* dst = &y.data[r * (A + C)];
            std::memcpy(dst, &a.data[r * A], A * sizeof(float));
            std::memcpy(dst + A, &b.data[r * C], C * sizeof(float));
        }
    }
};

std::unique_ptr<Op> make_concat() { return std::make_unique<ConcatOp>(); }

}  // namespace ddpm
