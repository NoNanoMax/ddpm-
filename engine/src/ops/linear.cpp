#include "ops/linear.h"
#include <cmath>

namespace ddpm {

class LinearOp : public Op {
  public:
    void forward(const std::vector<const Tensor*>& in,
                 std::vector<Tensor*>& out) const override {
        const Tensor& x = *in[0];
        const Tensor& W = *in[1];
        const Tensor* b = in.size() > 2 ? in[2] : nullptr;
        Tensor& y = *out[0];

        int64_t B = x.shape.values[0];
        int64_t K = x.shape.values[1];
        int64_t N = W.shape.values[0];  // out

        for (int64_t i = 0; i < B; ++i) {
            const float* xr = &x.data[i * K];
            float* yr = &y.data[i * N];
            for (int64_t j = 0; j < N; ++j) {
                float s = b ? b->data[j] : 0.0f;
                const float* wr = &W.data[j * K];
                for (int64_t k = 0; k < K; ++k)
                    s += xr[k] * wr[k];
                yr[j] = s;
            }
        }
    }
};

std::unique_ptr<Op> make_linear() { return std::make_unique<LinearOp>(); }

}  // namespace ddpm
