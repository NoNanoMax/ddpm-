#include "ops/attention.h"

#include <algorithm>
#include <cmath>

namespace ddpm {

class AttentionOp : public Op {
  public:
    void forward(const std::vector<const Tensor*>& in,
                 std::vector<Tensor*>& out) const override {
        const Tensor& x = *in[0];
        Tensor& y = *out[0];

        int64_t B = x.shape.values[0];
        int64_t HW = x.shape.values[1] * x.shape.values[2];
        int64_t C = x.shape.values[3] / 3;

        // рабочие buffers (v0: пересоздаём на каждом вызове — оптимизируем в v1)
        std::vector<float> q(B * HW * C), k(B * C * HW), v(B * HW * C), a(B * HW * HW);
        for (int64_t n = 0; n < B; ++n)
            for (int64_t s = 0; s < HW; ++s)
                for (int64_t c = 0; c < C; ++c) {
                    const float* p = &x.data[(n * HW + s) * 3 * C + c];
                    q[(n * HW + s) * C + c] = p[0];
                    k[(n * C + c) * HW + s] = p[C];
                    v[(n * HW + s) * C + c] = p[2 * C];
                }

        float inv_sqrt_c = 1.0f / std::sqrt(float(C));
        for (int64_t n = 0; n < B; ++n) {
            for (int64_t i = 0; i < HW; ++i) {
                const float* qi = &q[(n * HW + i) * C];
                float* ai = &a[(n * HW + i) * HW];
                float amax = -1e30f;
                for (int64_t j = 0; j < HW; ++j) {
                    const float* kj = &k[(n * C) * HW + j];
                    float s = 0;
                    for (int64_t c = 0; c < C; ++c)
                        s += qi[c] * kj[c * HW];
                    ai[j] = s * inv_sqrt_c;
                    amax = std::max(amax, ai[j]);
                }
                double sum = 0;
                for (int64_t j = 0; j < HW; ++j) {
                    ai[j] = std::exp(ai[j] - amax);
                    sum += ai[j];
                }
                float inv_sum = 1.0f / float(sum);
                float* yi = &y.data[(n * HW + i) * C];
                for (int64_t c = 0; c < C; ++c) yi[c] = 0;
                for (int64_t j = 0; j < HW; ++j) {
                    float w = ai[j] * inv_sum;
                    const float* vj = &v[(n * HW + j) * C];
                    for (int64_t c = 0; c < C; ++c)
                        yi[c] += w * vj[c];
                }
            }
        }
    }
};

std::unique_ptr<Op> make_attention() { return std::make_unique<AttentionOp>(); }

}  // namespace ddpm
