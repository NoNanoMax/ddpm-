#include "ops/groupnorm.h"

#include <cmath>
#include <numeric>

namespace ddpm {

class GroupNormOp : public Op {
  public:
    explicit GroupNormOp(int groups) : g_(groups) {}

    void forward(const std::vector<const Tensor*>& in,
                 std::vector<Tensor*>& out) const override {
        const Tensor& x = *in[0];
        const Tensor& gamma = *in[1];  // (C)
        const Tensor& beta = in.size() > 2 ? *in[2] : Tensor{};  // (C)
        Tensor& y = *out[0];

        int64_t B = x.shape.values[0];
        int64_t spatial = x.shape.values[1] * x.shape.values[2];
        int64_t C = x.shape.values[3];
        int64_t per_group = spatial * (C / g_);

        for (int64_t n = 0; n < B; ++n) {
            const float* xb = &x.data[n * spatial * C];
            float* yb = &y.data[n * spatial * C];
            for (int64_t g = 0; g < g_; ++g) {
                // каналы группы: [g*(C/g), (g+1)*(C/g)) — в NHWC каналы НЕ
                // смежны, блок канала на каждой пространственной позиции
                int64_t c0 = g * (C / g_);
                double sum = 0, sum2 = 0;
                for (int64_t s = 0; s < spatial; ++s)
                    for (int64_t c = c0; c < c0 + C / g_; ++c) {
                        float v = xb[s * C + c];
                        sum += v;
                        sum2 += double(v) * v;
                    }
                double mean = sum / per_group;
                double var = sum2 / per_group - mean * mean;
                float inv = 1.0f / std::sqrt(var + 1e-5f);
                for (int64_t s = 0; s < spatial; ++s)
                    for (int64_t c = c0; c < c0 + C / g_; ++c) {
                        float gma = gamma.data[c];
                        float bta = beta.data.empty() ? 0.0f : beta.data[c];
                        yb[s * C + c] = (xb[s * C + c] - mean) * inv * gma + bta;
                    }
            }
        }
    }

  private:
    int g_;
};

std::unique_ptr<Op> make_groupnorm(int groups) {
    return std::make_unique<GroupNormOp>(groups);
}

}  // namespace ddpm
