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
                const float* p = xb + g * (C / g_) * spatial;
                // mean/var по (spatial, C/g)
                double sum = 0, sum2 = 0;
                for (int64_t i = 0; i < C / g_; ++i)
                    for (int64_t s = 0; s < spatial; ++s) {
                        float v = p[i * spatial + s];
                        sum += v;
                        sum2 += double(v) * v;
                    }
                double mean = sum / per_group;
                double var = sum2 / per_group - mean * mean;
                float inv = 1.0f / std::sqrt(var + 1e-5f);
                for (int64_t i = 0; i < C / g_; ++i) {
                    float gma = gamma.data[g * (C / g_) + i];
                    float bta = beta.data.empty() ? 0.0f : beta.data[g * (C / g_) + i];
                    float* q = yb + g * (C / g_) * spatial + i * spatial;
                    const float* r = p + i * spatial;
                    for (int64_t s = 0; s < spatial; ++s)
                        q[s] = (r[s] - mean) * inv * gma + bta;
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
