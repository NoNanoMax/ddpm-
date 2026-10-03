#include "ops/sincos.h"

#include <cmath>

namespace ddpm {

class SincosOp : public Op {
  public:
    explicit SincosOp(int half_dim) : half_(half_dim) {
        for (int i = 0; i < half_; ++i) {
            double f = std::exp(-static_cast<double>(i) *
                                std::log(10000.0) / (half_ - 1));
            freq_.push_back(static_cast<float>(f));
        }
    }

    void forward(const std::vector<const Tensor*>& in,
                 std::vector<Tensor*>& out) const override {
        const Tensor& t = *in[0];
        Tensor& y = *out[0];
        int64_t B = t.numel();
        for (int64_t i = 0; i < B; ++i) {
            float tv = static_cast<float>(t.at(i));
            for (int j = 0; j < half_; ++j) {
                float arg = tv * freq_[j];
                y.at(i * 2 * half_ + j) = std::sin(arg);
                y.at(i * 2 * half_ + half_ + j) = std::cos(arg);
            }
        }
    }

  private:
    int half_;
    std::vector<float> freq_;
};

std::unique_ptr<Op> make_sincos(int half_dim) {
    return std::make_unique<SincosOp>(half_dim);
}

}  // namespace ddpm
