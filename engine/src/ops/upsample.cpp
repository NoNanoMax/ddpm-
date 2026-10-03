#include "ops/upsample.h"

namespace ddpm {

class UpsampleOp : public Op {
  public:
    explicit UpsampleOp(int scale) : s_(scale) {}

    void forward(const std::vector<const Tensor*>& in,
                 std::vector<Tensor*>& out) const override {
        const Tensor& x = *in[0];
        Tensor& y = *out[0];
        int64_t B = x.shape.values[0];
        int64_t H = x.shape.values[1], Wd = x.shape.values[2], C = x.shape.values[3];
        for (int64_t n = 0; n < B; ++n)
            for (int64_t h = 0; h < H; ++h)
                for (int64_t w = 0; w < Wd; ++w) {
                    const float* p = &x.data[(n * H + h) * Wd * C + w * C];
                    for (int64_t c = 0; c < C; ++c) {
                        float v = p[c];
                        for (int dh = 0; dh < s_; ++dh)
                            for (int dw = 0; dw < s_; ++dw)
                                y.data[((n * s_ * H + h * s_ + dh) * s_ * Wd + w * s_ + dw) * C + c] = v;
                    }
                }
    }

  private:
    int s_;
};

std::unique_ptr<Op> make_upsample(int scale) {
    return std::make_unique<UpsampleOp>(scale);
}

}  // namespace ddpm
