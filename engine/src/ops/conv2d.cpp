#include "ops/conv2d.h"

namespace ddpm {

class Conv2dOp : public Op {
  public:
    Conv2dOp(int kh, int kw, int stride, int pad)
        : kh_(kh), kw_(kw), stride_(stride), pad_(pad) {}

    void forward(const std::vector<const Tensor*>& in,
                 std::vector<Tensor*>& out) const override {
        const Tensor& x = *in[0];
        const Tensor& W = *in[1];
        const Tensor* b = in.size() > 2 ? in[2] : nullptr;
        Tensor& y = *out[0];

        int64_t B = x.shape.values[0];
        int64_t H = x.shape.values[1], Wd = x.shape.values[2], I = x.shape.values[3];
        int64_t O = W.shape.values[0];
        int64_t Ho = y.shape.values[1], Wo = y.shape.values[2];

        for (int64_t n = 0; n < B; ++n) {
            const float* xbase = &x.data[n * H * Wd * I];
            float* ybase = &y.data[n * Ho * Wo * O];
            for (int64_t oh = 0; oh < Ho; ++oh) {
                for (int64_t ow = 0; ow < Wo; ++ow) {
                    for (int64_t o = 0; o < O; ++o) {
                        float s = b ? b->data[o] : 0.0f;
                        for (int fkh = 0; fkh < kh_; ++fkh) {
                            int64_t h = oh * stride_ - pad_ + fkh;
                            if (h < 0 || h >= H) continue;
                            for (int fkw = 0; fkw < kw_; ++fkw) {
                                int64_t w = ow * stride_ - pad_ + fkw;
                                if (w < 0 || w >= Wd) continue;
                                const float* xp = xbase + (h * Wd + w) * I;
                                const float* wp = &W.data[((o * I + 0) * kh_ + fkh) * kw_ + fkw];
                                for (int64_t i = 0; i < I; ++i)
                                    s += xp[i] * wp[i * kh_ * kw_];
                            }
                        }
                        ybase[(oh * Wo + ow) * O + o] = s;
                    }
                }
            }
        }
    }

  private:
    int kh_, kw_, stride_, pad_;
};

std::unique_ptr<Op> make_conv2d(int kh, int kw, int stride, int pad) {
    return std::make_unique<Conv2dOp>(kh, kw, stride, pad);
}

}  // namespace ddpm
