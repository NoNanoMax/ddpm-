#include "ops/add.h"

namespace ddpm {

class AddOp : public Op {
  public:
    void forward(const std::vector<const Tensor*>& in,
                 std::vector<Tensor*>& out) const override {
        const Tensor& x = *in[0];
        const Tensor& y = *in[1];
        Tensor& z = *out[0];
        for (size_t i = 0; i < x.numel(); ++i)
            z.data[i] = x.data[i] + y.data[i];
    }
};

class AddBOp : public Op {
  public:
    void forward(const std::vector<const Tensor*>& in,
                 std::vector<Tensor*>& out) const override {
        const Tensor& x = *in[0];  // (B,H,W,C)
        const Tensor& t = *in[1];  // (B,C)
        Tensor& z = *out[0];
        int64_t B = x.shape.values[0];
        int64_t spatial = x.shape.values[1] * x.shape.values[2];
        int64_t C = x.shape.values[3];
        for (int64_t n = 0; n < B; ++n)
            for (int64_t s = 0; s < spatial; ++s)
                for (int64_t c = 0; c < C; ++c)
                    z.data[(n * spatial + s) * C + c] =
                        x.data[(n * spatial + s) * C + c] + t.data[n * C + c];
    }
};

std::unique_ptr<Op> make_add() { return std::make_unique<AddOp>(); }
std::unique_ptr<Op> make_addb() { return std::make_unique<AddBOp>(); }

}  // namespace ddpm
