#include "ops/identity.h"

#include <cstring>

namespace ddpm {

class IdentityOp : public Op {
  public:
    void forward(const std::vector<const Tensor*>& in,
                 std::vector<Tensor*>& out) const override {
        std::memcpy(out[0]->data.data(), in[0]->data.data(),
                    in[0]->numel() * sizeof(float));
    }
};

std::unique_ptr<Op> make_identity() { return std::make_unique<IdentityOp>(); }

}  // namespace ddpm
