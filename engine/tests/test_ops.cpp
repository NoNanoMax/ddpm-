// Unit-тесты операций v0.
// Linear проверяется против РУЧНОГО расчёта (не против PyTorch)
// Parity vs PyTorch — отдельный скрипт.
#include <cmath>
#include <cstdio>
#include <cstdlib>

#include "ops/concat.h"
#include "ops/linear.h"
#include "ops/silu.h"
#include "ops/sincos.h"

using namespace ddpm;

static int failures = 0;

#define CHECK(cond, msg) \
    do { \
        if (!(cond)) { \
            std::fprintf(stderr, "FAIL: %s\n", msg); \
            ++failures; \
        } \
    } while (0)

void test_linear() {
    // x: (2, 3)  W: (2, 3)  b: (2)
    Tensor x({2, 3});
    x.data = {1, 2, 3, 4, 5, 6};
    Tensor W({2, 3});
    W.data = {1, 0, 1, 0, 1, 0};  // y0 = x0+x2, y1 = x1
    Tensor b({2});
    b.data = {10, 20};

    auto op = make_linear();
    Tensor y({2, 2});
    std::vector<const Tensor*> ins = {&x, &W, &b};
    std::vector<Tensor*> outs = {&y};
    op->forward(ins, outs);

    CHECK(y.data[0] == 14.0f, "linear y[0][0] == 14 (1+3+10)");
    CHECK(y.data[1] == 22.0f, "linear y[0][1] == 22 (2+20)");
    CHECK(y.data[2] == 20.0f, "linear y[1][0] == 20 (4+6+10)");
    CHECK(y.data[3] == 25.0f, "linear y[1][1] == 25 (5+20)");
}

void test_silu() {
    Tensor x({3});
    x.data = {0.0f, 1.0f, -1.0f};
    auto op = make_silu();
    Tensor y({3});
    std::vector<const Tensor*> ins = {&x};
    std::vector<Tensor*> outs = {&y};
    op->forward(ins, outs);

    CHECK(y.data[0] == 0.0f, "silu(0) == 0");
    float e1 = 1.0f / (1.0f + std::exp(-1.0f));
    CHECK(std::abs(y.data[1] - e1) < 1e-6f, "silu(1)");
    float em1 = -1.0f / (1.0f + std::exp(1.0f));
    CHECK(std::abs(y.data[2] - em1) < 1e-6f, "silu(-1)");
}

void test_concat() {
    Tensor a({2, 2});
    a.data = {1, 2, 3, 4};
    Tensor b({2, 3});
    b.data = {5, 6, 7, 8, 9, 10};
    auto op = make_concat();
    Tensor y({2, 5});
    std::vector<const Tensor*> ins = {&a, &b};
    std::vector<Tensor*> outs = {&y};
    op->forward(ins, outs);
    const float exp_[] = {1, 2, 5, 6, 7, 3, 4, 8, 9, 10};
    for (int i = 0; i < 10; ++i)
        CHECK(y.data[i] == exp_[i], "concat layout");
}

void test_sincos() {
    Tensor t({2});
    t.data = {0.0f, 1.0f};
    auto op = make_sincos(2);  // half_dim=2 → out (2,4)
    Tensor y({2, 4});
    std::vector<const Tensor*> ins = {&t};
    std::vector<Tensor*> outs = {&y};
    op->forward(ins, outs);
    // f = [1.0, exp(-ln(10000))] = [1.0, 0.0001]
    CHECK(std::abs(y.data[0] - 0.0f) < 1e-7f, "sin(0)");
    CHECK(std::abs(y.data[2] - 1.0f) < 1e-7f, "cos(0)");
    CHECK(std::abs(y.data[1] - std::sin(0.0f)) < 1e-6f, "sin(t*f0), t=0");
    CHECK(std::abs(y.data[4] - std::sin(1.0f)) < 1e-6f, "sin(1*1.0)");
    CHECK(std::abs(y.data[5] - std::sin(0.0001f)) < 1e-6f, "sin(1*0.0001)");
    CHECK(std::abs(y.data[7] - std::cos(0.0001f)) < 1e-6f, "cos(1*0.0001)");
}

int main() {
    test_linear();
    test_silu();
    test_concat();
    test_sincos();
    if (failures) {
        std::fprintf(stderr, "%d failures\n", failures);
        return 1;
    }
    std::printf("all ops tests passed\n");
    return 0;
}
