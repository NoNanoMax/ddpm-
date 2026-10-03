// CLI движка v0.
//
// Использование:
//   ddpm-engine run <model.ddpm> --n <batch> --t <time> \
//       [--in x.bin] [--out out.bin]
//   ddpm-engine sample <model.ddpm> --n 4096 --t 256 \
//       [--beta-start 0.0001] [--beta-end 0.02] [--seed 42] [--out samples.bin]
//
// run:    один forward (parity-тесты, smoke)
// sample: полный reverse process: x_T ~ N(0,I) → T шагов → x_0 (out: (n, in_dim))
#include <chrono>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <random>
#include <string>

#include "model.h"

using namespace ddpm;

namespace {
void usage() {
    std::fprintf(stderr,
                 "usage: ddpm-engine run <model.ddpm> --n <batch> --t <time> "
                 "[--in x.bin] [--out out.bin]\n");
}
std::string arg_after(const char* name, int argc, char** argv,
                      std::string def = "") {
    for (int i = 1; i < argc - 1; ++i)
        if (std::string(argv[i]) == name) return argv[i + 1];
    return def;
}
}  // namespace

int main(int argc, char** argv) {
    if (argc < 3) {
        usage();
        return 1;
    }
    std::string mode = argv[1];
    std::string path = argv[2];

    if (mode == "sample") {
        int n = std::stoi(arg_after("--n", argc, argv, "4096"));
        int T = std::stoi(arg_after("--t", argc, argv, "256"));
        double bs = std::stod(arg_after("--beta-start", argc, argv, "0.0001"));
        double be = std::stod(arg_after("--beta-end", argc, argv, "0.02"));
        unsigned seed = std::stoul(arg_after("--seed", argc, argv, "42"));
        std::string out_path = arg_after("--out", argc, argv, "samples.bin");

        Model model = Model::load(path);

        // schedule: beta = linspace(bs, be, T)
        std::vector<double> beta(T), alpha(T), abar(T);
        for (int i = 0; i < T; ++i) {
            beta[i] = bs + (be - bs) * (T > 1 ? static_cast<double>(i) / (T - 1) : 0.0);
            alpha[i] = 1.0 - beta[i];
            abar[i] = (i == 0) ? alpha[i] : abar[i - 1] * alpha[i];
        }

        // x_T ~ N(0,I): Box-Muller
        std::mt19937 rng(seed);
        std::uniform_real_distribution<double> uni(0.0, 1.0);
        auto gauss = [&]() {
            double u1 = uni(rng), u2 = uni(rng);
            return std::sqrt(-2.0 * std::log(u1)) * std::cos(2.0 * M_PI * u2);
        };

        int in_dim = 2;  // v0: toy (B, 2)
        std::vector<float> x(static_cast<size_t>(n) * in_dim);
        for (auto& v : x) v = static_cast<float>(gauss());

        Tensor x_t({n, in_dim});
        x_t.data = std::move(x);
        Tensor t_t({n});

        printf("sampling %d samples in %d steps...\n", n, T);
        auto t0 = std::chrono::steady_clock::now();
        for (int t = T - 1; t >= 0; --t) {
            for (auto& v : t_t.data) v = static_cast<float>(t);
            auto out = model.run({{"x_t", x_t}, {"t", t_t}});
            const Tensor& pred = out.at(0);
            double c1 = 1.0 / std::sqrt(alpha[t]);
            double c2 = beta[t] / std::sqrt(1.0 - abar[t]);
            double sigma = std::sqrt(beta[t]);
            for (size_t i = 0; i < x_t.data.size(); ++i) {
                float z = (t > 0) ? static_cast<float>(gauss()) : 0.0f;
                x_t.data[i] = static_cast<float>(c1 * (x_t.data[i] - c2 * pred.data[i])
                                                 + sigma * z);
            }
        }
        auto t1 = std::chrono::steady_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

        std::ofstream f(out_path, std::ios::binary);
        f.write(reinterpret_cast<const char*>(x_t.data.data()),
                static_cast<std::streamsize>(x_t.data.size() * sizeof(float)));
        printf("done: %s (%lld floats), %.1f ms total, %.2f ms/step\n",
               out_path.c_str(), static_cast<long long>(x_t.data.size()), ms, ms / T);
        return 0;
    }

    if (mode != "run") {
        usage();
        return 1;
    }
    int n = std::stoi(arg_after("--n", argc, argv, "8"));
    int t = std::stoi(arg_after("--t", argc, argv, "17"));
    std::string in_path = arg_after("--in", argc, argv);
    std::string out_path = arg_after("--out", argc, argv, "out.bin");

    Model model = Model::load(path);
    printf("loaded %s: %zu inputs, %zu outputs\n", path.c_str(),
           model.inputs().size(), model.outputs().size());

    // вход x_t: (n, 2)
    std::vector<float> x(static_cast<size_t>(n) * 2);
    if (!in_path.empty()) {
        std::ifstream f(in_path, std::ios::binary);
        f.read(reinterpret_cast<char*>(x.data()), x.size() * sizeof(float));
    } else {
        std::mt19937 rng(0);
        std::normal_distribution<float> dist(0, 1);
        for (auto& v : x) v = dist(rng);
    }

    // вход t: (n,)
    std::vector<float> tv(x.size() / 2, static_cast<float>(t));

    Tensor x_t({n, 2});
    x_t.data = std::move(x);
    Tensor t_t({n});
    t_t.data = std::move(tv);

    auto out = model.run({{"x_t", x_t}, {"t", t_t}});
    const Tensor& y = out.at(0);

    std::ofstream f(out_path, std::ios::binary);
    f.write(reinterpret_cast<const char*>(y.data.data()),
            static_cast<std::streamsize>(y.numel() * sizeof(float)));

    printf("output: %s -> %s (%lld floats)\n",
           in_path.empty() ? "generated" : in_path.c_str(), out_path.c_str(),
           static_cast<long long>(y.numel()));
    return 0;
}
