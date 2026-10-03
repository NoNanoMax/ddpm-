#pragma once
// Тензор v0: shape + плотный массив float32, row-major (как numpy)
//
// Замечение на будущее: сейчас память = std::vector<float>. Когда появится
// CUDA-бэкенд, data станет указателем на память бэкенда (host/device),
// и появится allocator
#include <cstdint>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace ddpm {

struct Dims {
    std::vector<int64_t> values;

    Dims() = default;
    Dims(std::initializer_list<int64_t> v) : values(v) {}

    int64_t numel() const {
        int64_t n = 1;
        for (auto d : values) n *= d;
        return n;
    }

    std::string str() const {
        std::ostringstream os;
        os << "(";
        for (size_t i = 0; i < values.size(); ++i)
            os << (i ? ", " : "") << values[i];
        os << ")";
        return os.str();
    }
};

struct Tensor {
    Dims shape;
    std::vector<float> data;  // row-major, numel() элементов

    Tensor() = default;
    explicit Tensor(Dims s) : shape(s), data(s.numel(), 0.0f) {}

    int64_t numel() const { return shape.numel(); }
    float& at(size_t i) { return data[i]; }
    float at(size_t i) const { return data[i]; }
};

}  // namespace ddpm
