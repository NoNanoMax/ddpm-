#pragma once
// Контракт операции.
//
// Это ГЛАВНАЯ граница абстракции: бэкенды (scalar → AVX2 → CUDA) реализуют
// только forward(), а всё остальное (граф, рантайм, тесты) от этого не зависит.
//
// Факторизация на уровне op (один op — N реализаций в разных бэкендах)
// появляется в v1: сейчас за каждой op один .cpp со scalar-кодом,
// он же служит эталоном для parity-тестов.
#include <string>
#include <vector>

#include "tensor.h"

namespace ddpm {

class Op {
  public:
    virtual ~Op() = default;
    // in: входы в порядке, объявленном в графе
    // out: выходы, уже выделенные рантаймом (заполняем данные)
    virtual void forward(const std::vector<const Tensor*>& in,
                         std::vector<Tensor*>& out) const = 0;
};

}  // namespace ddpm
