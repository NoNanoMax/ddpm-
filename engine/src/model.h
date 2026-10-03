#pragma once
// Загрузка .ddpm и прогон графа.
//
// Формат файла (.ddpm v0):
//   DDPM-V0                          
//   input <name> <ndim> <d0> ...     
//   output <name>                    
//   tensor <name> <ndim> <d0> ...    
//   op <idx> <type> <in...> <out> {k=v ...}   
//   weights                          
//   <raw f32: все tensor в порядке объявления>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "op.h"
#include "tensor.h"

namespace ddpm {

struct OpDef {
    std::string type;
    std::vector<std::string> inputs;
    std::string output;
    std::map<std::string, int> attrs;
};

class Model {
  public:
    static Model load(const std::string& path);

    // in: имена входов → данные (batch произвольный). Выводит выход(и) графа.
    const std::vector<Tensor>& run(const std::map<std::string, Tensor>& in);

    const std::vector<std::string>& inputs() const { return inputs_; }
    const std::vector<std::string>& outputs() const { return outputs_; }

  private:
    std::vector<std::string> inputs_;
    std::vector<std::string> outputs_;
    std::map<std::string, Tensor> weights_;
    std::vector<std::string> weight_order_;  // порядок = порядок записи в файле!
    // (std::map итерируется лексикографически — для чтения blob'а не годится)
    std::vector<OpDef> opdefs_;
    std::vector<std::unique_ptr<Op>> ops_;
    std::vector<Tensor> cache_;  // результат последнего run()
};

}  // namespace ddpm
