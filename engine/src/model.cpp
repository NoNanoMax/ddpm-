#include "model.h"

#include <fstream>
#include <sstream>
#include <stdexcept>

#include "ops/concat.h"
#include "ops/linear.h"
#include "ops/silu.h"
#include "ops/sincos.h"

namespace ddpm {

namespace {

// shape выходов op по shape'ам входов (v0: только 2D-кейсы, достаточно для toy)
Dims infer_shape(const OpDef& def,
                 const std::map<std::string, Tensor>& vars,
                 int batch) {
    if (def.type == "linear") {
        const Tensor& W = vars.at(def.inputs[1]);
        return {static_cast<int64_t>(batch), W.shape.values[0]};
    }
    if (def.type == "silu")
        return vars.at(def.inputs[0]).shape;
    if (def.type == "concat") {
        const Tensor& a = vars.at(def.inputs[0]);
        const Tensor& b = vars.at(def.inputs[1]);
        return {static_cast<int64_t>(batch),
                a.shape.values[1] + b.shape.values[1]};
    }
    if (def.type == "sincos")
        return {static_cast<int64_t>(batch),
                2 * static_cast<int64_t>(def.attrs.at("half_dim"))};
    throw std::runtime_error("unknown op type: " + def.type);
}

OpDef parse_op_line(const std::string& line) {
    std::istringstream is(line);
    std::string kind, idx_s, type;
    is >> kind >> idx_s >> type;
    OpDef def;
    def.type = type;
    std::string tok;
    while (is >> tok) {
        if (tok == "{") {
            while (is >> tok && tok != "}") {
                auto pos = tok.find('=');
                def.attrs[tok.substr(0, pos)] = std::stoi(tok.substr(pos + 1));
            }
        } else {
            if (!tok.empty() && tok.front() == '{') {
                tok = tok.substr(1);
                auto pos = tok.find('=');
                def.attrs[tok.substr(0, pos)] = std::stoi(tok.substr(pos + 1));
                if (tok.back() == '}') continue;
                tok.clear();
                continue;
            }
            def.inputs.push_back(tok);
        }
    }
    def.output = def.inputs.back();
    def.inputs.pop_back();
    return def;
}

}  // namespace

Model Model::load(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("cannot open " + path);

    Model m;
    std::string line;
    while (std::getline(f, line)) {
        if (line.empty()) continue;
        if (line == "weights") break;
        std::istringstream is(line);
        std::string kind;
        is >> kind;
        if (kind == "DDPM-V0") {
            continue;
        } else if (kind == "input" || kind == "tensor") {
            std::string name;
            int ndim = 0;
            is >> name >> ndim;
            Dims d;
            for (int i = 0; i < ndim; ++i) {
                int64_t v;
                is >> v;
                d.values.push_back(v);
            }
            if (kind == "input")
                m.inputs_.push_back(name);
            else {
                m.weights_.emplace(name, Tensor(d));
                m.weight_order_.push_back(name);  // порядок записи в файл
            }
        } else if (kind == "output") {
            is >> line;
            m.outputs_.push_back(line);
        } else if (kind == "op") {
            m.opdefs_.push_back(parse_op_line(line));
        } else {
            throw std::runtime_error("bad header line: " + line);
        }
    }

    // сырые веса: всё, что после маркера "weights\n"
    f.seekg(0, std::ios::cur);
    size_t payload_bytes = 0;
    {
        auto end = f.tellg();
        f.seekg(0, std::ios::end);
        auto total = f.tellg();
        f.seekg(end, std::ios::beg);
        payload_bytes = static_cast<size_t>(total - end);
    }
    std::vector<float> buf(payload_bytes / sizeof(float));
    if (!buf.empty())
        f.read(reinterpret_cast<char*>(buf.data()),
               static_cast<std::streamsize>(payload_bytes));

    size_t off = 0;
    for (const auto& name : m.weight_order_) {  // В порядке файла, не map'а!
        Tensor& t = m.weights_.at(name);
        size_t n = static_cast<size_t>(t.numel());
        std::copy(buf.begin() + off, buf.begin() + off + n, t.data.begin());
        off += n;
    }
    if (off != buf.size())
        throw std::runtime_error("weight blob size mismatch");

    // строим ops
    m.ops_.reserve(m.opdefs_.size());
    for (const auto& d : m.opdefs_) {
        if (d.type == "linear") m.ops_.push_back(make_linear());
        else if (d.type == "silu") m.ops_.push_back(make_silu());
        else if (d.type == "concat") m.ops_.push_back(make_concat());
        else if (d.type == "sincos")
            m.ops_.push_back(make_sincos(d.attrs.at("half_dim")));
        else
            throw std::runtime_error("unknown op: " + d.type);
    }
    return m;
}

const std::vector<Tensor>& Model::run(
    const std::map<std::string, Tensor>& in) {
    int batch = static_cast<int>(in.begin()->second.shape.values[0]);

    std::map<std::string, Tensor> vars = weights_;
    for (const auto& [n, t] : in) vars[n] = t;

    for (size_t i = 0; i < opdefs_.size(); ++i) {
        const auto& d = opdefs_[i];
        vars[d.output] = Tensor(infer_shape(d, vars, batch));
        std::vector<const Tensor*> ins;
        for (const auto& n : d.inputs) ins.push_back(&vars[n]);
        std::vector<Tensor*> outs = {&vars[d.output]};
        ops_[i]->forward(ins, outs);
    }

    std::vector<Tensor> res;
    for (const auto& name : outputs_) res.push_back(vars.at(name));
    cache_ = std::move(res);
    return cache_;
}

}  // namespace ddpm
