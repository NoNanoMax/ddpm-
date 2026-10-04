#include "model.h"

#include <fstream>
#include <sstream>
#include <stdexcept>

#include "ops/add.h"
#include "ops/attention.h"
#include "ops/concat.h"
#include "ops/conv2d.h"
#include "ops/groupnorm.h"
#include "ops/identity.h"
#include "ops/linear.h"
#include "ops/silu.h"
#include "ops/sincos.h"
#include "ops/upsample.h"

namespace ddpm {

namespace {

// shape выходов op по shape'ам входов
Dims infer_shape(const OpDef& def,
                 const std::map<std::string, Tensor>& vars,
                 int batch) {
    if (def.type == "linear") {
        const Tensor& W = vars.at(def.inputs[1]);
        return {static_cast<int64_t>(batch), W.shape.values[0]};
    }
    if (def.type == "silu" || def.type == "groupnorm" || def.type == "add" || def.type == "identity")
        return vars.at(def.inputs[0]).shape;
    if (def.type == "addb")
        return vars.at(def.inputs[0]).shape;
    if (def.type == "conv2d") {
        const Tensor& x = vars.at(def.inputs[0]);
        int kh = def.attrs.at("kh"), kw = def.attrs.at("kw");
        int stride = def.attrs.at("stride"), pad = def.attrs.at("pad");
        int64_t H = x.shape.values[1], Wd = x.shape.values[2];
        int64_t Ho = (H + 2 * pad - kh) / stride + 1;
        int64_t Wo = (Wd + 2 * pad - kw) / stride + 1;
        const Tensor& W = vars.at(def.inputs[1]);
        return {static_cast<int64_t>(batch), Ho, Wo, W.shape.values[0]};
    }
    if (def.type == "upsample") {
        const Tensor& x = vars.at(def.inputs[0]);
        int s = def.attrs.at("scale");
        return {static_cast<int64_t>(batch), x.shape.values[1] * s,
                x.shape.values[2] * s, x.shape.values[3]};
    }
    if (def.type == "attention") {
        const Tensor& x = vars.at(def.inputs[0]);
        Dims d = x.shape;  // (B,H,W,3C) → (B,H,W,C)
        d.values.back() /= 3;
        return d;
    }
    if (def.type == "concat") {
        Dims d = vars.at(def.inputs[0]).shape;
        d.values.back() += vars.at(def.inputs[1]).shape.values.back();
        return d;
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
    bool in_attrs = false;
    auto parse_attr = [&](std::string t) {
        if (!t.empty() && t.front() == '{') t = t.substr(1);
        if (!t.empty() && t.back() == '}') t.pop_back();
        auto pos = t.find('=');
        def.attrs[t.substr(0, pos)] = std::stoi(t.substr(pos + 1));
    };
    while (is >> tok) {
        if (in_attrs) {
            parse_attr(tok);
            continue;
        }
        if (!tok.empty() && tok.front() == '{') {
            in_attrs = true;
            parse_attr(tok);
            continue;
        }
        def.inputs.push_back(tok);
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
    for (int idx = 0; idx < static_cast<int>(m.opdefs_.size()); ++idx) {
        const auto& d = m.opdefs_[idx];
        if (d.type == "linear") m.ops_.push_back(make_linear());
        else if (d.type == "silu") m.ops_.push_back(make_silu());
        else if (d.type == "concat") m.ops_.push_back(make_concat());
        else if (d.type == "add") m.ops_.push_back(make_add());
        else if (d.type == "addb") m.ops_.push_back(make_addb());
        else if (d.type == "upsample") m.ops_.push_back(make_upsample(d.attrs.at("scale")));
        else if (d.type == "attention") m.ops_.push_back(make_attention());
        else if (d.type == "identity") m.ops_.push_back(make_identity());
        else if (d.type == "groupnorm") m.ops_.push_back(make_groupnorm(d.attrs.at("groups")));
        else if (d.type == "conv2d") {
            auto get = [&](const char* k) {
                auto it = d.attrs.find(k);
                if (it == d.attrs.end())
                    throw std::runtime_error("conv2d op #" + std::to_string(idx) +
                                             ": no attr " + k);
                return it->second;
            };
            m.ops_.push_back(make_conv2d(get("kh"), get("kw"), get("stride"), get("pad")));
        }
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
