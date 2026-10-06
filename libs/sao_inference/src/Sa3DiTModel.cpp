#include "sao_inference/Sa3DiTModel.h"

#include <array>
#include <iostream>
#include <stdexcept>

namespace sao {

Sa3DiTModel::Sa3DiTModel(const std::string& onnx_path, const std::vector<std::string>& input_names, const std::string& output_name,
                         const std::string& migraphx_cache_dir, bool use_cuda, bool use_coreml, bool use_migraphx)
    : m_model(onnx_path, migraphx_cache_dir, use_cuda, use_coreml, use_migraphx)
    , m_input_names(input_names)
    , m_output_name(output_name)
{
    if (m_input_names.size() != 7)
        throw std::runtime_error("Sa3DiTModel: expected 7 input names, got " + std::to_string(m_input_names.size()));
    std::cout << "[Sa3DiTModel] Loaded " << onnx_path << std::endl;
}

std::vector<float> Sa3DiTModel::forward(
    const std::vector<float>& x,
    const std::vector<float>& t,
    const std::vector<float>& cross_attn_cond,
    const std::vector<float>& global_embed,
    const std::vector<float>& local_add_cond,
    const std::vector<float>& control,
    const std::vector<float>& tf_inpaint_mask,
    const Dims& d)
{
    const auto B = static_cast<size_t>(d.batch);
    const auto C = static_cast<size_t>(d.channels);
    const auto T = static_cast<size_t>(d.frames);
    auto check = [](const char* name, size_t got, size_t want) {
        if (got != want)
            throw std::runtime_error(std::string("Sa3DiTModel: ") + name + " has " + std::to_string(got) + " values, expected " + std::to_string(want));
    };
    check("x", x.size(), B * C * T);
    check("t", t.size(), B);
    check("cross_attn_cond", cross_attn_cond.size(), B * static_cast<size_t>(d.cross_tokens) * static_cast<size_t>(d.cond_dim));
    check("global_embed", global_embed.size(), B * static_cast<size_t>(d.global_dim));
    check("local_add_cond", local_add_cond.size(), B * static_cast<size_t>(d.local_add_channels) * T);
    check("control", control.size(), B * static_cast<size_t>(d.control_channels) * T);
    check("tf_inpaint_mask", tf_inpaint_mask.size(), B * T);

    const std::array<int64_t, 3> x_shape = {d.batch, d.channels, d.frames};
    const std::array<int64_t, 1> t_shape = {d.batch};
    const std::array<int64_t, 3> cross_shape = {d.batch, d.cross_tokens, d.cond_dim};
    const std::array<int64_t, 2> global_shape = {d.batch, d.global_dim};
    const std::array<int64_t, 3> local_shape = {d.batch, d.local_add_channels, d.frames};
    const std::array<int64_t, 3> control_shape = {d.batch, d.control_channels, d.frames};
    const std::array<int64_t, 3> tf_shape = {d.batch, 1, d.frames};

    auto tensor = [&](const std::vector<float>& v, const int64_t* shape, size_t rank) {
        return Ort::Value::CreateTensor<float>(m_model.memory_info(), const_cast<float*>(v.data()), v.size(), shape, rank);
    };
    std::vector<Ort::Value> inputs;
    inputs.reserve(7);
    inputs.push_back(tensor(x, x_shape.data(), x_shape.size()));
    inputs.push_back(tensor(t, t_shape.data(), t_shape.size()));
    inputs.push_back(tensor(cross_attn_cond, cross_shape.data(), cross_shape.size()));
    inputs.push_back(tensor(global_embed, global_shape.data(), global_shape.size()));
    inputs.push_back(tensor(local_add_cond, local_shape.data(), local_shape.size()));
    inputs.push_back(tensor(control, control_shape.data(), control_shape.size()));
    inputs.push_back(tensor(tf_inpaint_mask, tf_shape.data(), tf_shape.size()));

    std::vector<const char*> names;
    names.reserve(7);
    for (const auto& n : m_input_names) names.push_back(n.c_str());

    auto outputs = m_model.run(names, inputs);
    if (outputs.empty()) throw std::runtime_error("Sa3DiTModel: the graph returned no outputs");
    const float* data = outputs[0].GetTensorData<float>();
    return std::vector<float>(data, data + B * C * T);
}

} // namespace sao
