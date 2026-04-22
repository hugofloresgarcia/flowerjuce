#include "sao_inference/DiTInpaintModel.h"
#include <cassert>
#include <iostream>
#include <algorithm>

namespace sao {

DiTInpaintModel::DiTInpaintModel(const std::string& onnx_path, bool use_cuda, bool use_coreml, bool use_migraphx)
    : m_model(onnx_path, use_cuda, use_coreml, use_migraphx)
{
    auto session = &m_model;

    std::cout << "[DiTInpaintModel] Loaded from " << onnx_path << std::endl;
    std::cout << "[DiTInpaintModel] ONNX model accepts variable input_add tensors" << std::endl;
}

std::vector<float> DiTInpaintModel::forward(
    const std::vector<float>& x,
    const std::vector<float>& t,
    const std::vector<float>& cross_attn_cond,
    const std::vector<float>& global_embed,
    const std::vector<InputAddTensor>& input_add,
    int batch_size,
    int seq_len,
    int latent_channels,
    int latent_length,
    int cond_dim)
{
    int C = latent_channels;
    int T = latent_length;

    assert(static_cast<int>(x.size()) == batch_size * C * T);
    assert(static_cast<int>(t.size()) == batch_size);
    assert(static_cast<int>(cross_attn_cond.size()) == batch_size * seq_len * cond_dim);
    assert(static_cast<int>(global_embed.size()) == batch_size * cond_dim);

    std::vector<const char*> input_names;
    std::vector<Ort::Value> inputs;

    std::array<int64_t, 3> x_shape = {batch_size, C, T};
    std::array<int64_t, 1> t_shape = {batch_size};
    std::array<int64_t, 3> cond_shape = {batch_size, static_cast<int64_t>(seq_len), cond_dim};
    std::array<int64_t, 2> global_shape = {batch_size, cond_dim};

    input_names.push_back("x");
    inputs.push_back(Ort::Value::CreateTensor<float>(
        m_model.memory_info(),
        const_cast<float*>(x.data()), x.size(),
        x_shape.data(), x_shape.size()
    ));

    input_names.push_back("t");
    inputs.push_back(Ort::Value::CreateTensor<float>(
        m_model.memory_info(),
        const_cast<float*>(t.data()), t.size(),
        t_shape.data(), t_shape.size()
    ));

    input_names.push_back("cross_attn_cond");
    inputs.push_back(Ort::Value::CreateTensor<float>(
        m_model.memory_info(),
        const_cast<float*>(cross_attn_cond.data()), cross_attn_cond.size(),
        cond_shape.data(), cond_shape.size()
    ));

    input_names.push_back("global_embed");
    inputs.push_back(Ort::Value::CreateTensor<float>(
        m_model.memory_info(),
        const_cast<float*>(global_embed.data()), global_embed.size(),
        global_shape.data(), global_shape.size()
    ));

    // Keep shapes alive for the duration of the call
    std::vector<std::array<int64_t, 3>> add_shapes;
    add_shapes.reserve(input_add.size());

    // Stable storage for c_str pointers
    std::vector<std::string> add_name_storage;
    add_name_storage.reserve(input_add.size());

    for (const auto& add : input_add) {
        assert(static_cast<int>(add.data.size()) == batch_size * add.channels * T);

        add_name_storage.push_back(add.name);
        input_names.push_back(add_name_storage.back().c_str());

        add_shapes.push_back({batch_size, static_cast<int64_t>(add.channels), T});

        inputs.push_back(Ort::Value::CreateTensor<float>(
            m_model.memory_info(),
            const_cast<float*>(add.data.data()), add.data.size(),
            add_shapes.back().data(), add_shapes.back().size()
        ));
    }

    auto outputs = m_model.run(input_names, inputs);
    assert(!outputs.empty());

    auto& out = outputs[0];
    size_t total = batch_size * C * T;
    const float* data = out.GetTensorData<float>();
    return std::vector<float>(data, data + total);
}

} // namespace sao
