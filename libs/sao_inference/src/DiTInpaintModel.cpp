#include "sao_inference/DiTInpaintModel.h"
#include <array>
#include <cassert>
#include <iostream>

namespace sao {

DiTInpaintModel::DiTInpaintModel(const std::string& onnx_path, bool use_cuda, bool use_coreml, bool use_migraphx)
    : m_model(onnx_path, use_cuda, use_coreml, use_migraphx)
{
    std::cout << "[DiTInpaintModel] Loaded from " << onnx_path
              << " (fused input_add_cond input)" << std::endl;
}

std::vector<float> DiTInpaintModel::forward(
    const std::vector<float>& x,
    const std::vector<float>& t,
    const std::vector<float>& cross_attn_cond,
    const std::vector<float>& global_embed,
    const std::vector<float>& input_add_cond,
    int batch_size,
    int seq_len,
    int latent_channels,
    int latent_length,
    int cond_dim,
    int input_add_total_channels)
{
    const int C = latent_channels;
    const int T = latent_length;

    assert(static_cast<int>(x.size()) == batch_size * C * T);
    assert(static_cast<int>(t.size()) == batch_size);
    assert(static_cast<int>(cross_attn_cond.size()) == batch_size * seq_len * cond_dim);
    assert(static_cast<int>(global_embed.size()) == batch_size * cond_dim);
    assert(static_cast<int>(input_add_cond.size()) == batch_size * input_add_total_channels * T);

    // ONNX input order matches scripts/export_zenon_dit.py exactly:
    //   ["x", "t", "cross_attn_cond", "global_embed", "input_add_cond"].
    static constexpr std::array<const char*, 5> kInputNames = {
        "x", "t", "cross_attn_cond", "global_embed", "input_add_cond"
    };

    std::array<int64_t, 3> x_shape    = {batch_size, C, T};
    std::array<int64_t, 1> t_shape    = {batch_size};
    std::array<int64_t, 3> cond_shape = {batch_size, static_cast<int64_t>(seq_len), cond_dim};
    std::array<int64_t, 2> global_shape = {batch_size, cond_dim};
    std::array<int64_t, 3> add_shape  = {batch_size, input_add_total_channels, T};

    std::vector<Ort::Value> inputs;
    inputs.reserve(kInputNames.size());

    inputs.push_back(Ort::Value::CreateTensor<float>(
        m_model.memory_info(),
        const_cast<float*>(x.data()), x.size(),
        x_shape.data(), x_shape.size()));

    inputs.push_back(Ort::Value::CreateTensor<float>(
        m_model.memory_info(),
        const_cast<float*>(t.data()), t.size(),
        t_shape.data(), t_shape.size()));

    inputs.push_back(Ort::Value::CreateTensor<float>(
        m_model.memory_info(),
        const_cast<float*>(cross_attn_cond.data()), cross_attn_cond.size(),
        cond_shape.data(), cond_shape.size()));

    inputs.push_back(Ort::Value::CreateTensor<float>(
        m_model.memory_info(),
        const_cast<float*>(global_embed.data()), global_embed.size(),
        global_shape.data(), global_shape.size()));

    inputs.push_back(Ort::Value::CreateTensor<float>(
        m_model.memory_info(),
        const_cast<float*>(input_add_cond.data()), input_add_cond.size(),
        add_shape.data(), add_shape.size()));

    std::vector<const char*> input_names(kInputNames.begin(), kInputNames.end());

    auto outputs = m_model.run(input_names, inputs);
    assert(!outputs.empty());

    auto& out = outputs[0];
    const size_t total = static_cast<size_t>(batch_size) * static_cast<size_t>(C) * static_cast<size_t>(T);
    const float* data = out.GetTensorData<float>();
    return std::vector<float>(data, data + total);
}

} // namespace sao
