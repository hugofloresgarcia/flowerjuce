#include "sao_inference/DiTModel.h"
#include <cassert>
#include <iostream>

namespace sao {

DiTModel::DiTModel(const std::string& onnx_path, bool use_cuda, bool use_coreml)
    : m_model(onnx_path, use_cuda, use_coreml)
{
}

std::vector<float> DiTModel::forward(
    const std::vector<float>& x,
    const std::vector<float>& t,
    const std::vector<float>& cross_attn_cond,
    const std::vector<float>& global_embed,
    int batch_size,
    int seq_len)
{
    constexpr int C = 64;
    constexpr int T = 256;
    constexpr int COND_DIM = 768;

    assert(static_cast<int>(x.size()) == batch_size * C * T);
    assert(static_cast<int>(t.size()) == batch_size);
    assert(static_cast<int>(cross_attn_cond.size()) == batch_size * seq_len * COND_DIM);
    assert(static_cast<int>(global_embed.size()) == batch_size * COND_DIM);

    std::array<int64_t, 3> x_shape = {batch_size, C, T};
    std::array<int64_t, 1> t_shape = {batch_size};
    std::array<int64_t, 3> cond_shape = {batch_size, static_cast<int64_t>(seq_len), COND_DIM};
    std::array<int64_t, 2> global_shape = {batch_size, COND_DIM};

    auto x_tensor = Ort::Value::CreateTensor<float>(
        m_model.memory_info(),
        const_cast<float*>(x.data()), x.size(),
        x_shape.data(), x_shape.size()
    );
    auto t_tensor = Ort::Value::CreateTensor<float>(
        m_model.memory_info(),
        const_cast<float*>(t.data()), t.size(),
        t_shape.data(), t_shape.size()
    );
    auto cond_tensor = Ort::Value::CreateTensor<float>(
        m_model.memory_info(),
        const_cast<float*>(cross_attn_cond.data()), cross_attn_cond.size(),
        cond_shape.data(), cond_shape.size()
    );
    auto global_tensor = Ort::Value::CreateTensor<float>(
        m_model.memory_info(),
        const_cast<float*>(global_embed.data()), global_embed.size(),
        global_shape.data(), global_shape.size()
    );

    std::vector<const char*> input_names = {"x", "t", "cross_attn_cond", "global_embed"};
    std::vector<Ort::Value> inputs;
    inputs.push_back(std::move(x_tensor));
    inputs.push_back(std::move(t_tensor));
    inputs.push_back(std::move(cond_tensor));
    inputs.push_back(std::move(global_tensor));

    auto outputs = m_model.run(input_names, inputs);
    assert(!outputs.empty());

    auto& out = outputs[0];
    size_t total = batch_size * C * T;
    const float* data = out.GetTensorData<float>();
    return std::vector<float>(data, data + total);
}

} // namespace sao
