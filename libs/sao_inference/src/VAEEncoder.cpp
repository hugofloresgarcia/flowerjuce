#include "sao_inference/VAEEncoder.h"
#include <cassert>
#include <iostream>

namespace sao {

VAEEncoder::VAEEncoder(const std::string& onnx_path, bool use_cuda, bool use_coreml, bool use_migraphx)
    : m_model(onnx_path, use_cuda, use_coreml, use_migraphx)
{
}

std::vector<float> VAEEncoder::encode(
    const std::vector<float>& audio,
    int num_samples,
    int latent_dim)
{
    constexpr int CHANNELS = 2;
    assert(static_cast<int>(audio.size()) == CHANNELS * num_samples);

    std::array<int64_t, 3> shape = {1, CHANNELS, static_cast<int64_t>(num_samples)};

    auto tensor = Ort::Value::CreateTensor<float>(
        m_model.memory_info(),
        const_cast<float*>(audio.data()), audio.size(),
        shape.data(), shape.size()
    );

    std::vector<const char*> input_names = {"audio"};
    std::vector<Ort::Value> inputs;
    inputs.push_back(std::move(tensor));

    auto outputs = m_model.run(input_names, inputs);
    assert(!outputs.empty());

    auto& out = outputs[0];
    auto info = out.GetTensorTypeAndShapeInfo();
    auto out_shape = info.GetShape();

    assert(out_shape.size() == 3);
    assert(out_shape[1] == latent_dim);

    size_t total = 1;
    for (auto d : out_shape) total *= d;

    const float* data = out.GetTensorData<float>();
    return std::vector<float>(data, data + total);
}

} // namespace sao
