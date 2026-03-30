#include "sao_inference/VAEDecoder.h"
#include <cassert>
#include <iostream>

namespace sao {

VAEDecoder::VAEDecoder(const std::string& onnx_path, float scale, bool use_cuda, bool use_coreml)
    : m_model(onnx_path, use_cuda, use_coreml)
    , m_scale(scale)
{
}

std::vector<float> VAEDecoder::decode(const std::vector<float>& latents, int latent_length)
{
    constexpr int LATENT_DIM = 64;
    assert(static_cast<int>(latents.size()) == LATENT_DIM * latent_length);

    // Apply scale before decoding
    std::vector<float> scaled(latents.size());
    for (size_t i = 0; i < latents.size(); ++i) {
        scaled[i] = latents[i] * m_scale;
    }

    std::array<int64_t, 3> shape = {1, LATENT_DIM, static_cast<int64_t>(latent_length)};

    auto tensor = Ort::Value::CreateTensor<float>(
        m_model.memory_info(),
        scaled.data(), scaled.size(),
        shape.data(), shape.size()
    );

    std::vector<const char*> input_names = {"latents"};
    std::vector<Ort::Value> inputs;
    inputs.push_back(std::move(tensor));

    auto outputs = m_model.run(input_names, inputs);
    assert(!outputs.empty());

    auto& out = outputs[0];
    auto info = out.GetTensorTypeAndShapeInfo();
    auto out_shape = info.GetShape();
    size_t total = 1;
    for (auto d : out_shape) total *= d;

    const float* data = out.GetTensorData<float>();
    return std::vector<float>(data, data + total);
}

} // namespace sao
