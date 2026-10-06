#include "sao_inference/VAEEncoder.h"
#include <cassert>
#include <iostream>

namespace sao {

VAEEncoder::VAEEncoder(const std::string& onnx_path, bool use_cuda, bool use_coreml, bool use_migraphx,
                       const std::string& migraphx_cache_dir)
    : m_model(onnx_path, migraphx_cache_dir, use_cuda, use_coreml, use_migraphx)
{
}

std::vector<float> VAEEncoder::encode(
    const std::vector<float>& audio,
    int num_samples,
    int latent_dim)
{
    return encode_batch(audio, 1, num_samples, latent_dim);
}

std::vector<float> VAEEncoder::encode_batch(
    const std::vector<float>& audio,
    int batch_size,
    int num_samples,
    int latent_dim)
{
    constexpr int CHANNELS = 2;
    assert(batch_size >= 1); // the graph's batch axis is static: the caller passes what it was exported with
    assert(num_samples > 0);
    assert(latent_dim > 0);
    assert(static_cast<int64_t>(audio.size()) == static_cast<int64_t>(batch_size) * CHANNELS * num_samples);

    std::array<int64_t, 3> shape = {
        static_cast<int64_t>(batch_size),
        CHANNELS,
        static_cast<int64_t>(num_samples),
    };

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
    assert(out_shape[0] == batch_size);
    assert(out_shape[1] == latent_dim);

    size_t total = 1;
    for (auto d : out_shape) total *= d;

    const float* data = out.GetTensorData<float>();
    return std::vector<float>(data, data + total);
}

} // namespace sao
