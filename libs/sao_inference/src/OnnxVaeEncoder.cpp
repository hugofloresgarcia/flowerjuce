#include "sao_inference/OnnxVaeEncoder.h"

namespace sao {

OnnxVaeEncoder::OnnxVaeEncoder(const std::string& onnx_path, bool use_cuda, bool use_coreml)
    : m_encoder(onnx_path, use_cuda, use_coreml)
{
}

std::vector<float> OnnxVaeEncoder::encode(
    const std::vector<float>& audio,
    int num_samples,
    int latent_dim)
{
    return m_encoder.encode(audio, num_samples, latent_dim);
}

} // namespace sao
