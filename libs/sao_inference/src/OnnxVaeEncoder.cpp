#include "sao_inference/OnnxVaeEncoder.h"

namespace sao {

OnnxVaeEncoder::OnnxVaeEncoder(const std::string& onnx_path, bool use_cuda, bool use_coreml, bool use_migraphx)
    : m_encoder(onnx_path, use_cuda, use_coreml, use_migraphx)
{
}

std::vector<float> OnnxVaeEncoder::encode(
    const std::vector<float>& audio,
    int num_samples,
    int latent_dim)
{
    return m_encoder.encode(audio, num_samples, latent_dim);
}

std::vector<float> OnnxVaeEncoder::encode_batch(
    const std::vector<float>& audio,
    int batch_size,
    int num_samples,
    int latent_dim)
{
    return m_encoder.encode_batch(audio, batch_size, num_samples, latent_dim);
}

} // namespace sao
