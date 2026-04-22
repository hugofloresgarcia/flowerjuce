#include "sao_inference/OnnxVaeDecoder.h"

namespace sao {

OnnxVaeDecoder::OnnxVaeDecoder(
    const std::string& onnx_path,
    float vae_scale,
    bool use_cuda,
    bool use_coreml,
    bool use_migraphx)
    : m_decoder(onnx_path, vae_scale, use_cuda, use_coreml, use_migraphx)
{
}

std::vector<float> OnnxVaeDecoder::decode(const std::vector<float>& latents, int latent_length)
{
    return m_decoder.decode(latents, latent_length);
}

} // namespace sao
