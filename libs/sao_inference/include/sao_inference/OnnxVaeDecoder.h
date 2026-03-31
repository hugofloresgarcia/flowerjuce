#pragma once

#include "sao_inference/IVaeDecoder.h"
#include "sao_inference/VAEDecoder.h"

#include <string>

namespace sao {

/// ONNX Runtime VAE decoder (wraps VAEDecoder).
class OnnxVaeDecoder final : public IVaeDecoder {
public:
    explicit OnnxVaeDecoder(const std::string& onnx_path, float vae_scale, bool use_cuda, bool use_coreml);

    std::vector<float> decode(const std::vector<float>& latents, int latent_length) override;

private:
    VAEDecoder m_decoder;
};

} // namespace sao
