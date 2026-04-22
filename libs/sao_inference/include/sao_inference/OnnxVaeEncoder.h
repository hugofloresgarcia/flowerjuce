#pragma once

#include "sao_inference/IVaeEncoder.h"
#include "sao_inference/VAEEncoder.h"

#include <memory>
#include <string>

namespace sao {

/// ONNX Runtime VAE encoder (wraps VAEEncoder).
class OnnxVaeEncoder final : public IVaeEncoder {
public:
    explicit OnnxVaeEncoder(const std::string& onnx_path, bool use_cuda, bool use_coreml, bool use_migraphx = false);

    std::vector<float> encode(
        const std::vector<float>& audio,
        int num_samples,
        int latent_dim) override;

private:
    VAEEncoder m_encoder;
};

} // namespace sao
