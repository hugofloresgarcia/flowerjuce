#pragma once

#include "sao_inference/IVaeDecoder.h"
#include "sao_inference/IVaeEncoder.h"

#include <memory>
#include <string>

namespace sao {

/// Opaque MLX weights + architecture (defined in MlxZenonVae.cpp).
struct MlxVaeBundle;

/// Load safetensors + vae_config.json into a shared bundle (encoder and decoder reuse tensors).
std::shared_ptr<MlxVaeBundle> load_mlx_vae_bundle(
    const std::string& weights_path,
    const std::string& config_path);

/// MLX Metal VAE encoder (mean-only output matches ONNX MeanOnlyVAEEncoder).
class MlxVaeEncoder final : public IVaeEncoder {
public:
    explicit MlxVaeEncoder(std::shared_ptr<MlxVaeBundle> bundle);
    ~MlxVaeEncoder() override;

    std::vector<float> encode(
        const std::vector<float>& audio,
        int num_samples,
        int latent_dim) override;

private:
    std::shared_ptr<MlxVaeBundle> m_bundle;
};

/// MLX Metal VAE decoder (applies vae_scale like VAEDecoder).
class MlxVaeDecoder final : public IVaeDecoder {
public:
    MlxVaeDecoder(std::shared_ptr<MlxVaeBundle> bundle, float vae_scale);
    ~MlxVaeDecoder() override;

    std::vector<float> decode(const std::vector<float>& latents, int latent_length) override;

private:
    std::shared_ptr<MlxVaeBundle> m_bundle;
    float m_scale;
};

} // namespace sao
