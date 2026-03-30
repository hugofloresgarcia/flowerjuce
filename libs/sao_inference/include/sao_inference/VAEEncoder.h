#pragma once

#include "OnnxModel.h"
#include <string>
#include <vector>

namespace sao {

/// VAE encoder wrapper around the ONNX-exported MeanOnlyVAEEncoder.
///
/// Encodes stereo audio waveforms to latent representations.
/// The ONNX model includes the mean-only VAE bottleneck (no stochastic
/// sampling), so the output is deterministic.
///
/// Manifest fields consumed: none directly (paths come from ZenonPipelineConfig).
class VAEEncoder {
public:
    /// Load the VAE encoder ONNX model.
    ///
    /// Args:
    ///     onnx_path: Path to zenon_vae_encoder.onnx.
    ///     use_cuda: If true, use CUDA execution provider.
    ///     use_coreml: If true (macOS), use CoreML execution provider.
    explicit VAEEncoder(const std::string& onnx_path, bool use_cuda = false, bool use_coreml = false);

    /// Encode stereo audio to latent.
    ///
    /// Args:
    ///     audio: Stereo audio, flat row-major (1, 2, num_samples).
    ///     num_samples: Number of audio samples (typically 524288).
    ///     latent_dim: Expected latent channels (typically 64).
    ///
    /// Returns:
    ///     Latent tensor, flat row-major (1, latent_dim, latent_length).
    ///     latent_length = num_samples / downsampling_ratio.
    std::vector<float> encode(
        const std::vector<float>& audio,
        int num_samples,
        int latent_dim
    );

private:
    OnnxModel m_model;
};

} // namespace sao
