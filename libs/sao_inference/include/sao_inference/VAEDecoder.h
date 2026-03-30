#pragma once

#include "OnnxModel.h"
#include <string>
#include <vector>

namespace sao {

/// VAE decoder wrapper around the ONNX-exported OobleckDecoder.
///
/// Decodes latent representations to stereo audio waveforms.
/// Note: the pretransform scale must be applied BEFORE calling decode().
class VAEDecoder {
public:
    /// Load the VAE decoder ONNX model.
    ///
    /// Args:
    ///     onnx_path: Path to vae_decoder.onnx.
    ///     scale: The pretransform scale factor (loaded from vae_scale.json).
    ///     use_cuda: If true, use CUDA execution provider.
    ///     use_coreml: If true (macOS), use CoreML execution provider.
    explicit VAEDecoder(const std::string& onnx_path, float scale = 1.0f, bool use_cuda = false, bool use_coreml = false);

    /// Decode latent to audio.
    ///
    /// Args:
    ///     latents: Latent tensor, flat row-major (1, 64, latent_length).
    ///     latent_length: Number of latent frames (typically 256).
    ///
    /// Returns:
    ///     Decoded audio, flat row-major (1, 2, audio_length).
    ///     audio_length = latent_length * downsampling_ratio (typically 256 * 2048 = 524288).
    std::vector<float> decode(const std::vector<float>& latents, int latent_length);

    float scale() const { return m_scale; }

private:
    OnnxModel m_model;
    float m_scale;
};

} // namespace sao
