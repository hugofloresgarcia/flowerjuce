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
    ///     use_migraphx: If true (Linux/ROCm), use MIGraphX execution provider.
    explicit VAEEncoder(const std::string& onnx_path, bool use_cuda = false, bool use_coreml = false, bool use_migraphx = false);

    /// Encode stereo audio to latent (single waveform; equivalent to `encode_batch` with batch_size=1).
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

    /// Encode a batch of stereo waveforms in a single ONNX Runtime call.
    ///
    /// The exported `zenon_vae_encoder.onnx` has a dynamic batch axis, so this is a single
    /// `Run()` on shape `(batch_size, 2, num_samples)` rather than `batch_size` separate calls.
    ///
    /// Args:
    ///     audio: Batched stereo audio, flat row-major (batch_size, 2, num_samples).
    ///         Length must equal batch_size * 2 * num_samples.
    ///     batch_size: Number of waveforms in the batch (>= 1).
    ///     num_samples: Number of audio samples per channel.
    ///     latent_dim: Expected latent channels.
    ///
    /// Returns:
    ///     Latent tensor (batch_size, latent_dim, latent_length) flat row-major.
    std::vector<float> encode_batch(
        const std::vector<float>& audio,
        int batch_size,
        int num_samples,
        int latent_dim
    );

private:
    OnnxModel m_model;
};

} // namespace sao
