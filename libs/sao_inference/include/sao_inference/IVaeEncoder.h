#pragma once

#include <string>
#include <vector>

namespace sao {

/// Abstract VAE encoder: stereo waveform to latent (same contract as VAEEncoder).
class IVaeEncoder {
public:
    virtual ~IVaeEncoder() = default;

    /// Args:
    ///     audio: Stereo audio, flat row-major (1, 2, num_samples).
    ///     num_samples: Number of audio samples per channel.
    ///     latent_dim: Expected latent channels (e.g. 64).
    ///
    /// Returns:
    ///     Latent (1, latent_dim, latent_length), flat row-major.
    virtual std::vector<float> encode(
        const std::vector<float>& audio,
        int num_samples,
        int latent_dim) = 0;
};

} // namespace sao
