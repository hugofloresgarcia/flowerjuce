#pragma once

#include <vector>

namespace sao {

/// Abstract VAE decoder: latent to stereo waveform (same contract as VAEDecoder).
class IVaeDecoder {
public:
    virtual ~IVaeDecoder() = default;

    /// Args:
    ///     latents: (1, 64, latent_length), flat row-major (pre-scale; same contract as VAEDecoder::decode).
    ///     latent_length: Latent time dimension (e.g. 256).
    ///
    /// Returns:
    ///     Audio (1, 2, sample_size), flat row-major.
    virtual std::vector<float> decode(
        const std::vector<float>& latents,
        int latent_length) = 0;
};

} // namespace sao
