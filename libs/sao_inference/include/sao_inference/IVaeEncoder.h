#pragma once

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace sao {

/// Abstract VAE encoder: stereo waveform to latent (same contract as VAEEncoder).
class IVaeEncoder {
public:
    virtual ~IVaeEncoder() = default;

    /// Encode a single stereo waveform.
    ///
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

    /// Encode a batch of stereo waveforms.
    ///
    /// The default implementation loops over `encode()` once per batch item — correct for any
    /// backend, but gives no throughput advantage over calling `encode()` directly. Backends
    /// that support a true batched forward pass (e.g. ONNX Runtime with a dynamic batch axis)
    /// should override this to do a single forward call.
    ///
    /// Args:
    ///     audio: Batched stereo audio, flat row-major (batch_size, 2, num_samples).
    ///         Length must equal batch_size * 2 * num_samples.
    ///     batch_size: Number of waveforms in the batch (>= 1).
    ///     num_samples: Number of audio samples per channel.
    ///     latent_dim: Expected latent channels (e.g. 64).
    ///
    /// Returns:
    ///     Latent tensor (batch_size, latent_dim, latent_length), flat row-major.
    ///     Length equals batch_size * latent_dim * latent_length.
    virtual std::vector<float> encode_batch(
        const std::vector<float>& audio,
        int batch_size,
        int num_samples,
        int latent_dim)
    {
        constexpr int CHANNELS = 2;
        assert(batch_size >= 1);
        assert(num_samples > 0);
        assert(static_cast<int64_t>(audio.size())
               == static_cast<int64_t>(batch_size) * CHANNELS * num_samples);

        const std::size_t per_item_audio =
            static_cast<std::size_t>(CHANNELS) * static_cast<std::size_t>(num_samples);
        std::vector<float> out;
        std::vector<float> single(per_item_audio);
        for (int b = 0; b < batch_size; ++b)
        {
            const std::size_t offset = static_cast<std::size_t>(b) * per_item_audio;
            std::copy(audio.begin() + static_cast<std::ptrdiff_t>(offset),
                      audio.begin() + static_cast<std::ptrdiff_t>(offset + per_item_audio),
                      single.begin());
            auto latent = encode(single, num_samples, latent_dim);
            out.insert(out.end(), latent.begin(), latent.end());
        }
        return out;
    }
};

} // namespace sao
