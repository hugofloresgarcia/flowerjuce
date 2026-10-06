#pragma once

#include <string>
#include <vector>

namespace sao {

/// C++ reimplementation of the Stable Audio 3 `NumberConditioner` with `fourier_features_type: expo`
/// (stable_audio_3/models/conditioners.py, blocks.py::ExpoFourierFeatures).
///
///   s      = (clamp(value, min_val, max_val) - min_val) / (max_val - min_val)
///   freqs  = exp(linspace(0, 1, dim/2) * (ln max_freq - ln min_freq) + ln min_freq)
///   feat   = [cos(2 pi s freqs), sin(2 pi s freqs)]                       (dim values, cos first)
///   embed  = W feat + b                                                   (features values)
///
/// Weights and constants come from the directory written by scripts/export_sa3_number_embedder.py:
///   config.json        {"fourier_features_type": "expo", "min_val", "max_val", "dim", "min_freq", "max_freq", "features"}
///   linear_weight.npy  float32 [features, dim]
///   linear_bias.npy    float32 [features]
///   freqs.npy          float32 [dim/2]   torch's own float32 frequency table (see NumberEmbedder.cpp for why it is not recomputed)
class NumberEmbedder {
public:
    /// Throws std::runtime_error if a file is missing, the config is not `expo`, or shapes disagree.
    explicit NumberEmbedder(const std::string& weights_dir);

    /// Embedding of a raw (unnormalised) value, e.g. seconds_total. Returns `features()` floats.
    std::vector<float> embed(float value) const;

    int features() const { return m_features; }
    int dim() const { return m_dim; }
    float min_val() const { return m_min_val; }
    float max_val() const { return m_max_val; }

private:
    float m_min_val = 0.0f;
    float m_max_val = 1.0f;
    int m_dim = 0;        ///< Fourier feature count (cos + sin)
    int m_features = 0;   ///< output size
    std::vector<float> m_freqs;          // (dim/2,)
    std::vector<float> m_linear_weight;  // (features, dim) row-major
    std::vector<float> m_linear_bias;    // (features,)
};

} // namespace sao
