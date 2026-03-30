#pragma once

#include <string>
#include <vector>
#include <cmath>

namespace sao {

/// C++ reimplementation of the NumberEmbedder from stable-audio-tools.
///
/// Architecture: LearnedPositionalEmbedding(dim=256) -> Linear(257, 768)
///
/// LearnedPositionalEmbedding:
///   weights: (half_dim=128,) learnable frequencies
///   forward(x): x -> (x, sin(x * weights * 2pi), cos(x * weights * 2pi)) -> (dim+1,)
///
/// Linear: (257,) -> (768,)
///
/// The NumberConditioner normalizes input: (val - min_val) / (max_val - min_val)
/// then passes through NumberEmbedder.
class NumberEmbedder {
public:
    /// Load weights from .npy files in the given directory.
    ///
    /// Expected files:
    ///   - learned_pos_embed_weights.npy: (128,) float32
    ///   - linear_weight.npy: (768, 257) float32
    ///   - linear_bias.npy: (768,) float32
    ///   - config.json: min_val, max_val, features, etc.
    ///
    /// Args:
    ///     weights_dir: Directory containing the .npy weight files.
    explicit NumberEmbedder(const std::string& weights_dir);

    /// Compute the embedding for a raw (unnormalized) float value.
    ///
    /// Args:
    ///     value: The raw input (e.g. seconds_total = 11.0).
    ///
    /// Returns:
    ///     Embedding vector of size (features,) = (768,).
    std::vector<float> embed(float value) const;

    int features() const { return m_features; }
    float min_val() const { return m_min_val; }
    float max_val() const { return m_max_val; }

private:
    /// Compute LearnedPositionalEmbedding: x -> (x, sin, cos)
    std::vector<float> learned_positional_embedding(float x) const;

    /// Apply linear layer: y = weight @ x + bias
    std::vector<float> linear_forward(const std::vector<float>& input) const;

    float m_min_val;
    float m_max_val;
    int m_features;
    int m_half_dim;

    std::vector<float> m_pos_weights;   // (half_dim,)
    std::vector<float> m_linear_weight; // (features, half_dim*2+1)
    std::vector<float> m_linear_bias;   // (features,)
};

} // namespace sao
