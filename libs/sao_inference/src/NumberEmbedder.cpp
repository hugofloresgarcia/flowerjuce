#include "sao_inference/NumberEmbedder.h"

#include <cnpy.h>
#include <nlohmann/json.hpp>

#include <cmath>
#include <fstream>
#include <iostream>
#include <stdexcept>

namespace sao {

namespace {

constexpr float kPi = 3.14159265358979323846f;

cnpy::NpyArray load_float32(const std::string& path, size_t ndim)
{
    cnpy::NpyArray arr = cnpy::npy_load(path);
    if (arr.word_size != sizeof(float))
        throw std::runtime_error("NumberEmbedder: " + path + " is not float32");
    if (arr.shape.size() != ndim)
        throw std::runtime_error("NumberEmbedder: " + path + " has " + std::to_string(arr.shape.size()) + " dims, expected " + std::to_string(ndim));
    return arr;
}

} // namespace

NumberEmbedder::NumberEmbedder(const std::string& weights_dir)
{
    std::ifstream config_file(weights_dir + "/config.json");
    if (!config_file)
        throw std::runtime_error("NumberEmbedder: cannot open " + weights_dir + "/config.json");
    const nlohmann::json cfg = nlohmann::json::parse(config_file);
    const std::string type = cfg.at("fourier_features_type").get<std::string>();
    if (type != "expo")
        throw std::runtime_error("NumberEmbedder: fourier_features_type '" + type + "' not supported (only 'expo')");
    m_min_val = cfg.at("min_val").get<float>();
    m_max_val = cfg.at("max_val").get<float>();
    m_dim = cfg.at("dim").get<int>();
    m_features = cfg.at("features").get<int>();
    const float min_freq = cfg.at("min_freq").get<float>();
    const float max_freq = cfg.at("max_freq").get<float>();
    if (m_dim <= 0 || m_dim % 2 != 0 || m_features <= 0 || !(m_max_val > m_min_val) || !(min_freq > 0.0f) || !(max_freq > min_freq))
        throw std::runtime_error("NumberEmbedder: inconsistent config.json");

    // The float32 frequency table exactly as torch computed it (exp(linspace(0,1,half)*(ln max - ln min) + ln min)).
    // It is loaded rather than recomputed: the Fourier arguments reach thousands of radians, so a 1-ulp difference
    // between std::exp and torch's exp moves the high-frequency features by ~1e-3.
    const int half = m_dim / 2;
    cnpy::NpyArray fr = load_float32(weights_dir + "/freqs.npy", 1);
    if (static_cast<int>(fr.shape[0]) != half)
        throw std::runtime_error("NumberEmbedder: freqs.npy must have dim/2 entries");
    m_freqs.assign(fr.data<float>(), fr.data<float>() + static_cast<size_t>(half));
    if (!(m_freqs.front() > 0.9f * min_freq && m_freqs.back() < 1.1f * max_freq))
        throw std::runtime_error("NumberEmbedder: freqs.npy does not match min_freq/max_freq in config.json");

    cnpy::NpyArray w = load_float32(weights_dir + "/linear_weight.npy", 2);
    if (static_cast<int>(w.shape[0]) != m_features || static_cast<int>(w.shape[1]) != m_dim)
        throw std::runtime_error("NumberEmbedder: linear_weight.npy shape does not match config (features x dim)");
    m_linear_weight.assign(w.data<float>(), w.data<float>() + static_cast<size_t>(m_features) * static_cast<size_t>(m_dim));

    cnpy::NpyArray b = load_float32(weights_dir + "/linear_bias.npy", 1);
    if (static_cast<int>(b.shape[0]) != m_features)
        throw std::runtime_error("NumberEmbedder: linear_bias.npy shape does not match config (features)");
    m_linear_bias.assign(b.data<float>(), b.data<float>() + static_cast<size_t>(m_features));

    std::cout << "[sao::NumberEmbedder] expo features: dim=" << m_dim << " features=" << m_features
              << " range=[" << m_min_val << ", " << m_max_val << "] freqs=[" << min_freq << ", " << max_freq << "]" << std::endl;
}

std::vector<float> NumberEmbedder::embed(float value) const
{
    const float clamped = std::max(m_min_val, std::min(m_max_val, value));
    const float s = (clamped - m_min_val) / (m_max_val - m_min_val);
    const int half = m_dim / 2;
    std::vector<float> feat(static_cast<size_t>(m_dim));
    for (int i = 0; i < half; ++i) {
        const float arg = ((s * m_freqs[static_cast<size_t>(i)]) * 2.0f) * kPi;   // torch: t * freqs * 2 * pi, float32, this order
        feat[static_cast<size_t>(i)] = std::cos(arg);
        feat[static_cast<size_t>(half + i)] = std::sin(arg);
    }
    std::vector<float> out(static_cast<size_t>(m_features));
    for (int o = 0; o < m_features; ++o) {
        const float* row = m_linear_weight.data() + static_cast<size_t>(o) * static_cast<size_t>(m_dim);
        double acc = m_linear_bias[static_cast<size_t>(o)];
        for (int i = 0; i < m_dim; ++i) acc += static_cast<double>(row[i]) * static_cast<double>(feat[static_cast<size_t>(i)]);
        out[static_cast<size_t>(o)] = static_cast<float>(acc);
    }
    return out;
}

} // namespace sao
