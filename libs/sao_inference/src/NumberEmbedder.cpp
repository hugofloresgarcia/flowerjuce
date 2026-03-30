#include "sao_inference/NumberEmbedder.h"

#include <cnpy.h>
#include <fstream>
#include <iostream>
#include <cassert>
#include <sstream>
#include <string>

namespace sao {

static constexpr float TWO_PI = 2.0f * 3.14159265358979323846f;

static float parse_json_float(const std::string& json, const std::string& key)
{
    auto pos = json.find("\"" + key + "\"");
    assert(pos != std::string::npos);
    pos = json.find(':', pos);
    assert(pos != std::string::npos);
    return std::stof(json.substr(pos + 1));
}

static int parse_json_int(const std::string& json, const std::string& key)
{
    auto pos = json.find("\"" + key + "\"");
    assert(pos != std::string::npos);
    pos = json.find(':', pos);
    assert(pos != std::string::npos);
    return std::stoi(json.substr(pos + 1));
}

NumberEmbedder::NumberEmbedder(const std::string& weights_dir)
{
    std::ifstream config_file(weights_dir + "/config.json");
    assert(config_file.is_open());
    std::stringstream buf;
    buf << config_file.rdbuf();
    std::string json = buf.str();

    m_min_val = parse_json_float(json, "min_val");
    m_max_val = parse_json_float(json, "max_val");
    m_features = parse_json_int(json, "features");
    m_half_dim = parse_json_int(json, "pos_embed_half_dim");

    std::cout << "[sao::NumberEmbedder] min_val=" << m_min_val
              << " max_val=" << m_max_val
              << " features=" << m_features
              << " half_dim=" << m_half_dim << std::endl;

    // Load LearnedPositionalEmbedding weights
    auto pos_data = cnpy::npy_load(weights_dir + "/learned_pos_embed_weights.npy");
    assert(pos_data.shape.size() == 1);
    assert(static_cast<int>(pos_data.shape[0]) == m_half_dim);
    m_pos_weights.assign(pos_data.data<float>(), pos_data.data<float>() + m_half_dim);

    // Load Linear weight and bias
    auto weight_data = cnpy::npy_load(weights_dir + "/linear_weight.npy");
    assert(weight_data.shape.size() == 2);
    int linear_in = static_cast<int>(weight_data.shape[1]);
    assert(static_cast<int>(weight_data.shape[0]) == m_features);
    assert(linear_in == m_half_dim * 2 + 1);
    m_linear_weight.assign(weight_data.data<float>(),
                           weight_data.data<float>() + m_features * linear_in);

    auto bias_data = cnpy::npy_load(weights_dir + "/linear_bias.npy");
    assert(bias_data.shape.size() == 1);
    assert(static_cast<int>(bias_data.shape[0]) == m_features);
    m_linear_bias.assign(bias_data.data<float>(), bias_data.data<float>() + m_features);
}

std::vector<float> NumberEmbedder::learned_positional_embedding(float x) const
{
    // Output: (x, sin(x * w[0] * 2pi), sin(x * w[1] * 2pi), ...,
    //             cos(x * w[0] * 2pi), cos(x * w[1] * 2pi), ...)
    // Total size: 1 + 2 * half_dim = dim + 1
    std::vector<float> result(1 + 2 * m_half_dim);
    result[0] = x;
    for (int i = 0; i < m_half_dim; ++i) {
        float freq = x * m_pos_weights[i] * TWO_PI;
        result[1 + i] = std::sin(freq);
        result[1 + m_half_dim + i] = std::cos(freq);
    }
    return result;
}

std::vector<float> NumberEmbedder::linear_forward(const std::vector<float>& input) const
{
    int in_size = static_cast<int>(input.size());
    assert(in_size == m_half_dim * 2 + 1);

    std::vector<float> output(m_features);
    for (int i = 0; i < m_features; ++i) {
        float sum = m_linear_bias[i];
        for (int j = 0; j < in_size; ++j) {
            sum += m_linear_weight[i * in_size + j] * input[j];
        }
        output[i] = sum;
    }
    return output;
}

std::vector<float> NumberEmbedder::embed(float value) const
{
    float clamped = std::max(m_min_val, std::min(m_max_val, value));
    float normalized = (clamped - m_min_val) / (m_max_val - m_min_val);
    auto pos_embed = learned_positional_embedding(normalized);
    return linear_forward(pos_embed);
}

} // namespace sao
