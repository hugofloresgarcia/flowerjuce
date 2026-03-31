#include "sao_inference/MlxZenonVae.h"

#include <nlohmann/json.hpp>

#include <cassert>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <unordered_map>
#include <vector>

#include "mlx/mlx.h"

namespace mx = mlx::core;

namespace sao {

struct MlxVaeArchConfig {
    std::vector<int> strides;
    std::vector<int> c_mults;
    int channels = 128;
    int encoder_latent_dim = 128;
    int decoder_latent_dim = 64;
};

struct MlxVaeBundle {
    std::unordered_map<std::string, mx::array> tensors;
    MlxVaeArchConfig arch;

    const mx::array& at(const std::string& key) const
    {
        auto it = tensors.find(key);
        if (it == tensors.end())
        {
            throw std::runtime_error("MlxVaeBundle: missing tensor '" + key + "'");
        }
        return it->second;
    }
};

namespace {

/// Copies `v` into a new MLX array (heap) with the given shape.
static mx::array vector_to_mlx_array(std::vector<float> v, const mx::Shape& shape)
{
    size_t expected = 1;
    for (auto d : shape)
    {
        expected *= static_cast<size_t>(d);
    }
    assert(v.size() == expected);

    float* p = new float[v.size()];
    std::memcpy(p, v.data(), v.size() * sizeof(float));
    return mx::array(
        p,
        shape,
        mx::float32,
        [](void* ptr) { delete[] static_cast<float*>(ptr); });
}

static MlxVaeArchConfig parse_vae_config_json(const std::string& path)
{
    std::ifstream file(path);
    if (!file.is_open())
    {
        throw std::runtime_error("MlxVaeBundle: cannot open config " + path);
    }
    nlohmann::json j = nlohmann::json::parse(file);
    MlxVaeArchConfig cfg;
    for (auto& v : j.at("strides"))
    {
        cfg.strides.push_back(v.get<int>());
    }
    for (auto& v : j.at("c_mults"))
    {
        cfg.c_mults.push_back(v.get<int>());
    }
    cfg.channels = j.at("channels").get<int>();
    cfg.encoder_latent_dim = j.at("encoder_latent_dim").get<int>();
    cfg.decoder_latent_dim = j.at("decoder_latent_dim").get<int>();
    return cfg;
}

static mx::array snake_activation(const mx::array& x, const mx::array& alpha, const mx::array& beta)
{
    mx::array a = mx::exp(alpha);
    mx::array b = mx::exp(beta);
    mx::array denom = mx::add(b, mx::array(1e-9f));
    mx::array s = mx::sin(mx::multiply(x, a));
    mx::array sq = mx::multiply(s, s);
    return mx::add(x, mx::divide(sq, denom));
}

static mx::array conv1d_forward(
    const mx::array& x,
    const mx::array& weight,
    const mx::array& bias,
    int stride,
    int padding,
    int dilation)
{
    mx::array w = mx::transpose(weight, {0, 2, 1});
    mx::array out = mx::conv1d(x, w, stride, padding, dilation);
    return mx::add(out, bias);
}

static mx::array residual_unit_forward(
    const mx::array& x,
    const MlxVaeBundle& bundle,
    const std::string& unit_prefix,
    int /*channels*/,
    int dilation)
{
    mx::array residual = x;
    int padding = (dilation * 6) / 2;
    const mx::array& alpha1 = bundle.at(unit_prefix + ".layers.0.alpha");
    const mx::array& beta1 = bundle.at(unit_prefix + ".layers.0.beta");
    mx::array h = snake_activation(x, alpha1, beta1);
    h = conv1d_forward(
        h,
        bundle.at(unit_prefix + ".layers.1.weight"),
        bundle.at(unit_prefix + ".layers.1.bias"),
        1,
        padding,
        dilation);
    const mx::array& alpha2 = bundle.at(unit_prefix + ".layers.2.alpha");
    const mx::array& beta2 = bundle.at(unit_prefix + ".layers.2.beta");
    h = snake_activation(h, alpha2, beta2);
    h = conv1d_forward(
        h,
        bundle.at(unit_prefix + ".layers.3.weight"),
        bundle.at(unit_prefix + ".layers.3.bias"),
        1,
        0,
        1);
    return mx::add(h, residual);
}

static mx::array encoder_block_forward(
    const mx::array& x_in,
    const MlxVaeBundle& bundle,
    const std::string& block_prefix,
    int in_channels,
    int out_channels,
    int stride)
{
    const int dilations[3] = {1, 3, 9};
    mx::array x = x_in;
    for (int i = 0; i < 3; ++i)
    {
        std::string unit_prefix = block_prefix + ".layers." + std::to_string(i);
        x = residual_unit_forward(x, bundle, unit_prefix, in_channels, dilations[i]);
    }
    const mx::array& sa = bundle.at(block_prefix + ".layers.3.alpha");
    const mx::array& sb = bundle.at(block_prefix + ".layers.3.beta");
    x = snake_activation(x, sa, sb);
    int pad = (stride + 1) / 2;
    x = conv1d_forward(
        x,
        bundle.at(block_prefix + ".layers.4.weight"),
        bundle.at(block_prefix + ".layers.4.bias"),
        stride,
        pad,
        1);
    (void)out_channels;
    return x;
}

static mx::array mlx_encode(const mx::array& audio_nlc, const MlxVaeBundle& bundle)
{
    const MlxVaeArchConfig& cfg = bundle.arch;
    const int channels = cfg.channels;
    const std::vector<int>& strides = cfg.strides;
    const std::vector<int>& c_mults = cfg.c_mults;

    std::vector<int> block_channels;
    block_channels.reserve(c_mults.size());
    for (int m : c_mults)
    {
        block_channels.push_back(channels * m);
    }

    mx::array x = conv1d_forward(
        audio_nlc,
        bundle.at("encoder.layers.0.weight"),
        bundle.at("encoder.layers.0.bias"),
        1,
        3,
        1);

    const size_t n_blocks = strides.size();
    for (size_t i = 0; i < n_blocks; ++i)
    {
        std::string prefix = "encoder.layers." + std::to_string(static_cast<int>(i) + 1);
        x = encoder_block_forward(
            x,
            bundle,
            prefix,
            block_channels[i],
            block_channels[i + 1],
            strides[i]);
    }

    int final_snake_idx = static_cast<int>(n_blocks) + 1;
    std::string fa_key = "encoder.layers." + std::to_string(final_snake_idx) + ".alpha";
    std::string fb_key = "encoder.layers." + std::to_string(final_snake_idx) + ".beta";
    x = snake_activation(x, bundle.at(fa_key), bundle.at(fb_key));

    int final_conv_idx = final_snake_idx + 1;
    std::string fw_key = "encoder.layers." + std::to_string(final_conv_idx) + ".weight";
    std::string fbi_key = "encoder.layers." + std::to_string(final_conv_idx) + ".bias";
    x = conv1d_forward(x, bundle.at(fw_key), bundle.at(fbi_key), 1, 1, 1);
    return x;
}

static mx::array mean_only_slice(const mx::array& enc_out, int latent_dim)
{
    int T = enc_out.shape(1);
    return mx::slice(enc_out, {0, 0, 0}, {1, T, latent_dim});
}

static mx::array decoder_block_forward(
    const mx::array& x_in,
    const MlxVaeBundle& bundle,
    const std::string& block_prefix,
    int /*in_channels*/,
    int out_channels,
    int stride)
{
    mx::array x = snake_activation(
        x_in,
        bundle.at(block_prefix + ".layers.0.alpha"),
        bundle.at(block_prefix + ".layers.0.beta"));

    mx::array w_up = mx::transpose(bundle.at(block_prefix + ".layers.1.weight"), {1, 2, 0});
    int kernel_size = stride * 2;
    int padding = (stride + 1) / 2;
    x = mx::conv_transpose1d(x, w_up, stride, padding);
    x = mx::add(x, bundle.at(block_prefix + ".layers.1.bias"));

    const int dilations[3] = {1, 3, 9};
    for (int i = 0; i < 3; ++i)
    {
        std::string unit_prefix = block_prefix + ".layers." + std::to_string(i + 2);
        x = residual_unit_forward(x, bundle, unit_prefix, out_channels, dilations[i]);
    }
    return x;
}

static mx::array mlx_decode(const mx::array& latent_nlc, const MlxVaeBundle& bundle)
{
    const MlxVaeArchConfig& cfg = bundle.arch;
    const int channels = cfg.channels;
    const std::vector<int>& strides = cfg.strides;
    const std::vector<int>& c_mults = cfg.c_mults;

    std::vector<int> block_channels;
    block_channels.reserve(c_mults.size());
    for (int m : c_mults)
    {
        block_channels.push_back(channels * m);
    }

    mx::array x = conv1d_forward(
        latent_nlc,
        bundle.at("decoder.layers.0.weight"),
        bundle.at("decoder.layers.0.bias"),
        1,
        3,
        1);

    const int L = static_cast<int>(strides.size());
    for (int i = 0; i < L; ++i)
    {
        int stride = strides[static_cast<size_t>(L - 1 - i)];
        int block_in = block_channels[static_cast<size_t>(L - i)];
        int block_out = block_channels[static_cast<size_t>(L - i - 1)];
        std::string prefix = "decoder.layers." + std::to_string(i + 1);
        x = decoder_block_forward(x, bundle, prefix, block_in, block_out, stride);
    }

    int final_snake_idx = L + 1;
    x = snake_activation(
        x,
        bundle.at("decoder.layers." + std::to_string(final_snake_idx) + ".alpha"),
        bundle.at("decoder.layers." + std::to_string(final_snake_idx) + ".beta"));
    mx::array wf = mx::transpose(
        bundle.at("decoder.layers." + std::to_string(final_snake_idx + 1) + ".weight"),
        {0, 2, 1});
    x = mx::conv1d(x, wf, 1, 3);
    std::string bias_key = "decoder.layers." + std::to_string(final_snake_idx + 1) + ".bias";
    auto bit = bundle.tensors.find(bias_key);
    if (bit != bundle.tensors.end())
    {
        x = mx::add(x, bit->second);
    }
    return x;
}

static std::vector<float> audio_chw_to_nlc(const std::vector<float>& audio, int num_samples)
{
    assert(static_cast<int>(audio.size()) == 2 * num_samples);
    std::vector<float> nlc(static_cast<size_t>(num_samples) * 2);
    for (int s = 0; s < num_samples; ++s)
    {
        nlc[static_cast<size_t>(s) * 2 + 0] = audio[static_cast<size_t>(s)];
        nlc[static_cast<size_t>(s) * 2 + 1] = audio[static_cast<size_t>(num_samples + s)];
    }
    return nlc;
}

static void copy_chw_to_vector(const mx::array& a, std::vector<float>& out)
{
    mx::array c = mx::contiguous(a);
    mx::eval(c);
    out.resize(c.size());
    const float* p = c.data<float>();
    std::memcpy(out.data(), p, out.size() * sizeof(float));
}

static void copy_nlc_audio_to_chw(const mx::array& a, int num_samples, std::vector<float>& out)
{
    mx::array c = mx::contiguous(a);
    mx::eval(c);
    assert(c.shape(0) == 1 && c.shape(1) == num_samples && c.shape(2) == 2);
    const float* p = c.data<float>();
    out.resize(static_cast<size_t>(2) * static_cast<size_t>(num_samples));
    for (int s = 0; s < num_samples; ++s)
    {
        out[static_cast<size_t>(s)] = p[static_cast<size_t>(s) * 2 + 0];
        out[static_cast<size_t>(num_samples + s)] = p[static_cast<size_t>(s) * 2 + 1];
    }
}

} // namespace

std::shared_ptr<MlxVaeBundle> load_mlx_vae_bundle(
    const std::string& weights_path,
    const std::string& config_path)
{
    auto bundle = std::make_shared<MlxVaeBundle>();
    bundle->arch = parse_vae_config_json(config_path);
    auto loaded = mx::load_safetensors(weights_path);
    bundle->tensors = std::move(loaded.first);
    return bundle;
}

MlxVaeEncoder::MlxVaeEncoder(std::shared_ptr<MlxVaeBundle> bundle)
    : m_bundle(std::move(bundle))
{
    if (!m_bundle)
    {
        throw std::runtime_error("MlxVaeEncoder: null bundle");
    }
}

MlxVaeEncoder::~MlxVaeEncoder() = default;

std::vector<float> MlxVaeEncoder::encode(
    const std::vector<float>& audio,
    int num_samples,
    int latent_dim)
{
    if (latent_dim != m_bundle->arch.decoder_latent_dim)
    {
        std::ostringstream msg;
        msg << "MlxVaeEncoder: latent_dim " << latent_dim
            << " != expected " << m_bundle->arch.decoder_latent_dim;
        throw std::runtime_error(msg.str());
    }

    std::vector<float> nlc_data = audio_chw_to_nlc(audio, num_samples);
    mx::array audio_mx = vector_to_mlx_array(std::move(nlc_data), {1, num_samples, 2});

    mx::array enc = mlx_encode(audio_mx, *m_bundle);
    mx::array mean = mean_only_slice(enc, latent_dim);
    mx::array chw = mx::transpose(mean, {0, 2, 1});

    std::vector<float> result;
    copy_chw_to_vector(chw, result);
    return result;
}

MlxVaeDecoder::MlxVaeDecoder(std::shared_ptr<MlxVaeBundle> bundle, float vae_scale)
    : m_bundle(std::move(bundle))
    , m_scale(vae_scale)
{
    if (!m_bundle)
    {
        throw std::runtime_error("MlxVaeDecoder: null bundle");
    }
}

MlxVaeDecoder::~MlxVaeDecoder() = default;

std::vector<float> MlxVaeDecoder::decode(const std::vector<float>& latents, int latent_length)
{
    constexpr int LATENT_DIM = 64;
    assert(static_cast<int>(latents.size()) == LATENT_DIM * latent_length);

    std::vector<float> scaled(latents.size());
    for (size_t i = 0; i < latents.size(); ++i)
    {
        scaled[i] = latents[i] * m_scale;
    }

    std::vector<float> nlc_data(static_cast<size_t>(latent_length) * static_cast<size_t>(LATENT_DIM));
    for (int t = 0; t < latent_length; ++t)
    {
        for (int c = 0; c < LATENT_DIM; ++c)
        {
            nlc_data[static_cast<size_t>(t) * static_cast<size_t>(LATENT_DIM) + static_cast<size_t>(c)] =
                scaled[static_cast<size_t>(c) * static_cast<size_t>(latent_length) + static_cast<size_t>(t)];
        }
    }

    mx::array latent_mx = vector_to_mlx_array(std::move(nlc_data), {1, latent_length, LATENT_DIM});

    mx::array audio_out = mlx_decode(latent_mx, *m_bundle);
    int num_samples = audio_out.shape(1);
    std::vector<float> result;
    copy_nlc_audio_to_chw(audio_out, num_samples, result);
    return result;
}

} // namespace sao
