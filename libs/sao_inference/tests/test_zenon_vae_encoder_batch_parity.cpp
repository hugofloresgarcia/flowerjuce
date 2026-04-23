#include "sao_inference/VAEEncoder.h"
#include <cnpy.h>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <cassert>
#include <string>
#include <vector>

/// Per-element max-abs error between two flat buffers.
static float max_abs_error(const float* a, const float* b, std::size_t n)
{
    float max_err = 0.0f;
    for (std::size_t i = 0; i < n; ++i) {
        float err = std::abs(a[i] - b[i]);
        if (err > max_err) max_err = err;
    }
    return max_err;
}

static std::vector<float> load_npy_flat(const std::string& path, std::vector<std::size_t>& out_shape)
{
    auto arr = cnpy::npy_load(path);
    out_shape = arr.shape;
    std::size_t total = 1;
    for (auto d : arr.shape) total *= d;
    const float* data = arr.data<float>();
    return std::vector<float>(data, data + total);
}

int main()
{
    std::string parity_dir = ZENON_PARITY_DATA_DIR;
    std::string onnx_path = ZENON_ONNX_DIR "/zenon_vae_encoder.onnx";

    std::cout << "=== Zenon VAE Encoder Batched Parity Test (B=2 only) ===" << std::endl;
    std::cout << "ONNX model: " << onnx_path << std::endl;
    std::cout << "Parity data: " << parity_dir << std::endl;

    sao::VAEEncoder encoder(onnx_path);

    std::vector<std::size_t> streamgen_audio_shape;
    std::vector<std::size_t> input_audio_shape;
    auto streamgen_audio = load_npy_flat(parity_dir + "/streamgen_audio.npy", streamgen_audio_shape);
    auto input_audio = load_npy_flat(parity_dir + "/input_audio.npy", input_audio_shape);

    assert(streamgen_audio_shape.size() == 3);
    assert(input_audio_shape.size() == 3);
    assert(streamgen_audio_shape == input_audio_shape);
    assert(streamgen_audio_shape[0] == 1);
    assert(streamgen_audio_shape[1] == 2);

    // Mean-only references (deterministic): the ONNX export exposes only the VAE mean,
    // whereas streamgen_latent.npy / input_latent.npy include the bottleneck noise term
    // (mean + std * randn). Use the mean-only dumps from
    // scripts/dump_zenon_encoder_mean_refs.py instead.
    std::vector<std::size_t> streamgen_latent_shape;
    std::vector<std::size_t> input_latent_shape;
    auto ref_streamgen_latent = load_npy_flat(
        parity_dir + "/streamgen_latent_encoder_mean_onnx.npy", streamgen_latent_shape);
    auto ref_input_latent = load_npy_flat(
        parity_dir + "/input_latent_encoder_mean_onnx.npy", input_latent_shape);

    assert(streamgen_latent_shape.size() == 3);
    assert(input_latent_shape.size() == 3);
    assert(streamgen_latent_shape == input_latent_shape);
    assert(streamgen_latent_shape[0] == 1);

    const int num_samples = static_cast<int>(streamgen_audio_shape[2]);
    const int latent_dim = static_cast<int>(streamgen_latent_shape[1]);
    const int latent_length = static_cast<int>(streamgen_latent_shape[2]);

    std::vector<float> combined_audio;
    combined_audio.reserve(streamgen_audio.size() + input_audio.size());
    combined_audio.insert(combined_audio.end(), streamgen_audio.begin(), streamgen_audio.end());
    combined_audio.insert(combined_audio.end(), input_audio.begin(), input_audio.end());

    auto latent_batched = encoder.encode_batch(combined_audio, 2, num_samples, latent_dim);

    const std::size_t per_latent =
        static_cast<std::size_t>(latent_dim) * static_cast<std::size_t>(latent_length);
    assert(ref_streamgen_latent.size() == per_latent);
    assert(ref_input_latent.size() == per_latent);
    assert(latent_batched.size() == 2 * per_latent);

    const float err_streamgen = max_abs_error(
        ref_streamgen_latent.data(), latent_batched.data(), per_latent);
    const float err_input = max_abs_error(
        ref_input_latent.data(), latent_batched.data() + per_latent, per_latent);

    std::cout << "Batched latent size: " << latent_batched.size()
              << " (2 * " << per_latent << ")" << std::endl;
    std::cout << "Max abs error (streamgen row vs reference): " << err_streamgen << std::endl;
    std::cout << "Max abs error (input row vs reference):     " << err_input << std::endl;

    // Single NN forward pass: ONNX Runtime B=2 vs Python reference (B=1, run twice).
    constexpr float THRESHOLD = 1e-4f;
    const float worst = std::max(err_streamgen, err_input);
    if (worst < THRESHOLD) {
        std::cout << "PASS (threshold=" << THRESHOLD << ")" << std::endl;
        return 0;
    }
    std::cerr << "FAIL: worst error " << worst << " >= threshold " << THRESHOLD << std::endl;
    return 1;
}
