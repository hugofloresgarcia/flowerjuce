#include "sao_inference/MlxZenonVae.h"
#include <cnpy.h>
#include <cmath>
#include <iostream>
#include <cassert>
#include <string>

static float max_abs_error(const std::vector<float>& a, const float* b, size_t n)
{
    float max_err = 0.0f;
    for (size_t i = 0; i < n; ++i)
    {
        float err = std::abs(a[i] - b[i]);
        if (err > max_err)
            max_err = err;
    }
    return max_err;
}

int main()
{
    std::string parity_dir = ZENON_PARITY_DATA_DIR;
    std::string weights = ZENON_MLX_VAE_WEIGHTS;
    std::string cfg = ZENON_MLX_VAE_CONFIG;

    std::cout << "=== Zenon VAE Encoder MLX Parity Test ===" << std::endl;
    std::cout << "MLX weights: " << weights << std::endl;
    std::cout << "Parity data: " << parity_dir << std::endl;

    auto bundle = sao::load_mlx_vae_bundle(weights, cfg);
    sao::MlxVaeEncoder encoder(bundle);

    auto ref_input = cnpy::npy_load(parity_dir + "/streamgen_audio.npy");
    // Mean-only latent; streamgen_latent.npy from dump_zenon_reference_tensors is VAE *sampled* latent.
    auto ref_output = cnpy::npy_load(parity_dir + "/streamgen_latent_encoder_mean_mlx.npy");

    size_t in_total = 1;
    for (auto d : ref_input.shape)
        in_total *= d;
    std::vector<float> audio_data(ref_input.data<float>(), ref_input.data<float>() + in_total);

    int num_samples = static_cast<int>(ref_input.shape[2]);
    int latent_dim = static_cast<int>(ref_output.shape[1]);

    auto result = encoder.encode(audio_data, num_samples, latent_dim);

    size_t out_total = 1;
    for (auto d : ref_output.shape)
        out_total *= d;
    assert(result.size() == out_total);

    float err = max_abs_error(result, ref_output.data<float>(), out_total);
    std::cout << "Max abs error: " << err << std::endl;

    // Tight: C++ MLX graph should match MLX Python; both use the same safetensors ops.
    constexpr float THRESHOLD = 1e-4f;
    if (err < THRESHOLD)
    {
        std::cout << "PASS (threshold=" << THRESHOLD << ")" << std::endl;
        return 0;
    }
    else
    {
        std::cerr << "FAIL: error " << err << " >= threshold " << THRESHOLD << std::endl;
        return 1;
    }
}
