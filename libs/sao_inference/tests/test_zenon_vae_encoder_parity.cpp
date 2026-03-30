#include "sao_inference/VAEEncoder.h"
#include <cnpy.h>
#include <cmath>
#include <iostream>
#include <cassert>
#include <string>

static float max_abs_error(const std::vector<float>& a, const float* b, size_t n)
{
    float max_err = 0.0f;
    for (size_t i = 0; i < n; ++i) {
        float err = std::abs(a[i] - b[i]);
        if (err > max_err) max_err = err;
    }
    return max_err;
}

int main()
{
    std::string parity_dir = ZENON_PARITY_DATA_DIR;
    std::string onnx_path = ZENON_ONNX_DIR "/zenon_vae_encoder.onnx";

    std::cout << "=== Zenon VAE Encoder Parity Test ===" << std::endl;
    std::cout << "ONNX model: " << onnx_path << std::endl;
    std::cout << "Parity data: " << parity_dir << std::endl;

    sao::VAEEncoder encoder(onnx_path);

    auto ref_input = cnpy::npy_load(parity_dir + "/streamgen_audio.npy");
    auto ref_output = cnpy::npy_load(parity_dir + "/streamgen_latent.npy");

    size_t in_total = 1;
    for (auto d : ref_input.shape) in_total *= d;
    std::vector<float> audio_data(ref_input.data<float>(), ref_input.data<float>() + in_total);

    int num_samples = static_cast<int>(ref_input.shape[2]);
    int latent_dim = static_cast<int>(ref_output.shape[1]);

    auto result = encoder.encode(audio_data, num_samples, latent_dim);

    size_t out_total = 1;
    for (auto d : ref_output.shape) out_total *= d;
    assert(result.size() == out_total);

    float err = max_abs_error(result, ref_output.data<float>(), out_total);
    std::cout << "Max abs error: " << err << std::endl;

    constexpr float THRESHOLD = 1e-3f;
    if (err < THRESHOLD) {
        std::cout << "PASS (threshold=" << THRESHOLD << ")" << std::endl;
        return 0;
    } else {
        std::cerr << "FAIL: error " << err << " >= threshold " << THRESHOLD << std::endl;
        return 1;
    }
}
