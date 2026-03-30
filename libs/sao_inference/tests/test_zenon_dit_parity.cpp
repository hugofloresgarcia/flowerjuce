#include "sao_inference/DiTInpaintModel.h"
#include <cnpy.h>
#include <cmath>
#include <iostream>
#include <cassert>
#include <string>

static float max_abs_error(const float* a, const float* b, size_t n)
{
    float max_err = 0.0f;
    for (size_t i = 0; i < n; ++i) {
        float err = std::abs(a[i] - b[i]);
        if (err > max_err) max_err = err;
    }
    return max_err;
}

static std::vector<float> load_npy_flat(const std::string& path)
{
    auto arr = cnpy::npy_load(path);
    size_t total = 1;
    for (auto d : arr.shape) total *= d;
    const float* data = arr.data<float>();
    return std::vector<float>(data, data + total);
}

int main()
{
    std::string parity_dir = ZENON_PARITY_DATA_DIR;
    std::string onnx_path = ZENON_ONNX_DIR "/zenon_dit.onnx";

    std::cout << "=== Zenon DiT (input_add) Parity Test ===" << std::endl;
    std::cout << "ONNX model: " << onnx_path << std::endl;

    sao::DiTInpaintModel dit(onnx_path);

    auto ref_x = load_npy_flat(parity_dir + "/dit_parity_x.npy");
    auto ref_t = load_npy_flat(parity_dir + "/dit_parity_t.npy");
    auto ref_cross = load_npy_flat(parity_dir + "/cond_input_cross_attn_cond.npy");
    auto ref_global = load_npy_flat(parity_dir + "/cond_input_global_cond.npy");
    auto ref_output = load_npy_flat(parity_dir + "/dit_parity_output.npy");

    int batch_size = 1;
    int latent_channels = 64;
    int latent_length = 256;
    int cond_dim = 768;
    int seq_len = static_cast<int>(ref_cross.size()) / (batch_size * cond_dim);

    std::vector<sao::InputAddTensor> input_add;

    auto load_add = [&](const std::string& name) {
        std::string path = parity_dir + "/dit_parity_input_add_" + name + ".npy";
        auto arr = cnpy::npy_load(path);
        sao::InputAddTensor add;
        add.name = name;
        add.channels = static_cast<int>(arr.shape[1]);
        size_t total = 1;
        for (auto d : arr.shape) total *= d;
        add.data = std::vector<float>(arr.data<float>(), arr.data<float>() + total);
        return add;
    };

    input_add.push_back(load_add("streamgen_latent"));
    input_add.push_back(load_add("inpaint_mask"));
    input_add.push_back(load_add("inpaint_masked_input"));

    auto result = dit.forward(
        ref_x, ref_t, ref_cross, ref_global,
        input_add, batch_size, seq_len, latent_channels, latent_length, cond_dim
    );

    assert(result.size() == ref_output.size());
    float err = max_abs_error(result.data(), ref_output.data(), result.size());
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
