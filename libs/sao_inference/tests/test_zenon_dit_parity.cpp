#include "sao_inference/DiTInpaintModel.h"
#include "sao_inference/ZenonPipelineConfig.h"
#include <cnpy.h>
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

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
    std::string manifest_path = ZENON_MANIFEST_PATH;

    std::cout << "=== Zenon DiT (fused input_add) Parity Test ===" << std::endl;

    // Pull dimensions from the manifest so the test matches whatever model
    // export_zenon_all.py produced (key order, channel total, ONNX path).
    auto config = sao::ZenonPipelineConfig::load(manifest_path);
    std::cout << "Manifest: " << manifest_path << std::endl;
    std::cout << "ONNX model: " << config.dit_onnx_path << std::endl;
    std::cout << "input_add_total_channels (from manifest): "
              << config.input_add_total_channels << std::endl;

    sao::DiTInpaintModel dit(config.dit_onnx_path);

    auto ref_x_b1 = load_npy_flat(parity_dir + "/dit_parity_x.npy");
    auto ref_t_b1 = load_npy_flat(parity_dir + "/dit_parity_t.npy");
    auto ref_cross_b1 = load_npy_flat(parity_dir + "/cond_input_cross_attn_cond.npy");
    auto ref_global_b1 = load_npy_flat(parity_dir + "/cond_input_global_cond.npy");
    auto ref_output_b1 = load_npy_flat(parity_dir + "/dit_parity_output.npy");

    // The fused-input-add reference dump saves the concatenated input_add_cond
    // tensor produced by Python's get_conditioning_inputs (already gated and
    // concatenated). Use it directly.
    auto input_add_cond_arr = cnpy::npy_load(parity_dir + "/dit_parity_input_add_cond.npy");
    assert(input_add_cond_arr.shape.size() == 3);
    const int input_add_total_channels = static_cast<int>(input_add_cond_arr.shape[1]);
    const int latent_length = static_cast<int>(input_add_cond_arr.shape[2]);
    const size_t add_per_batch = static_cast<size_t>(input_add_total_channels) * static_cast<size_t>(latent_length);
    std::vector<float> input_add_cond_b1(input_add_cond_arr.data<float>(),
                                         input_add_cond_arr.data<float>() + add_per_batch);

    // The DiT ONNX is exported with a static batch dimension of 2 (always-on
    // CFG batch doubling — see scripts/export_zenon_dit.py). The reference
    // dump is for a single (B=1) `_forward` call, so duplicate every input
    // along the batch axis and check that both halves match the reference.
    const int batch_size = 2;
    auto duplicate_batch = [](const std::vector<float>& src) {
        std::vector<float> out(src.size() * 2);
        std::memcpy(out.data(),               src.data(), src.size() * sizeof(float));
        std::memcpy(out.data() + src.size(),  src.data(), src.size() * sizeof(float));
        return out;
    };
    auto ref_x      = duplicate_batch(ref_x_b1);
    auto ref_t      = duplicate_batch(ref_t_b1);
    auto ref_cross  = duplicate_batch(ref_cross_b1);
    auto ref_global = duplicate_batch(ref_global_b1);
    auto input_add_cond = duplicate_batch(input_add_cond_b1);

    if (input_add_total_channels != config.input_add_total_channels) {
        std::cerr << "FAIL: reference tensor has " << input_add_total_channels
                  << " input_add channels but manifest declares "
                  << config.input_add_total_channels << std::endl;
        return 2;
    }

    const int latent_channels = config.latent_dim;
    const int cond_dim = config.cond_token_dim;
    const int seq_len = static_cast<int>(ref_cross.size()) / (batch_size * cond_dim);

    std::cout << "Inputs: B=" << batch_size << " C=" << latent_channels
              << " T=" << latent_length << " S=" << seq_len
              << " input_add=" << input_add_total_channels
              << " (B=1 reference duplicated for static-B=2 ONNX)" << std::endl;

    auto result = dit.forward(
        ref_x, ref_t, ref_cross, ref_global, input_add_cond,
        batch_size, seq_len, latent_channels, latent_length, cond_dim,
        input_add_total_channels
    );

    // Both halves of the batch were fed identical inputs; both should match
    // the (B=1) PyTorch reference. Check both.
    const size_t per_batch = ref_output_b1.size();
    assert(result.size() == 2 * per_batch);
    float err0 = max_abs_error(result.data(),               ref_output_b1.data(), per_batch);
    float err1 = max_abs_error(result.data() + per_batch,   ref_output_b1.data(), per_batch);
    float err = std::max(err0, err1);
    std::cout << "Max abs error: batch0=" << err0 << " batch1=" << err1
              << " (max=" << err << ")" << std::endl;

    constexpr float THRESHOLD = 1e-3f;
    if (err < THRESHOLD) {
        std::cout << "PASS (threshold=" << THRESHOLD << ")" << std::endl;
        return 0;
    } else {
        std::cerr << "FAIL: error " << err << " >= threshold " << THRESHOLD << std::endl;
        return 1;
    }
}
