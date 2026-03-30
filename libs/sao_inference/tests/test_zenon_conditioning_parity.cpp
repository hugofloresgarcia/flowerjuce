#include "sao_inference/InpaintConditioningAssembler.h"
#include "sao_inference/NumberEmbedder.h"
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
    std::string weights_dir = ZENON_WEIGHTS_DIR "/number_embedder_zenon";

    std::cout << "=== Zenon Conditioning Assembly Parity Test ===" << std::endl;

    sao::NumberEmbedder embedder(weights_dir);

    auto t5_embed = load_npy_flat(parity_dir + "/t5_embeddings_masked.npy");
    auto ref_cross_attn = load_npy_flat(parity_dir + "/cond_input_cross_attn_cond.npy");
    auto ref_global = load_npy_flat(parity_dir + "/cond_input_global_cond.npy");

    float seconds_total = 11.0f;
    auto seconds_embed = embedder.embed(seconds_total);

    int embed_dim = 768;
    int t5_seq_len = 64;
    int latent_channels = 64;
    int latent_length = 256;

    auto streamgen_latent = load_npy_flat(parity_dir + "/streamgen_latent.npy");
    auto inpaint_mask_full = load_npy_flat(parity_dir + "/inpaint_mask.npy");
    auto inpaint_masked_input = load_npy_flat(parity_dir + "/inpaint_masked_input.npy");

    std::vector<float> inpaint_mask(latent_length);
    for (int i = 0; i < latent_length; ++i) {
        inpaint_mask[i] = inpaint_mask_full[i];
    }

    std::vector<sao::InputAddKeyDescriptor> descriptors = {
        {"streamgen_latent", latent_channels, sao::MaskRule::multiply_by_mask},
        {"inpaint_mask", 1, sao::MaskRule::pass_through},
        {"inpaint_masked_input", latent_channels, sao::MaskRule::pass_through},
    };

    auto cond = sao::assemble_inpaint_conditioning(
        t5_embed, t5_seq_len,
        seconds_embed, embed_dim,
        streamgen_latent, inpaint_mask, inpaint_masked_input,
        descriptors, latent_channels, latent_length
    );

    assert(cond.cross_attn_seq_len == t5_seq_len + 1);
    assert(static_cast<int>(cond.cross_attn_cond.size()) == cond.cross_attn_seq_len * embed_dim);
    assert(static_cast<int>(cond.global_embed.size()) == embed_dim);
    assert(cond.input_add.size() == 3);

    float cross_err = max_abs_error(cond.cross_attn_cond, ref_cross_attn.data(), ref_cross_attn.size());
    float global_err = max_abs_error(cond.global_embed, ref_global.data(), ref_global.size());

    std::cout << "Cross-attn max error: " << cross_err << std::endl;
    std::cout << "Global embed max error: " << global_err << std::endl;
    std::cout << "Input add keys: " << cond.input_add.size() << std::endl;

    constexpr float THRESHOLD = 1e-4f;
    bool pass = cross_err < THRESHOLD && global_err < THRESHOLD;

    if (pass) {
        std::cout << "PASS (threshold=" << THRESHOLD << ")" << std::endl;
        return 0;
    } else {
        std::cerr << "FAIL" << std::endl;
        return 1;
    }
}
