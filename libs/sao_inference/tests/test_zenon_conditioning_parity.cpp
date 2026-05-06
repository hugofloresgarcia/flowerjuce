#include "sao_inference/InpaintConditioningAssembler.h"
#include "sao_inference/NumberEmbedder.h"
#include "sao_inference/ZenonPipelineConfig.h"
#include <cnpy.h>
#include <cmath>
#include <iostream>
#include <cassert>
#include <string>
#include <vector>

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
    std::string manifest_path = ZENON_MANIFEST_PATH;

    std::cout << "=== Zenon Conditioning Assembly Parity Test ===" << std::endl;

    // Drive the test from the manifest so it auto-adapts to the model
    // currently exported (key set, channel total, gate source).
    auto config = sao::ZenonPipelineConfig::load(manifest_path);

    sao::NumberEmbedder embedder(weights_dir);

    auto t5_embed = load_npy_flat(parity_dir + "/t5_embeddings_masked.npy");
    auto ref_cross_attn = load_npy_flat(parity_dir + "/cond_input_cross_attn_cond.npy");
    auto ref_global = load_npy_flat(parity_dir + "/cond_input_global_cond.npy");
    auto ref_input_add_cond = load_npy_flat(parity_dir + "/cond_input_input_add_cond.npy");

    const float seconds_total = 11.0f;
    auto seconds_embed = embedder.embed(seconds_total);

    const int embed_dim = config.cond_token_dim;
    const int t5_seq_len = 64;
    const int latent_channels = config.latent_dim;
    const int latent_length = config.latent_length;

    auto streamgen_latent = load_npy_flat(parity_dir + "/streamgen_latent.npy");
    auto inpaint_mask_full = load_npy_flat(parity_dir + "/inpaint_mask.npy");
    auto inpaint_masked_input = load_npy_flat(parity_dir + "/inpaint_masked_input.npy");

    // inpaint_mask is dumped as (1, 1, T); the assembler expects a flat (T,)
    // gate tensor.
    std::vector<float> inpaint_mask(inpaint_mask_full.begin(),
                                    inpaint_mask_full.begin() + latent_length);

    auto cond = sao::assemble_inpaint_conditioning(
        t5_embed, t5_seq_len,
        seconds_embed, embed_dim,
        streamgen_latent, inpaint_mask, inpaint_masked_input,
        config.input_add_keys, config.gate_input_add_key,
        latent_channels, latent_length
    );

    assert(cond.cross_attn_seq_len == t5_seq_len + 1);
    assert(static_cast<int>(cond.cross_attn_cond.size()) == cond.cross_attn_seq_len * embed_dim);
    assert(static_cast<int>(cond.global_embed.size()) == embed_dim);
    assert(cond.input_add.size() == config.input_add_keys.size());
    assert(cond.input_add_total_channels == config.input_add_total_channels);
    assert(static_cast<int>(cond.input_add_concat.size())
           == config.input_add_total_channels * latent_length);

    float cross_err = max_abs_error(cond.cross_attn_cond, ref_cross_attn.data(), ref_cross_attn.size());
    float global_err = max_abs_error(cond.global_embed, ref_global.data(), ref_global.size());

    // The reference tensor is shape (1, sum_channels, T) flat — same layout
    // as cond.input_add_concat.
    if (ref_input_add_cond.size() != cond.input_add_concat.size()) {
        std::cerr << "FAIL: input_add_cond size mismatch — reference "
                  << ref_input_add_cond.size() << " vs assembled "
                  << cond.input_add_concat.size() << std::endl;
        return 2;
    }
    float input_add_err = max_abs_error(cond.input_add_concat,
                                        ref_input_add_cond.data(),
                                        ref_input_add_cond.size());

    std::cout << "Cross-attn max error:        " << cross_err << std::endl;
    std::cout << "Global embed max error:      " << global_err << std::endl;
    std::cout << "Input add concat max error:  " << input_add_err << std::endl;
    std::cout << "Input add keys: " << cond.input_add.size()
              << " (gate=" << config.gate_input_add_key << ")" << std::endl;

    constexpr float THRESHOLD = 1e-4f;
    bool pass = cross_err < THRESHOLD && global_err < THRESHOLD && input_add_err < THRESHOLD;

    if (pass) {
        std::cout << "PASS (threshold=" << THRESHOLD << ")" << std::endl;
        return 0;
    } else {
        std::cerr << "FAIL" << std::endl;
        return 1;
    }
}
