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

    if (!pass) {
        std::cerr << "FAIL on default (V=0) parity check" << std::endl;
        return 1;
    }
    std::cout << "PASS default V=0 (threshold=" << THRESHOLD << ")" << std::endl;

    // ---- Second sub-test: tf_inpaint_mask override for V = -3 ----
    // We can't compare against a Python reference here (Nithya's inference path
    // never sets tf_inpaint_mask differently from inpaint_mask), so we hand-
    // compute the expected gating: for the streamgen_latent channels of
    // input_add_concat, frames in [tf_keep_frames, T) must be zero; everything
    // else must equal the V=0 result. The other channels are gated by the same
    // tf gate (mask_rule=pass_through for inpaint_mask / inpaint_masked_input /
    // tf_inpaint_mask), so they're identical to the V=0 case.
    std::cout << "\n=== V<0 sub-test (tf_inpaint_mask override) ===" << std::endl;

    // Find the keep prefix length used in the existing reference data so we
    // know what tf_keep_frames falls out of K + V.
    int keep_frames = 0;
    for (int i = 0; i < latent_length; ++i) {
        if (inpaint_mask[i] >= 0.5f) ++keep_frames;
    }
    constexpr int V = -3;
    int tf_keep_frames = keep_frames + V;
    if (tf_keep_frames < 0) tf_keep_frames = 0;
    if (tf_keep_frames > latent_length) tf_keep_frames = latent_length;
    std::cout << "  keep_frames=" << keep_frames
              << "  V=" << V
              << "  tf_keep_frames=" << tf_keep_frames << std::endl;

    std::vector<float> tf_mask(latent_length, 0.0f);
    for (int i = 0; i < tf_keep_frames; ++i) tf_mask[i] = 1.0f;

    auto cond_v = sao::assemble_inpaint_conditioning(
        t5_embed, t5_seq_len,
        seconds_embed, embed_dim,
        streamgen_latent, inpaint_mask, inpaint_masked_input,
        config.input_add_keys, config.gate_input_add_key,
        latent_channels, latent_length,
        &tf_mask
    );

    // Locate streamgen_latent and tf_inpaint_mask blocks inside the concat tensor.
    int sl_offset = -1;
    int sl_channels = 0;
    int tf_offset = -1;
    int tf_channels = 0;
    {
        int offset = 0;
        for (const auto& desc : config.input_add_keys) {
            if (desc.name == "streamgen_latent") {
                sl_offset = offset;
                sl_channels = desc.channels;
            } else if (desc.name == "tf_inpaint_mask") {
                tf_offset = offset;
                tf_channels = desc.channels;
            }
            offset += desc.channels;
        }
    }
    if (sl_offset < 0) {
        std::cerr << "FAIL: streamgen_latent not found in manifest input_add_keys" << std::endl;
        return 3;
    }

    // Verify three things:
    //   (a) streamgen_latent block: zeros at t >= tf_keep_frames; equals V=0 result for t < tf_keep_frames.
    //   (b) tf_inpaint_mask block (if present): exactly matches the supplied tf_mask.
    //   (c) every other channel block (inpaint_mask, inpaint_masked_input, ...): identical to V=0.
    float zero_violation = 0.0f;
    float sl_nonzero_diff = 0.0f;
    float tf_diff = 0.0f;
    float other_diff = 0.0f;
    const int total_ch = config.input_add_total_channels;
    for (int c = 0; c < total_ch; ++c) {
        const bool is_sl = (c >= sl_offset && c < sl_offset + sl_channels);
        const bool is_tf = (tf_offset >= 0 && c >= tf_offset && c < tf_offset + tf_channels);
        for (int t = 0; t < latent_length; ++t) {
            const size_t idx = static_cast<size_t>(c) * latent_length + t;
            const float v0 = cond.input_add_concat[idx];
            const float vV = cond_v.input_add_concat[idx];
            if (is_sl) {
                if (t >= tf_keep_frames) {
                    if (std::abs(vV) > zero_violation) zero_violation = std::abs(vV);
                } else {
                    const float d = std::abs(vV - v0);
                    if (d > sl_nonzero_diff) sl_nonzero_diff = d;
                }
            } else if (is_tf) {
                const float d = std::abs(vV - tf_mask[t]);
                if (d > tf_diff) tf_diff = d;
            } else {
                const float d = std::abs(vV - v0);
                if (d > other_diff) other_diff = d;
            }
        }
    }
    std::cout << "  streamgen_latent zero region max abs:    " << zero_violation << std::endl;
    std::cout << "  streamgen_latent kept region max delta:  " << sl_nonzero_diff << std::endl;
    std::cout << "  tf_inpaint_mask channel max delta vs supplied: " << tf_diff << std::endl;
    std::cout << "  other channels max delta vs V=0:         " << other_diff << std::endl;

    bool pass_v = zero_violation < 1e-6f
                  && sl_nonzero_diff < 1e-6f
                  && tf_diff < 1e-6f
                  && other_diff < 1e-6f;
    if (!pass_v) {
        std::cerr << "FAIL on V=" << V << " sub-test" << std::endl;
        return 4;
    }
    std::cout << "PASS V=" << V << " sub-test" << std::endl;
    return 0;
}
