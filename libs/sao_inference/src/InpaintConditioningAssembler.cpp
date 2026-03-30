#include "sao_inference/InpaintConditioningAssembler.h"
#include <cassert>
#include <iostream>

namespace sao {

static void apply_mask_rule(
    std::vector<float>& data,
    const std::vector<float>& mask,
    MaskRule rule,
    int channels,
    int length)
{
    assert(static_cast<int>(mask.size()) == length);

    if (rule == MaskRule::pass_through) {
        return;
    }

    assert(static_cast<int>(data.size()) == channels * length);

    for (int c = 0; c < channels; ++c) {
        for (int t = 0; t < length; ++t) {
            float m = mask[t];
            float factor = (rule == MaskRule::multiply_by_mask) ? m : (1.0f - m);
            data[c * length + t] *= factor;
        }
    }
}

InpaintConditioning assemble_inpaint_conditioning(
    const std::vector<float>& t5_embeddings,
    int t5_seq_len,
    const std::vector<float>& seconds_total_embed,
    int embed_dim,
    const std::vector<float>& streamgen_latent,
    const std::vector<float>& inpaint_mask,
    const std::vector<float>& inpaint_masked_input,
    const std::vector<InputAddKeyDescriptor>& key_descriptors,
    int latent_channels,
    int latent_length)
{
    InpaintConditioning cond;

    cond.cross_attn_seq_len = t5_seq_len + 1;
    cond.cross_attn_cond.reserve(cond.cross_attn_seq_len * embed_dim);
    cond.cross_attn_cond.insert(cond.cross_attn_cond.end(),
                                 t5_embeddings.begin(), t5_embeddings.end());
    cond.cross_attn_cond.insert(cond.cross_attn_cond.end(),
                                 seconds_total_embed.begin(), seconds_total_embed.end());

    cond.global_embed = seconds_total_embed;

    for (const auto& desc : key_descriptors) {
        InputAddTensor add;
        add.name = desc.name;
        add.channels = desc.channels;

        if (desc.name == "streamgen_latent") {
            add.data = streamgen_latent;
        } else if (desc.name == "inpaint_mask") {
            add.data = inpaint_mask;
        } else if (desc.name == "inpaint_masked_input") {
            add.data = inpaint_masked_input;
        } else {
            std::cerr << "[InpaintConditioningAssembler] Unknown input_add key: "
                      << desc.name << ", filling with zeros" << std::endl;
            add.data.resize(desc.channels * latent_length, 0.0f);
        }

        assert(static_cast<int>(add.data.size()) == desc.channels * latent_length);

        apply_mask_rule(add.data, inpaint_mask, desc.mask_rule, desc.channels, latent_length);

        cond.input_add.push_back(std::move(add));
    }

    return cond;
}

} // namespace sao
