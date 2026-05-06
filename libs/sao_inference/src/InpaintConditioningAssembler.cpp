#include "sao_inference/InpaintConditioningAssembler.h"
#include <cassert>
#include <iostream>
#include <stdexcept>

namespace sao {

static void apply_mask_rule(
    std::vector<float>& data,
    const std::vector<float>& gate,
    MaskRule rule,
    int channels,
    int length)
{
    assert(static_cast<int>(gate.size()) == length);

    if (rule == MaskRule::pass_through) {
        return;
    }

    assert(static_cast<int>(data.size()) == channels * length);

    for (int c = 0; c < channels; ++c) {
        for (int t = 0; t < length; ++t) {
            const float m = gate[t];
            const float factor = (rule == MaskRule::multiply_by_mask) ? m : (1.0f - m);
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
    const std::string& gate_input_add_key,
    int latent_channels,
    int latent_length,
    const std::vector<float>* tf_inpaint_mask_override)
{
    if (tf_inpaint_mask_override != nullptr
        && static_cast<int>(tf_inpaint_mask_override->size()) != latent_length) {
        throw std::runtime_error(
            "[InpaintConditioningAssembler] tf_inpaint_mask_override has length "
            + std::to_string(tf_inpaint_mask_override->size())
            + " but expected " + std::to_string(latent_length));
    }

    InpaintConditioning cond;

    // ---- Cross-attention: T5 tokens followed by the seconds_total embed ----
    cond.cross_attn_seq_len = t5_seq_len + 1;
    cond.cross_attn_cond.reserve(cond.cross_attn_seq_len * embed_dim);
    cond.cross_attn_cond.insert(cond.cross_attn_cond.end(),
                                 t5_embeddings.begin(), t5_embeddings.end());
    cond.cross_attn_cond.insert(cond.cross_attn_cond.end(),
                                 seconds_total_embed.begin(), seconds_total_embed.end());

    // ---- Global conditioning ----
    cond.global_embed = seconds_total_embed;

    // ---- Per-key input_add tensors ----
    // Build them in manifest-declared order (which is also the channel order
    // of the final concatenated tensor). For "tf_inpaint_mask" we use the
    // caller-supplied override when provided (StreamGen "Future visibility"
    // knob path); otherwise we fall back to a copy of inpaint_mask, matching
    // sat-zenon/stable_audio_tools/models/diffusion.py:202-207 ("tf_inpaint_mask
    // not found in input_add_cond, using inpaint_mask instead").
    cond.input_add.reserve(key_descriptors.size());
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
        } else if (desc.name == "tf_inpaint_mask") {
            if (tf_inpaint_mask_override != nullptr) {
                add.data = *tf_inpaint_mask_override;
            } else {
                add.data = inpaint_mask;
            }
        } else {
            std::cerr << "[InpaintConditioningAssembler] Unknown input_add key '"
                      << desc.name << "', filling with zeros" << std::endl;
            add.data.assign(static_cast<size_t>(desc.channels) * latent_length, 0.0f);
        }

        if (static_cast<int>(add.data.size()) != desc.channels * latent_length) {
            throw std::runtime_error(
                "[InpaintConditioningAssembler] Size mismatch for key '" + desc.name
                + "': expected " + std::to_string(desc.channels * latent_length)
                + ", got " + std::to_string(add.data.size()));
        }
        cond.input_add.push_back(std::move(add));
    }

    // ---- Locate the gate tensor by name ----
    // The gate is the data buffer of one of the keys we just built. By default
    // the new fused-input-add manifests use "tf_inpaint_mask" (== inpaint_mask
    // at inference); legacy manifests use "inpaint_mask". If the manifest
    // doesn't specify, fall back to inpaint_mask.
    const std::vector<float>* gate = nullptr;
    if (!gate_input_add_key.empty()) {
        for (const auto& add : cond.input_add) {
            if (add.name == gate_input_add_key) {
                gate = &add.data;
                break;
            }
        }
        if (gate == nullptr) {
            std::cerr << "[InpaintConditioningAssembler] gate_input_add_key '"
                      << gate_input_add_key
                      << "' not found among input_add keys; falling back to inpaint_mask"
                      << std::endl;
        }
    }
    if (gate == nullptr) {
        gate = &inpaint_mask;
    }

    // ---- Apply mask rules to each per-key tensor ----
    for (size_t i = 0; i < cond.input_add.size(); ++i) {
        const auto& desc = key_descriptors[i];
        apply_mask_rule(cond.input_add[i].data, *gate, desc.mask_rule, desc.channels, latent_length);
    }

    // ---- Concatenate per-key tensors along channel axis ----
    cond.input_add_total_channels = total_channels(cond.input_add);
    cond.input_add_concat = concat_input_add(cond.input_add, latent_length);

    return cond;
}

} // namespace sao
