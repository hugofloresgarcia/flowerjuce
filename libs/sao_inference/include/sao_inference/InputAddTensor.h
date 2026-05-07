#pragma once

#include <cassert>
#include <string>
#include <vector>

namespace sao {

/// A named per-key tensor used during conditioning assembly.
///
/// Each input_add key (e.g. "streamgen_latent", "inpaint_mask") has a name,
/// channel count, and flat data buffer of shape (1, channels, T). The
/// assembler builds these per-key tensors, applies mask gating, and then
/// concatenates them along the channel axis into a single tensor that the
/// fused `to_input_add_embed` Linear consumes inside the DiT ONNX graph.
struct InputAddTensor {
    std::string name;           // logical key name (e.g. "streamgen_latent")
    std::vector<float> data;    // flat row-major (1, channels, T)
    int channels;               // channel count for this key
};

/// Mask gating rules for input_add tensors during conditioning assembly.
///
/// Driven by the "mask_rule" field in zenon_pipeline_manifest.json.
/// See PORTING.md for the full specification.
enum class MaskRule {
    pass_through,           // no mask applied (e.g. inpaint_mask, inpaint_masked_input)
    multiply_by_mask,       // tensor *= gate (e.g. streamgen_latent)
    multiply_by_complement, // tensor *= (1 - gate) (default for unknown keys)
};

/// Descriptor for one input_add key, loaded from the manifest.
struct InputAddKeyDescriptor {
    std::string name;
    int channels;
    MaskRule mask_rule;
};

/// Sum the channel counts of an ordered set of input_add tensors.
inline int total_channels(const std::vector<InputAddTensor>& tensors)
{
    int total = 0;
    for (const auto& t : tensors) total += t.channels;
    return total;
}

/// Concatenate per-key input_add tensors along the channel axis, producing
/// a flat (1, sum_channels, latent_length) buffer.
///
/// The order of `tensors` IS the channel order in the output and MUST match
/// the model's declared `input_add_ids` order (the C++ assembler builds them
/// in that order, and Python's `get_conditioning_inputs` concatenates
/// `input_add_cond.values()` in the same order — see
/// sat-zenon/stable_audio_tools/models/diffusion.py line 217).
///
/// Each tensor's data is stored channel-major: all time steps for channel 0,
/// then all time steps for channel 1, etc. Concatenation along the channel
/// axis is therefore a simple sequential append of the flat buffers.
///
/// Args:
///     tensors: Per-key tensors, each shape (1, channels, latent_length).
///     latent_length: T (every tensor must agree).
///
/// Returns:
///     Concatenated flat buffer of size sum(channels) * latent_length.
inline std::vector<float> concat_input_add(
    const std::vector<InputAddTensor>& tensors,
    int latent_length)
{
    std::vector<float> out;
    out.reserve(static_cast<size_t>(total_channels(tensors)) * latent_length);
    for (const auto& t : tensors) {
        assert(static_cast<int>(t.data.size()) == t.channels * latent_length);
        out.insert(out.end(), t.data.begin(), t.data.end());
    }
    return out;
}

} // namespace sao
