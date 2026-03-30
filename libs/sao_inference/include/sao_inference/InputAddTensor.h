#pragma once

#include <string>
#include <vector>

namespace sao {

/// A named tensor for the DiT input_add mechanism.
///
/// Each input_add key (e.g. "streamgen_latent", "inpaint_mask") has a name,
/// channel count, and flat data buffer. The DiT ONNX model expects one
/// named input per key, with shape (B, channels, T).
///
/// These are auto-discovered from the ONNX model's input list at load time
/// and driven by the pipeline manifest at runtime.
struct InputAddTensor {
    std::string name;           // ONNX input name (e.g. "streamgen_latent")
    std::vector<float> data;    // flat row-major (B, channels, T)
    int channels;               // channel count for this key
};

/// Mask gating rules for input_add tensors during conditioning assembly.
///
/// Driven by the "mask_rule" field in zenon_pipeline_manifest.json.
/// See PORTING.md for the full specification.
enum class MaskRule {
    pass_through,           // no mask applied (e.g. inpaint_mask, inpaint_masked_input)
    multiply_by_mask,       // tensor *= mask (e.g. streamgen_latent)
    multiply_by_complement, // tensor *= (1 - mask) (default for unknown keys)
};

/// Descriptor for one input_add key, loaded from the manifest.
struct InputAddKeyDescriptor {
    std::string name;
    int channels;
    MaskRule mask_rule;
};

} // namespace sao
