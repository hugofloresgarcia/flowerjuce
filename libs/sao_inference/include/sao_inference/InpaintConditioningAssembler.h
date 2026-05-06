#pragma once

#include "ConditioningAssembler.h"
#include "InputAddTensor.h"
#include <string>
#include <vector>

namespace sao {

/// Assembled conditioning tensors for the inpainting pipeline.
///
/// Extends the base Conditioning with both per-key input_add tensors (kept
/// for parity-test introspection) and the channel-concatenated tensor that
/// the fused `to_input_add_embed` Linear inside the DiT consumes.
struct InpaintConditioning : Conditioning {
    /// Per-key tensors after mask gating, in manifest declared order.
    std::vector<InputAddTensor> input_add;
    /// Channel-concatenated form of `input_add`, flat (1, sum_channels, T).
    /// This is what gets fed to DiTInpaintModel::forward as `input_add_cond`.
    std::vector<float> input_add_concat;
    /// Sum of channels across all input_add tensors.
    int input_add_total_channels = 0;
};

/// Assemble inpainting conditioning with mask gating + channel concatenation.
///
/// Combines standard cross-attention/global conditioning (same as SAO)
/// with input_add tensors. Applies mask gating rules:
///   - pass_through: tensor is used as-is
///   - multiply_by_mask: tensor *= gate
///   - multiply_by_complement: tensor *= (1 - gate)
///
/// `gate_input_add_key` selects which input_add entry's data acts as the
/// gate. The new fused-input-add Zenon model uses "tf_inpaint_mask"; the
/// older model used "inpaint_mask".
///
/// `tf_inpaint_mask_override`:
///   - nullptr (default): replicate Nithya's reference inference
///     (sat-zenon/stable_audio_tools/models/diffusion.py:202-207). When the
///     manifest declares "tf_inpaint_mask" but no explicit value is supplied
///     we fall back to using `inpaint_mask` for the tf channel.
///   - non-null: use the supplied per-frame binary mask as `tf_inpaint_mask`.
///     Caller is responsible for shape (length == latent_length) and value
///     range ({0, 1}). This is how the StreamGen "Future visibility" knob
///     plumbs a shifted mask down to the model. The training-time semantics
///     are documented in sat-zenon/stable_audio_tools/models/inpainting.py
///     lines 103-115.
///
/// Mirrors ConditionedDiffusionModelWrapper.get_conditioning_inputs() from
/// sat-zenon/stable_audio_tools/models/diffusion.py (lines 196-217).
///
/// Args:
///     t5_embeddings: T5 encoder output, flat (1, t5_seq_len, embed_dim).
///     t5_seq_len: Number of T5 tokens (typically 64).
///     seconds_total_embed: NumberEmbedder output, (embed_dim,).
///     embed_dim: Conditioning embedding dimension (typically 768).
///     streamgen_latent: Encoded streamgen audio, flat (1, C, T).
///     inpaint_mask: Binary mask, flat (1, 1, T). 1=keep, 0=regenerate.
///     inpaint_masked_input: input_latent * mask, flat (1, C, T).
///     key_descriptors: input_add key descriptors from manifest, in declared order.
///     gate_input_add_key: Which key's data acts as the gate ("tf_inpaint_mask"
///         on new models, "inpaint_mask" on legacy). Must appear in `key_descriptors`.
///     latent_channels: C (from manifest).
///     latent_length: T (from manifest).
///     tf_inpaint_mask_override: Optional explicit tf_inpaint_mask; see class doc above.
///         Null = fall back to a copy of `inpaint_mask` (Nithya's reference behaviour).
///
/// Returns:
///     Assembled InpaintConditioning with mask gating + channel concatenation.
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
    const std::vector<float>* tf_inpaint_mask_override = nullptr
);

} // namespace sao
