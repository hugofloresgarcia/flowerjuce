#pragma once

#include "ConditioningAssembler.h"
#include "InputAddTensor.h"
#include <vector>

namespace sao {

/// Assembled conditioning tensors for the inpainting pipeline.
///
/// Extends the base Conditioning with input_add tensors that are passed
/// to the DiT alongside the standard cross-attention and global conditioning.
struct InpaintConditioning : Conditioning {
    std::vector<InputAddTensor> input_add;
};

/// Assemble inpainting conditioning with mask gating.
///
/// Combines standard cross-attention/global conditioning (same as SAO)
/// with input_add tensors. Applies mask gating rules from the manifest:
///   - pass_through: tensor is used as-is
///   - multiply_by_mask: tensor *= inpaint_mask
///   - multiply_by_complement: tensor *= (1 - inpaint_mask)
///
/// Mirrors ConditionedDiffusionModelWrapper.get_conditioning_inputs() from
/// sat-zenon/stable_audio_tools/models/diffusion.py (lines 190-224).
///
/// Args:
///     t5_embeddings: T5 encoder output, flat (1, t5_seq_len, embed_dim).
///     t5_seq_len: Number of T5 tokens (typically 64).
///     seconds_total_embed: NumberEmbedder output, (embed_dim,).
///     embed_dim: Conditioning embedding dimension (typically 768).
///     streamgen_latent: Encoded streamgen audio, flat (1, C, T).
///     inpaint_mask: Binary mask, flat (1, 1, T). 1=keep, 0=regenerate.
///     inpaint_masked_input: input_latent * mask, flat (1, C, T).
///     key_descriptors: input_add key descriptors from manifest.
///     latent_channels: C (from manifest).
///     latent_length: T (from manifest).
///
/// Returns:
///     Assembled InpaintConditioning with mask gating applied.
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
    int latent_length
);

} // namespace sao
