#pragma once

#include <vector>

namespace sao {

/// Assembled conditioning tensors ready for DiT input.
struct Conditioning {
    std::vector<float> cross_attn_cond; // (1, S, 768) flat row-major
    int cross_attn_seq_len;             // S = T5 tokens + number tokens
    std::vector<float> global_embed;    // (1, 768) flat
};

/// Assembles conditioning tensors from T5 embeddings and number embeddings.
///
/// Mirrors ConditionedDiffusionModelWrapper.get_conditioning_inputs():
/// - cross_attn_cond: concat T5 embeddings (1, 64, 768) + seconds_total embed (1, 1, 768)
///   -> (1, 65, 768)
/// - global_cond: seconds_total embed squeezed to (1, 768)
///
/// For the SAO Small config:
/// - cross_attn_cond_ids: ["prompt", "seconds_total"]
/// - global_cond_ids: ["seconds_total"]
/// - No prepend_cond_ids, no input_concat_ids
Conditioning assemble_conditioning(
    const std::vector<float>& t5_embeddings,   // (1, 64, 768) flat
    int t5_seq_len,                             // 64
    const std::vector<float>& seconds_total_embed,  // (768,)
    int embed_dim                               // 768
);

} // namespace sao
