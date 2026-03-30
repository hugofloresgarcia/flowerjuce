#include "sao_inference/ConditioningAssembler.h"
#include <cassert>

namespace sao {

Conditioning assemble_conditioning(
    const std::vector<float>& t5_embeddings,
    int t5_seq_len,
    const std::vector<float>& seconds_total_embed,
    int embed_dim)
{
    assert(static_cast<int>(t5_embeddings.size()) == t5_seq_len * embed_dim);
    assert(static_cast<int>(seconds_total_embed.size()) == embed_dim);

    Conditioning cond;
    int total_seq_len = t5_seq_len + 1;
    cond.cross_attn_seq_len = total_seq_len;

    // cross_attn_cond: concat T5 (64 tokens) + seconds_total (1 token) along seq dim
    cond.cross_attn_cond.reserve(total_seq_len * embed_dim);
    cond.cross_attn_cond.insert(cond.cross_attn_cond.end(),
                                 t5_embeddings.begin(), t5_embeddings.end());
    cond.cross_attn_cond.insert(cond.cross_attn_cond.end(),
                                 seconds_total_embed.begin(), seconds_total_embed.end());

    // global_embed: just the seconds_total embed (squeezed to (768,))
    cond.global_embed = seconds_total_embed;

    return cond;
}

} // namespace sao
