#include "sao_inference/Tokenizer.h"
#include <sentencepiece_processor.h>
#include <cassert>
#include <iostream>

namespace sao {

Tokenizer::Tokenizer(const std::string& model_path, int max_length)
    : m_processor(std::make_unique<sentencepiece::SentencePieceProcessor>())
    , m_max_length(max_length)
{
    auto status = m_processor->Load(model_path);
    assert(status.ok());
    std::cout << "[sao::Tokenizer] Loaded " << model_path
              << " (vocab_size=" << m_processor->GetPieceSize() << ")" << std::endl;
}

Tokenizer::~Tokenizer() = default;

TokenizedInput Tokenizer::tokenize(const std::string& text) const
{
    std::vector<int> pieces;
    auto status = m_processor->Encode(text, &pieces);
    assert(status.ok());

    // Append EOS
    pieces.push_back(EOS_TOKEN_ID);

    // Truncate if needed
    if (static_cast<int>(pieces.size()) > m_max_length) {
        pieces.resize(m_max_length);
        pieces.back() = EOS_TOKEN_ID;
    }

    TokenizedInput result;
    result.input_ids.resize(m_max_length, PAD_TOKEN_ID);
    result.attention_mask.resize(m_max_length, 0);

    for (size_t i = 0; i < pieces.size(); ++i) {
        result.input_ids[i] = static_cast<int64_t>(pieces[i]);
        result.attention_mask[i] = 1;
    }

    return result;
}

} // namespace sao
