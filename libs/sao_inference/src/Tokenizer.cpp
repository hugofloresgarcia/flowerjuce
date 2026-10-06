#include "sao_inference/Tokenizer.h"

#include <sentencepiece_processor.h>

#include <iostream>
#include <stdexcept>

namespace sao {

Tokenizer::Tokenizer(const std::string& model_path, const TokenizerOptions& options)
    : m_processor(std::make_unique<sentencepiece::SentencePieceProcessor>())
    , m_options(options)
{
    if (m_options.max_length <= 0)
        throw std::runtime_error("Tokenizer: max_length must be > 0");
    const auto status = m_processor->Load(model_path);
    if (!status.ok())
        throw std::runtime_error("Tokenizer: cannot load " + model_path + ": " + status.ToString());
    std::cout << "[sao::Tokenizer] Loaded " << model_path << " (vocab_size=" << m_processor->GetPieceSize()
              << ", max_length=" << m_options.max_length << ", bos=" << (m_options.add_bos ? "yes" : "no")
              << ", eos=" << (m_options.add_eos ? "yes" : "no") << ")" << std::endl;
}

Tokenizer::~Tokenizer() = default;

int Tokenizer::vocab_size() const
{
    return m_processor->GetPieceSize();
}

TokenizedInput Tokenizer::tokenize(const std::string& text) const
{
    std::vector<int> pieces;
    const auto status = m_processor->Encode(text, &pieces);
    if (!status.ok())
        throw std::runtime_error("Tokenizer: encode failed: " + status.ToString());

    std::vector<int> ids;
    ids.reserve(pieces.size() + 2);
    if (m_options.add_bos) ids.push_back(m_options.bos_id);
    ids.insert(ids.end(), pieces.begin(), pieces.end());
    if (m_options.add_eos) ids.push_back(m_options.eos_id);

    const size_t max_len = static_cast<size_t>(m_options.max_length);
    if (ids.size() > max_len) {
        ids.resize(max_len);
        if (m_options.add_eos) ids.back() = m_options.eos_id;
    }

    TokenizedInput result;
    result.input_ids.assign(max_len, static_cast<int64_t>(m_options.pad_id));
    result.attention_mask.assign(max_len, 0);
    for (size_t i = 0; i < ids.size(); ++i) {
        result.input_ids[i] = static_cast<int64_t>(ids[i]);
        result.attention_mask[i] = 1;
    }
    return result;
}

} // namespace sao
