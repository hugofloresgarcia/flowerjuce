#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace sentencepiece { class SentencePieceProcessor; }

namespace sao {

/// Fixed-length tokenization result.
struct TokenizedInput {
    std::vector<int64_t> input_ids;       // (max_length,)
    std::vector<int64_t> attention_mask;  // (max_length,) 1 = real token
};

/// How a SentencePiece model is wrapped into a model's input convention. Values come from the manifest.
struct TokenizerOptions {
    int max_length = 256;   ///< fixed sequence length (truncate / right-pad to this)
    bool add_bos = false;   ///< prepend bos_id (T5Gemma: false)
    bool add_eos = false;   ///< append eos_id   (T5Gemma: false; T5: true)
    int pad_id = 0;
    int bos_id = 2;
    int eos_id = 1;
};

/// SentencePiece tokenizer producing fixed-length `input_ids` / `attention_mask`.
///
/// For the T5Gemma conditioner the HF `GemmaTokenizer` output equals the raw SentencePiece ids with no
/// BOS/EOS, right-padded with 0 and truncated to `max_length` (verified in scripts/export_sa3_text_encoder.py).
class Tokenizer {
public:
    /// Throws std::runtime_error if the model file cannot be loaded or the options are inconsistent.
    Tokenizer(const std::string& model_path, const TokenizerOptions& options);
    ~Tokenizer();

    /// Encode `text`: pieces (+ optional BOS/EOS), truncated to `max_length` (an EOS, if requested, is kept
    /// as the last token), right-padded with `pad_id`.
    TokenizedInput tokenize(const std::string& text) const;

    const TokenizerOptions& options() const { return m_options; }
    int max_length() const { return m_options.max_length; }
    int vocab_size() const;

private:
    std::unique_ptr<sentencepiece::SentencePieceProcessor> m_processor;
    TokenizerOptions m_options;
};

} // namespace sao
