#pragma once

#include <string>
#include <vector>
#include <memory>

namespace sentencepiece { class SentencePieceProcessor; }

namespace sao {

/// Tokenization result with T5-style padding.
struct TokenizedInput {
    std::vector<int64_t> input_ids;      // (max_length,)
    std::vector<int64_t> attention_mask;  // (max_length,)
};

/// T5 tokenizer wrapping SentencePiece.
///
/// Loads a .model file (same format as HuggingFace T5 tokenizer),
/// tokenizes text, appends EOS token, pads to max_length.
class Tokenizer {
public:
    /// Load a SentencePiece .model file.
    ///
    /// Args:
    ///     model_path: Path to the .model file (e.g. t5_tokenizer.model).
    ///     max_length: Maximum sequence length (default 64, matching T5 config).
    explicit Tokenizer(const std::string& model_path, int max_length = 64);
    ~Tokenizer();

    /// Tokenize a text string.
    ///
    /// Produces input_ids and attention_mask with T5 conventions:
    /// - Tokens from SentencePiece encoding
    /// - EOS token (id=1) appended
    /// - Padded with pad_token (id=0) to max_length
    /// - Truncated to max_length if necessary
    ///
    /// Args:
    ///     text: The input text to tokenize.
    ///
    /// Returns:
    ///     TokenizedInput with input_ids and attention_mask.
    TokenizedInput tokenize(const std::string& text) const;

    int max_length() const { return m_max_length; }

private:
    std::unique_ptr<sentencepiece::SentencePieceProcessor> m_processor;
    int m_max_length;
    static constexpr int EOS_TOKEN_ID = 1;
    static constexpr int PAD_TOKEN_ID = 0;
};

} // namespace sao
