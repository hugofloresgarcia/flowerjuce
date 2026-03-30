#pragma once

#include "OnnxModel.h"
#include <string>
#include <vector>

namespace sao {

/// T5 text encoder wrapper around the ONNX-exported T5EncoderModel.
///
/// Takes pre-tokenized input IDs and returns text embeddings.
/// For tokenization, use SentencePiece C++ API or pre-tokenize in Python.
class T5Encoder {
public:
    /// Load the T5 encoder ONNX model.
    ///
    /// Args:
    ///     onnx_path: Path to t5_encoder.onnx.
    ///     use_cuda: If true, use CUDA execution provider.
    ///     use_coreml: If true (macOS), use CoreML execution provider.
    explicit T5Encoder(const std::string& onnx_path, bool use_cuda = false, bool use_coreml = false);

    /// Run the T5 encoder on pre-tokenized input.
    ///
    /// Args:
    ///     input_ids: Token IDs, shape (seq_len,). Will be padded/truncated to max_length=64.
    ///     attention_mask: Mask, shape (seq_len,). 1 for real tokens, 0 for padding.
    ///
    /// Returns:
    ///     Text embeddings as flat vector, shape (1, 64, 768) row-major.
    std::vector<float> encode(
        const std::vector<int64_t>& input_ids,
        const std::vector<int64_t>& attention_mask
    );

private:
    OnnxModel m_model;
};

} // namespace sao
