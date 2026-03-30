#pragma once

#include <memory>
#include <string>
#include <vector>
#include <cassert>

#include <onnxruntime_cxx_api.h>

namespace sao {

/// Lightweight wrapper around an ONNX Runtime inference session.
///
/// Loads a .onnx model file and provides a run() method that takes
/// named input tensors and returns named output tensors.
class OnnxModel {
public:
    /// Load an ONNX model from the given file path.
    ///
    /// Args:
    ///     model_path: Path to the .onnx file.
    ///     use_cuda: If true, attempt to use the CUDA execution provider.
    ///     use_coreml: If true (macOS only), use the CoreML execution provider
    ///                 for GPU + Apple Neural Engine acceleration.
    ///
    /// Raises:
    ///     Ort::Exception if the model cannot be loaded.
    explicit OnnxModel(const std::string& model_path, bool use_cuda = false, bool use_coreml = false);

    /// Run inference with the given input tensors.
    ///
    /// Args:
    ///     input_names: Names of the input tensors (must match model inputs).
    ///     input_tensors: The input Ort::Value tensors.
    ///
    /// Returns:
    ///     Vector of output Ort::Value tensors.
    std::vector<Ort::Value> run(
        const std::vector<const char*>& input_names,
        const std::vector<Ort::Value>& input_tensors
    );

    /// Get the ONNX Runtime memory info (for creating input tensors).
    const Ort::MemoryInfo& memory_info() const { return m_memory_info; }

private:
    Ort::Env m_env;
    Ort::SessionOptions m_session_options;
    std::unique_ptr<Ort::Session> m_session;
    Ort::MemoryInfo m_memory_info;
    std::vector<std::string> m_output_names_owned;
    std::vector<const char*> m_output_names;
};

} // namespace sao
