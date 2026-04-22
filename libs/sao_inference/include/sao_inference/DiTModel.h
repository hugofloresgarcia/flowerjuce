#pragma once

#include "OnnxModel.h"
#include <string>
#include <vector>

namespace sao {

/// DiT denoiser wrapper around the ONNX-exported DiTInferenceWrapper.
///
/// Runs a single denoiser forward pass. Supports dynamic batch size
/// (B=1 for no CFG, B=2 for CFG batch doubling).
class DiTModel {
public:
    /// Load the DiT ONNX model.
    ///
    /// Args:
    ///     onnx_path: Path to dit_step.onnx.
    ///     use_cuda: If true, use CUDA execution provider.
    ///     use_coreml: If true (macOS), use CoreML execution provider.
    ///     use_migraphx: If true (Linux/ROCm), use MIGraphX execution provider.
    explicit DiTModel(const std::string& onnx_path, bool use_cuda = false, bool use_coreml = false, bool use_migraphx = false);

    /// Run a single denoiser step.
    ///
    /// Args:
    ///     x: Noisy latent, flat row-major (B, 64, 256).
    ///     t: Timestep, (B,).
    ///     cross_attn_cond: Cross-attention conditioning, flat (B, S, 768).
    ///     global_embed: Global conditioning, flat (B, 768).
    ///     batch_size: Number of batches (1 or 2).
    ///     seq_len: Cross-attention sequence length (typically 65).
    ///
    /// Returns:
    ///     Predicted velocity, flat row-major (B, 64, 256).
    std::vector<float> forward(
        const std::vector<float>& x,
        const std::vector<float>& t,
        const std::vector<float>& cross_attn_cond,
        const std::vector<float>& global_embed,
        int batch_size,
        int seq_len
    );

private:
    OnnxModel m_model;
};

} // namespace sao
