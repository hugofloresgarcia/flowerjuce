#pragma once

#include "OnnxModel.h"
#include "InputAddTensor.h"
#include <string>
#include <vector>

namespace sao {

/// DiT denoiser with fused input_add support, wrapping the ONNX-exported
/// ZenonDiTInferenceWrapper.
///
/// The ONNX graph's `to_input_add_embed` is a single fused Linear that
/// consumes a pre-concatenated tensor of all input_add keys along the
/// channel axis (in the model's declared `input_add_ids` order). The C++
/// side passes that single tensor as the ONNX input named `input_add_cond`.
///
/// ONNX inputs (fixed order):
///     x, t, cross_attn_cond, global_embed, input_add_cond
/// ONNX output:
///     predicted_v
///
/// Supports dynamic batch size (B=1 for no CFG, B=2 for CFG batch doubling).
class DiTInpaintModel {
public:
    /// Load the DiT ONNX model.
    ///
    /// Args:
    ///     onnx_path: Path to zenon_dit.onnx.
    ///     use_cuda: If true, use CUDA execution provider.
    ///     use_coreml: If true (macOS), use CoreML execution provider.
    ///     use_migraphx: If true (Linux/ROCm), use MIGraphX execution provider.
    explicit DiTInpaintModel(const std::string& onnx_path, bool use_cuda = false, bool use_coreml = false, bool use_migraphx = false);

    /// Run a single denoiser step with fused input_add conditioning.
    ///
    /// Args:
    ///     x: Noisy latent, flat row-major (B, C, T).
    ///     t: Timestep, (B,).
    ///     cross_attn_cond: Cross-attention conditioning, flat (B, S, cond_dim).
    ///     global_embed: Global conditioning, flat (B, global_dim).
    ///     input_add_cond: Concatenated input_add tensor, flat
    ///         (B, input_add_total_channels, T), in the model's declared key order.
    ///     batch_size: Number of batches (1 or 2).
    ///     seq_len: Cross-attention sequence length (typically 65).
    ///     latent_channels: Number of latent channels (from manifest).
    ///     latent_length: Number of latent frames (from manifest).
    ///     cond_dim: Conditioning token dimension (from manifest).
    ///     input_add_total_channels: sum of channels across all input_add keys (from manifest).
    ///
    /// Returns:
    ///     Predicted velocity, flat row-major (B, C, T).
    std::vector<float> forward(
        const std::vector<float>& x,
        const std::vector<float>& t,
        const std::vector<float>& cross_attn_cond,
        const std::vector<float>& global_embed,
        const std::vector<float>& input_add_cond,
        int batch_size,
        int seq_len,
        int latent_channels,
        int latent_length,
        int cond_dim,
        int input_add_total_channels
    );

private:
    OnnxModel m_model;
};

} // namespace sao
