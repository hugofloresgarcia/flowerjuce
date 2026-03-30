#pragma once

#include "OnnxModel.h"
#include "InputAddTensor.h"
#include <string>
#include <vector>

namespace sao {

/// DiT denoiser with input_add support, wrapping the ONNX-exported
/// ZenonDiTInferenceWrapper.
///
/// Extends the base DiTModel by accepting a variable-length set of named
/// input_add tensors (e.g. streamgen_latent, inpaint_mask, inpaint_masked_input).
/// The ONNX model includes the to_input_add_embed projections + sum, so
/// the C++ side provides raw (unprojected) tensors.
///
/// At construction time, queries ONNX Runtime for the model's input names
/// to discover which input_add keys the model expects. At forward time,
/// validates that the provided tensors match the expected names.
///
/// Supports dynamic batch size (B=1 for no CFG, B=2 for CFG batch doubling).
class DiTInpaintModel {
public:
    /// Load the DiT ONNX model and discover its input names.
    ///
    /// Args:
    ///     onnx_path: Path to zenon_dit.onnx.
    ///     use_cuda: If true, use CUDA execution provider.
    ///     use_coreml: If true (macOS), use CoreML execution provider.
    explicit DiTInpaintModel(const std::string& onnx_path, bool use_cuda = false, bool use_coreml = false);

    /// Run a single denoiser step with input_add conditioning.
    ///
    /// Args:
    ///     x: Noisy latent, flat row-major (B, C, T).
    ///     t: Timestep, (B,).
    ///     cross_attn_cond: Cross-attention conditioning, flat (B, S, cond_dim).
    ///     global_embed: Global conditioning, flat (B, global_dim).
    ///     input_add: Variable-length set of named tensors, each flat (B, C_key, T).
    ///     batch_size: Number of batches (1 or 2).
    ///     seq_len: Cross-attention sequence length (typically 65).
    ///     latent_channels: Number of latent channels (from manifest).
    ///     latent_length: Number of latent frames (from manifest).
    ///     cond_dim: Conditioning token dimension (from manifest).
    ///
    /// Returns:
    ///     Predicted velocity, flat row-major (B, C, T).
    std::vector<float> forward(
        const std::vector<float>& x,
        const std::vector<float>& t,
        const std::vector<float>& cross_attn_cond,
        const std::vector<float>& global_embed,
        const std::vector<InputAddTensor>& input_add,
        int batch_size,
        int seq_len,
        int latent_channels,
        int latent_length,
        int cond_dim
    );

private:
    OnnxModel m_model;
    std::vector<std::string> m_input_names;
};

} // namespace sao
