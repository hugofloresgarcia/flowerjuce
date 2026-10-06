#pragma once

#include "OnnxModel.h"

#include <string>
#include <vector>

namespace sao {

/// The sa3-tap DiT denoiser: wraps the static-shape ONNX graph written by scripts/export_sa3_dit.py.
///
/// ONNX inputs, in the order the manifest lists them (`dit.inputs`):
///     x               float32 [B, C, T]
///     t               float32 [B]
///     cross_attn_cond float32 [B, S, cond_dim]      S = prompt tokens + 1 (seconds_total)
///     global_embed    float32 [B, global_dim]
///     local_add_cond  float32 [B, local_add_channels, T]   cat(inpaint_mask, inpaint_masked_input)
///     control         float32 [B, control_channels, T]     the arm's control, already gated by tf_inpaint_mask
///     tf_inpaint_mask float32 [B, 1, T]
/// ONNX output: `v` float32 [B, C, T] (rectified-flow velocity; cfg_scale 1 inside the graph).
///
/// The batch axis is static (the graph is compiled for `Dims::batch`, 2 for single-axis CFG).
class Sa3DiTModel {
public:
    struct Dims {
        int batch = 2;
        int channels = 256;          ///< C = latent_dim
        int frames = 130;            ///< T
        int cross_tokens = 257;      ///< S
        int cond_dim = 768;
        int global_dim = 768;
        int local_add_channels = 257;
        int control_channels = 2;
    };

    /// Args:
    ///     onnx_path: the graph for `dims.batch` (manifest `dit.onnx_by_batch`).
    ///     input_names: the 7 ONNX input names in feed order (manifest `dit.inputs`).
    ///     migraphx_cache_dir: per-arm compiled-program cache (manifest `dit.migraphx_cache_dir`).
    /// Throws std::runtime_error if `input_names` does not have 7 entries.
    Sa3DiTModel(const std::string& onnx_path, const std::vector<std::string>& input_names, const std::string& output_name,
                const std::string& migraphx_cache_dir, bool use_cuda, bool use_coreml, bool use_migraphx);

    /// One denoiser call. Every tensor is flat row-major with the shapes above for `dims`; sizes are checked.
    /// Returns `v`, flat [B, C, T].
    std::vector<float> forward(
        const std::vector<float>& x,
        const std::vector<float>& t,
        const std::vector<float>& cross_attn_cond,
        const std::vector<float>& global_embed,
        const std::vector<float>& local_add_cond,
        const std::vector<float>& control,
        const std::vector<float>& tf_inpaint_mask,
        const Dims& dims);

private:
    OnnxModel m_model;
    std::vector<std::string> m_input_names;
    std::string m_output_name;
};

} // namespace sao
