#pragma once

#include "InputAddTensor.h"
#include <string>
#include <vector>

namespace sao {

/// Distribution shift parameters for rectified flow sampling.
///
/// Loaded from the "distribution_shift" object in the manifest.
/// When present, the time schedule is shifted via time_shift() before sampling.
struct DistributionShiftConfig {
    float base_shift = 0.5f;
    float max_shift = 1.15f;
    int min_length = 256;
    int max_length = 4096;
    bool enabled = false;
};

/// Pipeline configuration loaded from zenon_pipeline_manifest.json.
///
/// This is the single source of truth for the C++ side. No dimensions,
/// key names, or architectural decisions should be hardcoded elsewhere.
///
/// Manifest fields:
///     sample_rate, sample_size, latent_dim, latent_length, downsampling_ratio,
///     cond_token_dim, global_cond_dim, embed_dim, cross_attn_cond_ids,
///     global_cond_ids, input_add_keys[], distribution_shift{}, diffusion_objective,
///     vae_scale, onnx_files{}, number_embedder_weights_dir
struct ZenonPipelineConfig {
    std::string manifest_dir;

    int sample_rate = 44100;
    int sample_size = 524288;
    int latent_dim = 64;
    int latent_length = 256;
    int downsampling_ratio = 2048;
    int cond_token_dim = 768;
    int global_cond_dim = 768;
    int embed_dim = 1024;

    std::vector<std::string> cross_attn_cond_ids;
    std::vector<std::string> global_cond_ids;
    std::vector<InputAddKeyDescriptor> input_add_keys;

    DistributionShiftConfig dist_shift;
    std::string diffusion_objective = "rectified_flow";
    float vae_scale = 1.0f;

    std::string t5_onnx_path;
    std::string dit_onnx_path;
    std::string vae_encoder_onnx_path;
    std::string vae_decoder_onnx_path;
    std::string number_embedder_weights_dir;

    /// When true, VAE uses MLX Metal (see manifest `mlx/` paths) instead of ONNX CPU.
    bool use_mlx_vae = false;
    std::string mlx_vae_weights_path;
    std::string mlx_vae_config_path;

    bool use_cuda = false;
    bool use_coreml = false;

    /// Load configuration from a zenon_pipeline_manifest.json file.
    ///
    /// Args:
    ///     manifest_path: Path to the manifest JSON file.
    ///
    /// Populates all fields and resolves ONNX file paths relative to the
    /// manifest's parent directory.
    static ZenonPipelineConfig load(const std::string& manifest_path);
};

} // namespace sao
