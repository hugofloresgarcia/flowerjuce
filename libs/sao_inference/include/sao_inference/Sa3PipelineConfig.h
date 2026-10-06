#pragma once

#include "ControlFeatureExtractor.h"
#include "Tokenizer.h"

#include <array>
#include <map>
#include <string>
#include <vector>

namespace sao {

/// Everything the SA3 pipeline and StreamGen read from `sa3_pipeline_manifest.json`
/// (written by scripts/write_sa3_manifest.py; schema "sa3_pipeline_manifest/1").
///
/// Paths are resolved to absolute paths at load time (the manifest stores them relative to its own directory).
/// Nothing here has a default that stands in for a missing field: `load` throws std::runtime_error on a missing
/// or inconsistent field, a wrong schema, or a referenced file that does not exist.
struct Sa3PipelineConfig {
    struct Window {
        int frames = 0;                 ///< latent frames per generation window (130)
        int samples = 0;                ///< frames * downsampling_ratio
        float seconds_total = 0.0f;     ///< the value fed to the seconds_total conditioner
        std::array<float, 2> future_visibility_seconds{{0.0f, 0.0f}};  ///< trained lookahead range (e.g. -4..0)
        std::array<int, 2> future_visibility_frames{{0, 0}};
    };
    struct Text {
        std::string onnx;
        int max_length = 0;
        int dim = 0;
        std::string tokenizer_model;
        TokenizerOptions tokenizer;
        std::string migraphx_cache_dir;
    };
    struct NumberEmbedderSpec {
        std::string weights_dir;
        float min_val = 0.0f;
        float max_val = 0.0f;
        int dim = 0;
        int features = 0;
    };
    struct LocalAddEntry {
        std::string id;
        int channels = 0;
    };
    struct DiT {
        std::map<int, std::string> onnx_by_batch;  ///< batch size -> graph path
        int batch = 0;                              ///< the batch the pipeline runs (CFG branches)
        std::vector<std::string> inputs;            ///< ONNX input names in feed order
        std::string output;
        int io_channels = 0;
        int embed_dim = 0;
        int depth = 0;
        int cond_token_dim = 0;
        int global_cond_dim = 0;
        int cross_attn_tokens = 0;                  ///< prompt tokens + 1 (seconds_total)
        std::vector<LocalAddEntry> local_add_layout;
        int local_add_channels = 0;
        std::string migraphx_cache_dir;
    };
    struct Control {
        std::string id;
        int dim = 0;
        float null_value = 0.0f;
        ControlFeatureConfig feature;
    };
    struct Vae {
        std::string decoder_onnx;
        std::string encoder_onnx;
        int latent_dim = 0;
        float scale = 1.0f;
        std::string migraphx_cache_dir;
    };
    struct LogSnrSchedule {
        float rate = 0.0f;
        float anchor_logsnr = 0.0f;
        float logsnr_end = 0.0f;
        float anchor_length = 0.0f;
    };
    struct Sampler {
        std::string type;               ///< "euler"
        int steps_default = 0;
        float sigma_max = 1.0f;
        float sigma_min = 0.0f;
        LogSnrSchedule schedule;
        std::string diffusion_objective; ///< "rf_denoiser"
    };
    struct Cfg {
        std::string mode;               ///< "single_axis_apg"
        std::vector<std::string> axes;  ///< selectable guided axes ("control", "context")
        std::string axis_default;
        float scale_default = 1.0f;
        float apg_scale = 1.0f;
    };

    // Backend selection: set by the caller (CLI flags), never by the manifest.
    bool use_cuda = false;
    bool use_coreml = false;
    bool use_migraphx = false;

    std::string manifest_path;
    std::string manifest_dir;
    std::string schema;
    std::string arm;
    std::string model_config_source;
    std::string checkpoint_path;
    long long global_step = 0;
    int sample_rate = 0;
    int audio_channels = 0;
    int latent_dim = 0;
    int downsampling_ratio = 0;
    Window window;
    Text text;
    NumberEmbedderSpec number_embedder;
    DiT dit;
    Control control;
    Vae vae;
    Sampler sampler;
    Cfg cfg;

    /// Parse and validate `path`. Throws std::runtime_error with the offending field or file in the message.
    static Sa3PipelineConfig load(const std::string& path);

    /// Path of the DiT graph for `batch`; throws if the manifest has none.
    const std::string& dit_onnx(int batch) const;
};

} // namespace sao
