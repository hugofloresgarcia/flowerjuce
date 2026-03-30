#pragma once

#include "T5Encoder.h"
#include "DiTModel.h"
#include "VAEDecoder.h"
#include "NumberEmbedder.h"
#include "ConditioningAssembler.h"
#include "RectifiedFlowSampler.h"

#include <memory>
#include <string>
#include <vector>

namespace sao {

struct PipelineConfig {
    std::string t5_onnx_path;
    std::string dit_onnx_path;
    std::string vae_onnx_path;
    std::string number_embedder_weights_dir;
    float vae_scale = 1.0f;
    bool use_cuda = false;
    bool use_coreml = false;
    int sample_rate = 44100;
    int latent_length = 256;
};

struct TimingReport {
    double model_load_ms = 0.0;
    double t5_encode_ms = 0.0;
    double conditioning_ms = 0.0;
    double sampling_total_ms = 0.0;
    std::vector<double> sampling_step_ms;
    double vae_decode_ms = 0.0;
    double total_ms = 0.0;
};

/// Full text-to-audio pipeline: text prompt -> stereo audio waveform.
///
/// Runs entirely in C++ with no Python dependencies.
/// Uses ONNX Runtime for NN forward passes and manual C++ for control flow.
class Pipeline {
public:
    /// Initialize the pipeline by loading all models and weights.
    ///
    /// Args:
    ///     config: Paths to ONNX models, weights, and configuration.
    explicit Pipeline(const PipelineConfig& config);

    /// Generate stereo audio from pre-tokenized input.
    ///
    /// Args:
    ///     input_ids: Pre-tokenized T5 input IDs (max_length=64).
    ///     attention_mask: Attention mask for input_ids.
    ///     seconds_total: Duration of the audio to generate (0-256).
    ///     seed: Random seed for noise generation (ignored if external_noise is provided).
    ///     steps: Number of sampling steps.
    ///     cfg_scale: Classifier-free guidance scale.
    ///     callback: Optional step callback for progress tracking.
    ///     external_noise: If non-empty, use this as the starting noise instead of RNG.
    ///                     Must be flat row-major (latent_channels * latent_length).
    ///     dump_steps_dir: If non-empty, save per-step latents as .npy to this directory.
    ///
    /// Returns:
    ///     Decoded stereo audio, flat row-major (2, audio_length).
    std::vector<float> generate(
        const std::vector<int64_t>& input_ids,
        const std::vector<int64_t>& attention_mask,
        float seconds_total,
        uint32_t seed,
        int steps = 8,
        float cfg_scale = 7.0f,
        StepCallback callback = nullptr,
        const std::vector<float>& external_noise = {},
        const std::string& dump_steps_dir = ""
    );

    const TimingReport& timing() const { return m_timing; }

private:
    std::unique_ptr<T5Encoder> m_t5;
    std::unique_ptr<DiTModel> m_dit;
    std::unique_ptr<VAEDecoder> m_vae;
    std::unique_ptr<NumberEmbedder> m_number_embedder;
    PipelineConfig m_config;
    TimingReport m_timing;
};

} // namespace sao
