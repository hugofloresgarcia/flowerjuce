#pragma once

#include "ZenonPipelineConfig.h"
#include "T5Encoder.h"
#include "DiTInpaintModel.h"
#include "VAEEncoder.h"
#include "VAEDecoder.h"
#include "NumberEmbedder.h"
#include "InpaintConditioningAssembler.h"
#include "InpaintSampler.h"

#include <memory>
#include <string>
#include <vector>

namespace sao {

struct ZenonTimingReport {
    double model_load_ms = 0.0;
    double vae_encode_ms = 0.0;
    double t5_encode_ms = 0.0;
    double conditioning_ms = 0.0;
    double sampling_total_ms = 0.0;
    std::vector<double> sampling_step_ms;
    double vae_decode_ms = 0.0;
    double total_ms = 0.0;
};

/// Full streamgen inpainting pipeline: two audio inputs -> stereo audio output.
///
/// Reads architecture configuration from zenon_pipeline_manifest.json.
/// Runs entirely in C++ with no Python dependencies.
/// Uses ONNX Runtime for NN forward passes and manual C++ for control flow.
///
/// Pipeline flow:
///   1. VAE encode streamgen_audio -> streamgen_latent
///   2. VAE encode input_audio -> input_latent
///   3. Build inpaint_mask (prefix mask based on keep_ratio)
///   4. Build inpaint_masked_input = input_latent * mask
///   5. T5 encode text prompt
///   6. NumberEmbedder encode seconds_total
///   7. Assemble conditioning (cross_attn + global + input_add with mask gating)
///   8. Rectified flow Euler sampling with CFG + distribution shift
///   9. VAE decode -> output audio
class ZenonPipeline {
public:
    /// Initialize the pipeline by loading the manifest and all models/weights.
    ///
    /// Args:
    ///     config: Pipeline configuration loaded from manifest.
    explicit ZenonPipeline(const ZenonPipelineConfig& config);

    /// Generate audio from pre-tokenized input and pre-encoded latents.
    ///
    /// This is the low-level entry point for maximum flexibility and
    /// parity testing (accepts .npy-loaded data).
    ///
    /// Args:
    ///     input_ids: Pre-tokenized T5 input IDs (max_length=64).
    ///     attention_mask: Attention mask for input_ids.
    ///     seconds_total: Duration of the audio (0-256).
    ///     streamgen_latent: Pre-encoded streamgen latent, flat (1, C, T).
    ///                       If empty, streamgen_audio must be provided.
    ///     input_latent: Pre-encoded input latent, flat (1, C, T).
    ///                   If empty, input_audio must be provided.
    ///     streamgen_audio: Raw streamgen audio, flat (1, 2, N). Used if streamgen_latent is empty.
    ///     input_audio: Raw input audio, flat (1, 2, N). Used if input_latent is empty.
    ///     keep_ratio: Fraction of latent frames to keep (0.0 to 1.0).
    ///     seed: Random seed for noise generation.
    ///     steps: Number of sampling steps.
    ///     cfg_scale: Classifier-free guidance scale.
    ///     callback: Optional step callback for progress tracking.
    ///     external_noise: If non-empty, use as starting noise instead of RNG.
    ///     dump_steps_dir: If non-empty, save per-step latents as .npy.
    ///
    /// Returns:
    ///     Decoded stereo audio, flat row-major (2, audio_length).
    std::vector<float> generate(
        const std::vector<int64_t>& input_ids,
        const std::vector<int64_t>& attention_mask,
        float seconds_total,
        const std::vector<float>& streamgen_latent,
        const std::vector<float>& input_latent,
        const std::vector<float>& streamgen_audio,
        const std::vector<float>& input_audio,
        float keep_ratio,
        uint32_t seed,
        int steps = 50,
        float cfg_scale = 7.0f,
        StepCallback callback = nullptr,
        const std::vector<float>& external_noise = {},
        const std::string& dump_steps_dir = ""
    );

    const ZenonTimingReport& timing() const { return m_timing; }
    const ZenonPipelineConfig& config() const { return m_config; }

private:
    std::unique_ptr<T5Encoder> m_t5;
    std::unique_ptr<DiTInpaintModel> m_dit;
    std::unique_ptr<VAEEncoder> m_vae_encoder;
    std::unique_ptr<VAEDecoder> m_vae_decoder;
    std::unique_ptr<NumberEmbedder> m_number_embedder;
    ZenonPipelineConfig m_config;
    ZenonTimingReport m_timing;
};

} // namespace sao
