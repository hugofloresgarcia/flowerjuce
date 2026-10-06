#pragma once

#include "IVaeDecoder.h"
#include "IVaeEncoder.h"
#include "NumberEmbedder.h"
#include "Sa3DiTModel.h"
#include "Sa3PipelineConfig.h"
#include "Sa3Sampler.h"
#include "T5Encoder.h"

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace sao {

/// High-level phase of `Sa3Pipeline::generate()` for progress UI (same shape as the Zenon pipeline's).
enum class Sa3InferencePhase : uint8_t {
    Encode = 0, ///< VAE encode (if needed), text encode, conditioning, noise prep.
    DiT = 1,    ///< Euler sampling; `dit_step_1_based` is 1..steps.
    Decode = 2, ///< VAE decode to waveform.
};
using Sa3PhaseProgressCallback = std::function<void(Sa3InferencePhase phase, int dit_step_1_based)>;

/// Which conditioning axis the second DiT branch nulls (manifest `cfg.axes`).
enum class CfgAxis : uint8_t {
    Control = 0, ///< the drum control -> `null_value` on the visible frames (training's control dropout)
    Context = 1, ///< the inpaint context -> zero latent, `inpaint_mask` kept (training's inpaint dropout)
};
CfgAxis parse_cfg_axis(const std::string& name);
const char* cfg_axis_name(CfgAxis axis);

struct Sa3LatentStats {
    double mean = 0.0;
    double std_dev = 0.0;
};

struct Sa3DiffusionStepStats {
    int step_index = 0;
    float t_curr = 0.0f;
    double mean_x = 0.0;
    double std_x = 0.0;
};

/// Diagnostics collected during the last `generate()`.
struct Sa3Diagnostics {
    Sa3LatentStats context_latent;
    Sa3LatentStats control;          ///< the gated control frames fed to the conditional branch
    Sa3LatentStats noise;
    Sa3LatentStats sampled_latent;
    int keep_frames = 0;             ///< inpaint context frames
    int tf_keep_frames = 0;          ///< control-visible frames (keep + future_visibility)
    std::vector<Sa3DiffusionStepStats> diffusion_steps;
};

struct Sa3TimingReport {
    double model_load_ms = 0.0;
    double vae_encode_ms = 0.0;
    double text_encode_ms = 0.0;
    double conditioning_ms = 0.0;
    double sampling_total_ms = 0.0;
    std::vector<double> sampling_step_ms;
    double vae_decode_ms = 0.0;
    double total_ms = 0.0;
};

/// Inputs of one generation window. Everything is flat row-major; sizes are checked against the manifest.
struct Sa3GenerateInputs {
    std::vector<int64_t> input_ids;        ///< [text.max_length] from `Tokenizer`
    std::vector<int64_t> attention_mask;   ///< [text.max_length]
    /// The arm's control over the window, ungated: [control.dim, window.frames] (frames straight from
    /// `ControlFeatureExtractor`). The pipeline applies `tf_inpaint_mask`.
    std::vector<float> control_frames;
    /// Inpaint context as a latent [latent_dim, window.frames] (the previous window's sampled latent, shifted by
    /// the hop). Empty -> `context_audio` is VAE-encoded; both empty -> zero latent.
    std::vector<float> context_latent;
    std::vector<float> context_audio;      ///< [2, window.samples] (padded/truncated if different)
    float keep_ratio = 0.5f;               ///< keep_frames = int(frames * keep_ratio): inpaint context prefix
    int future_visibility_frames = 0;      ///< tf_keep_frames = clamp(keep_frames + V, 0, frames); trained range in manifest
    uint32_t seed = 0;
    int steps = 8;
    float cfg_scale = 4.0f;
    CfgAxis cfg_axis = CfgAxis::Control;
    std::vector<float> external_noise;     ///< if non-empty, [latent_dim, frames] used instead of the seeded RNG
    std::string dump_steps_dir;            ///< if non-empty, per-step `step_XX_x_after.npy` are written there
    /// If non-null, [text.max_length, text.dim] text embeddings (pad positions already the learned pad vector,
    /// i.e. exactly what `T5Encoder::encode` returns) -> the text encoder is skipped.
    const std::vector<float>* precomputed_text = nullptr;
    Sa3StepCallback step_callback = nullptr;
    Sa3StepTrace step_trace = nullptr;     ///< parity / debugging: per-step x, v_cond, v_uncond, v_cfg
    Sa3PhaseProgressCallback phase_callback = nullptr;
};

/// The sa3-tap pipeline: drum control + inpaint context + prompt -> one stereo window.
///
/// Reads everything from `sa3_pipeline_manifest.json` (`Sa3PipelineConfig`); ONNX Runtime for the NN graphs,
/// C++ for the rest. Same structure as the Zenon pipeline it replaces:
///   1. VAE encode the context audio (only if no context latent was given)
///   2. inpaint_mask (prefix keep_frames) and tf_inpaint_mask (prefix keep_frames + V)
///   3. inpaint_masked_input = context_latent * inpaint_mask
///   4. text encode (T5Gemma, learned pad vector in-graph) unless precomputed
///   5. seconds_total embedding (expo number embedder)
///   6. assemble the conditional branch: cross_attn = [text; seconds], global = seconds,
///      local_add = [inpaint_mask; inpaint_masked_input], control = control_frames * tf, tf
///      and the guided-axis-nulled branch (control -> null_value * tf, or masked input -> 0)
///   7. noise, LogSNR schedule, Euler loop with single-axis APG CFG (2 DiT passes per step)
///   8. VAE decode
class Sa3Pipeline {
public:
    /// Loads all graphs and weights named in `config` (backend flags `use_cuda`/`use_coreml`/`use_migraphx` are
    /// fields of the config, set by the caller, not by the manifest). Throws on any load failure.
    explicit Sa3Pipeline(const Sa3PipelineConfig& config);

    /// Generate one window. Returns stereo audio, flat [2, window.samples].
    std::vector<float> generate(const Sa3GenerateInputs& in);

    /// Run one DiT call and one VAE decode on zeros so the first real `generate()` does not pay the
    /// session warm-up (MIGraphX loads its compiled programs lazily on the first run).
    void warmup();

    /// Text embeddings from the last `generate()` that ran the text encoder ([max_length, dim]).
    const std::vector<float>& last_text_embeddings() const { return m_last_text; }
    /// The final latent of the last `generate()` ([latent_dim, frames]): the next window's context source.
    const std::vector<float>& last_sampled_latent() const { return m_last_sampled; }
    /// The two DiT branches of the last `generate()` (cond, uncond), for parity tests.
    const Sa3BranchInputs& last_cond_branch() const { return m_last_cond; }
    const Sa3BranchInputs& last_uncond_branch() const { return m_last_uncond; }

    const Sa3TimingReport& timing() const { return m_timing; }
    const Sa3Diagnostics& diagnostics() const { return m_diagnostics; }
    const Sa3PipelineConfig& config() const { return m_config; }

    void set_verbose(bool verbose) { m_verbose = verbose; }
    bool verbose() const { return m_verbose; }
    void set_collect_step_diagnostics(bool enabled) { m_collect_step_diagnostics = enabled; }
    bool collect_step_diagnostics() const { return m_collect_step_diagnostics; }

private:
    Sa3DiTModel::Dims dit_dims() const;

    Sa3PipelineConfig m_config;
    std::unique_ptr<T5Encoder> m_text;
    std::unique_ptr<Sa3DiTModel> m_dit;
    std::unique_ptr<IVaeEncoder> m_vae_encoder;
    std::unique_ptr<IVaeDecoder> m_vae_decoder;
    std::unique_ptr<NumberEmbedder> m_number_embedder;
    std::vector<float> m_seconds_embed;    ///< seconds_total is fixed per manifest, so embed once
    Sa3TimingReport m_timing;
    Sa3Diagnostics m_diagnostics;
    std::vector<float> m_last_text;
    std::vector<float> m_last_sampled;
    Sa3BranchInputs m_last_cond;
    Sa3BranchInputs m_last_uncond;
    bool m_verbose = true;
    bool m_collect_step_diagnostics = false;
};

} // namespace sao
