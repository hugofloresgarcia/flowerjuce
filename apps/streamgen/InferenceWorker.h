#pragma once

#include "InferenceSnapshot.h"
#include "TimeRuler.h"
#include "StreamGenProcessor.h"

#include <juce_core/juce_core.h>

#include <sao_inference/ZenonPipeline.h>
#include <sao_inference/ZenonPipelineConfig.h>
#include <sao_inference/Tokenizer.h>

#include <atomic>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace streamgen {

class GenerationTimelineStore;

/// Background thread that runs the ZenonPipeline inference loop.
///
/// Polls the GenerationScheduler for jobs, snapshots audio from the processor's
/// ring buffers, runs the full pipeline (VAE encode -> T5 -> DiT -> VAE decode),
/// and writes the result back with overlap-add crossfade.
///
/// Thread-safety:
///   - Reads from processor (snapshot_streamgen_audio_for_vae / snapshot_input_audio_for_vae)
///   - Writes to processor output ring buffer (write_output)
///   - Reads/writes atomics shared with UI (prompt, timing, status)
class InferenceWorker : public juce::Thread {
public:
    /// Args:
    ///     processor: The audio processor owning ring buffers and scheduler.
    explicit InferenceWorker(StreamGenProcessor& processor);
    ~InferenceWorker() override;

    /// Load the Zenon pipeline from a manifest file and the tokenizer.
    /// Must be called before starting the thread.
    ///
    /// Args:
    ///     manifest_path: Path to zenon_pipeline_manifest.json.
    ///     use_cuda: Whether to use CUDA execution provider.
    ///     use_coreml: Whether to use CoreML execution provider (macOS).
    ///
    /// Args:
    ///     use_mlx_vae: When true, load Zenon VAE from MLX Metal (requires sao_inference built with SAO_ENABLE_MLX).
    ///
    /// Returns:
    ///     true if loading succeeded.
    bool load_pipeline(
        const std::string& manifest_path,
        bool use_cuda,
        bool use_coreml = false,
        bool use_mlx_vae = false);

    /// Set the text prompt. Thread-safe — can be called from UI thread.
    ///
    /// Args:
    ///     prompt: New text prompt for T5 conditioning.
    void set_prompt(const std::string& prompt);

    /// Get the most recent stage timing snapshot. Thread-safe.
    StageTiming last_timing() const;

    /// Full timing + latent/diffusion diagnostics from the last completed job. Thread-safe.
    InferenceSnapshot last_snapshot() const;

    /// When true, `generate()` prints stdout timing (default off for GUI; CLI uses verbose pipeline).
    void set_pipeline_verbose(bool verbose);

    /// Human-readable inference phase for UI: `[idle]`, `[enc]`, `[dit-K]`, `[dec]`.
    ///
    /// Reads atomics updated from the worker thread during `generate()` only.
    juce::String inference_phase_display() const;

    /// Crossfade length in samples for overlap-add writes.
    std::atomic<int> crossfade_samples{4410}; // 100ms @ 44100

    /// Whether the pipeline is loaded and ready.
    bool is_loaded() const { return m_pipeline != nullptr; }

    void run() override;

    /// Process a single generation job synchronously. Public so the CLI can
    /// call it directly instead of using the worker thread loop.
    ///
    /// Args:
    ///     job: The generation job to process.
    void process_job(const GenerationJob& job);

private:
    void update_tokenization();

    StreamGenProcessor& m_processor;
    GenerationTimelineStore* m_timeline = nullptr;

    std::unique_ptr<sao::ZenonPipeline> m_pipeline;
    std::unique_ptr<sao::Tokenizer> m_tokenizer;
    sao::ZenonPipelineConfig m_config;

    // Prompt state (UI writes, worker reads)
    std::mutex m_prompt_mutex;
    std::string m_prompt = "percussion";
    bool m_prompt_dirty = true;

    // Cached tokenization
    std::vector<int64_t> m_input_ids;
    std::vector<int64_t> m_attention_mask;

    /// Masked T5 embeddings (64 * cond_token_dim) when input_ids/mask match m_t5_cache_*.
    std::vector<float> m_cached_t5_masked;
    std::vector<int64_t> m_t5_cache_ids;
    std::vector<int64_t> m_t5_cache_mask;
    bool m_have_t5_cache = false;

    // Timing + snapshot (worker writes, UI reads)
    mutable std::mutex m_worker_state_mutex;
    StageTiming m_last_timing;
    InferenceSnapshot m_last_snapshot;
    bool m_pipeline_verbose = false;

    /// UI phase: 0 idle, 1 enc, 2 dit, 3 dec (matches pipeline coarse phases).
    std::atomic<int> m_inference_phase_kind{0};
    /// 1-based DiT step when `m_inference_phase_kind == 2`.
    std::atomic<int> m_inference_dit_step{0};
};

} // namespace streamgen
