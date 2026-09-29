#pragma once

#include "MusicalTime.h"
#include "TimeRuler.h"
#include <vector>
#include <mutex>

namespace streamgen {

class GenerationTimelineStore;

/// Decides when to trigger new generation jobs and manages the job queue.
///
/// Called from the audio thread (via advance()) and the worker thread (via pop_job()).
/// The audio thread only writes to atomics; the worker thread holds m_queue_mutex
/// briefly to pop jobs.
class GenerationScheduler {
public:
    GenerationScheduler() = default;

    /// Optional timeline for UI overlays (audio thread records scheduled jobs).
    void set_timeline_store(GenerationTimelineStore* store) { m_timeline = store; }

    /// Configure the scheduler with model constants.
    ///
    /// Args:
    ///     constants: Model window and timing parameters from the Zenon config.
    void configure(const ModelConstants& constants);

    /// Seconds of new audio one generation writes: the non-kept tail of the model window,
    /// `window_seconds * (1 - keep_ratio)`. This is the upper bound for `hop_seconds`; a longer
    /// hop leaves gaps between generations.
    ///
    /// Args:
    ///     constants: Model constants; `sample_rate` must be > 0.
    ///     keep_ratio: Kept fraction of the window, in [0, 1).
    /// Returns:
    ///     Generated length in seconds, > 0.
    static double generated_seconds(const ModelConstants& constants, float keep_ratio);

    /// `generated_seconds()` for this scheduler's configured constants and current `keep_ratio`.
    double generated_seconds() const;

    /// Set `hop_seconds` to `generated_seconds()` so each hop advances by exactly the generated
    /// (non-kept) fraction. Used at configure time and by the CLI when no `--hop` is given.
    void sync_hop_seconds_to_keep_ratio();

    /// Lower `hop_seconds` to `generated_seconds()` when it exceeds it; never raises it.
    /// Called when `keep_ratio` changes so a hop the user chose survives unless it would leave gaps.
    void clamp_hop_seconds_to_generated_length();

    /// Length of one launch-quantize step in seconds (`quantize_launch_beats` beats at `bpm`), or 0
    /// when musical time is off or quantize is off. Generations wait for this grid after each hop,
    /// so a step longer than `generated_seconds()` leaves gaps whatever the hop.
    double launch_quantize_step_seconds() const;

    /// Sample rate of the live audio device (callback / ring). If unset (0), hop/delay use model rate.
    void set_playback_sample_rate(int hz);

    /// Advance the absolute sample position. Called from the audio thread every block.
    /// Checks if a new generation should be triggered based on hop interval.
    ///
    /// Args:
    ///     num_samples: Number of samples in this audio block.
    void advance(int num_samples);

    /// Reset transport counters, clear the job queue, and zero UI status. Call only when
    /// the audio callback is not running (no concurrent advance()).
    void reset_session();

    /// Pop the next job from the queue. Called from the worker thread.
    /// If multiple jobs are queued and skip_stale is true, returns only the most recent.
    ///
    /// Args:
    ///     job: Output parameter for the job to process.
    ///
    /// Returns:
    ///     true if a job was available, false if queue is empty.
    bool pop_job(GenerationJob& job);

    /// Current absolute sample position (audio thread writes, others read).
    int64_t absolute_sample_pos() const { return m_absolute_sample_pos.load(std::memory_order_relaxed); }

    /// Device sample rate if set (after audio starts), otherwise model rate.
    int effective_playback_rate_hz() const
    {
        return m_playback_sample_rate > 0 ? m_playback_sample_rate : m_constants.sample_rate;
    }

    // --- Parameters (set from UI thread, read from audio/worker threads) ---
    /// Wall-clock hop (seconds) between generation triggers. Default is overwritten in
    /// `configure()` via `sync_hop_seconds_to_keep_ratio()` (≈ half the ~11.9s window for keep 0.5).
    std::atomic<float> hop_seconds{3.0f};
    std::atomic<float> keep_ratio{0.5f};
    /// Signed offset (in latent frames) for tf_inpaint_mask. 0 = match keep_ratio
    /// (Nithya's reference behaviour). Negative shrinks sax visibility; positive lookahead
    /// is OOD for the current checkpoint and is clamped at 0 by the UI.
    std::atomic<int> future_visibility_frames{0};
    std::atomic<int> steps{8};
    std::atomic<float> cfg_scale{7.0f};
    /// Seconds after keep_end where generated drums land in the timeline (0 = immediate).
    /// Absolute wall-clock: independent of BPM and not snapped to the beat grid.
    std::atomic<float> schedule_delay_seconds{0.0f};
    std::atomic<bool> generation_enabled{false};

    /// When true, launch quantization is active and the ruler/UI show bars/beats.
    /// Hop and land delay are always wall-clock seconds regardless of this flag.
    std::atomic<bool> musical_time_enabled{false};
    /// Beats per minute (clamped on use to [20, 400]).
    std::atomic<float> bpm{kStreamGenDefaultBpm};
    /// Time signature: beats per bar (numerator) and denominator (display / future).
    std::atomic<int> time_sig_numerator{4};
    std::atomic<int> time_sig_denominator{4};
    /// Launch quantization: 0 = off; else enqueue keep_end snaps forward to multiples of N beats.
    std::atomic<int> quantize_launch_beats{0};

    /// Shared generation status for UI readout.
    GenerationStatus status;

private:
    void enqueue_job(int64_t keep_end_sample);

    ModelConstants m_constants;
    int m_playback_sample_rate = 0;
    std::atomic<int64_t> m_absolute_sample_pos{0};
    int64_t m_last_trigger_sample = 0;
    std::atomic<int64_t> m_next_job_id{0};

    /// Quantized enqueue waiting for playhead (audio thread only). -1 = none.
    int64_t m_pending_keep_end_sample = -1;
    float m_pending_snap_bpm = 0.0f;
    int m_pending_snap_quantize = 0;

    std::mutex m_queue_mutex;
    std::vector<GenerationJob> m_queue;

    GenerationTimelineStore* m_timeline = nullptr;
};

} // namespace streamgen
