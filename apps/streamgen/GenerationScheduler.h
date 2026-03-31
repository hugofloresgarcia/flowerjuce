#pragma once

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
    std::atomic<float> hop_seconds{3.0f};
    std::atomic<float> keep_ratio{0.5f};
    std::atomic<int> steps{8};
    std::atomic<float> cfg_scale{7.0f};
    /// Seconds after keep_end where generated drums land in the timeline (0 = immediate).
    std::atomic<float> schedule_delay_seconds{0.0f};
    std::atomic<bool> generation_enabled{false};

    /// When true, hop and schedule delay use beats + BPM; ruler/UI show bars/beats.
    std::atomic<bool> musical_time_enabled{false};
    /// Beats per minute (clamped on use to [20, 400]).
    std::atomic<float> bpm{120.0f};
    /// Time signature: beats per bar (numerator) and denominator (display / future).
    std::atomic<int> time_sig_numerator{4};
    std::atomic<int> time_sig_denominator{4};
    /// Hop interval in quarter-note beats (legacy mirror of hop_bars * time_sig_numerator; UI updates both).
    std::atomic<float> hop_beats{4.0f};
    /// Land delay in beats (legacy mirror of schedule_delay_bars * time_sig_numerator).
    std::atomic<float> schedule_delay_beats{0.0f};
    /// Hop length in **bars** (musical mode). Allowed: 0.5, 1, 2 (fraction of a bar with current numerator).
    std::atomic<float> hop_bars{1.0f};
    /// Output land delay in **whole bars** after keep_end (musical mode). Allowed: 0, 1, 2.
    std::atomic<float> schedule_delay_bars{0.0f};
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
