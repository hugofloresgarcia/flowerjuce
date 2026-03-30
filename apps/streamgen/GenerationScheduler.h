#pragma once

#include "TimeRuler.h"
#include <vector>
#include <mutex>

namespace streamgen {

/// Decides when to trigger new generation jobs and manages the job queue.
///
/// Called from the audio thread (via advance()) and the worker thread (via pop_job()).
/// The audio thread only writes to atomics; the worker thread holds m_queue_mutex
/// briefly to pop jobs.
class GenerationScheduler {
public:
    GenerationScheduler() = default;

    /// Configure the scheduler with model constants.
    ///
    /// Args:
    ///     constants: Model window and timing parameters from the Zenon config.
    void configure(const ModelConstants& constants);

    /// Advance the absolute sample position. Called from the audio thread every block.
    /// Checks if a new generation should be triggered based on hop interval.
    ///
    /// Args:
    ///     num_samples: Number of samples in this audio block.
    void advance(int num_samples);

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

    // --- Parameters (set from UI thread, read from audio/worker threads) ---
    std::atomic<float> hop_seconds{3.0f};
    std::atomic<float> keep_ratio{0.5f};
    std::atomic<int> steps{8};
    std::atomic<float> cfg_scale{7.0f};
    std::atomic<bool> generation_enabled{false};

    /// Shared generation status for UI readout.
    GenerationStatus status;

private:
    void enqueue_job();

    ModelConstants m_constants;
    std::atomic<int64_t> m_absolute_sample_pos{0};
    int64_t m_last_trigger_sample = 0;
    std::atomic<int64_t> m_next_job_id{0};

    std::mutex m_queue_mutex;
    std::vector<GenerationJob> m_queue;
};

} // namespace streamgen
