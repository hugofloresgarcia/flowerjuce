#pragma once

#include "TimeRuler.h"

#include <chrono>
#include <cstdint>
#include <mutex>
#include <vector>

namespace streamgen {

/// Fraction of the visible timeline width that lies to the **left** of live playhead
/// (past). Playhead is drawn at this horizontal position; the remainder shows future time.
inline constexpr float k_timeline_playhead_past_fraction = 0.75f;

/// One generation job as recorded for timeline UI (scheduled and optionally completed).
struct JobTimelineRecord {
    int64_t job_id = -1;
    GenerationJob job;
    int64_t scheduled_steady_ns = 0;
    int64_t scheduled_system_ms = 0;
    bool has_completed = false;
    int64_t completed_steady_ns = 0;
    int64_t completed_system_ms = 0;
    double inference_ms = 0.0;
    int64_t gen_samples = 0;
};

/// Thread-safe append-only history of scheduled/completed jobs for waveform overlays.
///
/// Audio thread calls record_scheduled; worker calls record_completed; UI calls
/// snapshot_intersecting.
class GenerationTimelineStore {
public:
    GenerationTimelineStore();

    /// Called from the audio thread when a job is enqueued.
    void record_scheduled(const GenerationJob& job);

    /// Called from the worker thread after output is written.
    void record_completed(int64_t job_id, const GenerationJob& job, double inference_ms, int64_t gen_samples);

    /// Returns jobs that overlap the visible sample window:
    /// [absolute_pos - past_samples, absolute_pos + future_samples] with past/future split
    /// given by k_timeline_playhead_past_fraction.
    /// Prunes entries that ended before the visible window (with margin).
    std::vector<JobTimelineRecord> snapshot_intersecting(
        int64_t absolute_pos,
        int sample_rate,
        float visible_seconds);

    /// Remove all job records (e.g. session reset). Thread-safe.
    void clear();

private:
    static int64_t steady_now_ns();
    static int64_t system_now_ms();
    static int64_t latest_relevant_sample(const JobTimelineRecord& r);
    void prune_locked(int64_t min_sample_to_keep);

    mutable std::mutex m_mutex;
    std::vector<JobTimelineRecord> m_entries;
    static constexpr size_t k_max_entries = 64;
};

} // namespace streamgen
