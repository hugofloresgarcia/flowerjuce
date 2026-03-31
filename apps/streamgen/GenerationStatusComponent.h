#pragma once

#include "TimeRuler.h"

#include <juce_gui_basics/juce_gui_basics.h>

namespace streamgen {

/// Shows generation queue status, generation count, measured latency,
/// and current job info. Updates via polling from the UI timer.
class GenerationStatusComponent : public juce::Component {
public:
    GenerationStatusComponent();

    /// Update the displayed status values.
    ///
    /// Args:
    ///     queue_depth: Number of jobs in the generation queue.
    ///     generation_count: Total number of completed generations.
    ///     last_latency_ms: Latency of the most recent generation in ms.
    ///     last_job_id: ID of the most recent generation job.
    ///     worker_busy: Whether the worker is currently processing a job.
    ///     source_label: Input source description (e.g. "Live Mic", "Simulation (file.wav)").
    ///     last_land_timeline: Optional one-line summary of where/when the last gen landed on the timeline.
    void update(int queue_depth, int64_t generation_count, double last_latency_ms,
                int64_t last_job_id, bool worker_busy, const juce::String& source_label,
                const juce::String& last_land_timeline = {});

    void paint(juce::Graphics& g) override;

private:
    int m_queue_depth = 0;
    int64_t m_generation_count = 0;
    double m_last_latency_ms = 0.0;
    int64_t m_last_job_id = -1;
    bool m_worker_busy = false;
    juce::String m_source_label = "Not started";
    juce::String m_last_land_timeline;
};

} // namespace streamgen
