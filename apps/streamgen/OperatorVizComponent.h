#pragma once

#include "StreamGenProcessor.h"

#include <array>
#include <juce_gui_basics/juce_gui_basics.h>

namespace streamgen {

class InferenceWorker;

/// Live “bridge” visualization: level meters, status lamps, pipeline strip, RT gauge.
/// Updated from the message thread only (`updateFrom` then `repaint`).
class OperatorVizComponent : public juce::Component {
public:
    OperatorVizComponent();

    /// Pulls latest telemetry and caches it for `paint`.
    void update_from(StreamGenProcessor& processor, InferenceWorker* worker);

    void paint(juce::Graphics& g) override;

private:
    float m_input_level = 0.0f;
    float m_output_level = 0.0f;

    bool m_audio_alive = false;
    bool m_pipeline_loaded = false;
    bool m_generation_enabled = false;
    bool m_worker_busy = false;
    bool m_simulation_playing = false;

    float m_wall_ms = 0.0f;
    float m_rt_factor = 0.0f;
    int m_gen_count = 0;
    int m_queue_depth = 0;
    int64_t m_last_job_id = -1;

    std::array<float, 5> m_stage_ms{};
    float m_stage_total_ms = 1.0f;

    std::array<float, 64> m_step_dt_ms{};
    int m_step_count = 0;

    double m_pulse = 0.0;
};

} // namespace streamgen
