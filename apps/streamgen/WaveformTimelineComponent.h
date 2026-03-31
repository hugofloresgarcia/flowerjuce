#pragma once

#include "GenerationTimelineStore.h"
#include "TimeRuler.h"

#include <juce_gui_basics/juce_gui_basics.h>
#include <vector>

namespace streamgen {

/// Which lane to draw for paired sax/drums job coloring.
enum class TimelineWaveRole {
    SaxInput,
    DrumsOutput
};

/// Scrolling waveform display synced to absolute time, with generation window
/// overlay markers showing scheduled jobs and completed regions.
class WaveformTimelineComponent : public juce::Component,
                                  public juce::SettableTooltipClient {
public:
    WaveformTimelineComponent();

    /// Update with per-pixel min/max buckets (one column per logical pixel) and timeline position.
    ///
    /// Args:
    ///     min_per_px: Minimum sample in each horizontal column (length num_px).
    ///     max_per_px: Maximum sample in each horizontal column (length num_px).
    ///     num_px: Number of horizontal buckets (may be less than getWidth(); paint stretches to full width).
    ///     absolute_pos: Live playhead sample index (maps to k_timeline_playhead_past_fraction across width).
    ///     sample_rate: Audio sample rate for time axis labels.
    ///     timeline_jobs: Optional job history for overlays (same pointer may be null).
    ///     role: Lane-specific labels only; both lanes use the same window fill + keep ticks behind the waveform.
    ///     drums_warm_min etc.: When all six optional pointers are non-null and role is DrumsOutput, draw
    ///     warmup (amber), model from ring (cyan), loop-hold (magenta).
    void update(
        const float* min_per_px,
        const float* max_per_px,
        int num_px,
        int64_t absolute_pos,
        int sample_rate,
        const std::vector<JobTimelineRecord>* timeline_jobs,
        TimelineWaveRole role,
        const float* drums_warm_min = nullptr,
        const float* drums_warm_max = nullptr,
        const float* drums_gen_min = nullptr,
        const float* drums_gen_max = nullptr,
        const float* drums_hold_min = nullptr,
        const float* drums_hold_max = nullptr);

    /// Set the label displayed in the top-left corner (e.g. "SAX INPUT", "DRUMS OUTPUT").
    void set_label(const juce::String& label) { m_label = label; }

    /// Set a source tag shown next to the label (e.g. "[SIM]", "[WARMUP]", "[GEN #5]").
    void set_source_tag(const juce::String& tag) { m_source_tag = tag; }

    /// Visible duration in seconds (how much time the waveform shows).
    void set_visible_duration(float seconds) { m_visible_seconds = seconds; }

    /// Ruler labels: wall-clock (MM:SS) vs bars/beats from session start.
    void set_time_axis_for_paint(bool musical, float bpm, int beats_per_bar, int time_sig_denominator);

    void paint(juce::Graphics& g) override;
    void mouseMove(const juce::MouseEvent& event) override;
    void mouseExit(const juce::MouseEvent& event) override;

private:
    juce::String tooltip_for_timeline_x(int x) const;
    std::vector<float> m_min_px;
    std::vector<float> m_max_px;
    int64_t m_absolute_pos = 0;
    int m_sample_rate = 44100;
    float m_visible_seconds = 30.0f;
    juce::String m_label;
    juce::String m_source_tag;

    const std::vector<JobTimelineRecord>* m_timeline_jobs = nullptr;
    TimelineWaveRole m_timeline_role = TimelineWaveRole::SaxInput;

    std::vector<float> m_drums_warm_min_px;
    std::vector<float> m_drums_warm_max_px;
    std::vector<float> m_drums_gen_min_px;
    std::vector<float> m_drums_gen_max_px;
    std::vector<float> m_drums_hold_min_px;
    std::vector<float> m_drums_hold_max_px;
    bool m_drums_source_split = false;

    bool m_paint_musical = false;
    float m_paint_bpm = 120.0f;
    int m_paint_beats_per_bar = 4;
    int m_paint_time_sig_d = 4;

    /// Smoothed vertical normalization (see paint); avoids frame-to-frame scale pumping.
    float m_waveform_display_gain = 1.0f;
};

} // namespace streamgen
