#pragma once

#include "TimeRuler.h"
#include "StreamGenProcessor.h"

#include <juce_gui_basics/juce_gui_basics.h>
#include <vector>

namespace streamgen {

/// Scrolling waveform display synced to absolute time, with generation window
/// overlay markers showing [WARM] and [GEN #N] regions.
///
/// Renders a mono waveform that scrolls rightward as time advances.
/// Overlays colored blocks for generation windows (dim for kept prefix,
/// bright for generated suffix).
class WaveformTimelineComponent : public juce::Component {
public:
    WaveformTimelineComponent();

    /// Update with fresh waveform data and timeline position.
    ///
    /// Args:
    ///     waveform: Mono waveform samples for the visible time range.
    ///     absolute_pos: Current absolute sample position (rightmost edge).
    ///     sample_rate: Audio sample rate for time axis labels.
    void update(const std::vector<float>& waveform, int64_t absolute_pos, int sample_rate);

    /// Set the label displayed in the top-left corner (e.g. "SAX INPUT", "DRUMS OUTPUT").
    void set_label(const juce::String& label) { m_label = label; }

    /// Set a source tag shown next to the label (e.g. "[SIM]", "[WARM]", "[GEN #5]").
    void set_source_tag(const juce::String& tag) { m_source_tag = tag; }

    /// Visible duration in seconds (how much time the waveform shows).
    void set_visible_duration(float seconds) { m_visible_seconds = seconds; }

    void paint(juce::Graphics& g) override;

private:
    std::vector<float> m_waveform;
    int64_t m_absolute_pos = 0;
    int m_sample_rate = 44100;
    float m_visible_seconds = 15.0f;
    juce::String m_label;
    juce::String m_source_tag;
};

} // namespace streamgen
