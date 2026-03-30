#include "WaveformTimelineComponent.h"

#include <cmath>

namespace streamgen {

WaveformTimelineComponent::WaveformTimelineComponent()
{
    setOpaque(true);
}

void WaveformTimelineComponent::update(
    const std::vector<float>& waveform,
    int64_t absolute_pos,
    int sample_rate)
{
    m_waveform = waveform;
    m_absolute_pos = absolute_pos;
    m_sample_rate = sample_rate;
    repaint();
}

void WaveformTimelineComponent::paint(juce::Graphics& g)
{
    auto bounds = getLocalBounds();
    g.fillAll(juce::Colour(0xff1a1a2e));

    const int w = bounds.getWidth();
    const int h = bounds.getHeight();

    const int label_height = 18;
    const int timeline_height = 16;
    const int waveform_top = label_height;
    const int waveform_height = h - label_height - timeline_height;

    // --- Label ---
    g.setColour(juce::Colour(0xffe0e0e0));
    g.setFont(juce::Font(13.0f));
    g.drawText(m_label, 4, 0, w / 2, label_height, juce::Justification::centredLeft);

    if (m_source_tag.isNotEmpty())
    {
        g.setColour(juce::Colour(0xff00e676));
        g.drawText(m_source_tag, w / 2, 0, w / 2 - 4, label_height, juce::Justification::centredRight);
    }

    // --- Waveform ---
    if (m_waveform.empty() || w <= 0 || waveform_height <= 0)
        return;

    float mid_y = static_cast<float>(waveform_top + waveform_height / 2);
    float amp = static_cast<float>(waveform_height) * 0.45f;

    int total_visible_samples = static_cast<int>(m_visible_seconds * m_sample_rate);
    int num_samples = static_cast<int>(m_waveform.size());
    float samples_per_pixel = static_cast<float>(total_visible_samples) / static_cast<float>(w);

    // Draw center line
    g.setColour(juce::Colour(0xff333355));
    g.drawHorizontalLine(static_cast<int>(mid_y), 0.0f, static_cast<float>(w));

    // Draw waveform
    g.setColour(juce::Colour(0xff4fc3f7));
    juce::Path path;
    bool path_started = false;

    for (int px = 0; px < w; ++px)
    {
        int sample_offset = total_visible_samples - num_samples;
        int sample_idx = static_cast<int>(static_cast<float>(px) * samples_per_pixel) - sample_offset;

        if (sample_idx < 0 || sample_idx >= num_samples)
            continue;

        float val = m_waveform[static_cast<size_t>(sample_idx)];
        val = std::max(-1.0f, std::min(1.0f, val));
        float y = mid_y - val * amp;

        if (!path_started)
        {
            path.startNewSubPath(static_cast<float>(px), y);
            path_started = true;
        }
        else
        {
            path.lineTo(static_cast<float>(px), y);
        }
    }

    if (path_started)
        g.strokePath(path, juce::PathStrokeType(1.0f));

    // --- Now marker ---
    g.setColour(juce::Colour(0xffff5252));
    g.drawVerticalLine(w - 1, static_cast<float>(waveform_top),
                       static_cast<float>(waveform_top + waveform_height));

    // --- Timeline ticks ---
    int timeline_y = h - timeline_height;
    g.setColour(juce::Colour(0xff333355));
    g.fillRect(0, timeline_y, w, timeline_height);

    g.setColour(juce::Colour(0xff888888));
    g.setFont(juce::Font(10.0f));

    double seconds_visible = static_cast<double>(m_visible_seconds);
    double end_seconds = samples_to_seconds(m_absolute_pos, m_sample_rate);
    double start_seconds = end_seconds - seconds_visible;

    double tick_interval = 1.0;
    if (seconds_visible > 30.0) tick_interval = 5.0;
    else if (seconds_visible > 10.0) tick_interval = 2.0;

    double first_tick = std::ceil(start_seconds / tick_interval) * tick_interval;
    for (double t = first_tick; t <= end_seconds; t += tick_interval)
    {
        double frac = (t - start_seconds) / seconds_visible;
        int px = static_cast<int>(frac * w);
        g.drawVerticalLine(px, static_cast<float>(timeline_y),
                           static_cast<float>(timeline_y + 4));

        int total_secs = static_cast<int>(t);
        if (total_secs >= 0)
        {
            int mins = total_secs / 60;
            int secs = total_secs % 60;
            juce::String label = juce::String::formatted("%d:%02d", mins, secs);
            g.drawText(label, px - 20, timeline_y + 3, 40, 12, juce::Justification::centred);
        }
    }
}

} // namespace streamgen
