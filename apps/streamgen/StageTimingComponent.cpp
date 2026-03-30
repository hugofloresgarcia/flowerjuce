#include "StageTimingComponent.h"

#include <algorithm>
#include <cmath>

namespace streamgen {

StageTimingComponent::StageTimingComponent()
{
    setOpaque(true);
}

void StageTimingComponent::update(const StageTiming& timing)
{
    m_timing = timing;
    repaint();
}

void StageTimingComponent::paint(juce::Graphics& g)
{
    auto bounds = getLocalBounds();
    g.fillAll(juce::Colour(0xff1a1a2e));

    const int w = bounds.getWidth();

    g.setColour(juce::Colour(0xffe0e0e0));
    g.setFont(juce::Font(11.0f));
    g.drawText("TIMING", 4, 2, w - 8, 14, juce::Justification::centredLeft);

    struct StageInfo {
        const char* name;
        double ms;
        juce::Colour colour;
    };

    StageInfo stages[] = {
        {"VAE Enc", m_timing.vae_encode_ms, juce::Colour(0xff42a5f5)},
        {"T5",      m_timing.t5_encode_ms,  juce::Colour(0xff66bb6a)},
        {"DiT",     m_timing.sampling_total_ms, juce::Colour(0xffef5350)},
        {"VAE Dec", m_timing.vae_decode_ms, juce::Colour(0xffab47bc)},
        {"Total",   m_timing.total_ms,      juce::Colour(0xffffa726)},
    };

    double max_ms = 1.0;
    for (const auto& s : stages)
        max_ms = std::max(max_ms, s.ms);

    const int bar_height = 14;
    const int label_width = 52;
    const int value_width = 52;
    const int bar_area_width = w - label_width - value_width - 12;
    int y = 20;

    g.setFont(juce::Font(10.0f));

    for (const auto& s : stages)
    {
        // Label
        g.setColour(juce::Colour(0xffaaaaaa));
        g.drawText(s.name, 4, y, label_width, bar_height, juce::Justification::centredRight);

        // Bar
        float bar_frac = static_cast<float>(s.ms / max_ms);
        int bar_width = static_cast<int>(bar_frac * bar_area_width);
        bar_width = std::max(bar_width, 1);

        g.setColour(s.colour.withAlpha(0.7f));
        g.fillRect(label_width + 6, y + 1, bar_width, bar_height - 2);

        // Value
        juce::String val = juce::String(s.ms, 1) + "ms";
        if (juce::String(s.name) == "DiT" && m_timing.steps > 0)
            val += " (" + juce::String(m_timing.steps) + " steps)";

        g.setColour(juce::Colour(0xffe0e0e0));
        g.drawText(val, label_width + bar_area_width + 8, y, value_width, bar_height,
                   juce::Justification::centredLeft);

        y += bar_height + 2;
    }
}

} // namespace streamgen
