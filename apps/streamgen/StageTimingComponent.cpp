#include "StageTimingComponent.h"
#include "LayerCakeLookAndFeel.h"

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
    auto& lf = getLookAndFeel();
    const juce::Colour panel = lf.findColour(juce::ComboBox::backgroundColourId);
    const juce::Colour terminal = lf.findColour(juce::Label::textColourId);
    const juce::Colour dim = terminal.withAlpha(0.5f);

    g.fillAll(panel);

    const int w = bounds.getWidth();

    g.setColour(terminal);
    g.setFont(juce::Font(juce::FontOptions(juce::Font::getDefaultMonospacedFontName(), 11.0f, juce::Font::plain)));
    g.drawText("TIMING", 4, 2, w - 8, 14, juce::Justification::centredLeft);

    struct StageInfo {
        const char* name;
        double ms;
        juce::Colour colour;
    };

    juce::Colour c_enc(0xff35c0ff), c_t5(0xff3cff9f), c_dit(0xffff564a), c_dec(0xfff45bff), c_tot(0xfff2b950);
    if (auto* lc = dynamic_cast<LayerCakeLookAndFeel*>(&lf))
    {
        c_enc = lc->getLayerColour(1).brighter(0.4f);
        c_t5 = lc->getLayerColour(4).brighter(0.4f);
        c_dit = lc->getLayerColour(0).brighter(0.4f);
        c_dec = lc->getLayerColour(5).brighter(0.4f);
        c_tot = lc->getLayerColour(2).brighter(0.4f);
    }

    StageInfo stages[] = {
        {"VAE Enc", m_timing.vae_encode_ms, c_enc},
        {"T5",      m_timing.t5_encode_ms,  c_t5},
        {"DiT",     m_timing.sampling_total_ms, c_dit},
        {"VAE Dec", m_timing.vae_decode_ms, c_dec},
        {"Total",   m_timing.total_ms,      c_tot},
    };

    double max_ms = 1.0;
    for (const auto& s : stages)
        max_ms = std::max(max_ms, s.ms);

    const int bar_height = 14;
    const int label_width = 52;
    const int value_width = 52;
    const int bar_area_width = w - label_width - value_width - 12;
    int y = 20;

    g.setFont(juce::Font(juce::FontOptions(juce::Font::getDefaultMonospacedFontName(), 10.0f, juce::Font::plain)));

    for (const auto& s : stages)
    {
        // Label
        g.setColour(dim);
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

        g.setColour(terminal);
        g.drawText(val, label_width + bar_area_width + 8, y, value_width, bar_height,
                   juce::Justification::centredLeft);

        y += bar_height + 2;
    }
}

} // namespace streamgen
