#pragma once

#include <juce_gui_basics/juce_gui_basics.h>
#include <functional>

namespace streamgen {

/// Simple two-fader mixer for sax passthrough and drums output volumes.
class MixerComponent : public juce::Component {
public:
    MixerComponent();

    void resized() override;
    void paint(juce::Graphics& g) override;

    std::function<void(float)> on_sax_gain_changed;
    std::function<void(float)> on_drums_gain_changed;

private:
    juce::Slider m_sax_slider;
    juce::Slider m_drums_slider;
    juce::Label m_sax_label;
    juce::Label m_drums_label;
};

} // namespace streamgen
