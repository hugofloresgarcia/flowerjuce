#pragma once

#include <juce_gui_basics/juce_gui_basics.h>
#include <functional>

namespace streamgen {

/// Three-fader mixer (sax passthrough, live percussion passthrough, drums output) plus a
/// drums-bus mute toggle.
class MixerComponent : public juce::Component {
public:
    MixerComponent();

    void resized() override;
    void paint(juce::Graphics& g) override;

    std::function<void(float)> on_streamgen_audio_gain_changed;
    std::function<void(float)> on_percussion_gain_changed;
    std::function<void(float)> on_drums_gain_changed;
    std::function<void(bool)> on_drums_mute_changed;

private:
    juce::Slider m_streamgen_audio_slider;
    juce::Slider m_percussion_slider;
    juce::Slider m_drums_slider;
    juce::Label m_streamgen_audio_label;
    juce::Label m_percussion_label;
    juce::Label m_drums_label;
    juce::ToggleButton m_drums_mute_toggle{"Mute"};
};

} // namespace streamgen
