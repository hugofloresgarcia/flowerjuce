#pragma once

#include <juce_gui_basics/juce_gui_basics.h>
#include <functional>

namespace streamgen {

/// Controls panel: prompt text field, hop/keep_ratio/steps/cfg sliders,
/// warm-start load button, simulate button, audio settings button,
/// and generation enable toggle.
class ControlsComponent : public juce::Component {
public:
    ControlsComponent();

    void resized() override;

    // --- Callbacks (set by the parent component) ---
    std::function<void(const juce::String&)> on_prompt_changed;
    std::function<void(float)> on_hop_changed;
    std::function<void(float)> on_keep_ratio_changed;
    std::function<void(int)> on_steps_changed;
    std::function<void(float)> on_cfg_changed;
    std::function<void()> on_warm_start_clicked;
    std::function<void()> on_simulate_clicked;
    std::function<void()> on_audio_settings_clicked;
    std::function<void(bool)> on_generation_enabled_changed;

    /// Set the prompt text (e.g. when loading saved state).
    void set_prompt(const juce::String& text) { m_prompt_editor.setText(text, false); }

private:
    juce::Label m_prompt_label;
    juce::TextEditor m_prompt_editor;

    juce::Label m_hop_label;
    juce::Slider m_hop_slider;

    juce::Label m_keep_ratio_label;
    juce::Slider m_keep_ratio_slider;

    juce::Label m_steps_label;
    juce::Slider m_steps_slider;

    juce::Label m_cfg_label;
    juce::Slider m_cfg_slider;

    juce::TextButton m_warm_start_button{"Load Warm Start"};
    juce::TextButton m_simulate_button{"Simulate..."};
    juce::TextButton m_audio_settings_button{"Audio Settings"};
    juce::ToggleButton m_generation_toggle{"Enable Generation"};
};

} // namespace streamgen
