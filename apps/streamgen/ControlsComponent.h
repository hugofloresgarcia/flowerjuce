#pragma once

#include "GenerationScheduler.h"
#include <juce_gui_basics/juce_gui_basics.h>
#include <functional>

namespace streamgen {

/// Controls panel: prompt text field, hop/keep_ratio/steps/cfg sliders,
/// warm-start load button, simulate button, audio settings button,
/// and generation enable toggle.
class ControlsComponent : public juce::Component {
public:
    ControlsComponent();

    void paint(juce::Graphics& g) override;
    void resized() override;

    // --- Callbacks (set by the parent component) ---
    std::function<void(const juce::String&)> on_prompt_changed;
    std::function<void(float)> on_hop_changed;
    std::function<void(float)> on_hop_beats_changed;
    std::function<void(float)> on_schedule_delay_changed;
    std::function<void(float)> on_schedule_delay_beats_changed;
    std::function<void(float)> on_keep_ratio_changed;
    std::function<void(int)> on_steps_changed;
    std::function<void(float)> on_cfg_changed;
    std::function<void(bool)> on_musical_time_changed;
    std::function<void(float)> on_bpm_changed;
    std::function<void(int, int)> on_time_signature_changed;
    std::function<void(int)> on_quantize_launch_changed;
    std::function<void()> on_warm_start_clicked;
    std::function<void(bool)> on_warm_route_toggled;
    std::function<void()> on_simulate_clicked;
    std::function<void()> on_audio_settings_clicked;
    std::function<void()> on_operator_clicked;
    std::function<void()> on_reset_clicked;
    std::function<void(bool)> on_generation_enabled_changed;

    /// Set the prompt text (e.g. when loading saved state).
    void set_prompt(const juce::String& text) { m_prompt_editor.setText(text, false); }

    /// Sync musical toggle, BPM, sig, quantize, hop, and land delay from scheduler (no callbacks).
    void sync_time_mode_from_scheduler(const GenerationScheduler& sched);

    /// After toggling musical mode, refresh hop/land labels, ranges, and values from scheduler.
    void sync_hop_delay_sliders_from_scheduler(const GenerationScheduler& sched);

    void set_warm_route_toggle(bool route_to_output, juce::NotificationType notification);

    void set_warm_route_enabled(bool enabled);

private:
    static int time_sig_combo_index_for(int numerator, int denominator);

    juce::Label m_prompt_label;
    juce::TextEditor m_prompt_editor;

    juce::ToggleButton m_musical_time_toggle{"Musical time"};

    juce::Label m_bpm_label;
    juce::Slider m_bpm_slider;

    juce::Label m_time_sig_label;
    juce::ComboBox m_time_sig_combo;

    juce::Label m_quantize_label;
    juce::ComboBox m_quantize_combo;

    juce::Label m_hop_label;
    juce::Slider m_hop_slider;

    juce::Label m_keep_ratio_label;
    juce::Slider m_keep_ratio_slider;

    juce::Label m_steps_label;
    juce::Slider m_steps_slider;

    juce::Label m_cfg_label;
    juce::Slider m_cfg_slider;

    juce::Label m_schedule_delay_label;
    juce::Slider m_schedule_delay_slider;

    juce::TextButton m_warm_start_button{"Load Warm Start"};
    juce::ToggleButton m_warm_route_toggle{"Warm out"};
    juce::TextButton m_simulate_button{"Simulate..."};
    juce::TextButton m_audio_settings_button{"Audio Settings"};
    juce::TextButton m_operator_button{"Operator"};
    juce::TextButton m_reset_button{"Reset"};
    juce::ToggleButton m_generation_toggle{"Enable Generation"};
};

} // namespace streamgen
