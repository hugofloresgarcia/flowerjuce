#pragma once

#include "GenerationScheduler.h"
#include "LayerCakeLookAndFeel.h"
#include <juce_gui_basics/juce_gui_basics.h>
#include <functional>

namespace streamgen {

class StreamGenProcessor;

/// Controls panel grouped into Prompt, Musical time & grid, Inference, and Session / I/O.
class ControlsComponent : public juce::Component {
public:
    ControlsComponent();
    ~ControlsComponent() override;

    void paint(juce::Graphics& g) override;
    void resized() override;

    // --- Callbacks (set by the parent component) ---
    std::function<void(const juce::String&)> on_prompt_changed;
    std::function<void(float)> on_hop_changed;
    std::function<void(float)> on_schedule_delay_changed;
    std::function<void(float)> on_keep_ratio_changed;
    /// Fires `tf_inpaint_mask` offset in latent frames (signed; UI clamps to [-N, 0]).
    std::function<void(int)> on_future_visibility_changed;
    std::function<void(int)> on_steps_changed;
    std::function<void(float)> on_cfg_changed;
    std::function<void(bool)> on_musical_time_changed;
    std::function<void(float)> on_bpm_changed;
    std::function<void(int, int)> on_time_signature_changed;
    std::function<void(int)> on_quantize_launch_changed;
    std::function<void(bool)> on_loop_last_generation_changed;
    std::function<void()> on_warmup_audio_clicked;
    std::function<void(bool)> on_warmup_audio_route_toggled;
    std::function<void()> on_simulate_clicked;
    std::function<void()> on_audio_settings_clicked;
    std::function<void()> on_reset_clicked;
    /// Fires the index (into the list passed to `set_model_choices`) of the picked manifest.
    std::function<void(int)> on_model_changed;
    std::function<void(bool)> on_generation_enabled_changed;
    std::function<void(bool)> on_click_track_enabled_changed;
    std::function<void(float)> on_click_track_volume_changed;

    /// Set the prompt text (e.g. when loading saved state).
    void set_prompt(const juce::String& text) { m_prompt_editor.setText(text, false); }

    /// Current prompt text. Used to re-apply the prompt to a freshly loaded pipeline.
    juce::String prompt_text() const { return m_prompt_editor.getText(); }

    /// Fill the model picker with the available manifests. Fires no callback.
    ///
    /// Args:
    ///     names: Display name per manifest, in the same order as the caller's manifest list.
    ///     selected_index: Index of the currently loaded manifest; must be within `names`.
    void set_model_choices(const juce::StringArray& names, int selected_index);

    /// Sync musical toggle, BPM, sig, quantize, hop, land delay, and loop-last toggle (no callbacks).
    void sync_time_mode_from_scheduler(const GenerationScheduler& sched, bool loop_last_generation_enabled);

    /// Set loop-last-gen toggle from code (e.g. auto-off when generation disabled).
    void set_loop_last_generation_toggle(bool on, juce::NotificationType notification);

    /// Refresh the hop and land-delay sliders (both in seconds) from the scheduler. No callbacks.
    void sync_hop_delay_controls_from_scheduler(const GenerationScheduler& sched);

    /// Supply the loaded model's timing constants so the Lookahead slider can convert
    /// seconds to latent frames. Call once the manifest has been loaded; until then the
    /// slider uses `ModelConstants` defaults.
    ///
    /// Args:
    ///     constants: Sample rate and downsampling ratio from the active pipeline manifest.
    void set_model_constants(const ModelConstants& constants);

    void set_warmup_audio_route_toggle(bool route_to_output, juce::NotificationType notification);

    void set_warmup_audio_route_enabled(bool enabled);

    /// Sync click metronome toggle and volume from processor atomics (no callbacks).
    void sync_click_track_from_processor(const StreamGenProcessor& processor);

private:
    /// Draws the model picker (box text and popup items) larger than the rest of the panel,
    /// which keeps the LayerCake defaults.
    class ModelComboLookAndFeel : public LayerCakeLookAndFeel {
    public:
        juce::Font getComboBoxFont(juce::ComboBox&) override;
        juce::Font getPopupMenuFont() override;
    };

    static int time_sig_combo_index_for(int numerator, int denominator);
    void layout_prompt_block(juce::Rectangle<int> inner);
    void layout_musical_rows(juce::Rectangle<int> inner);
    void layout_inference_rows(juce::Rectangle<int> inner);
    void layout_session_rows(juce::Rectangle<int> inner);
    void refresh_click_track_control_enabled_state();

    /// Seconds per latent frame for the Lookahead slider, from the loaded manifest.
    double lookahead_frames_per_second() const;

    juce::GroupComponent m_group_prompt{"Prompt"};
    juce::Label m_prompt_label;
    juce::TextEditor m_prompt_editor;

    juce::GroupComponent m_group_musical{"Musical time & grid"};
    juce::ToggleButton m_musical_time_toggle{"Musical time"};

    juce::Label m_bpm_label;
    juce::Slider m_bpm_slider;

    juce::Label m_time_sig_label;
    juce::ComboBox m_time_sig_combo;

    juce::Label m_quantize_label;
    juce::ComboBox m_quantize_combo;

    juce::Label m_hop_label;
    juce::Slider m_hop_slider;

    juce::Label m_schedule_delay_label;
    juce::Slider m_schedule_delay_slider;

    juce::ToggleButton m_click_track_toggle{"Click track"};
    juce::Label m_click_vol_label;
    juce::Slider m_click_volume_slider;

    juce::GroupComponent m_group_inference{"Inference"};
    juce::Label m_keep_ratio_label;
    juce::Slider m_keep_ratio_slider;

    juce::Label m_future_visibility_label;
    juce::Slider m_future_visibility_slider;

    juce::Label m_steps_label;
    juce::Slider m_steps_slider;

    juce::Label m_cfg_label;
    juce::Slider m_cfg_slider;

    juce::GroupComponent m_group_session{"Session & I/O"};
    juce::ToggleButton m_generation_toggle{"Enable Generation"};
    juce::ToggleButton m_loop_last_gen_toggle{"Loop last gen"};
    juce::TextButton m_warmup_audio_button{"Load Warmup Audio"};
    juce::ToggleButton m_warmup_audio_route_toggle{"Warmup out"};
    juce::TextButton m_simulate_button{"Simulate..."};
    juce::TextButton m_audio_settings_button{"Audio Settings"};
    juce::TextButton m_reset_button{"Reset"};
    /// Declared before `m_model_combo` so it is destroyed after the combo that points at it.
    ModelComboLookAndFeel m_model_combo_look_and_feel;
    juce::Label m_model_label;
    juce::ComboBox m_model_combo;

    /// Timing constants backing the Lookahead seconds <-> latent-frame conversion.
    ModelConstants m_model_constants;
};

} // namespace streamgen
