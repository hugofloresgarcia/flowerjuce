#include "ControlsComponent.h"
#include "LayerCakeLookAndFeel.h"

namespace streamgen {

namespace {

struct TimeSigPreset {
    int n;
    int d;
    const char* label;
};

constexpr TimeSigPreset k_time_sig_presets[] = {
    {4, 4, "4/4"},
    {3, 4, "3/4"},
    {2, 4, "2/4"},
    {6, 8, "6/8"},
    {5, 4, "5/4"},
    {7, 8, "7/8"},
};

constexpr int k_time_sig_preset_count = static_cast<int>(sizeof(k_time_sig_presets) / sizeof(k_time_sig_presets[0]));

constexpr int k_quantize_item_ids[] = {0, 1, 2, 4, 8};
constexpr int k_quantize_item_count = static_cast<int>(sizeof(k_quantize_item_ids) / sizeof(k_quantize_item_ids[0]));

} // namespace

ControlsComponent::ControlsComponent()
{
    setOpaque(true);

    m_prompt_label.setText("Prompt:", juce::dontSendNotification);
    addAndMakeVisible(m_prompt_label);

    m_prompt_editor.setMultiLine(false);
    m_prompt_editor.setText("percussion");
    m_prompt_editor.onReturnKey = [this]()
    {
        if (on_prompt_changed)
            on_prompt_changed(m_prompt_editor.getText());
    };
    m_prompt_editor.onFocusLost = [this]()
    {
        if (on_prompt_changed)
            on_prompt_changed(m_prompt_editor.getText());
    };
    addAndMakeVisible(m_prompt_editor);

    m_musical_time_toggle.onClick = [this]()
    {
        if (on_musical_time_changed)
            on_musical_time_changed(m_musical_time_toggle.getToggleState());
    };
    addAndMakeVisible(m_musical_time_toggle);

    m_bpm_label.setText("BPM", juce::dontSendNotification);
    addAndMakeVisible(m_bpm_label);

    m_bpm_slider.setRange(20.0, 400.0, 0.5);
    m_bpm_slider.setValue(120.0);
    m_bpm_slider.setTextBoxStyle(juce::Slider::TextBoxRight, false, 52, 20);
    m_bpm_slider.onValueChange = [this]()
    {
        if (on_bpm_changed)
            on_bpm_changed(static_cast<float>(m_bpm_slider.getValue()));
    };
    addAndMakeVisible(m_bpm_slider);

    m_time_sig_label.setText("Sig", juce::dontSendNotification);
    addAndMakeVisible(m_time_sig_label);

    for (int i = 0; i < k_time_sig_preset_count; ++i)
        m_time_sig_combo.addItem(k_time_sig_presets[i].label, i + 1);
    m_time_sig_combo.setSelectedItemIndex(0, juce::dontSendNotification);
    m_time_sig_combo.onChange = [this]()
    {
        const int idx = m_time_sig_combo.getSelectedItemIndex();
        if (idx < 0 || idx >= k_time_sig_preset_count)
            return;
        const auto& pre = k_time_sig_presets[idx];
        if (on_time_signature_changed)
            on_time_signature_changed(pre.n, pre.d);
    };
    addAndMakeVisible(m_time_sig_combo);

    m_quantize_label.setText("Launch Q", juce::dontSendNotification);
    addAndMakeVisible(m_quantize_label);

    m_quantize_combo.addItem("Off", 1);
    m_quantize_combo.addItem("1 beat", 2);
    m_quantize_combo.addItem("2 beats", 3);
    m_quantize_combo.addItem("4 beats", 4);
    m_quantize_combo.addItem("8 beats", 5);
    m_quantize_combo.setSelectedId(1, juce::dontSendNotification);
    m_quantize_combo.onChange = [this]()
    {
        const int idx = m_quantize_combo.getSelectedItemIndex();
        if (idx < 0 || idx >= k_quantize_item_count)
            return;
        if (on_quantize_launch_changed)
            on_quantize_launch_changed(k_quantize_item_ids[idx]);
    };
    addAndMakeVisible(m_quantize_combo);

    m_hop_label.setText("Hop (s):", juce::dontSendNotification);
    addAndMakeVisible(m_hop_label);

    m_hop_slider.setRange(0.5, 10.0, 0.1);
    m_hop_slider.setValue(3.0);
    m_hop_slider.setTextBoxStyle(juce::Slider::TextBoxRight, false, 50, 20);
    m_hop_slider.onValueChange = [this]()
    {
        const float v = static_cast<float>(m_hop_slider.getValue());
        if (m_musical_time_toggle.getToggleState())
        {
            if (on_hop_beats_changed)
                on_hop_beats_changed(v);
        }
        else
        {
            if (on_hop_changed)
                on_hop_changed(v);
        }
    };
    addAndMakeVisible(m_hop_slider);

    m_keep_ratio_label.setText("Keep:", juce::dontSendNotification);
    addAndMakeVisible(m_keep_ratio_label);

    m_keep_ratio_slider.setRange(0.0, 0.9, 0.05);
    m_keep_ratio_slider.setValue(0.5);
    m_keep_ratio_slider.setTextBoxStyle(juce::Slider::TextBoxRight, false, 50, 20);
    m_keep_ratio_slider.onValueChange = [this]()
    {
        if (on_keep_ratio_changed)
            on_keep_ratio_changed(static_cast<float>(m_keep_ratio_slider.getValue()));
    };
    addAndMakeVisible(m_keep_ratio_slider);

    m_steps_label.setText("Steps:", juce::dontSendNotification);
    addAndMakeVisible(m_steps_label);

    m_steps_slider.setRange(1.0, 50.0, 1.0);
    m_steps_slider.setValue(8.0);
    m_steps_slider.setNumDecimalPlacesToDisplay(0);
    m_steps_slider.setSliderSnapsToMousePosition(true);
    m_steps_slider.setVelocityBasedMode(false);
    m_steps_slider.setTextBoxStyle(juce::Slider::TextBoxRight, false, 52, 20);
    m_steps_slider.onValueChange = [this]()
    {
        if (on_steps_changed)
            on_steps_changed(static_cast<int>(m_steps_slider.getValue()));
    };
    addAndMakeVisible(m_steps_slider);

    m_cfg_label.setText("CFG:", juce::dontSendNotification);
    addAndMakeVisible(m_cfg_label);

    m_cfg_slider.setRange(0.0, 15.0, 0.5);
    m_cfg_slider.setValue(7.0);
    m_cfg_slider.setTextBoxStyle(juce::Slider::TextBoxRight, false, 40, 20);
    m_cfg_slider.onValueChange = [this]()
    {
        if (on_cfg_changed)
            on_cfg_changed(static_cast<float>(m_cfg_slider.getValue()));
    };
    addAndMakeVisible(m_cfg_slider);

    m_schedule_delay_label.setText("Land delay (s):", juce::dontSendNotification);
    addAndMakeVisible(m_schedule_delay_label);

    m_schedule_delay_slider.setRange(0.0, 30.0, 0.25);
    m_schedule_delay_slider.setValue(0.0);
    m_schedule_delay_slider.setTextBoxStyle(juce::Slider::TextBoxRight, false, 52, 20);
    m_schedule_delay_slider.onValueChange = [this]()
    {
        const float v = static_cast<float>(m_schedule_delay_slider.getValue());
        if (m_musical_time_toggle.getToggleState())
        {
            if (on_schedule_delay_beats_changed)
                on_schedule_delay_beats_changed(v);
        }
        else
        {
            if (on_schedule_delay_changed)
                on_schedule_delay_changed(v);
        }
    };
    addAndMakeVisible(m_schedule_delay_slider);

    LayerCakeLookAndFeel::setControlButtonType(m_warm_start_button, LayerCakeLookAndFeel::ControlButtonType::Preset);
    m_warm_start_button.onClick = [this]() { if (on_warm_start_clicked) on_warm_start_clicked(); };
    addAndMakeVisible(m_warm_start_button);

    m_warm_route_toggle.onClick = [this]()
    {
        if (on_warm_route_toggled)
            on_warm_route_toggled(m_warm_route_toggle.getToggleState());
    };
    addAndMakeVisible(m_warm_route_toggle);

    LayerCakeLookAndFeel::setControlButtonType(m_simulate_button, LayerCakeLookAndFeel::ControlButtonType::Trigger);
    m_simulate_button.onClick = [this]() { if (on_simulate_clicked) on_simulate_clicked(); };
    addAndMakeVisible(m_simulate_button);

    LayerCakeLookAndFeel::setControlButtonType(m_audio_settings_button, LayerCakeLookAndFeel::ControlButtonType::Clock);
    m_audio_settings_button.onClick = [this]() { if (on_audio_settings_clicked) on_audio_settings_clicked(); };
    addAndMakeVisible(m_audio_settings_button);

    LayerCakeLookAndFeel::setControlButtonType(m_operator_button, LayerCakeLookAndFeel::ControlButtonType::Pattern);
    m_operator_button.onClick = [this]() { if (on_operator_clicked) on_operator_clicked(); };
    addAndMakeVisible(m_operator_button);

    LayerCakeLookAndFeel::setControlButtonType(m_reset_button, LayerCakeLookAndFeel::ControlButtonType::Record);
    m_reset_button.onClick = [this]() { if (on_reset_clicked) on_reset_clicked(); };
    addAndMakeVisible(m_reset_button);

    m_generation_toggle.onClick = [this]()
    {
        if (on_generation_enabled_changed)
            on_generation_enabled_changed(m_generation_toggle.getToggleState());
    };
    addAndMakeVisible(m_generation_toggle);
}

void ControlsComponent::paint(juce::Graphics& g)
{
    g.fillAll(getLookAndFeel().findColour(juce::ComboBox::backgroundColourId));
    g.setColour(getLookAndFeel().findColour(juce::ComboBox::outlineColourId).withAlpha(0.45f));
    g.drawHorizontalLine(0, 0.0f, static_cast<float>(getWidth()));
}

void ControlsComponent::sync_time_mode_from_scheduler(const GenerationScheduler& sched)
{
    const bool musical = sched.musical_time_enabled.load(std::memory_order_relaxed);
    m_musical_time_toggle.setToggleState(musical, juce::dontSendNotification);

    m_bpm_slider.setValue(static_cast<double>(sched.bpm.load(std::memory_order_relaxed)), juce::dontSendNotification);

    const int sig_n = sched.time_sig_numerator.load(std::memory_order_relaxed);
    const int sig_d = sched.time_sig_denominator.load(std::memory_order_relaxed);
    const int sig_idx = time_sig_combo_index_for(sig_n, sig_d);
    m_time_sig_combo.setSelectedItemIndex(sig_idx, juce::dontSendNotification);

    const int q = sched.quantize_launch_beats.load(std::memory_order_relaxed);
    int q_combo_idx = 0;
    for (int i = 0; i < k_quantize_item_count; ++i)
    {
        if (k_quantize_item_ids[i] == q)
        {
            q_combo_idx = i;
            break;
        }
    }
    m_quantize_combo.setSelectedItemIndex(q_combo_idx, juce::dontSendNotification);

    sync_hop_delay_sliders_from_scheduler(sched);
}

void ControlsComponent::sync_hop_delay_sliders_from_scheduler(const GenerationScheduler& sched)
{
    const bool musical = sched.musical_time_enabled.load(std::memory_order_relaxed);
    if (musical)
    {
        m_hop_label.setText("Hop (beats):", juce::dontSendNotification);
        m_hop_slider.setRange(0.25, 32.0, 0.25);
        m_hop_slider.setValue(static_cast<double>(sched.hop_beats.load(std::memory_order_relaxed)),
                              juce::dontSendNotification);
        m_schedule_delay_label.setText("Land delay (beats):", juce::dontSendNotification);
        m_schedule_delay_slider.setRange(0.0, 16.0, 0.25);
        m_schedule_delay_slider.setValue(
            static_cast<double>(sched.schedule_delay_beats.load(std::memory_order_relaxed)),
            juce::dontSendNotification);
    }
    else
    {
        m_hop_label.setText("Hop (s):", juce::dontSendNotification);
        m_hop_slider.setRange(0.5, 10.0, 0.1);
        m_hop_slider.setValue(static_cast<double>(sched.hop_seconds.load(std::memory_order_relaxed)),
                              juce::dontSendNotification);
        m_schedule_delay_label.setText("Land delay (s):", juce::dontSendNotification);
        m_schedule_delay_slider.setRange(0.0, 30.0, 0.25);
        m_schedule_delay_slider.setValue(
            static_cast<double>(sched.schedule_delay_seconds.load(std::memory_order_relaxed)),
            juce::dontSendNotification);
    }
}

int ControlsComponent::time_sig_combo_index_for(int numerator, int denominator)
{
    for (int i = 0; i < k_time_sig_preset_count; ++i)
    {
        if (k_time_sig_presets[i].n == numerator && k_time_sig_presets[i].d == denominator)
            return i;
    }
    return 0;
}

void ControlsComponent::set_warm_route_toggle(bool route_to_output, juce::NotificationType notification)
{
    m_warm_route_toggle.setToggleState(route_to_output, notification);
}

void ControlsComponent::set_warm_route_enabled(bool enabled)
{
    m_warm_route_toggle.setEnabled(enabled);
}

void ControlsComponent::resized()
{
    auto bounds = getLocalBounds().reduced(4);

    const int row_height = 26;
    const int label_width = 60;
    const int button_width = 120;
    const int spacing = 4;
    const int musical_row_label_w = 88;
    const int bpm_label_w = 36;
    const int bpm_slider_w = 108;
    const int sig_label_w = 28;
    const int sig_combo_w = 72;
    const int quant_label_w = 56;
    const int quant_combo_w = 96;
    const int schedule_label_w = 132;

    // Row 1: Prompt (full width)
    auto row = bounds.removeFromTop(row_height);
    m_prompt_label.setBounds(row.removeFromLeft(label_width));
    m_prompt_editor.setBounds(row);
    bounds.removeFromTop(spacing);

    // Row 2: Musical time + BPM + sig + quantize
    row = bounds.removeFromTop(row_height);
    m_musical_time_toggle.setBounds(row.removeFromLeft(musical_row_label_w));
    row.removeFromLeft(spacing);
    m_bpm_label.setBounds(row.removeFromLeft(bpm_label_w));
    m_bpm_slider.setBounds(row.removeFromLeft(bpm_slider_w));
    row.removeFromLeft(spacing);
    m_time_sig_label.setBounds(row.removeFromLeft(sig_label_w));
    m_time_sig_combo.setBounds(row.removeFromLeft(sig_combo_w));
    row.removeFromLeft(spacing);
    m_quantize_label.setBounds(row.removeFromLeft(quant_label_w));
    m_quantize_combo.setBounds(row.removeFromLeft(quant_combo_w));
    bounds.removeFromTop(spacing);

    // Row 3: Hop + Keep ratio
    row = bounds.removeFromTop(row_height);
    auto left_half = row.removeFromLeft(row.getWidth() / 2 - spacing / 2);
    auto right_half = row;

    m_hop_label.setBounds(left_half.removeFromLeft(label_width));
    m_hop_slider.setBounds(left_half);

    m_keep_ratio_label.setBounds(right_half.removeFromLeft(label_width));
    m_keep_ratio_slider.setBounds(right_half);
    bounds.removeFromTop(spacing);

    // Row 4: Steps + CFG
    row = bounds.removeFromTop(row_height);
    left_half = row.removeFromLeft(row.getWidth() / 2 - spacing / 2);
    right_half = row;

    m_steps_label.setBounds(left_half.removeFromLeft(label_width));
    m_steps_slider.setBounds(left_half);

    m_cfg_label.setBounds(right_half.removeFromLeft(label_width));
    m_cfg_slider.setBounds(right_half);
    bounds.removeFromTop(spacing);

    row = bounds.removeFromTop(row_height);
    m_schedule_delay_label.setBounds(row.removeFromLeft(schedule_label_w));
    m_schedule_delay_slider.setBounds(row);
    bounds.removeFromTop(spacing);

    // Row: Buttons
    row = bounds.removeFromTop(row_height);
    m_generation_toggle.setBounds(row.removeFromLeft(160));
    row.removeFromLeft(spacing);
    m_warm_start_button.setBounds(row.removeFromLeft(button_width));
    row.removeFromLeft(spacing);
    m_warm_route_toggle.setBounds(row.removeFromLeft(100));
    row.removeFromLeft(spacing);
    m_simulate_button.setBounds(row.removeFromLeft(button_width));
    row.removeFromLeft(spacing);
    m_audio_settings_button.setBounds(row.removeFromLeft(button_width));
    bounds.removeFromTop(spacing);

    row = bounds.removeFromTop(row_height);
    m_operator_button.setBounds(row.removeFromLeft(button_width));
    row.removeFromLeft(spacing);
    m_reset_button.setBounds(row.removeFromLeft(88));
}

} // namespace streamgen
