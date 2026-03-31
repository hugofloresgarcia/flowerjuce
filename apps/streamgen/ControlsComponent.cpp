#include "ControlsComponent.h"
#include "StreamGenProcessor.h"
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

constexpr int k_hop_bar_combo_id_half = 1;
constexpr int k_hop_bar_combo_id_one = 2;
constexpr int k_hop_bar_combo_id_two = 3;
constexpr int k_hop_bar_combo_id_three = 4;
constexpr int k_hop_bar_combo_id_four = 5;

constexpr int k_delay_bar_combo_id_zero = 1;
constexpr int k_delay_bar_combo_id_one = 2;
constexpr int k_delay_bar_combo_id_two = 3;
constexpr int k_delay_bar_combo_id_three = 4;
constexpr int k_delay_bar_combo_id_four = 5;

float hop_bars_for_combo_id(int combo_id)
{
    if (combo_id == k_hop_bar_combo_id_half)
        return 0.5f;
    if (combo_id == k_hop_bar_combo_id_one)
        return 1.0f;
    if (combo_id == k_hop_bar_combo_id_two)
        return 2.0f;
    if (combo_id == k_hop_bar_combo_id_three)
        return 3.0f;
    if (combo_id == k_hop_bar_combo_id_four)
        return 4.0f;
    return 1.0f;
}

int combo_id_for_hop_bars(float hop_bars)
{
    if (hop_bars <= 0.625f)
        return k_hop_bar_combo_id_half;
    if (hop_bars <= 1.5f)
        return k_hop_bar_combo_id_one;
    if (hop_bars <= 2.5f)
        return k_hop_bar_combo_id_two;
    if (hop_bars <= 3.5f)
        return k_hop_bar_combo_id_three;
    return k_hop_bar_combo_id_four;
}

float delay_bars_for_combo_id(int combo_id)
{
    if (combo_id == k_delay_bar_combo_id_one)
        return 1.0f;
    if (combo_id == k_delay_bar_combo_id_two)
        return 2.0f;
    if (combo_id == k_delay_bar_combo_id_three)
        return 3.0f;
    if (combo_id == k_delay_bar_combo_id_four)
        return 4.0f;
    return 0.0f;
}

int combo_id_for_delay_bars(float delay_bars)
{
    if (delay_bars <= 0.5f)
        return k_delay_bar_combo_id_zero;
    if (delay_bars <= 1.5f)
        return k_delay_bar_combo_id_one;
    if (delay_bars <= 2.5f)
        return k_delay_bar_combo_id_two;
    if (delay_bars <= 3.5f)
        return k_delay_bar_combo_id_three;
    return k_delay_bar_combo_id_four;
}

} // namespace

ControlsComponent::ControlsComponent()
{
    setOpaque(true);

    addAndMakeVisible(m_group_prompt);
    addAndMakeVisible(m_group_musical);
    addAndMakeVisible(m_group_inference);
    addAndMakeVisible(m_group_session);

    m_group_prompt.toBack();
    m_group_musical.toBack();
    m_group_inference.toBack();
    m_group_session.toBack();

    m_prompt_label.setText("Prompt", juce::dontSendNotification);
    addAndMakeVisible(m_prompt_label);

    m_prompt_editor.setMultiLine(true, true);
    m_prompt_editor.setReturnKeyStartsNewLine(true);
    m_prompt_editor.setText("percussion");
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
        refresh_musical_hop_delay_visibility();
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
        if (on_hop_changed)
            on_hop_changed(static_cast<float>(m_hop_slider.getValue()));
    };
    addAndMakeVisible(m_hop_slider);

    m_hop_bars_combo.addItem("Hop: 1/2 bar", k_hop_bar_combo_id_half);
    m_hop_bars_combo.addItem("Hop: 1 bar", k_hop_bar_combo_id_one);
    m_hop_bars_combo.addItem("Hop: 2 bars", k_hop_bar_combo_id_two);
    m_hop_bars_combo.addItem("Hop: 3 bars", k_hop_bar_combo_id_three);
    m_hop_bars_combo.addItem("Hop: 4 bars", k_hop_bar_combo_id_four);
    m_hop_bars_combo.setSelectedId(k_hop_bar_combo_id_one, juce::dontSendNotification);
    m_hop_bars_combo.onChange = [this]()
    {
        if (on_hop_bars_changed)
            on_hop_bars_changed(hop_bars_for_combo_id(m_hop_bars_combo.getSelectedId()));
    };
    addAndMakeVisible(m_hop_bars_combo);

    m_schedule_delay_label.setText("Land delay (s):", juce::dontSendNotification);
    addAndMakeVisible(m_schedule_delay_label);

    m_schedule_delay_slider.setRange(0.0, 30.0, 0.25);
    m_schedule_delay_slider.setValue(0.0);
    m_schedule_delay_slider.setTextBoxStyle(juce::Slider::TextBoxRight, false, 52, 20);
    m_schedule_delay_slider.onValueChange = [this]()
    {
        if (on_schedule_delay_changed)
            on_schedule_delay_changed(static_cast<float>(m_schedule_delay_slider.getValue()));
    };
    addAndMakeVisible(m_schedule_delay_slider);

    m_schedule_delay_bars_combo.addItem("Land: 0 bars", k_delay_bar_combo_id_zero);
    m_schedule_delay_bars_combo.addItem("Land: 1 bar", k_delay_bar_combo_id_one);
    m_schedule_delay_bars_combo.addItem("Land: 2 bars", k_delay_bar_combo_id_two);
    m_schedule_delay_bars_combo.addItem("Land: 3 bars", k_delay_bar_combo_id_three);
    m_schedule_delay_bars_combo.addItem("Land: 4 bars", k_delay_bar_combo_id_four);
    m_schedule_delay_bars_combo.setSelectedId(k_delay_bar_combo_id_zero, juce::dontSendNotification);
    m_schedule_delay_bars_combo.onChange = [this]()
    {
        if (on_schedule_delay_bars_changed)
            on_schedule_delay_bars_changed(delay_bars_for_combo_id(m_schedule_delay_bars_combo.getSelectedId()));
    };
    addAndMakeVisible(m_schedule_delay_bars_combo);

    m_click_track_toggle.onClick = [this]()
    {
        if (on_click_track_enabled_changed)
            on_click_track_enabled_changed(m_click_track_toggle.getToggleState());
    };
    addAndMakeVisible(m_click_track_toggle);

    m_click_vol_label.setText("Click gain", juce::dontSendNotification);
    addAndMakeVisible(m_click_vol_label);

    m_click_volume_slider.setRange(0.0, 1.0, 0.01);
    m_click_volume_slider.setValue(0.35);
    m_click_volume_slider.setTextBoxStyle(juce::Slider::TextBoxRight, false, 44, 20);
    m_click_volume_slider.onValueChange = [this]()
    {
        if (on_click_track_volume_changed)
            on_click_track_volume_changed(static_cast<float>(m_click_volume_slider.getValue()));
    };
    addAndMakeVisible(m_click_volume_slider);

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

    LayerCakeLookAndFeel::setControlButtonType(m_warmup_audio_button, LayerCakeLookAndFeel::ControlButtonType::Preset);
    m_warmup_audio_button.onClick = [this]()
    {
        if (on_warmup_audio_clicked)
            on_warmup_audio_clicked();
    };
    addAndMakeVisible(m_warmup_audio_button);

    m_warmup_audio_route_toggle.onClick = [this]()
    {
        if (on_warmup_audio_route_toggled)
            on_warmup_audio_route_toggled(m_warmup_audio_route_toggle.getToggleState());
    };
    addAndMakeVisible(m_warmup_audio_route_toggle);

    m_loop_last_gen_toggle.setToggleState(true, juce::dontSendNotification);
    m_loop_last_gen_toggle.onClick = [this]()
    {
        if (on_loop_last_generation_changed)
            on_loop_last_generation_changed(m_loop_last_gen_toggle.getToggleState());
    };
    addAndMakeVisible(m_loop_last_gen_toggle);

    LayerCakeLookAndFeel::setControlButtonType(m_simulate_button, LayerCakeLookAndFeel::ControlButtonType::Trigger);
    m_simulate_button.onClick = [this]()
    {
        if (on_simulate_clicked)
            on_simulate_clicked();
    };
    addAndMakeVisible(m_simulate_button);

    LayerCakeLookAndFeel::setControlButtonType(m_audio_settings_button, LayerCakeLookAndFeel::ControlButtonType::Clock);
    m_audio_settings_button.onClick = [this]()
    {
        if (on_audio_settings_clicked)
            on_audio_settings_clicked();
    };
    addAndMakeVisible(m_audio_settings_button);

    LayerCakeLookAndFeel::setControlButtonType(m_reset_button, LayerCakeLookAndFeel::ControlButtonType::Record);
    m_reset_button.onClick = [this]()
    {
        if (on_reset_clicked)
            on_reset_clicked();
    };
    addAndMakeVisible(m_reset_button);

    m_generation_toggle.onClick = [this]()
    {
        if (on_generation_enabled_changed)
            on_generation_enabled_changed(m_generation_toggle.getToggleState());
    };
    addAndMakeVisible(m_generation_toggle);

    refresh_musical_hop_delay_visibility();
}

void ControlsComponent::refresh_musical_hop_delay_visibility()
{
    const bool musical = m_musical_time_toggle.getToggleState();
    m_hop_slider.setVisible(!musical);
    m_hop_bars_combo.setVisible(musical);
    m_schedule_delay_slider.setVisible(!musical);
    m_schedule_delay_bars_combo.setVisible(musical);
    if (musical)
    {
        m_hop_label.setText("Hop (bars):", juce::dontSendNotification);
        m_schedule_delay_label.setText("Land delay (bars):", juce::dontSendNotification);
    }
    else
    {
        m_hop_label.setText("Hop (s):", juce::dontSendNotification);
        m_schedule_delay_label.setText("Land delay (s):", juce::dontSendNotification);
    }
    refresh_click_track_control_enabled_state();
}

void ControlsComponent::refresh_click_track_control_enabled_state()
{
    const bool musical = m_musical_time_toggle.getToggleState();
    m_click_track_toggle.setEnabled(musical);
    m_click_vol_label.setEnabled(musical);
    m_click_volume_slider.setEnabled(musical);
}

void ControlsComponent::sync_click_track_from_processor(const StreamGenProcessor& processor)
{
    m_click_track_toggle.setToggleState(
        processor.click_track_enabled.load(std::memory_order_relaxed),
        juce::dontSendNotification);
    m_click_volume_slider.setValue(
        static_cast<double>(processor.click_track_volume.load(std::memory_order_relaxed)),
        juce::dontSendNotification);
    refresh_click_track_control_enabled_state();
}

void ControlsComponent::set_hop_bars_combo_from_value(float hop_bars)
{
    const int id = combo_id_for_hop_bars(hop_bars);
    m_hop_bars_combo.setSelectedId(id, juce::dontSendNotification);
}

void ControlsComponent::set_schedule_delay_bars_combo_from_value(float delay_bars)
{
    const int id = combo_id_for_delay_bars(delay_bars);
    m_schedule_delay_bars_combo.setSelectedId(id, juce::dontSendNotification);
}

void ControlsComponent::paint(juce::Graphics& g)
{
    g.fillAll(getLookAndFeel().findColour(juce::ComboBox::backgroundColourId));
    g.setColour(getLookAndFeel().findColour(juce::ComboBox::outlineColourId).withAlpha(0.45f));
    g.drawHorizontalLine(0, 0.0f, static_cast<float>(getWidth()));
}

void ControlsComponent::sync_time_mode_from_scheduler(const GenerationScheduler& sched, bool loop_last_generation_enabled)
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

    m_loop_last_gen_toggle.setToggleState(loop_last_generation_enabled, juce::dontSendNotification);

    sync_hop_delay_controls_from_scheduler(sched);
    refresh_musical_hop_delay_visibility();
}

void ControlsComponent::set_loop_last_generation_toggle(bool on, juce::NotificationType notification)
{
    m_loop_last_gen_toggle.setToggleState(on, notification);
}

void ControlsComponent::sync_hop_delay_controls_from_scheduler(const GenerationScheduler& sched)
{
    const bool musical = sched.musical_time_enabled.load(std::memory_order_relaxed);
    if (musical)
    {
        set_hop_bars_combo_from_value(sched.hop_bars.load(std::memory_order_relaxed));
        set_schedule_delay_bars_combo_from_value(sched.schedule_delay_bars.load(std::memory_order_relaxed));
    }
    else
    {
        m_hop_slider.setRange(0.5, 10.0, 0.1);
        m_hop_slider.setValue(static_cast<double>(sched.hop_seconds.load(std::memory_order_relaxed)),
                              juce::dontSendNotification);
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

void ControlsComponent::set_warmup_audio_route_toggle(bool route_to_output, juce::NotificationType notification)
{
    m_warmup_audio_route_toggle.setToggleState(route_to_output, notification);
}

void ControlsComponent::set_warmup_audio_route_enabled(bool enabled)
{
    m_warmup_audio_route_toggle.setEnabled(enabled);
}

void ControlsComponent::layout_prompt_block(juce::Rectangle<int> inner)
{
    const int label_h = 18;
    auto label_row = inner.removeFromTop(label_h);
    m_prompt_label.setBounds(label_row);
    inner.removeFromTop(4);
    m_prompt_editor.setBounds(inner);
}

void ControlsComponent::layout_musical_rows(juce::Rectangle<int> inner)
{
    const int row_h = 26;
    const int spacing = 4;
    const int musical_toggle_w = 100;
    const int bpm_lbl = 36;
    const int bpm_sl = 108;
    const int sig_lbl = 28;
    const int sig_cb = 72;
    const int q_lbl = 56;
    const int q_cb = 96;
    const int hop_lbl = 96;

    auto row = inner.removeFromTop(row_h);
    m_musical_time_toggle.setBounds(row.removeFromLeft(musical_toggle_w));
    row.removeFromLeft(spacing);
    m_bpm_label.setBounds(row.removeFromLeft(bpm_lbl));
    m_bpm_slider.setBounds(row.removeFromLeft(bpm_sl));
    row.removeFromLeft(spacing);
    m_time_sig_label.setBounds(row.removeFromLeft(sig_lbl));
    m_time_sig_combo.setBounds(row.removeFromLeft(sig_cb));
    row.removeFromLeft(spacing);
    m_quantize_label.setBounds(row.removeFromLeft(q_lbl));
    m_quantize_combo.setBounds(row.removeFromLeft(q_cb));

    inner.removeFromTop(spacing);
    row = inner.removeFromTop(row_h);
    m_hop_label.setBounds(row.removeFromLeft(hop_lbl));
    const auto hop_val_bounds = row.removeFromLeft(juce::jmax(160, row.getWidth() / 2 - 60));
    m_hop_slider.setBounds(hop_val_bounds);
    m_hop_bars_combo.setBounds(hop_val_bounds);
    row.removeFromLeft(spacing);
    m_schedule_delay_label.setBounds(row.removeFromLeft(112));
    const auto delay_val_bounds = row;
    m_schedule_delay_slider.setBounds(delay_val_bounds);
    m_schedule_delay_bars_combo.setBounds(delay_val_bounds);

    inner.removeFromTop(spacing);
    row = inner.removeFromTop(row_h);
    m_click_track_toggle.setBounds(row.removeFromLeft(108));
    row.removeFromLeft(spacing);
    m_click_vol_label.setBounds(row.removeFromLeft(72));
    m_click_volume_slider.setBounds(row.removeFromLeft(juce::jmin(160, row.getWidth())));
}

void ControlsComponent::layout_inference_rows(juce::Rectangle<int> inner)
{
    const int row_h = 26;
    const int spacing = 4;
    const int lbl = 52;

    auto row1 = inner.removeFromTop(row_h);
    auto left = row1.removeFromLeft(row1.getWidth() / 2 - spacing / 2);
    auto right = row1;
    m_keep_ratio_label.setBounds(left.removeFromLeft(lbl));
    m_keep_ratio_slider.setBounds(left);
    m_steps_label.setBounds(right.removeFromLeft(lbl));
    m_steps_slider.setBounds(right);

    inner.removeFromTop(spacing);
    auto row2 = inner.removeFromTop(row_h);
    m_cfg_label.setBounds(row2.removeFromLeft(lbl));
    m_cfg_slider.setBounds(row2);
}

void ControlsComponent::layout_session_rows(juce::Rectangle<int> inner)
{
    const int row_h = 26;
    const int spacing = 4;
    const int btn_w = 112;
    const int small_toggle_w = 92;

    auto row = inner.removeFromTop(row_h);
    m_generation_toggle.setBounds(row.removeFromLeft(168));
    row.removeFromLeft(spacing);
    m_loop_last_gen_toggle.setBounds(row.removeFromLeft(130));
    row.removeFromLeft(spacing);
    m_warmup_audio_button.setBounds(row.removeFromLeft(btn_w));
    row.removeFromLeft(spacing);
    m_warmup_audio_route_toggle.setBounds(row.removeFromLeft(small_toggle_w));
    row.removeFromLeft(spacing);
    m_simulate_button.setBounds(row.removeFromLeft(btn_w));
    row.removeFromLeft(spacing);
    m_audio_settings_button.setBounds(row.removeFromLeft(btn_w));

    inner.removeFromTop(spacing);
    row = inner.removeFromTop(row_h);
    m_reset_button.setBounds(row.removeFromLeft(88));
}

void ControlsComponent::resized()
{
    auto bounds = getLocalBounds().reduced(6);

    const int group_inset_top = 16;
    const int group_inset_bottom = 8;
    const int group_inset_x = 10;
    const int inter_group = 8;
    const int row_h = 26;
    const int row_spacing = 4;
    const int grid_gutter = 10;
    constexpr int k_prompt_editor_min_height = 76;

    const int prompt_block = group_inset_top + 18 + 4 + k_prompt_editor_min_height + group_inset_bottom;
    const int musical_block = group_inset_top + row_h + row_spacing + row_h + row_spacing + row_h + group_inset_bottom;
    const int inference_block = group_inset_top + row_h + row_spacing + row_h + group_inset_bottom;
    const int session_block = group_inset_top + row_h + row_spacing + row_h + group_inset_bottom;
    const int grid_row_height = juce::jmax(musical_block, inference_block);

    auto pr = bounds.removeFromTop(prompt_block);
    m_group_prompt.setBounds(pr);
    layout_prompt_block(pr.reduced(group_inset_x, group_inset_top).withTrimmedBottom(group_inset_bottom));

    bounds.removeFromTop(inter_group);
    auto grid_bounds = bounds.removeFromTop(grid_row_height);
    const int half_w = (grid_bounds.getWidth() - grid_gutter) / 2;
    auto col_left = grid_bounds.removeFromLeft(half_w);
    grid_bounds.removeFromLeft(grid_gutter);
    auto col_right = grid_bounds;

    m_group_musical.setBounds(col_left);
    layout_musical_rows(col_left.reduced(group_inset_x, group_inset_top).withTrimmedBottom(group_inset_bottom));

    m_group_inference.setBounds(col_right);
    layout_inference_rows(col_right.reduced(group_inset_x, group_inset_top).withTrimmedBottom(group_inset_bottom));

    bounds.removeFromTop(inter_group);
    auto sess = bounds.removeFromTop(session_block);
    m_group_session.setBounds(sess);
    layout_session_rows(sess.reduced(group_inset_x, group_inset_top).withTrimmedBottom(group_inset_bottom));
}

} // namespace streamgen
