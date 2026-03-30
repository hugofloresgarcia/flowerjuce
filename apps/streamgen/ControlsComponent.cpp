#include "ControlsComponent.h"

namespace streamgen {

ControlsComponent::ControlsComponent()
{
    // --- Prompt ---
    m_prompt_label.setText("Prompt:", juce::dontSendNotification);
    m_prompt_label.setColour(juce::Label::textColourId, juce::Colour(0xffe0e0e0));
    addAndMakeVisible(m_prompt_label);

    m_prompt_editor.setMultiLine(false);
    m_prompt_editor.setText("percussion");
    m_prompt_editor.setColour(juce::TextEditor::backgroundColourId, juce::Colour(0xff2a2a4e));
    m_prompt_editor.setColour(juce::TextEditor::textColourId, juce::Colour(0xffe0e0e0));
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

    // --- Hop ---
    m_hop_label.setText("Hop (s):", juce::dontSendNotification);
    m_hop_label.setColour(juce::Label::textColourId, juce::Colour(0xffe0e0e0));
    addAndMakeVisible(m_hop_label);

    m_hop_slider.setRange(0.5, 10.0, 0.1);
    m_hop_slider.setValue(3.0);
    m_hop_slider.setTextBoxStyle(juce::Slider::TextBoxRight, false, 50, 20);
    m_hop_slider.setColour(juce::Slider::textBoxTextColourId, juce::Colour(0xffe0e0e0));
    m_hop_slider.onValueChange = [this]()
    {
        if (on_hop_changed)
            on_hop_changed(static_cast<float>(m_hop_slider.getValue()));
    };
    addAndMakeVisible(m_hop_slider);

    // --- Keep ratio ---
    m_keep_ratio_label.setText("Keep:", juce::dontSendNotification);
    m_keep_ratio_label.setColour(juce::Label::textColourId, juce::Colour(0xffe0e0e0));
    addAndMakeVisible(m_keep_ratio_label);

    m_keep_ratio_slider.setRange(0.1, 0.9, 0.05);
    m_keep_ratio_slider.setValue(0.5);
    m_keep_ratio_slider.setTextBoxStyle(juce::Slider::TextBoxRight, false, 50, 20);
    m_keep_ratio_slider.setColour(juce::Slider::textBoxTextColourId, juce::Colour(0xffe0e0e0));
    m_keep_ratio_slider.onValueChange = [this]()
    {
        if (on_keep_ratio_changed)
            on_keep_ratio_changed(static_cast<float>(m_keep_ratio_slider.getValue()));
    };
    addAndMakeVisible(m_keep_ratio_slider);

    // --- Steps ---
    m_steps_label.setText("Steps:", juce::dontSendNotification);
    m_steps_label.setColour(juce::Label::textColourId, juce::Colour(0xffe0e0e0));
    addAndMakeVisible(m_steps_label);

    m_steps_slider.setRange(1, 50, 1);
    m_steps_slider.setValue(8);
    m_steps_slider.setTextBoxStyle(juce::Slider::TextBoxRight, false, 40, 20);
    m_steps_slider.setColour(juce::Slider::textBoxTextColourId, juce::Colour(0xffe0e0e0));
    m_steps_slider.onValueChange = [this]()
    {
        if (on_steps_changed)
            on_steps_changed(static_cast<int>(m_steps_slider.getValue()));
    };
    addAndMakeVisible(m_steps_slider);

    // --- CFG ---
    m_cfg_label.setText("CFG:", juce::dontSendNotification);
    m_cfg_label.setColour(juce::Label::textColourId, juce::Colour(0xffe0e0e0));
    addAndMakeVisible(m_cfg_label);

    m_cfg_slider.setRange(1.0, 15.0, 0.5);
    m_cfg_slider.setValue(7.0);
    m_cfg_slider.setTextBoxStyle(juce::Slider::TextBoxRight, false, 40, 20);
    m_cfg_slider.setColour(juce::Slider::textBoxTextColourId, juce::Colour(0xffe0e0e0));
    m_cfg_slider.onValueChange = [this]()
    {
        if (on_cfg_changed)
            on_cfg_changed(static_cast<float>(m_cfg_slider.getValue()));
    };
    addAndMakeVisible(m_cfg_slider);

    // --- Buttons ---
    m_warm_start_button.onClick = [this]() { if (on_warm_start_clicked) on_warm_start_clicked(); };
    addAndMakeVisible(m_warm_start_button);

    m_simulate_button.onClick = [this]() { if (on_simulate_clicked) on_simulate_clicked(); };
    addAndMakeVisible(m_simulate_button);

    m_audio_settings_button.onClick = [this]() { if (on_audio_settings_clicked) on_audio_settings_clicked(); };
    addAndMakeVisible(m_audio_settings_button);

    m_generation_toggle.setColour(juce::ToggleButton::textColourId, juce::Colour(0xffe0e0e0));
    m_generation_toggle.setColour(juce::ToggleButton::tickColourId, juce::Colour(0xff00e676));
    m_generation_toggle.onClick = [this]()
    {
        if (on_generation_enabled_changed)
            on_generation_enabled_changed(m_generation_toggle.getToggleState());
    };
    addAndMakeVisible(m_generation_toggle);
}

void ControlsComponent::resized()
{
    auto bounds = getLocalBounds().reduced(4);

    const int row_height = 26;
    const int label_width = 60;
    const int button_width = 120;
    const int spacing = 4;

    // Row 1: Prompt (full width)
    auto row = bounds.removeFromTop(row_height);
    m_prompt_label.setBounds(row.removeFromLeft(label_width));
    m_prompt_editor.setBounds(row);
    bounds.removeFromTop(spacing);

    // Row 2: Hop + Keep ratio
    row = bounds.removeFromTop(row_height);
    auto left_half = row.removeFromLeft(row.getWidth() / 2 - spacing / 2);
    auto right_half = row;

    m_hop_label.setBounds(left_half.removeFromLeft(label_width));
    m_hop_slider.setBounds(left_half);

    m_keep_ratio_label.setBounds(right_half.removeFromLeft(label_width));
    m_keep_ratio_slider.setBounds(right_half);
    bounds.removeFromTop(spacing);

    // Row 3: Steps + CFG
    row = bounds.removeFromTop(row_height);
    left_half = row.removeFromLeft(row.getWidth() / 2 - spacing / 2);
    right_half = row;

    m_steps_label.setBounds(left_half.removeFromLeft(label_width));
    m_steps_slider.setBounds(left_half);

    m_cfg_label.setBounds(right_half.removeFromLeft(label_width));
    m_cfg_slider.setBounds(right_half);
    bounds.removeFromTop(spacing);

    // Row 4: Buttons
    row = bounds.removeFromTop(row_height);
    m_generation_toggle.setBounds(row.removeFromLeft(160));
    row.removeFromLeft(spacing);
    m_warm_start_button.setBounds(row.removeFromLeft(button_width));
    row.removeFromLeft(spacing);
    m_simulate_button.setBounds(row.removeFromLeft(button_width));
    row.removeFromLeft(spacing);
    m_audio_settings_button.setBounds(row.removeFromLeft(button_width));
}

} // namespace streamgen
