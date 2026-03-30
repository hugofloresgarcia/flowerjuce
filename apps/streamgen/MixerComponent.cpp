#include "MixerComponent.h"

namespace streamgen {

MixerComponent::MixerComponent()
{
    m_sax_label.setText("Sax", juce::dontSendNotification);
    m_sax_label.setColour(juce::Label::textColourId, juce::Colour(0xffe0e0e0));
    m_sax_label.setJustificationType(juce::Justification::centred);
    addAndMakeVisible(m_sax_label);

    m_sax_slider.setSliderStyle(juce::Slider::LinearVertical);
    m_sax_slider.setRange(0.0, 2.0, 0.01);
    m_sax_slider.setValue(1.0);
    m_sax_slider.setTextBoxStyle(juce::Slider::TextBoxBelow, false, 40, 16);
    m_sax_slider.setColour(juce::Slider::textBoxTextColourId, juce::Colour(0xffe0e0e0));
    m_sax_slider.onValueChange = [this]()
    {
        if (on_sax_gain_changed)
            on_sax_gain_changed(static_cast<float>(m_sax_slider.getValue()));
    };
    addAndMakeVisible(m_sax_slider);

    m_drums_label.setText("Drums", juce::dontSendNotification);
    m_drums_label.setColour(juce::Label::textColourId, juce::Colour(0xffe0e0e0));
    m_drums_label.setJustificationType(juce::Justification::centred);
    addAndMakeVisible(m_drums_label);

    m_drums_slider.setSliderStyle(juce::Slider::LinearVertical);
    m_drums_slider.setRange(0.0, 2.0, 0.01);
    m_drums_slider.setValue(1.0);
    m_drums_slider.setTextBoxStyle(juce::Slider::TextBoxBelow, false, 40, 16);
    m_drums_slider.setColour(juce::Slider::textBoxTextColourId, juce::Colour(0xffe0e0e0));
    m_drums_slider.onValueChange = [this]()
    {
        if (on_drums_gain_changed)
            on_drums_gain_changed(static_cast<float>(m_drums_slider.getValue()));
    };
    addAndMakeVisible(m_drums_slider);
}

void MixerComponent::paint(juce::Graphics& g)
{
    g.fillAll(juce::Colour(0xff1a1a2e));
    g.setColour(juce::Colour(0xffe0e0e0));
    g.setFont(juce::Font(11.0f));
    g.drawText("MIXER", getLocalBounds().removeFromTop(16), juce::Justification::centred);
}

void MixerComponent::resized()
{
    auto bounds = getLocalBounds().reduced(2);
    bounds.removeFromTop(18);

    const int fader_width = bounds.getWidth() / 2;

    auto left = bounds.removeFromLeft(fader_width);
    auto right = bounds;

    const int label_height = 16;

    m_sax_label.setBounds(left.removeFromTop(label_height));
    m_sax_slider.setBounds(left);

    m_drums_label.setBounds(right.removeFromTop(label_height));
    m_drums_slider.setBounds(right);
}

} // namespace streamgen
