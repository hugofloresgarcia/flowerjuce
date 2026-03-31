#include "MixerComponent.h"

namespace streamgen {

MixerComponent::MixerComponent()
{
    m_streamgen_audio_label.setText("Streamgen", juce::dontSendNotification);
    m_streamgen_audio_label.setJustificationType(juce::Justification::centred);
    addAndMakeVisible(m_streamgen_audio_label);

    m_streamgen_audio_slider.setSliderStyle(juce::Slider::LinearVertical);
    m_streamgen_audio_slider.setRange(0.0, 2.0, 0.01);
    m_streamgen_audio_slider.setValue(0.0);
    m_streamgen_audio_slider.setTextBoxStyle(juce::Slider::TextBoxBelow, false, 40, 16);
    m_streamgen_audio_slider.onValueChange = [this]()
    {
        if (on_streamgen_audio_gain_changed)
            on_streamgen_audio_gain_changed(static_cast<float>(m_streamgen_audio_slider.getValue()));
    };
    addAndMakeVisible(m_streamgen_audio_slider);

    m_drums_label.setText("Drums", juce::dontSendNotification);
    m_drums_label.setJustificationType(juce::Justification::centred);
    addAndMakeVisible(m_drums_label);

    m_drums_slider.setSliderStyle(juce::Slider::LinearVertical);
    m_drums_slider.setRange(0.0, 2.0, 0.01);
    m_drums_slider.setValue(1.0);
    m_drums_slider.setTextBoxStyle(juce::Slider::TextBoxBelow, false, 40, 16);
    m_drums_slider.onValueChange = [this]()
    {
        if (on_drums_gain_changed)
            on_drums_gain_changed(static_cast<float>(m_drums_slider.getValue()));
    };
    addAndMakeVisible(m_drums_slider);
}

void MixerComponent::paint(juce::Graphics& g)
{
    auto& lf = getLookAndFeel();
    g.fillAll(lf.findColour(juce::ComboBox::backgroundColourId));
    g.setColour(lf.findColour(juce::Label::textColourId));
    g.setFont(juce::Font(juce::FontOptions(juce::Font::getDefaultMonospacedFontName(), 11.0f, juce::Font::plain)));
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

    m_streamgen_audio_label.setBounds(left.removeFromTop(label_height));
    m_streamgen_audio_slider.setBounds(left);

    m_drums_label.setBounds(right.removeFromTop(label_height));
    m_drums_slider.setBounds(right);
}

} // namespace streamgen
