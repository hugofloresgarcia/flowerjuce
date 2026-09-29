#include "MixerComponent.h"

namespace streamgen {

namespace {

void setup_fader(
    juce::Label& label,
    juce::Slider& slider,
    const juce::String& name,
    double initial_gain,
    std::function<void(float)>& callback,
    juce::Component& parent)
{
    label.setText(name, juce::dontSendNotification);
    label.setJustificationType(juce::Justification::centred);
    parent.addAndMakeVisible(label);

    slider.setSliderStyle(juce::Slider::LinearVertical);
    slider.setRange(0.0, 2.0, 0.01);
    slider.setValue(initial_gain);
    slider.setTextBoxStyle(juce::Slider::TextBoxBelow, false, 40, 16);
    slider.onValueChange = [&slider, &callback]()
    {
        if (callback)
            callback(static_cast<float>(slider.getValue()));
    };
    parent.addAndMakeVisible(slider);
}

} // namespace

MixerComponent::MixerComponent()
{
    setup_fader(m_streamgen_audio_label, m_streamgen_audio_slider, "Sax", 0.0, on_streamgen_audio_gain_changed, *this);
    setup_fader(m_percussion_label, m_percussion_slider, "Perc", 0.0, on_percussion_gain_changed, *this);
    setup_fader(m_drums_label, m_drums_slider, "Drums", 1.0, on_drums_gain_changed, *this);

    m_drums_mute_toggle.setToggleState(false, juce::dontSendNotification);
    m_drums_mute_toggle.setTooltip("Silence the drums bus (model output, loop hold, warmup) on the speakers");
    m_drums_mute_toggle.onClick = [this]()
    {
        if (on_drums_mute_changed)
            on_drums_mute_changed(m_drums_mute_toggle.getToggleState());
    };
    addAndMakeVisible(m_drums_mute_toggle);
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

    const int label_height = 16;
    const int mute_height = 20;
    const int fader_width = bounds.getWidth() / 3;

    auto sax = bounds.removeFromLeft(fader_width);
    auto perc = bounds.removeFromLeft(fader_width);
    auto drums = bounds;

    m_streamgen_audio_label.setBounds(sax.removeFromTop(label_height));
    m_streamgen_audio_slider.setBounds(sax);

    m_percussion_label.setBounds(perc.removeFromTop(label_height));
    m_percussion_slider.setBounds(perc);

    m_drums_label.setBounds(drums.removeFromTop(label_height));
    m_drums_mute_toggle.setBounds(drums.removeFromBottom(mute_height));
    m_drums_slider.setBounds(drums);
}

} // namespace streamgen
