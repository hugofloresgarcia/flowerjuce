#pragma once

#include <juce_gui_basics/juce_gui_basics.h>
#include <array>

class LayerCakeLookAndFeel : public juce::LookAndFeel_V4
{
public:
    enum class ControlButtonType
    {
        Unknown = 0,
        Trigger,
        Record,
        Clock,
        Pattern,
        Preset
    };

    static const juce::Identifier kControlButtonTypeProperty;

    LayerCakeLookAndFeel();

    juce::Font getTextButtonFont(juce::TextButton&, int buttonHeight) override;
    juce::Font getLabelFont(juce::Label&) override;
    juce::Font getComboBoxFont(juce::ComboBox&) override;
    juce::Font getPopupMenuFont() override;
    juce::Typeface::Ptr getTypefaceForFont(const juce::Font& font) override;

    void drawButtonBackground(juce::Graphics&,
                              juce::Button&,
                              const juce::Colour& backgroundColour,
                              bool shouldDrawButtonAsHighlighted,
                              bool shouldDrawButtonAsDown) override;

    void drawComboBox(juce::Graphics&,
                      int width,
                      int height,
                      bool isButtonDown,
                      int buttonX,
                      int buttonY,
                      int buttonW,
                      int buttonH,
                      juce::ComboBox&) override;

    void positionComboBoxText(juce::ComboBox&, juce::Label&) override;

    void drawLinearSlider(juce::Graphics&,
                          int x,
                          int y,
                          int width,
                          int height,
                          float sliderPos,
                          float minSliderPos,
                          float maxSliderPos,
                          juce::Slider::SliderStyle,
                          juce::Slider&) override;

    void drawTickBox(juce::Graphics&,
                     juce::Component&,
                     float x,
                     float y,
                     float w,
                     float h,
                     bool ticked,
                     bool isEnabled,
                     bool shouldDrawButtonAsHighlighted,
                     bool shouldDrawButtonAsDown) override;

    static void setControlButtonType(juce::Button& button, ControlButtonType type);
    static ControlButtonType getControlButtonType(const juce::Button& button);

    juce::Colour getTerminalColour() const noexcept { return m_terminal; }
    juce::Colour getPanelColour() const noexcept { return m_panel; }
    juce::Colour getWaveformColour() const noexcept { return m_waveform_colour; }
    juce::Colour getKnobLabelColour() const noexcept { return m_knob_label_colour; }
    juce::Colour getKnobRecorderIdleColour() const noexcept;
    juce::Colour getKnobRecorderArmedColour() const noexcept;
    juce::Colour getKnobRecorderRecordingColour() const noexcept;
    juce::Colour getKnobRecorderPlayingColour() const noexcept;
    juce::Colour getLayerColour(size_t index) const noexcept;

    juce::Colour getControlAccentColour(ControlButtonType type) const noexcept;
    juce::Colour getControlDisabledBorderColour(ControlButtonType type) const noexcept;
    juce::Colour getControlDisabledFillColour() const noexcept { return m_disabled_button_fill; }

private:
    static juce::Font makeMonoFont(float size, juce::Font::FontStyleFlags style = juce::Font::plain);
    juce::Colour control_fill_colour(ControlButtonType type,
                                     bool isEnabled,
                                     bool isActive,
                                     bool isHighlighted,
                                     bool isDown) const noexcept;
    juce::Colour control_border_colour(ControlButtonType type, bool isEnabled, bool isActive = false) const noexcept;

    juce::Colour m_background;
    juce::Colour m_panel;
    juce::Colour m_border;
    juce::Colour m_terminal;
    juce::Colour m_scanline;
    juce::Colour m_accentCyan;
    juce::Colour m_accentMagenta;
    juce::Colour m_controlRed;
    juce::Colour m_controlGreen;
    juce::Colour m_controlYellow;
    juce::Colour m_controlCyan;
    juce::Colour m_controlMagenta;
    juce::Colour m_waveform_colour;
    juce::Colour m_knob_label_colour;
    juce::Colour m_disabled_button_fill;
    std::array<juce::Colour, 6> m_layer_colours;
};

