#include "LayerCakeLookAndFeel.h"

namespace
{
constexpr float kButtonBorderThickness = 1.6f;
constexpr float kInnerBorderThickness = 0.9f;
const juce::Colour kSoftWhite(0xfff4f4f2);

juce::Colour getAccentForButton(const juce::Button& button)
{
    return button.findColour(juce::TextButton::buttonOnColourId);
}
} // namespace

const juce::Identifier LayerCakeLookAndFeel::kControlButtonTypeProperty("layercake.controlButtonType");

LayerCakeLookAndFeel::LayerCakeLookAndFeel()
{
    m_background = juce::Colours::black;
    m_panel = juce::Colour(0xff050d17);
    m_border = juce::Colour(0xff2a3147);
    m_terminal = kSoftWhite;
    m_scanline = juce::Colour(0x3300b5ff);
    m_accentCyan = juce::Colour(0xff35c0ff);
    m_accentMagenta = juce::Colour(0xfff45bff);
    m_controlRed = juce::Colour(0xffff564a);
    m_controlGreen = juce::Colour(0xff3cff9f);
    m_controlYellow = juce::Colour(0xfff8d24b);
    m_controlCyan = juce::Colour(0xff35c0ff);
    m_controlMagenta = juce::Colour(0xfff45bff);
    m_waveform_colour = juce::Colour(0xffefefef);
    m_knob_label_colour = m_terminal.brighter(0.4f);
    m_disabled_button_fill = juce::Colours::black.withAlpha(0.85f);
    m_layer_colours = {
        juce::Colour(0xfff25f5c).darker(0.8),
        juce::Colour(0xff35c0ff).darker(0.8),
        juce::Colour(0xfff2b950).darker(0.8),
        juce::Colour(0xff7d6bff).darker(0.8),
        juce::Colour(0xff5aff8c).darker(0.8),
        juce::Colour(0xfff45bff).darker(0.8)
    };

    setColour(juce::ResizableWindow::backgroundColourId, m_background);

    setColour(juce::TextButton::buttonColourId, m_panel);
    setColour(juce::TextButton::textColourOffId, m_terminal);
    setColour(juce::TextButton::textColourOnId, m_terminal);
    setColour(juce::TextButton::buttonOnColourId, m_accentCyan);

    setColour(juce::ToggleButton::textColourId, m_terminal);
    setColour(juce::ToggleButton::tickColourId, m_controlGreen);
    setColour(juce::ToggleButton::tickDisabledColourId, m_border);

    setColour(juce::TextEditor::backgroundColourId, m_panel);
    setColour(juce::TextEditor::textColourId, m_terminal);
    setColour(juce::TextEditor::outlineColourId, m_border);
    setColour(juce::TextEditor::focusedOutlineColourId, m_accentCyan.withAlpha(0.85f));
    setColour(juce::TextEditor::highlightColourId, m_accentCyan.withAlpha(0.35f));
    setColour(juce::TextEditor::highlightedTextColourId, m_terminal);

    setColour(juce::Label::textColourId, m_terminal);
    setColour(juce::Label::backgroundColourId, juce::Colours::transparentBlack);
    setColour(juce::Label::outlineColourId, juce::Colours::transparentBlack);

    // Default knob color is a muted gray - LFOs provide the color pop
    const juce::Colour kDefaultKnobGray(0xff5a6070);
    setColour(juce::Slider::thumbColourId, kDefaultKnobGray);
    setColour(juce::Slider::trackColourId, kDefaultKnobGray.brighter(0.15f));
    setColour(juce::Slider::rotarySliderFillColourId, kDefaultKnobGray.withAlpha(0.7f));
    setColour(juce::Slider::rotarySliderOutlineColourId, m_border);
    setColour(juce::Slider::backgroundColourId, m_panel.darker(0.4f));
    setColour(juce::Slider::textBoxTextColourId, m_terminal);
    setColour(juce::Slider::textBoxBackgroundColourId, m_background);
    setColour(juce::Slider::textBoxOutlineColourId, m_border);

    setColour(juce::ComboBox::backgroundColourId, m_panel);
    setColour(juce::ComboBox::textColourId, m_terminal);
    setColour(juce::ComboBox::outlineColourId, m_border);
    setColour(juce::ComboBox::arrowColourId, m_terminal);

    setColour(juce::PopupMenu::backgroundColourId, m_panel);
    setColour(juce::PopupMenu::textColourId, m_terminal);
    setColour(juce::PopupMenu::highlightedBackgroundColourId, m_accentCyan);
    setColour(juce::PopupMenu::highlightedTextColourId, m_background);

    setColour(juce::ProgressBar::foregroundColourId, m_accentCyan);
    setColour(juce::ProgressBar::backgroundColourId, m_panel);
}

juce::Font LayerCakeLookAndFeel::makeMonoFont(float size, juce::Font::FontStyleFlags style)
{
    juce::FontOptions options(juce::Font::getDefaultMonospacedFontName(), size, static_cast<int>(style));
    return juce::Font(options);
}

juce::Font LayerCakeLookAndFeel::getTextButtonFont(juce::TextButton&, int buttonHeight)
{
    const float size = juce::jlimit(10.0f, 20.0f, buttonHeight * 0.65f);
    return makeMonoFont(size, juce::Font::bold);
}

juce::Font LayerCakeLookAndFeel::getLabelFont(juce::Label&)
{
    return makeMonoFont(14.0f);
}

juce::Font LayerCakeLookAndFeel::getComboBoxFont(juce::ComboBox&)
{
    return makeMonoFont(14.0f);
}

juce::Font LayerCakeLookAndFeel::getPopupMenuFont()
{
    return makeMonoFont(13.0f);
}

juce::Typeface::Ptr LayerCakeLookAndFeel::getTypefaceForFont(const juce::Font& font)
{
    // Force monospaced font for the entire UI.  Delegate to the parent LookAndFeel so JUCE
    // resolves the placeholder name (e.g. "<Default Monospaced>" → "DejaVu Sans Mono") and
    // handles platform-specific style fallbacks before handing off to FreeType / CoreText.
    juce::Font mono(font);
    mono.setTypefaceName(juce::Font::getDefaultMonospacedFontName());
    return juce::LookAndFeel::getTypefaceForFont(mono);
}

void LayerCakeLookAndFeel::drawButtonBackground(juce::Graphics& g,
                                                juce::Button& button,
                                                const juce::Colour& backgroundColour,
                                                bool shouldDrawButtonAsHighlighted,
                                                bool shouldDrawButtonAsDown)
{
    auto bounds = button.getLocalBounds().toFloat().reduced(0.5f);
    const float radius = juce::jmin(6.0f, bounds.getHeight() * 0.45f);
    const auto controlType = getControlButtonType(button);

    if (controlType == ControlButtonType::Unknown)
    {
        auto accent = getAccentForButton(button);

        auto fillColour = juce::Colours::black.withAlpha(0.9f);
        if (button.getToggleState() || shouldDrawButtonAsDown)
            fillColour = accent.withAlpha(0.25f);
        else if (shouldDrawButtonAsHighlighted)
            fillColour = backgroundColour.withAlpha(0.3f);

        g.setColour(fillColour);
        g.fillRoundedRectangle(bounds, radius);

        g.setColour(m_border);
        g.drawRoundedRectangle(bounds, radius, kButtonBorderThickness);

        g.setColour(accent.withAlpha(0.85f));
        g.drawRoundedRectangle(bounds.reduced(1.5f), juce::jmax(2.0f, radius - 2.0f), kInnerBorderThickness);
    }
    else
    {
        const bool isEnabled = button.isEnabled();
        const bool isActive = button.getToggleState();
        auto fillColour = control_fill_colour(controlType,
                                              isEnabled,
                                              isActive,
                                              shouldDrawButtonAsHighlighted,
                                              shouldDrawButtonAsDown);
        auto borderColour = control_border_colour(controlType, isEnabled, isActive);

        g.setColour(fillColour);
        g.fillRoundedRectangle(bounds, radius);

        g.setColour(borderColour);
        g.drawRoundedRectangle(bounds, radius, kButtonBorderThickness);

        g.setColour(borderColour.withAlpha(juce::jlimit(0.2f, 1.0f, borderColour.getFloatAlpha() + 0.1f)));
        g.drawRoundedRectangle(bounds.reduced(1.5f), juce::jmax(2.0f, radius - 2.0f), kInnerBorderThickness);
    }
}

void LayerCakeLookAndFeel::drawComboBox(juce::Graphics& g,
                                        int width,
                                        int height,
                                        bool isButtonDown,
                                        int buttonX,
                                        int buttonY,
                                        int buttonW,
                                        int buttonH,
                                        juce::ComboBox& box)
{
    juce::ignoreUnused(buttonX, buttonY, buttonW, buttonH);
    auto bounds = juce::Rectangle<int>(0, 0, width, height).toFloat().reduced(0.5f);
    const float radius = juce::jmin(6.0f, bounds.getHeight() * 0.45f);
    juce::Colour fill = box.findColour(juce::ComboBox::backgroundColourId);
    if (isButtonDown)
        fill = fill.brighter(0.08f);
    g.setColour(fill);
    g.fillRoundedRectangle(bounds, radius);
    g.setColour(box.findColour(juce::ComboBox::outlineColourId));
    g.drawRoundedRectangle(bounds, radius, kButtonBorderThickness);
    g.setColour(m_accentCyan.withAlpha(0.75f));
    g.drawRoundedRectangle(bounds.reduced(1.5f), juce::jmax(2.0f, radius - 2.0f), kInnerBorderThickness);

    juce::Rectangle<int> arrowZone(width - 28, 0, 20, height);
    juce::Path path;
    path.startNewSubPath(static_cast<float>(arrowZone.getX()) + 3.0f, static_cast<float>(arrowZone.getCentreY()) - 2.0f);
    path.lineTo(static_cast<float>(arrowZone.getCentreX()), static_cast<float>(arrowZone.getCentreY()) + 3.0f);
    path.lineTo(static_cast<float>(arrowZone.getRight()) - 3.0f, static_cast<float>(arrowZone.getCentreY()) - 2.0f);
    g.setColour(box.findColour(juce::ComboBox::arrowColourId).withAlpha(box.isEnabled() ? 0.95f : 0.25f));
    g.strokePath(path, juce::PathStrokeType(2.0f));
}

void LayerCakeLookAndFeel::positionComboBoxText(juce::ComboBox& box, juce::Label& label)
{
    label.setBounds(1, 1, box.getWidth() - 28, box.getHeight() - 2);
    label.setFont(getComboBoxFont(box));
    label.setColour(juce::Label::textColourId, box.findColour(juce::ComboBox::textColourId));
}

void LayerCakeLookAndFeel::drawLinearSlider(juce::Graphics& g,
                                            int x,
                                            int y,
                                            int width,
                                            int height,
                                            float sliderPos,
                                            float minSliderPos,
                                            float maxSliderPos,
                                            juce::Slider::SliderStyle style,
                                            juce::Slider& slider)
{
    if (slider.isBar())
    {
        g.setColour(slider.findColour(juce::Slider::trackColourId));
        g.fillRect(slider.isHorizontal() ? juce::Rectangle<float>(static_cast<float>(x), static_cast<float>(y) + 0.5f,
                                                                    sliderPos - static_cast<float>(x), static_cast<float>(height) - 1.0f)
                                         : juce::Rectangle<float>(static_cast<float>(x) + 0.5f, sliderPos,
                                                                  static_cast<float>(width) - 1.0f,
                                                                  static_cast<float>(y) + (static_cast<float>(height) - sliderPos)));

        drawLinearSliderOutline(g, x, y, width, height, style, slider);
        return;
    }

    const bool isTwoVal =
        (style == juce::Slider::SliderStyle::TwoValueVertical || style == juce::Slider::SliderStyle::TwoValueHorizontal);
    const bool isThreeVal =
        (style == juce::Slider::SliderStyle::ThreeValueVertical || style == juce::Slider::SliderStyle::ThreeValueHorizontal);

    const float trackWidth =
        juce::jmin(5.0f, slider.isHorizontal() ? static_cast<float>(height) * 0.22f : static_cast<float>(width) * 0.22f);

    juce::Point<float> startPoint(slider.isHorizontal() ? static_cast<float>(x) : static_cast<float>(x) + static_cast<float>(width) * 0.5f,
                                 slider.isHorizontal() ? static_cast<float>(y) + static_cast<float>(height) * 0.5f
                                                       : static_cast<float>(height + y));

    juce::Point<float> endPoint(slider.isHorizontal() ? static_cast<float>(width + x) : startPoint.x,
                                slider.isHorizontal() ? startPoint.y : static_cast<float>(y));

    juce::Path backgroundTrack;
    backgroundTrack.startNewSubPath(startPoint);
    backgroundTrack.lineTo(endPoint);
    g.setColour(slider.findColour(juce::Slider::backgroundColourId));
    g.strokePath(backgroundTrack, {trackWidth, juce::PathStrokeType::curved, juce::PathStrokeType::rounded});

    juce::Path valueTrack;
    juce::Point<float> minPoint;
    juce::Point<float> maxPoint;
    juce::Point<float> thumbPoint;

    if (isTwoVal || isThreeVal)
    {
        minPoint = {slider.isHorizontal() ? minSliderPos : static_cast<float>(width) * 0.5f,
                    slider.isHorizontal() ? static_cast<float>(height) * 0.5f : minSliderPos};

        if (isThreeVal)
            thumbPoint = {slider.isHorizontal() ? sliderPos : static_cast<float>(width) * 0.5f,
                          slider.isHorizontal() ? static_cast<float>(height) * 0.5f : sliderPos};

        maxPoint = {slider.isHorizontal() ? maxSliderPos : static_cast<float>(width) * 0.5f,
                    slider.isHorizontal() ? static_cast<float>(height) * 0.5f : maxSliderPos};
    }
    else
    {
        const float kx = slider.isHorizontal() ? sliderPos : (static_cast<float>(x) + static_cast<float>(width) * 0.5f);
        const float ky = slider.isHorizontal() ? (static_cast<float>(y) + static_cast<float>(height) * 0.5f) : sliderPos;
        minPoint = startPoint;
        maxPoint = {kx, ky};
    }

    const int thumbWidth = getSliderThumbRadius(slider);

    valueTrack.startNewSubPath(minPoint);
    valueTrack.lineTo(isThreeVal ? thumbPoint : maxPoint);
    g.setColour(slider.findColour(juce::Slider::trackColourId));
    g.strokePath(valueTrack, {trackWidth, juce::PathStrokeType::curved, juce::PathStrokeType::rounded});

    if (!isTwoVal)
    {
        const juce::Point<float> centre = isThreeVal ? thumbPoint : maxPoint;
        const float d = static_cast<float>(thumbWidth);
        g.setColour(slider.findColour(juce::Slider::thumbColourId));
        g.fillEllipse(juce::Rectangle<float>(d, d).withCentre(centre));
        g.setColour(m_accentCyan.withAlpha(slider.isEnabled() ? 0.5f : 0.22f));
        g.drawEllipse(juce::Rectangle<float>(d, d).withCentre(centre), kInnerBorderThickness);
    }

    if (isTwoVal || isThreeVal)
    {
        const float sr = juce::jmin(trackWidth, (slider.isHorizontal() ? static_cast<float>(height) : static_cast<float>(width)) * 0.4f);
        const juce::Colour pointerColour = slider.findColour(juce::Slider::thumbColourId);

        if (slider.isHorizontal())
        {
            drawPointer(g, minSliderPos - sr,
                        juce::jmax(0.0f, static_cast<float>(y) + static_cast<float>(height) * 0.5f - trackWidth * 2.0f),
                        trackWidth * 2.0f, pointerColour, 2);
            drawPointer(g, maxSliderPos - trackWidth,
                        juce::jmin(static_cast<float>(y + height) - trackWidth * 2.0f,
                                   static_cast<float>(y) + static_cast<float>(height) * 0.5f),
                        trackWidth * 2.0f, pointerColour, 4);
        }
        else
        {
            drawPointer(g, juce::jmax(0.0f, static_cast<float>(x) + static_cast<float>(width) * 0.5f - trackWidth * 2.0f),
                        minSliderPos - trackWidth, trackWidth * 2.0f, pointerColour, 1);
            drawPointer(g,
                        juce::jmin(static_cast<float>(x + width) - trackWidth * 2.0f,
                                   static_cast<float>(x) + static_cast<float>(width) * 0.5f),
                        maxSliderPos - sr, trackWidth * 2.0f, pointerColour, 3);
        }
    }
}

void LayerCakeLookAndFeel::drawTickBox(juce::Graphics& g,
                                      juce::Component& component,
                                      float x,
                                      float y,
                                      float w,
                                      float h,
                                      const bool ticked,
                                      const bool isEnabled,
                                      const bool shouldDrawButtonAsHighlighted,
                                      const bool shouldDrawButtonAsDown)
{
    juce::ignoreUnused(shouldDrawButtonAsDown);
    juce::Rectangle<float> tickBounds(x, y, w, h);
    const float radius = juce::jmin(4.0f, h * 0.28f);

    g.setColour(m_panel.darker(0.15f));
    g.fillRoundedRectangle(tickBounds, radius);

    if (shouldDrawButtonAsHighlighted && isEnabled)
    {
        g.setColour(m_accentCyan.withAlpha(0.12f));
        g.fillRoundedRectangle(tickBounds.reduced(1.0f), juce::jmax(1.5f, radius - 1.0f));
    }

    g.setColour(component.findColour(juce::ToggleButton::tickDisabledColourId));
    g.drawRoundedRectangle(tickBounds, radius, kButtonBorderThickness);
    g.setColour(m_accentCyan.withAlpha(isEnabled ? 0.45f : 0.2f));
    g.drawRoundedRectangle(tickBounds.reduced(1.2f), juce::jmax(2.0f, radius - 1.2f), kInnerBorderThickness);

    if (ticked)
    {
        juce::Colour tickCol = component.findColour(juce::ToggleButton::tickColourId);
        if (!isEnabled)
            tickCol = tickCol.withMultipliedAlpha(0.45f);
        g.setColour(tickCol);
        auto tick = getTickShape(0.75f);
        g.fillPath(tick, tick.getTransformToScaleToFit(tickBounds.reduced(4, 5).toFloat(), false));
    }
}

void LayerCakeLookAndFeel::setControlButtonType(juce::Button& button, ControlButtonType type)
{
    button.getProperties().set(kControlButtonTypeProperty, static_cast<int>(type));
}

LayerCakeLookAndFeel::ControlButtonType LayerCakeLookAndFeel::getControlButtonType(const juce::Button& button)
{
    const auto value = button.getProperties()[kControlButtonTypeProperty];
    if (value.isVoid())
        return ControlButtonType::Unknown;
    return static_cast<ControlButtonType>(static_cast<int>(value));
}

juce::Colour LayerCakeLookAndFeel::getLayerColour(size_t index) const noexcept
{
    if (m_layer_colours.empty())
        return m_terminal;
    return m_layer_colours[index % m_layer_colours.size()];
}

juce::Colour LayerCakeLookAndFeel::getKnobRecorderIdleColour() const noexcept
{
    return m_knob_label_colour.withAlpha(0.35f);
}

juce::Colour LayerCakeLookAndFeel::getKnobRecorderArmedColour() const noexcept
{
    return getControlAccentColour(ControlButtonType::Record).brighter(0.35f);
}

juce::Colour LayerCakeLookAndFeel::getKnobRecorderRecordingColour() const noexcept
{
    return getControlAccentColour(ControlButtonType::Record);
}

juce::Colour LayerCakeLookAndFeel::getKnobRecorderPlayingColour() const noexcept
{
    return getControlAccentColour(ControlButtonType::Clock);
}

juce::Colour LayerCakeLookAndFeel::getControlAccentColour(ControlButtonType type) const noexcept
{
    switch (type)
    {
        case ControlButtonType::Trigger: return m_controlCyan;
        case ControlButtonType::Record:  return m_controlRed;
        case ControlButtonType::Clock:   return m_controlGreen;
        case ControlButtonType::Pattern: return m_controlYellow;
        case ControlButtonType::Preset:  return m_controlMagenta;
        case ControlButtonType::Unknown:
        default:                         return m_accentCyan;
    }
}

juce::Colour LayerCakeLookAndFeel::getControlDisabledBorderColour(ControlButtonType type) const noexcept
{
    return control_border_colour(type, false);
}

juce::Colour LayerCakeLookAndFeel::control_fill_colour(ControlButtonType type,
                                                       bool isEnabled,
                                                       bool isActive,
                                                       bool isHighlighted,
                                                       bool isDown) const noexcept
{
    if (!isEnabled)
        return m_disabled_button_fill;

    // Gray when inactive, colored when active
    const juce::Colour inactiveGray(0xff3a3a3a);
    auto accent = getControlAccentColour(type);
    
    if (isActive || isDown)
        return accent.withAlpha(0.45f);
    if (isHighlighted)
        return inactiveGray.brighter(0.15f);
    return inactiveGray;
}

juce::Colour LayerCakeLookAndFeel::control_border_colour(ControlButtonType type, bool isEnabled, bool isActive) const noexcept
{
    // Gray border when inactive, colored when active
    const juce::Colour inactiveGray(0xff5a5a5a);
    
    if (!isEnabled)
        return inactiveGray.withAlpha(0.4f);
    
    if (isActive)
    {
        auto accent = getControlAccentColour(type);
        return accent.withAlpha(0.95f);
    }
    
    return inactiveGray;
}

