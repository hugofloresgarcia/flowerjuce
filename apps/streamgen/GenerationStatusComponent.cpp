#include "GenerationStatusComponent.h"
#include "LayerCakeLookAndFeel.h"

namespace streamgen {

GenerationStatusComponent::GenerationStatusComponent()
{
    setOpaque(true);
}

void GenerationStatusComponent::update(
    int queue_depth,
    int64_t generation_count,
    double last_latency_ms,
    int64_t last_job_id,
    bool worker_busy,
    const juce::String& source_label,
    const juce::String& last_land_timeline)
{
    m_queue_depth = queue_depth;
    m_generation_count = generation_count;
    m_last_latency_ms = last_latency_ms;
    m_last_job_id = last_job_id;
    m_worker_busy = worker_busy;
    m_source_label = source_label;
    m_last_land_timeline = last_land_timeline;
    repaint();
}

void GenerationStatusComponent::paint(juce::Graphics& g)
{
    auto bounds = getLocalBounds();
    auto& lf = getLookAndFeel();
    const juce::Colour panel = lf.findColour(juce::ComboBox::backgroundColourId);
    const juce::Colour terminal = lf.findColour(juce::Label::textColourId);
    juce::Colour dim = terminal.withAlpha(0.5f);
    juce::Colour idle_grey = terminal.withAlpha(0.42f);
    juce::Colour busy_on(0xff3cff9f);
    juce::Colour gen_c(0xff35c0ff);
    juce::Colour lat_y(0xfff2b950);
    juce::Colour land_m(0xfff45bff);
    if (auto* lc = dynamic_cast<LayerCakeLookAndFeel*>(&lf))
    {
        busy_on = lc->getControlAccentColour(LayerCakeLookAndFeel::ControlButtonType::Clock);
        gen_c = lc->getLayerColour(1).brighter(0.45f);
        lat_y = lc->getLayerColour(2).brighter(0.45f);
        land_m = lc->getLayerColour(5).brighter(0.45f);
    }

    g.fillAll(panel);

    const int w = bounds.getWidth();

    g.setColour(terminal);
    g.setFont(juce::Font(juce::FontOptions(juce::Font::getDefaultMonospacedFontName(), 11.0f, juce::Font::plain)));
    g.drawText("STATUS", 4, 2, w - 8, 14, juce::Justification::centredLeft);

    g.setFont(juce::Font(juce::FontOptions(juce::Font::getDefaultMonospacedFontName(), 10.0f, juce::Font::plain)));
    int y = 20;
    const int row_height = 14;

    auto draw_row = [&](const juce::String& label, const juce::String& value, juce::Colour value_colour)
    {
        g.setColour(dim);
        g.drawText(label, 4, y, w / 2, row_height, juce::Justification::centredLeft);
        g.setColour(value_colour);
        g.drawText(value, w / 2, y, w / 2 - 4, row_height, juce::Justification::centredRight);
        y += row_height + 1;
    };

    const juce::Colour busy_colour = m_worker_busy ? busy_on : idle_grey;
    draw_row("Worker:", m_worker_busy ? "BUSY" : "IDLE", busy_colour);

    draw_row("Queue:", juce::String(m_queue_depth), terminal);

    draw_row("Gen #:", juce::String(m_generation_count), gen_c);

    juce::String latency_str = (m_last_latency_ms > 0.0)
        ? juce::String(m_last_latency_ms, 0) + "ms"
        : "--";
    draw_row("Latency:", latency_str, lat_y);

    draw_row("Job ID:", (m_last_job_id >= 0) ? juce::String(m_last_job_id) : "--", terminal);

    if (m_last_land_timeline.isNotEmpty())
        draw_row("Land @:", m_last_land_timeline, land_m);

    y += 4;
    g.setColour(dim);
    g.drawText("Source:", 4, y, w / 3, row_height, juce::Justification::centredLeft);
    g.setColour(busy_on);
    g.drawText(m_source_label, 4, y + row_height, w - 8, row_height,
               juce::Justification::centredLeft);
}

} // namespace streamgen
