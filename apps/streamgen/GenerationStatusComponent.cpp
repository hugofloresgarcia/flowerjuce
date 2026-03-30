#include "GenerationStatusComponent.h"

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
    const juce::String& source_label)
{
    m_queue_depth = queue_depth;
    m_generation_count = generation_count;
    m_last_latency_ms = last_latency_ms;
    m_last_job_id = last_job_id;
    m_worker_busy = worker_busy;
    m_source_label = source_label;
    repaint();
}

void GenerationStatusComponent::paint(juce::Graphics& g)
{
    auto bounds = getLocalBounds();
    g.fillAll(juce::Colour(0xff1a1a2e));

    const int w = bounds.getWidth();

    g.setColour(juce::Colour(0xffe0e0e0));
    g.setFont(juce::Font(11.0f));
    g.drawText("STATUS", 4, 2, w - 8, 14, juce::Justification::centredLeft);

    g.setFont(juce::Font(10.0f));
    int y = 20;
    const int row_height = 14;

    auto draw_row = [&](const juce::String& label, const juce::String& value, juce::Colour value_colour)
    {
        g.setColour(juce::Colour(0xffaaaaaa));
        g.drawText(label, 4, y, w / 2, row_height, juce::Justification::centredLeft);
        g.setColour(value_colour);
        g.drawText(value, w / 2, y, w / 2 - 4, row_height, juce::Justification::centredRight);
        y += row_height + 1;
    };

    // Worker state indicator
    juce::Colour busy_colour = m_worker_busy ? juce::Colour(0xff00e676) : juce::Colour(0xff888888);
    draw_row("Worker:", m_worker_busy ? "BUSY" : "IDLE", busy_colour);

    draw_row("Queue:", juce::String(m_queue_depth), juce::Colour(0xffe0e0e0));

    draw_row("Gen #:", juce::String(m_generation_count), juce::Colour(0xff4fc3f7));

    juce::String latency_str = (m_last_latency_ms > 0.0)
        ? juce::String(m_last_latency_ms, 0) + "ms"
        : "--";
    draw_row("Latency:", latency_str, juce::Colour(0xffffa726));

    draw_row("Job ID:", (m_last_job_id >= 0) ? juce::String(m_last_job_id) : "--",
             juce::Colour(0xffe0e0e0));

    y += 4;
    g.setColour(juce::Colour(0xffaaaaaa));
    g.drawText("Source:", 4, y, w / 3, row_height, juce::Justification::centredLeft);
    g.setColour(juce::Colour(0xff00e676));
    g.drawText(m_source_label, 4, y + row_height, w - 8, row_height,
               juce::Justification::centredLeft);
}

} // namespace streamgen
