#include "OperatorVizComponent.h"
#include "InferenceSnapshot.h"
#include "InferenceWorker.h"

#include <cmath>

namespace streamgen {

namespace {

const juce::Colour kPanelTop{0xff0a1522};
const juce::Colour kPanelBot{0xff050810};
const juce::Colour kFrame{0xff2a6a72};
const juce::Colour kLabel{0xff88ccd8};
const juce::Colour kDim{0xff334455};
const juce::Colour kLedOn{0xff00ffcc};
const juce::Colour kLedOff{0xff1a2830};
const juce::Colour kWarn{0xffffaa44};

juce::Colour meter_colour(float level)
{
    const juce::Colour lo{0xff00cc66};
    const juce::Colour mid{0xffffff44};
    const juce::Colour hi{0xffff3333};
    if (level < 0.5f)
        return lo.interpolatedWith(mid, level * 2.f);
    return mid.interpolatedWith(hi, (level - 0.5f) * 2.f);
}

float rms_to_level(float rms)
{
    if (rms <= 1e-9f)
        return 0.f;
    const float db = 20.f * std::log10(rms);
    return juce::jlimit(0.f, 1.f, (db + 60.f) / 60.f);
}

void draw_corner_brackets(juce::Graphics& g, juce::Rectangle<float> r, juce::Colour c, float len)
{
    const float x0 = r.getX();
    const float y0 = r.getY();
    const float x1 = r.getRight();
    const float y1 = r.getBottom();
    g.setColour(c);
    g.drawLine(x0, y0, x0 + len, y0, 1.2f);
    g.drawLine(x0, y0, x0, y0 + len, 1.2f);
    g.drawLine(x1 - len, y0, x1, y0, 1.2f);
    g.drawLine(x1, y0, x1, y0 + len, 1.2f);
    g.drawLine(x0, y1 - len, x0, y1, 1.2f);
    g.drawLine(x0, y1, x0 + len, y1, 1.2f);
    g.drawLine(x1 - len, y1, x1, y1, 1.2f);
    g.drawLine(x1, y1 - len, x1, y1, 1.2f);
}

} // namespace

OperatorVizComponent::OperatorVizComponent()
{
    setOpaque(false);
}

void OperatorVizComponent::update_from(StreamGenProcessor& processor, InferenceWorker* worker)
{
    AudioThreadTelemetry::Snapshot audio;
    processor.audio_telemetry().copy_snapshot(audio);
    m_input_level = rms_to_level(static_cast<float>(audio.ema_input_rms));
    m_output_level = rms_to_level(static_cast<float>(audio.ema_output_rms));

    m_audio_alive = audio.callback_count > 0;

    auto& sched = processor.scheduler();
    m_generation_enabled = sched.generation_enabled.load(std::memory_order_relaxed);
    m_simulation_playing = processor.simulation_playing.load(std::memory_order_relaxed);
    m_queue_depth = sched.status.queue_depth.load(std::memory_order_relaxed);
    m_last_job_id = sched.status.last_job_id.load(std::memory_order_relaxed);
    m_gen_count = static_cast<int>(sched.status.generation_count.load(std::memory_order_relaxed));

    if (worker != nullptr && worker->is_loaded())
    {
        m_pipeline_loaded = true;
        m_worker_busy = sched.status.worker_busy.load(std::memory_order_relaxed);
        InferenceSnapshot snap = worker->last_snapshot();
        const auto& tm = snap.timing;
        m_wall_ms = static_cast<float>(snap.wall_clock_ms);
        const double win_sec = processor.constants().window_seconds();
        const double total_s = tm.total_ms / 1000.0;
        m_rt_factor = static_cast<float>(total_s > 0.0 ? win_sec / total_s : 0.0);

        m_stage_ms[0] = static_cast<float>(tm.vae_encode_ms);
        m_stage_ms[1] = static_cast<float>(tm.t5_encode_ms);
        m_stage_ms[2] = static_cast<float>(tm.conditioning_ms);
        m_stage_ms[3] = static_cast<float>(tm.sampling_total_ms);
        m_stage_ms[4] = static_cast<float>(tm.vae_decode_ms);
        m_stage_total_ms = m_stage_ms[0] + m_stage_ms[1] + m_stage_ms[2] + m_stage_ms[3] + m_stage_ms[4];
        if (m_stage_total_ms < 1.f)
            m_stage_total_ms = 1.f;

        const int n = juce::jmin(64, static_cast<int>(tm.sampling_step_ms.size()));
        m_step_count = n;
        for (int i = 0; i < n; ++i)
            m_step_dt_ms[static_cast<size_t>(i)] = static_cast<float>(tm.sampling_step_ms[static_cast<size_t>(i)]);
    }
    else
    {
        m_pipeline_loaded = false;
        m_worker_busy = false;
        m_wall_ms = 0.f;
        m_rt_factor = 0.f;
        m_stage_total_ms = 1.f;
        for (auto& s : m_stage_ms)
            s = 0.f;
        m_step_count = 0;
    }

    m_pulse += 0.15;
    if (m_pulse > 6.28318530718)
        m_pulse -= 6.28318530718;

    repaint();
}

void OperatorVizComponent::paint(juce::Graphics& g)
{
    auto bounds = getLocalBounds().toFloat().reduced(4.f);

    juce::ColourGradient grad(kPanelTop, bounds.getX(), bounds.getY(),
                              kPanelBot, bounds.getX(), bounds.getBottom(), false);
    g.setGradientFill(grad);
    g.fillRoundedRectangle(bounds, 6.f);

    g.setColour(kFrame.withAlpha(0.35f));
    g.drawRoundedRectangle(bounds, 6.f, 1.f);
    draw_corner_brackets(g, bounds, kFrame.withAlpha(0.85f), 12.f);

    const int title_h = 22;
    auto row = bounds.removeFromTop(static_cast<float>(title_h));
    g.setColour(kLabel);
    g.setFont(juce::Font(juce::Font::getDefaultSansSerifFontName(), 15.f, juce::Font::bold));
    g.drawText("OPERATOR BRIDGE — STREAMGEN", row.removeFromLeft(row.getWidth() * 0.55f),
               juce::Justification::centredLeft, true);

    juce::String rt_txt = "RT factor — ";
    rt_txt << juce::String(m_rt_factor, 2) << "x   wall ";
    rt_txt << juce::String(m_wall_ms, 0) << " ms";
    g.setColour(kWarn.withAlpha(0.95f));
    g.setFont(juce::Font(juce::Font::getDefaultSansSerifFontName(), 13.f, juce::Font::plain));
    g.drawText(rt_txt, row, juce::Justification::centredRight, true);

    bounds.removeFromTop(6.f);

    const float meter_w = 36.f;
    const float gap = 14.f;
    const float meter_h = juce::jmax(80.f, bounds.getHeight() * 0.42f);

    auto meter_row = bounds.removeFromTop(meter_h);
    auto meter_area = meter_row.removeFromLeft(meter_w * 2.f + gap);

    auto in_col = meter_area.removeFromLeft(meter_w);
    auto out_col = meter_area.removeFromLeft(meter_w);
    meter_area.removeFromLeft(gap);

    auto draw_meter = [&](juce::Rectangle<float> col, float level, const juce::String& name)
    {
        g.setColour(kDim);
        g.fillRoundedRectangle(col, 3.f);
        const float fill_h = col.getHeight() * level;
        auto fill = col.withTop(col.getBottom() - fill_h);
        juce::ColourGradient mgrad(meter_colour(level).withAlpha(0.95f), fill.getX(), fill.getBottom(),
                                   meter_colour(level).withAlpha(0.25f), fill.getX(), fill.getY(), false);
        g.setGradientFill(mgrad);
        g.fillRoundedRectangle(fill, 3.f);
        g.setColour(kFrame);
        g.drawRoundedRectangle(col, 3.f, 1.f);
        g.setColour(kLabel);
        g.setFont(11.f);
        g.drawText(name, col.translated(0, col.getHeight() + 4).withHeight(14.f),
                   juce::Justification::centred, true);
    };

    draw_meter(in_col, m_input_level, "IN");
    draw_meter(out_col, m_output_level, "OUT");

    // Status lamps
    auto lamp_area = meter_row;
    const float lamp_r = 7.f;
    const float lamp_sp = 6.f;
    auto draw_lamp = [&](float x, float y, bool on, const juce::String& lab)
    {
        const float pulse = m_worker_busy && lab == "RUN" ? (0.65f + 0.35f * static_cast<float>(std::sin(m_pulse))) : 1.f;
        juce::Colour c = on ? kLedOn : kLedOff;
        if (on && lab == "RUN")
            c = c.withMultipliedBrightness(pulse);
        g.setColour(c.withAlpha(on ? 0.95f : 0.35f));
        g.fillEllipse(x - lamp_r, y - lamp_r, lamp_r * 2.f, lamp_r * 2.f);
        g.setColour(kFrame.withAlpha(0.8f));
        g.drawEllipse(x - lamp_r, y - lamp_r, lamp_r * 2.f, lamp_r * 2.f, 1.f);
        g.setColour(kLabel.withAlpha(0.9f));
        g.setFont(10.f);
        g.drawText(lab, juce::Rectangle<float>(x - 28.f, y + lamp_r + 2.f, 56.f, 12.f),
                   juce::Justification::centred, true);
    };

    const float lx0 = lamp_area.getX() + 24.f;
    const float ly = lamp_area.getY() + lamp_area.getHeight() * 0.35f;
    float lx = lx0;
    draw_lamp(lx, ly, m_audio_alive, "SIG");
    lx += 56.f + lamp_sp;
    draw_lamp(lx, ly, m_pipeline_loaded, "LNK");
    lx += 56.f + lamp_sp;
    draw_lamp(lx, ly, m_generation_enabled, "ARM");
    lx += 56.f + lamp_sp;
    draw_lamp(lx, ly, m_worker_busy, "RUN");
    lx += 56.f + lamp_sp;
    draw_lamp(lx, ly, m_simulation_playing, "SIM");

    g.setColour(kDim.withAlpha(0.9f));
    g.setFont(10.f);
    g.drawText("GEN #" + juce::String(m_gen_count) + "  Q:" + juce::String(m_queue_depth)
                   + "  JOB:" + juce::String(static_cast<int>(m_last_job_id)),
               lamp_area.removeFromBottom(18.f), juce::Justification::centredLeft, true);

    bounds.removeFromTop(8.f);

    // Pipeline stage strip
    auto strip = bounds.removeFromTop(52.f);
    g.setColour(kLabel.withAlpha(0.85f));
    g.setFont(11.f);
    g.drawText("pipeline ms (last job)", strip.removeFromTop(12.f), juce::Justification::centredLeft, true);

    const char* labels[] = {"VAEe", "T5", "COND", "DiT", "VAEd"};
    const juce::Colour cols[] = {
        juce::Colour(0xff4488cc),
        juce::Colour(0xff66ccaa),
        juce::Colour(0xffccaa44),
        juce::Colour(0xffcc66ff),
        juce::Colour(0xff44cc88),
    };

    auto bar_row = strip;
    const float bw = (bar_row.getWidth() - 8.f * 4.f) / 5.f;
    for (int i = 0; i < 5; ++i)
    {
        auto cell = bar_row.removeFromLeft(bw);
        if (i < 4)
            bar_row.removeFromLeft(8.f);

        auto label_rect = cell.removeFromBottom(10.f);
        const float frac = m_stage_ms[static_cast<size_t>(i)] / m_stage_total_ms;
        auto inner = cell.reduced(1.f, 4.f);
        g.setColour(kDim);
        g.fillRoundedRectangle(inner, 3.f);
        auto fill = inner.withHeight(inner.getHeight() * frac);
        fill = fill.withBottom(inner.getBottom());
        g.setColour(cols[static_cast<size_t>(i)].withAlpha(0.92f));
        g.fillRoundedRectangle(fill, 3.f);
        g.setColour(kFrame.withAlpha(0.6f));
        g.drawRoundedRectangle(inner, 3.f, 1.f);
        g.setColour(kLabel);
        g.setFont(10.f);
        g.drawText(labels[static_cast<size_t>(i)], label_rect, juce::Justification::centred, true);
        g.drawText(juce::String(m_stage_ms[static_cast<size_t>(i)], 0),
                   inner, juce::Justification::centred, true);
    }

    bounds.removeFromTop(6.f);

    // Diffusion step spark
    auto spark = bounds.removeFromTop(juce::jmax(28.f, bounds.getHeight() * 0.35f));
    g.setColour(kLabel.withAlpha(0.85f));
    g.setFont(11.f);
    g.drawText("DiT step dt (ms)", spark.removeFromTop(12.f), juce::Justification::centredLeft, true);

    if (m_step_count > 0)
    {
        float max_dt = 1.f;
        for (int i = 0; i < m_step_count; ++i)
            max_dt = juce::jmax(max_dt, m_step_dt_ms[static_cast<size_t>(i)]);
        auto spark_row = spark.reduced(2.f, 2.f);
        const float step_w = (spark_row.getWidth() - static_cast<float>(m_step_count - 1) * 2.f)
            / static_cast<float>(m_step_count);
        for (int i = 0; i < m_step_count; ++i)
        {
            const float dt = m_step_dt_ms[static_cast<size_t>(i)];
            const float h = spark_row.getHeight() * (dt / max_dt);
            auto cell = spark_row.withX(spark_row.getX() + static_cast<float>(i) * (step_w + 2.f)).withWidth(step_w);
            auto bar = cell.withTop(cell.getBottom() - h);
            g.setColour(kLedOn.withAlpha(0.35f));
            g.fillRect(cell);
            g.setColour(kLedOn.withAlpha(0.95f));
            g.fillRoundedRectangle(bar.reduced(1.f, 0.f), 2.f);
        }
    }
    else
    {
        g.setColour(kDim);
        g.setFont(11.f);
        g.drawText("— awaiting inference —", spark, juce::Justification::centred, true);
    }

    // Subtle scanlines (CRT-ish)
    g.setColour(juce::Colours::white.withAlpha(0.03f));
    for (float y = 4.f; y < static_cast<float>(getHeight()) - 4.f; y += 4.f)
        g.drawHorizontalLine(static_cast<int>(y), 4.f, static_cast<float>(getWidth()) - 4.f);
}

} // namespace streamgen
