#include "WaveformTimelineComponent.h"
#include "MusicalTime.h"
#include "LayerCakeLookAndFeel.h"

#include <algorithm>
#include <cmath>
#include <string>

namespace streamgen {

namespace {

float sample_to_x(
    int64_t sample,
    int64_t abs_pos,
    int sample_rate,
    float visible_seconds,
    float width)
{
    const double now_sec = static_cast<double>(abs_pos) / static_cast<double>(sample_rate);
    const double vis = static_cast<double>(visible_seconds);
    const double past_frac = static_cast<double>(k_timeline_playhead_past_fraction);
    double start_sec = now_sec - past_frac * vis;
    double t_sec = static_cast<double>(sample) / static_cast<double>(sample_rate);
    double frac = (t_sec - start_sec) / vis;
    if (frac < 0.0)
        frac = 0.0;
    if (frac > 1.0)
        frac = 1.0;
    return static_cast<float>(frac * width);
}

juce::Colour palette_fill_colour(int64_t job_id, juce::LookAndFeel& lf)
{
    if (auto* lc = dynamic_cast<LayerCakeLookAndFeel*>(&lf))
        return lc->getLayerColour(static_cast<size_t>(static_cast<uint64_t>(job_id < 0 ? -job_id : job_id)))
            .brighter(0.55f)
            .withAlpha(0.24f);
    static const juce::uint32 hex[] = {
        0xffe53935, 0xfffb8c00, 0xfffdd835, 0xff43a047, 0xff00897b,
        0xff039be5, 0xff3949ab, 0xff8e24aa, 0xffd81b60, 0xff6d4c41,
        0xff546e7a, 0xff78909c,
    };
    size_t i = static_cast<size_t>(static_cast<uint64_t>(job_id < 0 ? -job_id : job_id) % 12u);
    return juce::Colour(hex[i]).withAlpha(0.22f);
}

juce::Colour palette_edge_colour(int64_t job_id, juce::LookAndFeel& lf)
{
    if (auto* lc = dynamic_cast<LayerCakeLookAndFeel*>(&lf))
        return lc->getLayerColour(static_cast<size_t>(static_cast<uint64_t>(job_id < 0 ? -job_id : job_id)))
            .brighter(0.35f)
            .withAlpha(0.95f);
    static const juce::uint32 hex[] = {
        0xffe53935, 0xfffb8c00, 0xfffdd835, 0xff43a047, 0xff00897b,
        0xff039be5, 0xff3949ab, 0xff8e24aa, 0xffd81b60, 0xff6d4c41,
        0xff546e7a, 0xff78909c,
    };
    size_t i = static_cast<size_t>(static_cast<uint64_t>(job_id < 0 ? -job_id : job_id) % 12u);
    return juce::Colour(hex[i]).withAlpha(0.95f);
}

juce::String format_latency(double inference_ms)
{
    if (inference_ms >= 1000.0)
        return juce::String(inference_ms / 1000.0, 1) + "s";
    return juce::String(static_cast<int>(inference_ms + 0.5)) + "ms";
}

juce::String format_clock_ms(int64_t system_ms_since_epoch)
{
    juce::Time t(static_cast<juce::int64>(system_ms_since_epoch));
    return t.formatted("%H:%M:%S");
}

/// Job timeline coloured regions use thin horizontal bands (not full waveform height) so
/// overlapping jobs (e.g. one job's generated span as the next job's input) stay readable.
constexpr float k_job_band_height_fraction = 0.16f;
constexpr float k_job_band_min_px = 12.0f;
constexpr float k_job_band_gap_px = 2.0f;

/// Drums lane: [keep] band height between [input] and [generated] (bottom band).
constexpr float k_keep_band_height_fraction = 0.12f;
constexpr float k_keep_band_min_px = 10.0f;

float job_highlight_band_height(float waveform_height)
{
    return juce::jmax(k_job_band_min_px, waveform_height * k_job_band_height_fraction);
}

float keep_highlight_band_height(float waveform_height)
{
    return juce::jmax(k_keep_band_min_px, waveform_height * k_keep_band_height_fraction);
}

void draw_region_tag(juce::Graphics& g, float x, float y, const juce::String& text, juce::Colour fg, juce::Colour tag_bg)
{
    juce::Font font(juce::FontOptions(juce::Font::getDefaultMonospacedFontName(), 10.0f, juce::Font::bold));
    g.setFont(font);
    const float pad = 3.0f;
    const float tw = static_cast<float>(font.getStringWidth(text)) + pad * 2.0f;
    const float th = 14.0f;
    g.setColour(tag_bg);
    g.fillRoundedRectangle(x, y, tw, th, 3.0f);
    g.setColour(fg);
    g.drawText(text, juce::roundToInt(x + pad), juce::roundToInt(y), juce::roundToInt(tw), juce::roundToInt(th),
               juce::Justification::centredLeft);
}

void draw_keep_boundary_ticks(
    juce::Graphics& g,
    float x_ke,
    float waveform_top,
    float waveform_height,
    juce::Colour edge)
{
    g.setColour(edge.withAlpha(0.7f));
    for (float y = waveform_top; y < waveform_top + waveform_height; y += 8.0f)
        g.drawLine(x_ke, y, x_ke, y + 4.0f, 1.0f);
}

/// Where new drums audio lands: output_start_sample() (keep_end + schedule delay). Wall clock when write finished.
void draw_generation_land_marker(
    juce::Graphics& g,
    float x,
    float wf_top,
    float wf_h,
    float panel_w,
    const JobTimelineRecord& rec,
    int sample_rate,
    bool musical_ruler,
    float bpm,
    int beats_per_bar,
    juce::LookAndFeel& lf)
{
    if (!rec.has_completed || rec.gen_samples <= 0)
        return;

    juce::Colour accent = palette_edge_colour(rec.job_id, lf);
    const juce::Colour terminal = lf.findColour(juce::Label::textColourId);
    const juce::Colour panel_fill = lf.findColour(juce::ComboBox::backgroundColourId);
    const float y_line_top = wf_top + 14.0f;
    const float y_line_bot = wf_top + wf_h - 3.0f;
    g.setColour(accent.withAlpha(0.95f));
    g.drawLine(x, y_line_top, x, y_line_bot, 3.0f);

    juce::Path head;
    const float tip_y = wf_top + 8.0f;
    head.addTriangle(x, tip_y - 6.0f, x - 6.0f, tip_y + 2.0f, x + 6.0f, tip_y + 2.0f);
    g.setColour(accent.brighter(0.15f));
    g.fillPath(head);
    g.setColour(terminal.withAlpha(0.9f));
    g.strokePath(head, juce::PathStrokeType(1.0f));

    const std::string t_samples = format_time(rec.job.output_start_sample(), sample_rate);
    juce::String line_a = "LAND #" + juce::String(rec.job_id) + " @ " + juce::String(t_samples);
    if (musical_ruler && beats_per_bar >= 1 && bpm > 0.0f)
    {
        const double bpm_d = static_cast<double>(juce::jlimit(20.0f, 400.0f, bpm));
        line_a += "  ";
        line_a += juce::String(format_bar_beat(rec.job.output_start_sample(), sample_rate, bpm_d, beats_per_bar));
    }
    juce::String line_b = "wall " + format_clock_ms(rec.completed_system_ms);

    juce::Font lbl_font(juce::FontOptions(juce::Font::getDefaultMonospacedFontName(), 9.0f, juce::Font::bold));
    g.setFont(lbl_font);
    const float tw = juce::jmax(160.0f, static_cast<float>(lbl_font.getStringWidth(line_a)) + 12.0f);
    float tx = x + 8.0f;
    if (tx + tw > panel_w - 4.0f)
        tx = x - tw - 6.0f;
    if (tx < 4.0f)
        tx = 4.0f;
    const float ty = wf_top + wf_h * 0.52f;
    g.setColour(panel_fill.withAlpha(0.88f));
    g.fillRoundedRectangle(tx, ty, tw, 30.0f, 4.0f);
    g.setColour(accent.brighter(0.25f));
    g.drawRoundedRectangle(tx, ty, tw, 30.0f, 4.0f, 1.0f);
    g.setColour(terminal.withAlpha(0.95f));
    g.drawText(line_a, juce::roundToInt(tx + 4), juce::roundToInt(ty + 3),
               juce::roundToInt(tw - 8), 12, juce::Justification::centredLeft);
    g.setFont(
        juce::Font(juce::FontOptions(juce::Font::getDefaultMonospacedFontName(), 8.5f, juce::Font::plain)));
    g.setColour(terminal.withAlpha(0.75f));
    g.drawText(line_b, juce::roundToInt(tx + 4), juce::roundToInt(ty + 15),
               juce::roundToInt(tw - 8), 12, juce::Justification::centredLeft);
}

} // namespace

WaveformTimelineComponent::WaveformTimelineComponent()
{
    setOpaque(true);
}

void WaveformTimelineComponent::update(
    const float* min_per_px,
    const float* max_per_px,
    int num_px,
    int64_t absolute_pos,
    int sample_rate,
    const std::vector<JobTimelineRecord>* timeline_jobs,
    TimelineWaveRole role,
    const float* drums_warm_min,
    const float* drums_warm_max,
    const float* drums_gen_min,
    const float* drums_gen_max)
{
    if (num_px <= 0 || min_per_px == nullptr || max_per_px == nullptr)
    {
        m_min_px.clear();
        m_max_px.clear();
        m_drums_warm_min_px.clear();
        m_drums_warm_max_px.clear();
        m_drums_gen_min_px.clear();
        m_drums_gen_max_px.clear();
        m_drums_source_split = false;
    }
    else
    {
        m_min_px.assign(min_per_px, min_per_px + static_cast<size_t>(num_px));
        m_max_px.assign(max_per_px, max_per_px + static_cast<size_t>(num_px));
        const bool want_split = role == TimelineWaveRole::DrumsOutput
            && drums_warm_min != nullptr && drums_warm_max != nullptr
            && drums_gen_min != nullptr && drums_gen_max != nullptr;
        if (want_split)
        {
            m_drums_warm_min_px.assign(drums_warm_min, drums_warm_min + static_cast<size_t>(num_px));
            m_drums_warm_max_px.assign(drums_warm_max, drums_warm_max + static_cast<size_t>(num_px));
            m_drums_gen_min_px.assign(drums_gen_min, drums_gen_min + static_cast<size_t>(num_px));
            m_drums_gen_max_px.assign(drums_gen_max, drums_gen_max + static_cast<size_t>(num_px));
            m_drums_source_split = true;
        }
        else
        {
            m_drums_warm_min_px.clear();
            m_drums_warm_max_px.clear();
            m_drums_gen_min_px.clear();
            m_drums_gen_max_px.clear();
            m_drums_source_split = false;
        }
    }
    m_absolute_pos = absolute_pos;
    m_sample_rate = sample_rate;
    m_timeline_jobs = timeline_jobs;
    m_timeline_role = role;
    repaint();
}

void WaveformTimelineComponent::set_time_axis_for_paint(
    bool musical,
    float bpm,
    int beats_per_bar,
    int time_sig_denominator)
{
    m_paint_musical = musical;
    m_paint_bpm = bpm;
    m_paint_beats_per_bar = juce::jmax(1, beats_per_bar);
    m_paint_time_sig_d = juce::jmax(1, time_sig_denominator);
}

juce::String WaveformTimelineComponent::tooltip_for_timeline_x(int x) const
{
    const int w = getWidth();
    if (w <= 0 || m_sample_rate <= 0)
        return {};

    const double now_sec = static_cast<double>(m_absolute_pos) / static_cast<double>(m_sample_rate);
    const double vis = static_cast<double>(m_visible_seconds);
    const double past = static_cast<double>(k_timeline_playhead_past_fraction);
    const double start_sec = now_sec - past * vis;
    const double frac = static_cast<double>(x) / static_cast<double>(w);
    const double t_sec = start_sec + frac * vis;
    const int64_t smp = static_cast<int64_t>(std::llround(t_sec * static_cast<double>(m_sample_rate)));

    juce::String tip = juce::String(format_time(smp, m_sample_rate));
    if (m_paint_musical)
    {
        const double bpm_d = static_cast<double>(juce::jlimit(20.0f, 400.0f, m_paint_bpm));
        tip += "  ";
        tip += juce::String(format_bar_beat(smp, m_sample_rate, bpm_d, m_paint_beats_per_bar));
    }
    return tip;
}

void WaveformTimelineComponent::mouseMove(const juce::MouseEvent& event)
{
    const int h = getHeight();
    const int timeline_height = 16;
    const int timeline_y = h - timeline_height;
    if (event.y >= timeline_y)
        setTooltip(tooltip_for_timeline_x(event.x));
    else
        setTooltip({});

    Component::mouseMove(event);
}

void WaveformTimelineComponent::mouseExit(const juce::MouseEvent& event)
{
    setTooltip({});
    Component::mouseExit(event);
}

void WaveformTimelineComponent::paint(juce::Graphics& g)
{
    auto bounds = getLocalBounds();
    auto& lf = getLookAndFeel();
    const juce::Colour panel = lf.findColour(juce::ComboBox::backgroundColourId);
    const juce::Colour border = lf.findColour(juce::ComboBox::outlineColourId);
    const juce::Colour terminal = lf.findColour(juce::Label::textColourId);
    juce::Colour accent_green(0xff3cff9f);
    juce::Colour accent_cyan(0xff35c0ff);
    juce::Colour accent_amber(0xfff2b950);
    juce::Colour wave_line = accent_cyan;
    if (auto* lc = dynamic_cast<LayerCakeLookAndFeel*>(&lf))
    {
        accent_green = lc->getControlAccentColour(LayerCakeLookAndFeel::ControlButtonType::Clock);
        accent_cyan = lc->getControlAccentColour(LayerCakeLookAndFeel::ControlButtonType::Trigger);
        accent_amber = lc->getControlAccentColour(LayerCakeLookAndFeel::ControlButtonType::Pattern);
        wave_line = lc->getWaveformColour();
    }
    const juce::Colour tag_backdrop = juce::Colours::black.withAlpha(0.55f);

    g.fillAll(panel);

    const int w = bounds.getWidth();
    const int h = bounds.getHeight();

    const int label_height = 18;
    const int timeline_height = 16;
    const int waveform_top = label_height;
    const int waveform_height = h - label_height - timeline_height;
    const int timeline_y = h - timeline_height;

    g.setColour(terminal);
    g.setFont(juce::Font(juce::FontOptions(juce::Font::getDefaultMonospacedFontName(), 13.0f, juce::Font::plain)));
    g.drawText(m_label, 4, 0, w / 2, label_height, juce::Justification::centredLeft);

    if (m_source_tag.isNotEmpty())
    {
        g.setColour(accent_green);
        g.drawText(m_source_tag, w / 2, 0, w / 2 - 4, label_height, juce::Justification::centredRight);
    }

    if (w <= 0 || waveform_height <= 0)
        return;

    const float mid_y = static_cast<float>(waveform_top + waveform_height / 2);
    const float amp = static_cast<float>(waveform_height) * 0.45f;

    const int num_buckets = static_cast<int>(std::min(m_min_px.size(), m_max_px.size()));

    g.setColour(border.withAlpha(0.75f));
    g.drawHorizontalLine(static_cast<int>(mid_y), 0.0f, static_cast<float>(w));

    // --- Timeline overlays: [input] / [keep] / [generated] tags + thin band fills ---
    if (m_timeline_jobs != nullptr)
    {
        const float wf_top = static_cast<float>(waveform_top);
        const float wf_h = static_cast<float>(waveform_height);
        const float band = job_highlight_band_height(wf_h);
        const float keep_band = keep_highlight_band_height(wf_h);
        const float gen_y = wf_top + wf_h - band;

        int timeline_job_index = 0;

        for (const auto& rec : *m_timeline_jobs)
        {
            juce::Colour fill = palette_fill_colour(rec.job_id, lf);
            juce::Colour edge = palette_edge_colour(rec.job_id, lf);

            const int64_t ws = rec.job.window_start_sample;
            const int64_t ke = rec.job.keep_end_sample;
            const int64_t we = rec.job.window_end_sample;
            const int64_t gen_len_samp = we - ke;
            const int64_t land_start = rec.job.output_start_sample();
            const int64_t land_end = land_start + gen_len_samp;

            const float x_ws = sample_to_x(ws, m_absolute_pos, m_sample_rate, m_visible_seconds, static_cast<float>(w));
            const float x_ke = sample_to_x(ke, m_absolute_pos, m_sample_rate, m_visible_seconds, static_cast<float>(w));
            const float x_we = sample_to_x(we, m_absolute_pos, m_sample_rate, m_visible_seconds, static_cast<float>(w));
            const float input_lo = std::min(x_ws, x_ke);
            const float input_hi = std::max(x_ws, x_ke);
            const float x_land_start = sample_to_x(land_start, m_absolute_pos, m_sample_rate, m_visible_seconds, static_cast<float>(w));
            const float x_land_end = sample_to_x(land_end, m_absolute_pos, m_sample_rate, m_visible_seconds, static_cast<float>(w));
            const float drums_gen_lo = std::min(x_land_start, x_land_end);
            const float drums_gen_hi = std::max(x_land_start, x_land_end);
            const float x_boundary_ke = x_ke;

            const float keep_y = wf_top + band + k_job_band_gap_px;

            if (m_timeline_role == TimelineWaveRole::SaxInput)
            {
                if (input_hi > input_lo)
                {
                    g.setColour(fill);
                    g.fillRect(input_lo, wf_top, input_hi - input_lo, band);
                    draw_region_tag(g, input_lo + 4.0f, wf_top + 2.0f, "[input]", terminal.withAlpha(0.95f), tag_backdrop);
                }
                draw_keep_boundary_ticks(g, x_boundary_ke, wf_top, wf_h, edge);
            }
            else
            {
                if (input_hi > input_lo)
                {
                    g.setColour(fill.withAlpha(0.22f));
                    g.fillRect(input_lo, wf_top, input_hi - input_lo, band);
                    draw_region_tag(g, input_lo + 4.0f, wf_top + 2.0f, "[input]", terminal.withAlpha(0.95f), tag_backdrop);

                    g.setColour(fill);
                    g.fillRect(input_lo, keep_y, input_hi - input_lo, keep_band);
                    draw_region_tag(g, input_lo + 4.0f, keep_y + 2.0f, "[keep]", terminal.withAlpha(0.95f), tag_backdrop);
                }
                if (drums_gen_hi > drums_gen_lo)
                {
                    g.setColour(fill);
                    g.fillRect(drums_gen_lo, gen_y, drums_gen_hi - drums_gen_lo, band);
                    draw_region_tag(g, drums_gen_lo + 4.0f, gen_y + 2.0f, "[generated]", terminal.withAlpha(0.95f), tag_backdrop);
                }
                draw_keep_boundary_ticks(g, x_boundary_ke, wf_top, wf_h, edge);
            }

            juce::String lbl = "#" + juce::String(rec.job_id);
            if (m_timeline_role == TimelineWaveRole::SaxInput)
                lbl += " @" + format_clock_ms(rec.scheduled_system_ms);
            else if (rec.has_completed && rec.gen_samples > 0)
            {
                lbl += " inf " + format_latency(rec.inference_ms);
                if (rec.completed_steady_ns > rec.scheduled_steady_ns)
                {
                    double arr_ms = static_cast<double>(rec.completed_steady_ns - rec.scheduled_steady_ns) / 1.0e6;
                    lbl += " arr " + format_latency(arr_ms);
                }
            }
            else
                lbl += " @" + format_clock_ms(rec.scheduled_system_ms);

            const int meta_y = waveform_top + static_cast<int>(band) + 6 + timeline_job_index * 15;
            g.setColour(terminal.withAlpha(0.85f));
            g.setFont(juce::Font(juce::FontOptions(juce::Font::getDefaultMonospacedFontName(), 10.0f, juce::Font::plain)));
            g.drawText(lbl, static_cast<int>(input_lo) + 2, meta_y, 280, 14, juce::Justification::centredLeft);
            ++timeline_job_index;
        }
    }

    // --- Waveform: pre-bucketed min/max per column (see fill_recent_*_waveform_buckets) ---
    if (num_buckets > 0)
    {
        const int n = juce::jmin(w, num_buckets);
        const float column_w = static_cast<float>(w) / static_cast<float>(juce::jmax(1, n));

        const bool split_drums = m_drums_source_split && m_timeline_role == TimelineWaveRole::DrumsOutput
            && static_cast<int>(m_drums_warm_min_px.size()) >= n
            && static_cast<int>(m_drums_warm_max_px.size()) >= n
            && static_cast<int>(m_drums_gen_min_px.size()) >= n
            && static_cast<int>(m_drums_gen_max_px.size()) >= n;

        float peak_abs = 0.0f;
        if (split_drums)
        {
            for (int px = 0; px < n; ++px)
            {
                const size_t p = static_cast<size_t>(px);
                peak_abs = std::max(peak_abs, std::abs(m_drums_warm_min_px[p]));
                peak_abs = std::max(peak_abs, std::abs(m_drums_warm_max_px[p]));
                peak_abs = std::max(peak_abs, std::abs(m_drums_gen_min_px[p]));
                peak_abs = std::max(peak_abs, std::abs(m_drums_gen_max_px[p]));
            }
        }
        else
        {
            for (int px = 0; px < n; ++px)
            {
                peak_abs = std::max(peak_abs, std::abs(m_min_px[static_cast<size_t>(px)]));
                peak_abs = std::max(peak_abs, std::abs(m_max_px[static_cast<size_t>(px)]));
            }
        }
        const float display_gain = std::min(1.0f / std::max(peak_abs, 1.0e-6f), 8.0f);

        if (split_drums)
        {
            const juce::Colour colour_warm = accent_amber;
            const juce::Colour colour_gen = accent_cyan;
            juce::Path wave_warm;
            for (int px = 0; px < n; ++px)
            {
                const size_t p = static_cast<size_t>(px);
                float mn = m_drums_warm_min_px[p];
                float mx = m_drums_warm_max_px[p];
                float peak = std::max(std::abs(mn), std::abs(mx));
                peak = std::max(0.0f, std::min(1.0f, peak * display_gain));
                const float y_hi = mid_y - peak * amp;
                const float y_lo = mid_y + peak * amp;
                const float x = (static_cast<float>(px) + 0.5f) * column_w;
                wave_warm.startNewSubPath(x, y_hi);
                wave_warm.lineTo(x, y_lo);
            }
            juce::Path wave_gen;
            for (int px = 0; px < n; ++px)
            {
                const size_t p = static_cast<size_t>(px);
                float mn = m_drums_gen_min_px[p];
                float mx = m_drums_gen_max_px[p];
                float peak = std::max(std::abs(mn), std::abs(mx));
                peak = std::max(0.0f, std::min(1.0f, peak * display_gain));
                const float y_hi = mid_y - peak * amp;
                const float y_lo = mid_y + peak * amp;
                const float x = (static_cast<float>(px) + 0.5f) * column_w;
                wave_gen.startNewSubPath(x, y_hi);
                wave_gen.lineTo(x, y_lo);
            }
            g.setColour(colour_warm);
            g.strokePath(wave_warm, juce::PathStrokeType(1.0f));
            g.setColour(colour_gen);
            g.strokePath(wave_gen, juce::PathStrokeType(1.0f));
        }
        else
        {
            const juce::Colour colour_wave = wave_line;
            juce::Path wave;
            for (int px = 0; px < n; ++px)
            {
                float mn = m_min_px[static_cast<size_t>(px)];
                float mx = m_max_px[static_cast<size_t>(px)];
                float peak = std::max(std::abs(mn), std::abs(mx));
                peak = std::max(0.0f, std::min(1.0f, peak * display_gain));
                const float y_hi = mid_y - peak * amp;
                const float y_lo = mid_y + peak * amp;
                const float x = (static_cast<float>(px) + 0.5f) * column_w;
                wave.startNewSubPath(x, y_hi);
                wave.lineTo(x, y_lo);
            }
            g.setColour(colour_wave);
            g.strokePath(wave, juce::PathStrokeType(1.0f));
        }
    }

    if (m_timeline_jobs != nullptr && m_timeline_role == TimelineWaveRole::DrumsOutput)
    {
        const float wf_top_f = static_cast<float>(waveform_top);
        const float wf_h_f = static_cast<float>(waveform_height);
        for (const auto& rec : *m_timeline_jobs)
        {
            if (!rec.has_completed || rec.gen_samples <= 0)
                continue;
            const float x_land = sample_to_x(rec.job.output_start_sample(), m_absolute_pos, m_sample_rate,
                                             m_visible_seconds, static_cast<float>(w));
            draw_generation_land_marker(g, x_land, wf_top_f, wf_h_f, static_cast<float>(w), rec, m_sample_rate,
                                        m_paint_musical, m_paint_bpm, m_paint_beats_per_bar, lf);
        }
    }

    const float x_now = static_cast<float>(k_timeline_playhead_past_fraction) * static_cast<float>(w);
    juce::Colour playhead_fill = accent_cyan;
    juce::Colour playhead_edge = accent_cyan.brighter(0.28f);
    // Playhead: line through waveform + triangle marker at top + cap on time ruler
    g.setColour(playhead_fill.withAlpha(0.92f));
    g.drawLine(x_now, static_cast<float>(waveform_top), x_now,
               static_cast<float>(waveform_top + waveform_height), 2.0f);
    {
        juce::Path notch;
        const float tip_y = static_cast<float>(waveform_top);
        notch.addTriangle(x_now - 6.0f, tip_y, x_now + 6.0f, tip_y, x_now, tip_y + 7.0f);
        g.setColour(playhead_fill);
        g.fillPath(notch);
        g.setColour(playhead_edge);
        g.strokePath(notch, juce::PathStrokeType(1.0f));
    }
    g.setColour(playhead_fill);
    g.setFont(juce::Font(juce::FontOptions(juce::Font::getDefaultMonospacedFontName(), 9.0f, juce::Font::bold)));
    g.drawText("NOW", juce::roundToInt(x_now) - 18, waveform_top + 3, 36, 12, juce::Justification::centred);

    g.setColour(panel.darker(0.35f));
    g.fillRect(0, timeline_y, w, timeline_height);

    g.setColour(terminal.withAlpha(0.55f));
    g.setFont(juce::Font(juce::FontOptions(juce::Font::getDefaultMonospacedFontName(), 10.0f, juce::Font::plain)));

    const double seconds_visible = static_cast<double>(m_visible_seconds);
    const double now_sec = samples_to_seconds(m_absolute_pos, m_sample_rate);
    const double past_frac = static_cast<double>(k_timeline_playhead_past_fraction);
    const double start_seconds = now_sec - past_frac * seconds_visible;
    const double axis_end_sec = start_seconds + seconds_visible;

    if (!m_paint_musical)
    {
        const double tick_interval = 1.0;
        double first_tick = std::ceil(start_seconds / tick_interval) * tick_interval;
        for (double t = first_tick; t <= axis_end_sec + 1e-6; t += tick_interval)
        {
            double frac = (t - start_seconds) / seconds_visible;
            int px = static_cast<int>(frac * w);
            g.drawVerticalLine(px, static_cast<float>(timeline_y),
                               static_cast<float>(timeline_y + 4));

            int total_secs = static_cast<int>(t);
            if (total_secs >= 0)
            {
                int mins = total_secs / 60;
                int secs = total_secs % 60;
                juce::String label = juce::String::formatted("%d:%02d", mins, secs);
                g.drawText(label, px - 20, timeline_y + 3, 40, 12, juce::Justification::centred);
            }
        }
    }
    else
    {
        const double bpm_d = static_cast<double>(juce::jlimit(20.0f, 400.0f, m_paint_bpm));
        const int bpb = m_paint_beats_per_bar;
        const double beat_now = samples_to_beats(m_absolute_pos, m_sample_rate, bpm_d);
        const double vis_beats = seconds_visible * (bpm_d / 60.0);
        const double start_beat = beat_now - past_frac * vis_beats;
        const double end_beat = start_beat + vis_beats;

        const int k_start = static_cast<int>(std::ceil(start_beat - 1e-9));
        const int k_end = static_cast<int>(std::floor(end_beat + 1e-9));
        for (int kb = k_start; kb <= k_end; ++kb)
        {
            const int64_t s_tick = beats_to_samples(static_cast<double>(kb), m_sample_rate, bpm_d);
            const float x = sample_to_x(s_tick, m_absolute_pos, m_sample_rate, m_visible_seconds, static_cast<float>(w));
            const int px = juce::roundToInt(x);
            const bool is_bar_line = (bpb > 0) && (kb % bpb == 0);
            const float tick_bot = is_bar_line ? static_cast<float>(timeline_y + 6) : static_cast<float>(timeline_y + 4);
            g.drawVerticalLine(px, static_cast<float>(timeline_y), tick_bot);

            if (is_bar_line && kb >= 0)
            {
                const int bar_num = kb / bpb + 1;
                juce::String label = juce::String(bar_num);
                g.drawText(label, px - 14, timeline_y + 3, 28, 12, juce::Justification::centred);
            }
        }

        juce::String sig_h = juce::String(m_paint_beats_per_bar) + "/" + juce::String(m_paint_time_sig_d);
        g.setColour(terminal.withAlpha(0.5f));
        g.setFont(juce::Font(juce::FontOptions(juce::Font::getDefaultMonospacedFontName(), 8.5f, juce::Font::plain)));
        g.drawText(sig_h, w - 40, timeline_y + 1, 36, 12, juce::Justification::centredRight);
    }

    g.setColour(playhead_fill);
    g.drawLine(x_now, static_cast<float>(timeline_y), x_now,
               static_cast<float>(timeline_y + timeline_height - 1), 2.0f);
}

} // namespace streamgen
