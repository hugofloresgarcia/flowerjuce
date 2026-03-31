#include "OperatorDashboardWindow.h"
#include "DiagnosticLogWriter.h"
#include "InferenceSnapshot.h"
#include "InferenceWorker.h"
#include "LayerCakeLookAndFeel.h"

#include <cmath>
#include <sstream>

namespace streamgen {

static juce::String dbfs(double rms)
{
    if (rms <= 1e-20)
        return "-inf";
    const double db = 20.0 * std::log10(rms);
    return juce::String(db, 2) + " dBFS";
}

static juce::String format_dashboard(
    StreamGenProcessor& processor,
    InferenceWorker* worker)
{
    std::ostringstream out;
    out << std::fixed;

    AudioThreadTelemetry::Snapshot audio;
    processor.audio_telemetry().copy_snapshot(audio);

    auto& sched = processor.scheduler();
    const auto& st = sched.status;

    out << "=== threads ===\n";
    out << "audio callback count: " << audio.callback_count << "\n";
    out << "absolute sample pos: " << sched.absolute_sample_pos() << "\n";
    if (worker != nullptr)
    {
        out << "inference worker thread running: " << (worker->isThreadRunning() ? "yes" : "no") << "\n";
        out << "pipeline loaded: " << (worker->is_loaded() ? "yes" : "no") << "\n";
        out << "worker busy: " << (st.worker_busy.load(std::memory_order_relaxed) ? "yes" : "no") << "\n";
    }
    else
    {
        out << "inference worker: (not loaded)\n";
    }

    out << "\n=== audio I/O (ring input / device output) ===\n";
    out << "sample rate: " << processor.current_sample_rate() << " Hz\n";
    out << "last block input RMS:  " << std::setprecision(6) << audio.last_block_input_rms
        << "  (" << dbfs(audio.last_block_input_rms).toStdString() << ")\n";
    out << "EMA input RMS:         " << std::setprecision(6) << audio.ema_input_rms
        << "  (" << dbfs(audio.ema_input_rms).toStdString() << ")\n";
    out << "last block output L/R: " << audio.last_block_output_l_rms << " / "
        << audio.last_block_output_r_rms << "\n";
    out << "EMA output RMS (stereo mean): " << std::setprecision(6) << audio.ema_output_rms << "\n";
    out << "simulation playing: " << (processor.simulation_playing.load(std::memory_order_relaxed) ? "yes" : "no")
        << "  sim buffer samples: " << processor.simulation_total_samples.load(std::memory_order_relaxed)
        << "  warm route: " << (processor.warm_start_playing.load(std::memory_order_relaxed) ? "yes" : "no") << "\n";

    out << "\n=== scheduler ===\n";
    out << "musical_time: " << (sched.musical_time_enabled.load(std::memory_order_relaxed) ? "yes" : "no")
        << "  bpm: " << std::setprecision(1) << sched.bpm.load(std::memory_order_relaxed)
        << "  time_sig: " << sched.time_sig_numerator.load(std::memory_order_relaxed) << "/"
        << sched.time_sig_denominator.load(std::memory_order_relaxed) << "\n";
    out << "quantize_launch_beats: " << sched.quantize_launch_beats.load(std::memory_order_relaxed) << "\n";
    if (sched.musical_time_enabled.load(std::memory_order_relaxed))
    {
        out << "hop (bars): " << std::setprecision(2) << sched.hop_bars.load(std::memory_order_relaxed)
            << "  land delay (bars): " << sched.schedule_delay_bars.load(std::memory_order_relaxed) << "\n";
        out << "hop (beats): " << std::setprecision(3) << sched.hop_beats.load(std::memory_order_relaxed)
            << "  schedule_delay (beats): " << sched.schedule_delay_beats.load(std::memory_order_relaxed) << "\n";
    }
    else
    {
        out << "hop (s): " << std::setprecision(3) << sched.hop_seconds.load(std::memory_order_relaxed)
            << "  schedule_delay (s): " << sched.schedule_delay_seconds.load(std::memory_order_relaxed) << "\n";
    }
    out << "keep_ratio: " << sched.keep_ratio.load(std::memory_order_relaxed) << "\n";
    out << "steps: " << sched.steps.load(std::memory_order_relaxed)
        << "  cfg: " << std::setprecision(2) << sched.cfg_scale.load(std::memory_order_relaxed) << "\n";
    out << "generation enabled: " << (sched.generation_enabled.load(std::memory_order_relaxed) ? "yes" : "no") << "\n";
    out << "queue depth: " << st.queue_depth.load(std::memory_order_relaxed)
        << "  gen count: " << st.generation_count.load(std::memory_order_relaxed);
    out << "  last job id: " << st.last_job_id.load(std::memory_order_relaxed) << "\n";
    out << "last latency (ms): " << std::setprecision(2) << st.last_latency_ms.load(std::memory_order_relaxed) << "\n";

    out << "\n=== pipeline (last completed job) ===\n";
    if (worker != nullptr && worker->is_loaded())
    {
        InferenceSnapshot snap = worker->last_snapshot();
        const auto& tm = snap.timing;
        const auto& d = snap.diagnostics;
        const ModelConstants& mc = processor.constants();

        out << "job id: " << snap.job.job_id
            << "  wall_clock_ms: " << std::setprecision(2) << snap.wall_clock_ms << "\n";
        out << "window samples [ " << snap.job.window_start_sample << ", "
            << snap.job.window_end_sample << " )  keep_end: " << snap.job.keep_end_sample
            << "  output_delay_smpl: " << snap.job.output_delay_samples
            << "  land: " << snap.job.output_start_sample() << "\n";
        out << "keep_ratio: " << snap.job.keep_ratio
            << "  steps: " << snap.job.steps
            << "  cfg: " << snap.job.cfg_scale << "\n";

        const double win_sec = mc.window_seconds();
        const double total_s = tm.total_ms / 1000.0;
        out << "realtime factor (window / total): " << std::setprecision(2)
            << (total_s > 0.0 ? win_sec / total_s : 0.0) << "\n";

        out << "stage ms — vae_enc: " << std::setprecision(2) << tm.vae_encode_ms
            << "  t5: " << tm.t5_encode_ms
            << "  conditioning: " << tm.conditioning_ms
            << "  sampling: " << tm.sampling_total_ms
            << "  vae_dec: " << tm.vae_decode_ms
            << "  total: " << tm.total_ms << "\n";

        out << "DiT step ms:";
        for (size_t i = 0; i < tm.sampling_step_ms.size(); ++i)
            out << " [" << i << "]=" << tm.sampling_step_ms[i];
        out << "\n";

        out << "latent grid: " << mc.latent_dim << " x " << mc.latent_length << " (channels x frames)\n";
        out << "T5 text tokens (fixed width): " << snap.t5_sequence_length
            << "  attention mask positions: " << snap.t5_attention_nonzero_positions << "\n";

        out << "latent mean / std — streamgen: " << std::setprecision(5) << d.streamgen_latent.mean
            << " / " << d.streamgen_latent.std_dev << "\n";
        out << "latent mean / std — input (drums): " << d.input_latent.mean << " / " << d.input_latent.std_dev << "\n";
        out << "latent mean / std — noise: " << d.noise.mean << " / " << d.noise.std_dev << "\n";
        out << "latent mean / std — sampled: " << d.sampled_latent.mean << " / " << d.sampled_latent.std_dev << "\n";

        out << "\n=== diffusion (per step) ===\n";
        for (size_t i = 0; i < d.diffusion_steps.size(); ++i)
        {
            const auto& stp = d.diffusion_steps[i];
            double dt_ms = 0.0;
            if (i < tm.sampling_step_ms.size())
                dt_ms = tm.sampling_step_ms[i];
            out << "step " << stp.step_index << "  t=" << std::setprecision(5) << stp.t_curr
                << "  mean(x)=" << stp.mean_x << "  std(x)=" << stp.std_x
                << "  dt_ms=" << dt_ms << "\n";
        }
    }
    else
    {
        out << "(pipeline not loaded — no inference snapshot)\n";
    }

    out << "\n=== logging ===\n";
    out << "RT-safe path: audio callback writes AudioThreadTelemetry atomics only (no DBG).\n";
    out << "Inference: worker updates snapshot after each job; UI/logger threads read.\n";

    return juce::String(out.str());
}

OperatorDashboardPanel::OperatorDashboardPanel(
    StreamGenProcessor& processor,
    std::function<InferenceWorker*()> get_worker)
    : m_processor(processor),
      m_get_worker(std::move(get_worker))
{
    m_body.setMultiLine(true, true);
    m_body.setReadOnly(true);
    m_body.setFont(
        juce::Font(juce::FontOptions(juce::Font::getDefaultMonospacedFontName(), 13.0f, juce::Font::plain)));
    m_body.setScrollbarsShown(true);
    m_viewport.setViewedComponent(&m_body, false);
    addAndMakeVisible(m_viz);
    addAndMakeVisible(m_viewport);

    m_trace_toggle.onClick = [this]() { on_trace_toggled(); };
    addAndMakeVisible(m_trace_toggle);

    m_file_log_toggle.onClick = [this]() { on_file_log_toggled(); };
    addAndMakeVisible(m_file_log_toggle);

    LayerCakeLookAndFeel::setControlButtonType(m_choose_log_button, LayerCakeLookAndFeel::ControlButtonType::Trigger);
    m_choose_log_button.onClick = [this]() { choose_log_file(); };
    addAndMakeVisible(m_choose_log_button);

    startTimerHz(6);
    m_viz.update_from(m_processor, m_get_worker());
    refresh_text();
}

OperatorDashboardPanel::~OperatorDashboardPanel()
{
    stopTimer();
    m_log_writer.reset();
}

void OperatorDashboardPanel::resized()
{
    auto bounds = getLocalBounds().reduced(4);

    const int row_height = 28;
    const int viz_height = 288;
    const int spacing = 4;

    auto top = bounds.removeFromTop(row_height);
    m_trace_toggle.setBounds(top.removeFromLeft(180));
    top.removeFromLeft(8);
    m_file_log_toggle.setBounds(top.removeFromLeft(100));
    top.removeFromLeft(8);
    m_choose_log_button.setBounds(top.removeFromLeft(150));

    bounds.removeFromTop(spacing);
    m_viz.setBounds(bounds.removeFromTop(viz_height));
    bounds.removeFromTop(spacing);
    m_viewport.setBounds(bounds);

    const int vw = juce::jmax(100, m_viewport.getWidth() - 24);
    const int lines = juce::jmax(48, static_cast<int>(m_body.getText().length()) / 70 + 20);
    m_body.setSize(vw, lines * 18);
}

void OperatorDashboardPanel::timerCallback()
{
    InferenceWorker* w = m_get_worker();
    m_viz.update_from(m_processor, w);
    refresh_text();
    ++m_trace_counter;
    if (m_trace_toggle.getToggleState() && m_trace_counter % 6 == 0)
        juce::Logger::writeToLog(format_dashboard(m_processor, w));
}

void OperatorDashboardPanel::refresh_text()
{
    m_body.setText(format_dashboard(m_processor, m_get_worker()));
    const int vw = juce::jmax(100, m_viewport.getWidth() - 24);
    const int lines = juce::jmax(48, static_cast<int>(m_body.getText().length()) / 70 + 20);
    m_body.setSize(vw, lines * 18);
}

void OperatorDashboardPanel::on_trace_toggled()
{
    if (!m_trace_toggle.getToggleState())
        m_trace_counter = 0;
}

void OperatorDashboardPanel::on_file_log_toggled()
{
    if (!m_file_log_toggle.getToggleState())
    {
        m_log_writer.reset();
        return;
    }

    if (!m_log_file.existsAsFile())
    {
        juce::File desktop = juce::File::getSpecialLocation(juce::File::userDesktopDirectory);
        m_log_file = desktop.getChildFile("streamgen_telemetry.csv");
    }

    m_log_writer = std::make_unique<DiagnosticLogWriter>(
        m_processor,
        m_get_worker,
        m_log_file);
    m_log_writer->startThread();
}

void OperatorDashboardPanel::choose_log_file()
{
    auto chooser = std::make_shared<juce::FileChooser>(
        "Telemetry log file",
        m_log_file.getParentDirectory().exists() ? m_log_file.getParentDirectory()
                                                 : juce::File::getSpecialLocation(juce::File::userDesktopDirectory),
        "*.csv");

    constexpr auto flags = juce::FileBrowserComponent::saveMode | juce::FileBrowserComponent::canSelectFiles;
    chooser->launchAsync(flags, [this, chooser](const juce::FileChooser& fc)
    {
        juce::File f = fc.getResult();
        if (f == juce::File())
            return;
        m_log_file = f;
        if (m_file_log_toggle.getToggleState())
        {
            m_log_writer.reset();
            m_file_log_toggle.setToggleState(false, juce::dontSendNotification);
        }
    });
}

OperatorDashboardWindow::OperatorDashboardWindow(
    StreamGenProcessor& processor,
    std::function<InferenceWorker*()> get_worker)
    : juce::DocumentWindow("streamgen operator", juce::Colours::black,
                           juce::DocumentWindow::closeButton)
{
    setContentOwned(new OperatorDashboardPanel(processor, std::move(get_worker)), true);
    setResizable(true, true);
    setResizeLimits(560, 520, 1400, 1200);
    centreWithSize(780, 720);
    setUsingNativeTitleBar(true);
}

} // namespace streamgen
