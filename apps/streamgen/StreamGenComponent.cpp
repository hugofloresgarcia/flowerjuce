#include "StreamGenComponent.h"
#include "MusicalTime.h"
#include "StreamGenDebugLog.h"

namespace streamgen {

namespace {

constexpr const char* k_default_warmup_audio_rel = "tests/streamgen_test_audio/drums_120bpm.wav";
constexpr const char* k_default_sim_rel = "tests/streamgen_test_audio/synth_120bpm_medium.wav";

/// Walk parents of `manifest_path` until `tests/streamgen_test_audio/` exists (repo layout).
juce::File find_repo_root_with_streamgen_tests(const juce::File& manifest_path)
{
    juce::File start(manifest_path.getFullPathName());
    if (!start.exists())
        return {};
    if (!start.isDirectory())
        start = start.getParentDirectory();
    for (int depth = 0; depth < 40; ++depth)
    {
        const juce::File test_audio_dir = start.getChildFile("tests/streamgen_test_audio");
        if (test_audio_dir.isDirectory())
            return start;
        const juce::File parent = start.getParentDirectory();
        if (parent == start)
            break;
        start = parent;
    }
    return {};
}

} // namespace

StreamGenComponent::StreamGenComponent(
    StreamGenProcessor& processor,
    juce::AudioDeviceManager& device_manager)
    : m_processor(processor),
      m_device_manager(device_manager)
{
    m_title_label.setText("streamgen live", juce::dontSendNotification);
    m_title_label.setJustificationType(juce::Justification::centred);
    juce::FontOptions title_options(juce::Font::getDefaultMonospacedFontName(), 17.0f, juce::Font::bold);
    m_title_label.setFont(juce::Font(title_options));
    addAndMakeVisible(m_title_label);

    m_sax_waveform.set_label("SAX INPUT");
    m_drums_waveform.set_label("DRUMS OUTPUT");
    addAndMakeVisible(m_sax_waveform);
    addAndMakeVisible(m_drums_waveform);
    addAndMakeVisible(m_stage_timing);
    addAndMakeVisible(m_gen_status);
    addAndMakeVisible(m_controls);
    addAndMakeVisible(m_mixer);

    // Wire up controls callbacks
    m_controls.on_prompt_changed = [this](const juce::String& text)
    {
        if (m_worker != nullptr)
            m_worker->set_prompt(text.toStdString());
    };

    m_controls.on_hop_changed = [this](float val)
    {
        m_processor.scheduler().hop_seconds.store(val, std::memory_order_relaxed);
    };

    m_controls.on_hop_bars_changed = [this](float bars)
    {
        auto& sched = m_processor.scheduler();
        sched.hop_bars.store(bars, std::memory_order_relaxed);
        const int bpb = juce::jmax(1, sched.time_sig_numerator.load(std::memory_order_relaxed));
        sched.hop_beats.store(bars * static_cast<float>(bpb), std::memory_order_relaxed);
    };

    m_controls.on_keep_ratio_changed = [this](float val)
    {
        m_processor.scheduler().keep_ratio.store(val, std::memory_order_relaxed);
    };

    m_controls.on_steps_changed = [this](int val)
    {
        m_processor.scheduler().steps.store(val, std::memory_order_relaxed);
    };

    m_controls.on_cfg_changed = [this](float val)
    {
        m_processor.scheduler().cfg_scale.store(val, std::memory_order_relaxed);
    };

    m_controls.on_schedule_delay_changed = [this](float val)
    {
        m_processor.scheduler().schedule_delay_seconds.store(val, std::memory_order_relaxed);
    };

    m_controls.on_schedule_delay_bars_changed = [this](float bars)
    {
        auto& sched = m_processor.scheduler();
        sched.schedule_delay_bars.store(bars, std::memory_order_relaxed);
        const int bpb = juce::jmax(1, sched.time_sig_numerator.load(std::memory_order_relaxed));
        sched.schedule_delay_beats.store(bars * static_cast<float>(bpb), std::memory_order_relaxed);
    };

    m_controls.on_musical_time_changed = [this](bool musical)
    {
        auto& sched = m_processor.scheduler();
        sched.musical_time_enabled.store(musical, std::memory_order_relaxed);
        if (musical && sched.quantize_launch_beats.load(std::memory_order_relaxed) == 0)
        {
            const int bar_beats = juce::jmax(1, sched.time_sig_numerator.load(std::memory_order_relaxed));
            sched.quantize_launch_beats.store(bar_beats, std::memory_order_relaxed);
        }
        m_controls.sync_time_mode_from_scheduler(
            sched,
            m_processor.loop_last_generation.load(std::memory_order_relaxed));
        m_controls.sync_click_track_from_processor(m_processor);
    };

    m_controls.on_bpm_changed = [this](float val)
    {
        m_processor.scheduler().bpm.store(val, std::memory_order_relaxed);
    };

    m_controls.on_time_signature_changed = [this](int n, int d)
    {
        auto& sched = m_processor.scheduler();
        sched.time_sig_numerator.store(n, std::memory_order_relaxed);
        sched.time_sig_denominator.store(d, std::memory_order_relaxed);
        const int bpb = juce::jmax(1, n);
        const float hb = sched.hop_bars.load(std::memory_order_relaxed);
        sched.hop_beats.store(hb * static_cast<float>(bpb), std::memory_order_relaxed);
        const float db = sched.schedule_delay_bars.load(std::memory_order_relaxed);
        sched.schedule_delay_beats.store(db * static_cast<float>(bpb), std::memory_order_relaxed);
    };

    m_controls.on_loop_last_generation_changed = [this](bool enabled)
    {
        m_processor.loop_last_generation.store(enabled, std::memory_order_relaxed);
        if (!enabled)
            m_processor.clear_drums_output_buffers();
    };

    m_controls.on_click_track_enabled_changed = [this](bool enabled)
    {
        m_processor.click_track_enabled.store(enabled, std::memory_order_relaxed);
    };

    m_controls.on_click_track_volume_changed = [this](float v)
    {
        m_processor.click_track_volume.store(v, std::memory_order_relaxed);
    };

    m_controls.on_quantize_launch_changed = [this](int beats)
    {
        m_processor.scheduler().quantize_launch_beats.store(beats, std::memory_order_relaxed);
    };

    m_controls.on_warmup_audio_clicked = [this]() { load_warmup_audio(); };
    m_controls.on_warmup_audio_route_toggled = [this](bool route_to_output)
    {
        m_processor.set_warmup_audio_playing(route_to_output);
    };
    m_controls.on_simulate_clicked = [this]() { show_simulation_window(); };
    m_controls.on_audio_settings_clicked = [this]() { show_audio_settings(); };

    m_controls.on_reset_clicked = [this]() { reset_session(); };

    m_controls.on_generation_enabled_changed = [this](bool enabled)
    {
        m_processor.scheduler().generation_enabled.store(enabled, std::memory_order_relaxed);
        if (!enabled)
        {
            m_processor.loop_last_generation.store(false, std::memory_order_relaxed);
            m_controls.set_loop_last_generation_toggle(false, juce::dontSendNotification);
            m_processor.clear_drums_output_buffers();
        }
        DBG("StreamGenComponent: generation " + juce::String(enabled ? "enabled" : "disabled"));
        streamgen_log("UI: generation_enabled=" + juce::String(enabled ? "true" : "false"));
    };

    m_mixer.on_sax_gain_changed = [this](float val)
    {
        m_processor.sax_gain.store(val, std::memory_order_relaxed);
    };

    m_mixer.on_drums_gain_changed = [this](float val)
    {
        m_processor.drums_gain.store(val, std::memory_order_relaxed);
    };

    m_controls.set_warmup_audio_route_enabled(false);

    m_controls.sync_time_mode_from_scheduler(
        m_processor.scheduler(),
        m_processor.loop_last_generation.load(std::memory_order_relaxed));
    m_controls.sync_click_track_from_processor(m_processor);

    setSize(1000, 720);
    startTimerHz(18);
}

StreamGenComponent::~StreamGenComponent()
{
    stopTimer();
    if (m_worker != nullptr)
        m_worker->stopThread(5000);
}

void StreamGenComponent::load_pipeline(
    const std::string& manifest_path,
    bool use_cuda,
    bool use_coreml,
    bool use_mlx_vae)
{
    streamgen_log("load_pipeline: removeAudioCallback (safe window for processor.configure / ring rebuild)");
    m_device_manager.removeAudioCallback(&m_processor);

    m_worker = std::make_unique<InferenceWorker>(m_processor);

    if (!m_worker->load_pipeline(manifest_path, use_cuda, use_coreml, use_mlx_vae))
    {
        DBG("StreamGenComponent: FAILED to load pipeline from " + juce::String(manifest_path));
        streamgen_log("load_pipeline: FAILED, re-attaching audio callback");
        m_worker.reset();
        reattach_audio_callback_after_pipeline_load();
        return;
    }

    // While the callback is detached, zero playhead/rings/timeline so the first block sees abs=0.
    // Warmup audio uses absolute_sample_pos % loop; this avoids a stale playhead after reload.
    m_processor.reset_timeline_and_transport();

    try_load_default_audio_from_repo(juce::File(manifest_path));

    reattach_audio_callback_after_pipeline_load();

    m_worker->startThread(juce::Thread::Priority::high);
    DBG("StreamGenComponent: inference worker started");
    streamgen_log("load_pipeline: worker thread started");
}

void StreamGenComponent::reset_session()
{
    streamgen_log("UI: reset_session (stop audio + worker, clear timeline/transport)");
    m_device_manager.removeAudioCallback(&m_processor);
    if (m_worker != nullptr)
        m_worker->stopThread(8000);
    m_processor.reset_timeline_and_transport();
    if (m_worker != nullptr)
        m_worker->startThread(juce::Thread::Priority::high);
    reattach_audio_callback_after_pipeline_load();
    if (m_simulation_window != nullptr)
        m_simulation_window->sync_from_processor();
}

void StreamGenComponent::reattach_audio_callback_after_pipeline_load()
{
    m_device_manager.addAudioCallback(&m_processor);

    juce::AudioIODevice* dev = m_device_manager.getCurrentAudioDevice();
    const bool playing = dev != nullptr && dev->isPlaying();

    streamgen_log(
        "reattach_audio: dev="
        + (dev != nullptr ? dev->getName() : juce::String("<null>"))
        + " playing=" + juce::String(playing ? "y" : "n")
        + " sr=" + juce::String(m_processor.current_sample_rate())
        + " abs_pos=" + juce::String(m_processor.scheduler().absolute_sample_pos()));

    if (dev == nullptr)
    {
        streamgen_log("reattach_audio: no device; restartLastAudioDevice()");
        m_device_manager.restartLastAudioDevice();
        dev = m_device_manager.getCurrentAudioDevice();
        const bool playing_after = dev != nullptr && dev->isPlaying();
        streamgen_log("reattach_audio: after restart dev="
            + (dev != nullptr ? dev->getName() : juce::String("<null>"))
            + " playing=" + juce::String(playing_after ? "y" : "n"));
        return;
    }

    if (!playing)
    {
        streamgen_log("reattach_audio: device not playing; closeAudioDevice + restartLastAudioDevice");
        m_device_manager.closeAudioDevice();
        m_device_manager.restartLastAudioDevice();
        dev = m_device_manager.getCurrentAudioDevice();
        streamgen_log("reattach_audio: after cycle dev="
            + (dev != nullptr ? dev->getName() : juce::String("<null>"))
            + " playing=" + juce::String(dev != nullptr && dev->isPlaying() ? "y" : "n"));
    }
}

void StreamGenComponent::resized()
{
    auto bounds = getLocalBounds();

    const int title_height = 30;
    const int sidebar_width = 160;
    // Controls: full-width prompt, (musical | inference) grid row, full-width session.
    const int controls_height = 360;
    const int waveform_min_height = 100;
    const int mixer_width = 80;

    // Title bar
    m_title_label.setBounds(bounds.removeFromTop(title_height));

    // Controls at the bottom
    m_controls.setBounds(bounds.removeFromBottom(controls_height));

    // Right sidebar: timing + status + mixer stacked vertically
    auto sidebar = bounds.removeFromRight(sidebar_width);
    m_mixer.setBounds(sidebar.removeFromBottom(180));
    m_gen_status.setBounds(sidebar.removeFromBottom(sidebar.getHeight() / 2));
    m_stage_timing.setBounds(sidebar);

    // Main area: two waveforms stacked
    int available_height = bounds.getHeight();
    int waveform_height = std::max(waveform_min_height, available_height / 2);

    m_sax_waveform.setBounds(bounds.removeFromTop(waveform_height));
    m_drums_waveform.setBounds(bounds);
}

void StreamGenComponent::paint(juce::Graphics& g)
{
    g.fillAll(getLookAndFeel().findColour(juce::ResizableWindow::backgroundColourId));
}

void StreamGenComponent::timerCallback()
{
    auto& sched_for_timer = m_processor.scheduler();
    const bool worker_busy = sched_for_timer.status.worker_busy.load(std::memory_order_relaxed);

    int sr = m_processor.current_sample_rate();
    int64_t abs_pos = sched_for_timer.absolute_sample_pos();

    static int ui_tick = 0;
    ++ui_tick;
    if (ui_tick <= 8 || ui_tick % 40 == 0)
    {
        auto* dev = m_device_manager.getCurrentAudioDevice();
        streamgen_log("UI timer: abs_pos=" + juce::String(abs_pos) + " sr=" + juce::String(sr)
            + " gen_en=" + juce::String(m_processor.scheduler().generation_enabled.load() ? 1 : 0)
            + " queue=" + juce::String(m_processor.scheduler().status.queue_depth.load())
            + " device=" + (dev != nullptr ? dev->getName() : juce::String("<none>"))
            + " playing=" + juce::String(dev != nullptr && dev->isPlaying() ? "y" : "n"));
    }
    const float visible_seconds = 30.0f;
    int visible_samples = static_cast<int>(visible_seconds * static_cast<float>(sr));

    m_sax_waveform.set_visible_duration(visible_seconds);
    m_drums_waveform.set_visible_duration(visible_seconds);

    auto& sched = sched_for_timer;
    const bool musical = sched.musical_time_enabled.load(std::memory_order_relaxed);
    const float bpm = sched.bpm.load(std::memory_order_relaxed);
    const int sig_n = sched.time_sig_numerator.load(std::memory_order_relaxed);
    const int sig_d = sched.time_sig_denominator.load(std::memory_order_relaxed);
    m_sax_waveform.set_time_axis_for_paint(musical, bpm, sig_n, sig_d);
    m_drums_waveform.set_time_axis_for_paint(musical, bpm, sig_n, sig_d);

    const bool skip_timeline_tick = worker_busy && (ui_tick % 2 != 0);
    if (!skip_timeline_tick)
    {
        if (m_processor.timeline_store() != nullptr)
            m_timeline_paint_cache = m_processor.timeline_store()->snapshot_intersecting(
                abs_pos, sr, visible_seconds);
        else
            m_timeline_paint_cache.clear();
    }

    // Waveforms: one bucket column per panel pixel (capped); ~30s window with playhead at
    // k_timeline_playhead_past_fraction (see GenerationTimelineStore.h).
    constexpr int k_waveform_buckets_max = 2048;
    const int panel_w = juce::jmax(1, juce::jmin(m_sax_waveform.getWidth(), m_drums_waveform.getWidth()));
    const int wave_w = juce::jmin(k_waveform_buckets_max, panel_w);
    if (wave_w != m_waveform_bucket_width)
    {
        m_waveform_bucket_width = wave_w;
        const size_t w = static_cast<size_t>(wave_w);
        m_sax_wave_min.resize(w);
        m_sax_wave_max.resize(w);
        m_drums_wave_min.resize(w);
        m_drums_wave_max.resize(w);
        m_drums_wave_warm_min.resize(w);
        m_drums_wave_warm_max.resize(w);
        m_drums_wave_gen_min.resize(w);
        m_drums_wave_gen_max.resize(w);
        m_drums_wave_hold_min.resize(w);
        m_drums_wave_hold_max.resize(w);
    }

    m_processor.fill_recent_input_waveform_buckets(
        visible_samples, wave_w, m_sax_wave_min.data(), m_sax_wave_max.data());
    m_sax_waveform.update(
        m_sax_wave_min.data(),
        m_sax_wave_max.data(),
        wave_w,
        abs_pos,
        sr,
        &m_timeline_paint_cache,
        TimelineWaveRole::SaxInput);

    m_processor.fill_recent_output_waveform_buckets(
        visible_samples, wave_w, m_drums_wave_min.data(), m_drums_wave_max.data());
    m_processor.fill_recent_drums_source_buckets(
        visible_samples,
        wave_w,
        m_drums_wave_warm_min.data(),
        m_drums_wave_warm_max.data(),
        m_drums_wave_gen_min.data(),
        m_drums_wave_gen_max.data(),
        m_drums_wave_hold_min.data(),
        m_drums_wave_hold_max.data(),
        &m_timeline_paint_cache);
    m_drums_waveform.update(
        m_drums_wave_min.data(),
        m_drums_wave_max.data(),
        wave_w,
        abs_pos,
        sr,
        &m_timeline_paint_cache,
        TimelineWaveRole::DrumsOutput,
        m_drums_wave_warm_min.data(),
        m_drums_wave_warm_max.data(),
        m_drums_wave_gen_min.data(),
        m_drums_wave_gen_max.data(),
        m_drums_wave_hold_min.data(),
        m_drums_wave_hold_max.data());

    const bool drums_source_layers = m_processor.warmup_audio_has_audio();
    const juce::String drums_wave_legend = drums_source_layers
        ? juce::String("  orange=warmup  cyan=model  magenta=loop hold")
        : juce::String("  cyan=model  magenta=loop hold");

    // Source tags
    if (m_processor.simulation_playing.load(std::memory_order_relaxed))
        m_sax_waveform.set_source_tag("[SIM]");
    else
        m_sax_waveform.set_source_tag("[LIVE]");

    const bool drums_hold = m_processor.drums_output_from_last_gen_hold.load(std::memory_order_relaxed);
    juce::String drums_tag;
    if (m_processor.warmup_audio_playing.load(std::memory_order_relaxed))
        drums_tag = "[WARMUP]" + drums_wave_legend;
    else if (m_processor.warmup_audio_has_audio())
        drums_tag = "[WARMUP ·]" + drums_wave_legend;
    else
    {
        int64_t gen_count = m_processor.scheduler().status.generation_count.load(std::memory_order_relaxed);
        if (gen_count > 0)
            drums_tag = "[GEN #" + juce::String(gen_count) + "]";
        else
            drums_tag = "[IDLE]";
    }
    if (drums_hold)
        drums_tag += " [HOLD]";
    m_drums_waveform.set_source_tag(drums_tag);

    // Update timing
    if (m_worker != nullptr)
    {
        auto timing = m_worker->last_timing();
        m_stage_timing.update(timing);
    }

    // Update status
    auto& status = m_processor.scheduler().status;
    juce::String source_label = "Not started";
    if (m_processor.simulation_playing.load(std::memory_order_relaxed))
        source_label = "Simulation";
    else if (abs_pos > 0)
        source_label = "Live Mic";

    juce::String land_timeline;
    if (m_worker != nullptr && m_worker->is_loaded())
    {
        InferenceSnapshot snap = m_worker->last_snapshot();
        if (snap.job.job_id >= 0)
        {
            land_timeline = juce::String(format_time(snap.job.output_start_sample(), sr))
                + "  #" + juce::String(snap.job.job_id);
            if (sched.musical_time_enabled.load(std::memory_order_relaxed))
            {
                const float bpm = sched.bpm.load(std::memory_order_relaxed);
                const int bpb = sched.time_sig_numerator.load(std::memory_order_relaxed);
                const double bpm_d = static_cast<double>(juce::jlimit(20.0f, 400.0f, bpm));
                const int bpb_cl = juce::jmax(1, bpb);
                land_timeline += "  ";
                land_timeline += juce::String(format_bar_beat(snap.job.output_start_sample(), sr, bpm_d, bpb_cl));
            }
        }
    }

    juce::String phase_label = "[idle]";
    if (m_worker != nullptr)
        phase_label = m_worker->inference_phase_display();

    m_gen_status.update(
        status.queue_depth.load(std::memory_order_relaxed),
        status.generation_count.load(std::memory_order_relaxed),
        status.last_latency_ms.load(std::memory_order_relaxed),
        status.last_job_id.load(std::memory_order_relaxed),
        status.worker_busy.load(std::memory_order_relaxed),
        source_label,
        land_timeline,
        drums_hold,
        phase_label);
}

void StreamGenComponent::show_audio_settings()
{
    auto* selector = new juce::AudioDeviceSelectorComponent(
        m_device_manager, 1, 1, 2, 2, false, false, true, true);
    selector->setSize(500, 400);

    juce::DialogWindow::LaunchOptions options;
    options.content.setOwned(selector);
    options.dialogTitle = "Audio Device Settings";
    options.dialogBackgroundColour = getLookAndFeel().findColour(juce::ComboBox::backgroundColourId);
    options.escapeKeyTriggersCloseButton = true;
    options.useNativeTitleBar = true;
    options.resizable = false;
    options.launchAsync();
}

void StreamGenComponent::show_simulation_window()
{
    if (m_simulation_window == nullptr)
        m_simulation_window = std::make_unique<SimulationWindow>(m_processor);

    m_simulation_window->sync_from_processor();
    m_simulation_window->setVisible(true);
    m_simulation_window->toFront(true);
}

void StreamGenComponent::load_warmup_audio()
{
    auto chooser = std::make_shared<juce::FileChooser>(
        "Load Warmup Audio WAV", juce::File(), "*.wav;*.aif;*.aiff;*.flac");

    chooser->launchAsync(juce::FileBrowserComponent::openMode
                         | juce::FileBrowserComponent::canSelectFiles,
        [this, chooser](const juce::FileChooser& fc)
        {
            auto result = fc.getResult();
            if (result == juce::File())
            {
                DBG("StreamGenComponent: warmup audio file chooser cancelled");
                return;
            }

            if (m_processor.load_warmup_audio(result, false))
            {
                m_controls.set_warmup_audio_route_enabled(true);
                m_controls.set_warmup_audio_route_toggle(false, juce::dontSendNotification);
                DBG("StreamGenComponent: warmup audio loaded: " + result.getFileName());
            }
        });
}

void StreamGenComponent::apply_default_120bpm_grid_preset()
{
    auto& sched = m_processor.scheduler();
    sched.musical_time_enabled.store(true, std::memory_order_relaxed);
    sched.bpm.store(120.0f, std::memory_order_relaxed);
    sched.time_sig_numerator.store(4, std::memory_order_relaxed);
    sched.time_sig_denominator.store(4, std::memory_order_relaxed);
    if (sched.quantize_launch_beats.load(std::memory_order_relaxed) == 0)
        sched.quantize_launch_beats.store(4, std::memory_order_relaxed);
    const int bpb = juce::jmax(1, sched.time_sig_numerator.load(std::memory_order_relaxed));
    const float hb = sched.hop_bars.load(std::memory_order_relaxed);
    sched.hop_beats.store(hb * static_cast<float>(bpb), std::memory_order_relaxed);
    const float db = sched.schedule_delay_bars.load(std::memory_order_relaxed);
    sched.schedule_delay_beats.store(db * static_cast<float>(bpb), std::memory_order_relaxed);
    m_controls.sync_time_mode_from_scheduler(
        sched,
        m_processor.loop_last_generation.load(std::memory_order_relaxed));
    m_controls.sync_click_track_from_processor(m_processor);
    streamgen_log(
        "default_120bpm_grid: musical=on bpm=120 sig=4/4 quant_launch="
        + juce::String(sched.quantize_launch_beats.load(std::memory_order_relaxed)));
}

void StreamGenComponent::try_load_default_audio_from_repo(const juce::File& manifest_file)
{
    juce::File manifest(manifest_file.getFullPathName());
    juce::File repo_root = find_repo_root_with_streamgen_tests(manifest);
    if (!repo_root.exists())
    {
        repo_root = manifest.getParentDirectory().getParentDirectory().getParentDirectory();
        streamgen_log(
            "try_load_default_audio: walk-up miss; fallback repo_root=" + repo_root.getFullPathName());
    }
    else
    {
        streamgen_log("try_load_default_audio: repo_root=" + repo_root.getFullPathName());
    }

    juce::File warm_file = repo_root.getChildFile(k_default_warmup_audio_rel);
    juce::File sim_file = repo_root.getChildFile(k_default_sim_rel);

    bool loaded_any = false;

    if (warm_file.existsAsFile())
    {
        if (m_processor.load_warmup_audio(warm_file, false))
        {
            m_controls.set_warmup_audio_route_enabled(true);
            m_controls.set_warmup_audio_route_toggle(false, juce::dontSendNotification);
            loaded_any = true;
            DBG("StreamGenComponent: default warmup audio loaded: " + warm_file.getFileName());
            streamgen_log("default warmup audio: " + warm_file.getFullPathName());
        }
    }
    else
    {
        DBG("StreamGenComponent: default warmup audio not found at " + warm_file.getFullPathName());
        streamgen_log("default warmup audio missing: " + warm_file.getFullPathName());
    }

    if (sim_file.existsAsFile())
    {
        if (m_processor.load_simulation_file(sim_file))
        {
            loaded_any = true;
            DBG("StreamGenComponent: default simulation sax loaded: " + sim_file.getFileName());
            streamgen_log("default sim: " + sim_file.getFullPathName());
            if (m_simulation_window != nullptr)
                m_simulation_window->sync_from_processor();
        }
    }
    else
    {
        DBG("StreamGenComponent: default simulation file not found at " + sim_file.getFullPathName());
        streamgen_log("default sim missing: " + sim_file.getFullPathName());
    }

    if (loaded_any)
        apply_default_120bpm_grid_preset();
}

} // namespace streamgen
