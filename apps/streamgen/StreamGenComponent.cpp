#include "StreamGenComponent.h"

namespace streamgen {

StreamGenComponent::StreamGenComponent(
    StreamGenProcessor& processor,
    juce::AudioDeviceManager& device_manager)
    : m_processor(processor),
      m_device_manager(device_manager)
{
    m_title_label.setText("StreamGen Live", juce::dontSendNotification);
    m_title_label.setColour(juce::Label::textColourId, juce::Colour(0xffe0e0e0));
    m_title_label.setFont(juce::Font(18.0f, juce::Font::bold));
    m_title_label.setJustificationType(juce::Justification::centred);
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

    m_controls.on_warm_start_clicked = [this]() { load_warm_start(); };
    m_controls.on_simulate_clicked = [this]() { show_simulation_window(); };
    m_controls.on_audio_settings_clicked = [this]() { show_audio_settings(); };

    m_controls.on_generation_enabled_changed = [this](bool enabled)
    {
        m_processor.scheduler().generation_enabled.store(enabled, std::memory_order_relaxed);
        DBG("StreamGenComponent: generation " + juce::String(enabled ? "enabled" : "disabled"));
    };

    m_mixer.on_sax_gain_changed = [this](float val)
    {
        m_processor.sax_gain.store(val, std::memory_order_relaxed);
    };

    m_mixer.on_drums_gain_changed = [this](float val)
    {
        m_processor.drums_gain.store(val, std::memory_order_relaxed);
    };

    setSize(1000, 600);
    startTimerHz(30);
}

StreamGenComponent::~StreamGenComponent()
{
    stopTimer();
    if (m_worker != nullptr)
        m_worker->stopThread(5000);
}

void StreamGenComponent::load_pipeline(const std::string& manifest_path, bool use_cuda, bool use_coreml)
{
    m_worker = std::make_unique<InferenceWorker>(m_processor);

    if (!m_worker->load_pipeline(manifest_path, use_cuda, use_coreml))
    {
        DBG("StreamGenComponent: FAILED to load pipeline from " + juce::String(manifest_path));
        m_worker.reset();
        return;
    }

    m_worker->startThread(juce::Thread::Priority::high);
    DBG("StreamGenComponent: inference worker started");
}

void StreamGenComponent::resized()
{
    auto bounds = getLocalBounds();

    const int title_height = 30;
    const int sidebar_width = 160;
    const int controls_height = 130;
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
    g.fillAll(juce::Colour(0xff0d0d1a));
}

void StreamGenComponent::timerCallback()
{
    int sr = m_processor.current_sample_rate();
    int64_t abs_pos = m_processor.scheduler().absolute_sample_pos();
    int visible_samples = static_cast<int>(15.0f * sr);

    // Update waveforms
    auto sax_data = m_processor.get_recent_input_waveform(visible_samples);
    m_sax_waveform.update(sax_data, abs_pos, sr);

    auto drums_data = m_processor.get_recent_output_waveform(visible_samples);
    m_drums_waveform.update(drums_data, abs_pos, sr);

    // Source tags
    if (m_processor.simulation_active.load(std::memory_order_relaxed))
        m_sax_waveform.set_source_tag("[SIM]");
    else
        m_sax_waveform.set_source_tag("[LIVE]");

    if (m_processor.warm_start_playing.load(std::memory_order_relaxed))
        m_drums_waveform.set_source_tag("[WARM]");
    else
    {
        int64_t gen_count = m_processor.scheduler().status.generation_count.load(std::memory_order_relaxed);
        if (gen_count > 0)
            m_drums_waveform.set_source_tag("[GEN #" + juce::String(gen_count) + "]");
        else
            m_drums_waveform.set_source_tag("[IDLE]");
    }

    // Update timing
    if (m_worker != nullptr)
    {
        auto timing = m_worker->last_timing();
        m_stage_timing.update(timing);
    }

    // Update status
    auto& status = m_processor.scheduler().status;
    juce::String source_label = "Not started";
    if (m_processor.simulation_active.load(std::memory_order_relaxed))
        source_label = "Simulation";
    else if (abs_pos > 0)
        source_label = "Live Mic";

    m_gen_status.update(
        status.queue_depth.load(std::memory_order_relaxed),
        status.generation_count.load(std::memory_order_relaxed),
        status.last_latency_ms.load(std::memory_order_relaxed),
        status.last_job_id.load(std::memory_order_relaxed),
        status.worker_busy.load(std::memory_order_relaxed),
        source_label);
}

void StreamGenComponent::show_audio_settings()
{
    auto* selector = new juce::AudioDeviceSelectorComponent(
        m_device_manager, 1, 1, 2, 2, false, false, true, true);
    selector->setSize(500, 400);

    juce::DialogWindow::LaunchOptions options;
    options.content.setOwned(selector);
    options.dialogTitle = "Audio Device Settings";
    options.dialogBackgroundColour = juce::Colour(0xff1a1a2e);
    options.escapeKeyTriggersCloseButton = true;
    options.useNativeTitleBar = true;
    options.resizable = false;
    options.launchAsync();
}

void StreamGenComponent::show_simulation_window()
{
    if (m_simulation_window == nullptr)
        m_simulation_window = std::make_unique<SimulationWindow>(m_processor);

    m_simulation_window->setVisible(true);
    m_simulation_window->toFront(true);
}

void StreamGenComponent::load_warm_start()
{
    auto chooser = std::make_shared<juce::FileChooser>(
        "Load Warm Start WAV", juce::File(), "*.wav;*.aif;*.aiff;*.flac");

    chooser->launchAsync(juce::FileBrowserComponent::openMode
                         | juce::FileBrowserComponent::canSelectFiles,
        [this, chooser](const juce::FileChooser& fc)
        {
            auto result = fc.getResult();
            if (result == juce::File())
            {
                DBG("StreamGenComponent: warm-start file chooser cancelled");
                return;
            }

            if (m_processor.load_warm_start(result))
            {
                DBG("StreamGenComponent: warm-start loaded: " + result.getFileName());
            }
        });
}

} // namespace streamgen
