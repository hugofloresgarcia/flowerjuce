#include "StreamGenProcessor.h"
#include "StreamGenComponent.h"
#include "StreamGenDebugLog.h"
#include "LayerCakeLookAndFeel.h"

#include <juce_core/juce_core.h>
#include <juce_gui_basics/juce_gui_basics.h>
#include <juce_audio_devices/juce_audio_devices.h>

namespace streamgen {

namespace {

constexpr const char* k_default_manifest_relative = "models/zenon/zenon_pipeline_manifest.json";

juce::File resolve_manifest_file(const std::string& manifest_arg)
{
    if (!manifest_arg.empty())
        return juce::File(juce::String(manifest_arg));

    const juce::String rel(k_default_manifest_relative);
    juce::File probe = juce::File::getCurrentWorkingDirectory().getChildFile(rel);
    if (probe.existsAsFile())
        return probe;

    juce::File walk = juce::File::getSpecialLocation(juce::File::currentExecutableFile);
    for (int depth = 0; depth < 24; ++depth)
    {
        juce::File candidate = walk.getChildFile(rel);
        if (candidate.existsAsFile())
            return candidate;
        juce::File parent = walk.getParentDirectory();
        if (parent == walk)
            break;
        walk = parent;
    }

    return juce::File::getCurrentWorkingDirectory().getChildFile(rel);
}

bool is_macos_platform()
{
    return juce::SystemStats::getOperatingSystemType() == juce::SystemStats::MacOSX;
}

/// initialise() can leave input/output channel bitmasks empty on some devices; the callback
/// then gets num_input_channels == 0 and live input is silent. Match juce::AudioAppComponent.
void ensure_audio_channel_masks(juce::AudioDeviceManager& dm, int num_input, int num_output)
{
    auto setup = dm.getAudioDeviceSetup();
    if (setup.inputChannels.countNumberOfSetBits() == num_input
        && setup.outputChannels.countNumberOfSetBits() == num_output)
        return;

    setup.inputChannels.clear();
    setup.outputChannels.clear();
    setup.inputChannels.setRange(0, num_input, true);
    setup.outputChannels.setRange(0, num_output, true);

    juce::String err = dm.setAudioDeviceSetup(setup, true);
    if (err.isNotEmpty())
        DBG("StreamGen: setAudioDeviceSetup failed: " + err);
}

} // namespace

class StreamGenApplication : public juce::JUCEApplication {
public:
    StreamGenApplication() {}

    const juce::String getApplicationName() override { return "StreamGen"; }
    const juce::String getApplicationVersion() override { return "0.1.0"; }
    bool moreThanOneInstanceAllowed() override { return false; }

    void initialise(const juce::String& command_line) override
    {
        juce::File log_file = juce::File::getCurrentWorkingDirectory().getChildFile("log.log");
        m_file_logger = std::make_unique<juce::FileLogger>(
            log_file,
            "StreamGen session",
            static_cast<juce::int64>(8) * 1024 * 1024);
        juce::Logger::setCurrentLogger(m_file_logger.get());

        juce::StringArray args = juce::StringArray::fromTokens(command_line, true);

        std::string manifest_arg;
        bool use_cuda = false;
        bool use_coreml = false;
        bool use_migraphx = false;
        bool use_mlx_vae = false;
        bool user_set_backend = false;

        for (int i = 0; i < args.size(); ++i)
        {
            if (args[i] == "--manifest" && i + 1 < args.size())
                manifest_arg = args[++i].toStdString();
            else if (args[i] == "--cuda")
            {
                use_cuda = true;
                user_set_backend = true;
            }
            else if (args[i] == "--coreml")
            {
                use_coreml = true;
                user_set_backend = true;
            }
            else if (args[i] == "--migraphx")
            {
                use_migraphx = true;
                user_set_backend = true;
            }
            else if (args[i] == "--mlx-vae")
            {
                use_mlx_vae = true;
                user_set_backend = true;
            }
        }

        if (!user_set_backend)
        {
            if (is_macos_platform())
            {
                use_coreml = true;
                use_mlx_vae = true;
                use_cuda = false;
            }
            else
            {
                use_cuda = true;
                use_coreml = false;
                use_mlx_vae = false;
            }
        }

        juce::File manifest_file = resolve_manifest_file(manifest_arg);
        const std::string manifest_path = manifest_file.getFullPathName().toStdString();

        streamgen_log(juce::String("StreamGen: manifest path=") + manifest_file.getFullPathName()
            + " exists=" + juce::String(manifest_file.existsAsFile() ? "true" : "false")
            + " macos=" + juce::String(is_macos_platform() ? "true" : "false")
            + " backends cuda=" + juce::String(use_cuda ? "true" : "false")
            + " coreml=" + juce::String(use_coreml ? "true" : "false")
            + " migraphx=" + juce::String(use_migraphx ? "true" : "false")
            + " mlx_vae=" + juce::String(use_mlx_vae ? "true" : "false"));

        m_processor = std::make_unique<StreamGenProcessor>();

        // Ring buffers must exist before the audio device can call back; load_pipeline()
        // (when manifest is set) runs after addAudioCallback and would be too late alone.
        ModelConstants defaults;
        m_processor->configure(defaults);

        m_device_manager.initialise(1, 2, nullptr, true);
        ensure_audio_channel_masks(m_device_manager, 1, 2);
        streamgen_log("StreamGen initialise: addAudioCallback (default device will open)");
        m_device_manager.addAudioCallback(m_processor.get());
        if (auto* dev = m_device_manager.getCurrentAudioDevice())
            streamgen_log("StreamGen initialise: current device=" + dev->getName()
                + " sr=" + juce::String(dev->getCurrentSampleRate())
                + " playing=" + juce::String(dev->isPlaying() ? "y" : "n"));
        else
            streamgen_log("StreamGen initialise: getCurrentAudioDevice() is null after addAudioCallback");

        m_layercake_look_and_feel = std::make_unique<LayerCakeLookAndFeel>();
        juce::LookAndFeel::setDefaultLookAndFeel(m_layercake_look_and_feel.get());

        m_main_window = std::make_unique<MainWindow>(
            getApplicationName(), *m_processor, m_device_manager);

        auto& component = dynamic_cast<StreamGenComponent&>(
            *m_main_window->getContentComponent());
        component.load_pipeline(manifest_path, use_cuda, use_coreml, use_mlx_vae, use_migraphx);
    }

    void shutdown() override
    {
        m_device_manager.removeAudioCallback(m_processor.get());
        m_main_window = nullptr;
        m_processor = nullptr;
        juce::LookAndFeel::setDefaultLookAndFeel(nullptr);
        m_layercake_look_and_feel.reset();
        juce::Logger::setCurrentLogger(nullptr);
        m_file_logger.reset();
    }

    void systemRequestedQuit() override { quit(); }
    void anotherInstanceStarted(const juce::String&) override {}

    class MainWindow : public juce::DocumentWindow {
    public:
        MainWindow(const juce::String& name,
                   StreamGenProcessor& processor,
                   juce::AudioDeviceManager& device_manager)
            : juce::DocumentWindow(name, juce::Colours::black,
                                   juce::DocumentWindow::allButtons),
              m_tooltip(this, 400)
        {
            setUsingNativeTitleBar(true);
            setContentOwned(new StreamGenComponent(processor, device_manager), true);
            setResizable(true, true);
            setResizeLimits(880, 660, 2400, 1600);
            centreWithSize(1000, 720);
            setVisible(true);
        }

        void closeButtonPressed() override
        {
            juce::JUCEApplication::getInstance()->systemRequestedQuit();
        }

    private:
        juce::TooltipWindow m_tooltip;
    };

private:
    std::unique_ptr<LayerCakeLookAndFeel> m_layercake_look_and_feel;
    std::unique_ptr<juce::FileLogger> m_file_logger;
    std::unique_ptr<StreamGenProcessor> m_processor;
    juce::AudioDeviceManager m_device_manager;
    std::unique_ptr<MainWindow> m_main_window;
};

} // namespace streamgen

START_JUCE_APPLICATION(streamgen::StreamGenApplication)
