#include "StreamGenProcessor.h"
#include "StreamGenComponent.h"

#include <juce_gui_basics/juce_gui_basics.h>
#include <juce_audio_devices/juce_audio_devices.h>

namespace streamgen {

class StreamGenApplication : public juce::JUCEApplication {
public:
    StreamGenApplication() {}

    const juce::String getApplicationName() override { return "StreamGen"; }
    const juce::String getApplicationVersion() override { return "0.1.0"; }
    bool moreThanOneInstanceAllowed() override { return false; }

    void initialise(const juce::String& command_line) override
    {
        juce::StringArray args = juce::StringArray::fromTokens(command_line, true);

        std::string manifest_path;
        bool use_cuda = false;
        bool use_coreml = false;

        for (int i = 0; i < args.size(); ++i)
        {
            if (args[i] == "--manifest" && i + 1 < args.size())
                manifest_path = args[i + 1].toStdString();
            else if (args[i] == "--cuda")
                use_cuda = true;
            else if (args[i] == "--coreml")
                use_coreml = true;
        }

        if (manifest_path.empty())
        {
            DBG("StreamGen: no --manifest provided. App will start without pipeline.");
            DBG("Usage: StreamGen --manifest /path/to/manifest.json [--cuda] [--coreml]");
        }

        m_processor = std::make_unique<StreamGenProcessor>();

        // If no manifest yet, configure with defaults so audio callback doesn't crash
        if (manifest_path.empty())
        {
            ModelConstants defaults;
            m_processor->configure(defaults);
        }

        m_device_manager.initialise(1, 2, nullptr, true);
        m_device_manager.addAudioCallback(m_processor.get());

        m_main_window = std::make_unique<MainWindow>(
            getApplicationName(), *m_processor, m_device_manager);

        if (!manifest_path.empty())
        {
            auto& component = dynamic_cast<StreamGenComponent&>(
                *m_main_window->getContentComponent());
            component.load_pipeline(manifest_path, use_cuda, use_coreml);
        }
    }

    void shutdown() override
    {
        m_device_manager.removeAudioCallback(m_processor.get());
        m_main_window = nullptr;
        m_processor = nullptr;
    }

    void systemRequestedQuit() override { quit(); }
    void anotherInstanceStarted(const juce::String&) override {}

    class MainWindow : public juce::DocumentWindow {
    public:
        MainWindow(const juce::String& name,
                   StreamGenProcessor& processor,
                   juce::AudioDeviceManager& device_manager)
            : juce::DocumentWindow(name, juce::Colour(0xff0d0d1a),
                                   juce::DocumentWindow::allButtons)
        {
            setUsingNativeTitleBar(true);
            setContentOwned(new StreamGenComponent(processor, device_manager), true);
            setResizable(true, true);
            setResizeLimits(800, 500, 2400, 1600);
            centreWithSize(1000, 600);
            setVisible(true);
        }

        void closeButtonPressed() override
        {
            juce::JUCEApplication::getInstance()->systemRequestedQuit();
        }
    };

private:
    std::unique_ptr<StreamGenProcessor> m_processor;
    juce::AudioDeviceManager m_device_manager;
    std::unique_ptr<MainWindow> m_main_window;
};

} // namespace streamgen

START_JUCE_APPLICATION(streamgen::StreamGenApplication)
