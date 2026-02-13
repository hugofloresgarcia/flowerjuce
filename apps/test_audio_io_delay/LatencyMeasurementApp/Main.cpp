#include <juce_core/juce_core.h>
#include <juce_gui_basics/juce_gui_basics.h>
#include <juce_audio_devices/juce_audio_devices.h>
#include <juce_audio_utils/juce_audio_utils.h>
#include "StartupDialog.h"
#include "MainComponent.h"

class LatencyMeasurementApplication : public juce::JUCEApplication {
public:
    const juce::String getApplicationName() override { return "Latency Measurement"; }
    const juce::String getApplicationVersion() override { return "1.0.0"; }
    bool moreThanOneInstanceAllowed() override { return true; }

    void initialise(const juce::String&) override
    {
        juce::AudioDeviceManager tempManager;
        tempManager.initialiseWithDefaultDevices(2, 2);
        
        auto dialog = std::make_unique<StartupDialog>(tempManager);
        auto* dialogPtr = dialog.get();
        
        juce::DialogWindow::LaunchOptions opts;
        opts.content.setNonOwned(dialog.release());
        opts.dialogTitle = "Audio Setup";
        opts.dialogBackgroundColour = juce::Colour(0xff2a2a2a);
        opts.escapeKeyTriggersCloseButton = false;
        opts.useNativeTitleBar = true;
        opts.resizable = false;
        
#if JUCE_MODAL_LOOPS_PERMITTED
        juce::Process::makeForegroundProcess();
        if (opts.runModal() == 1 && dialogPtr && dialogPtr->wasOkClicked()) {
            auto deviceSetup = dialogPtr->getDeviceSetup();
            mainWindow = std::make_unique<MainWindow>("Latency Measurement", deviceSetup);
        } else {
            quit();
        }
        delete dialogPtr;
#endif
    }

    void shutdown() override { mainWindow = nullptr; }
    void systemRequestedQuit() override { quit(); }
    void anotherInstanceStarted(const juce::String&) override {}

    class MainWindow : public juce::DocumentWindow {
    public:
        MainWindow(const juce::String& name, const juce::AudioDeviceManager::AudioDeviceSetup& setup)
            : DocumentWindow(name, juce::Colours::darkgrey, allButtons)
        {
            setUsingNativeTitleBar(true);
            auto* main = new MainComponent();
            main->applyDeviceSetup(setup);
            setContentOwned(main, true);
            setResizable(true, false);
            centreWithSize(getWidth(), getHeight());
            setVisible(true);
        }
        void closeButtonPressed() override { JUCEApplication::getInstance()->systemRequestedQuit(); }
    };

private:
    std::unique_ptr<MainWindow> mainWindow;
};

START_JUCE_APPLICATION(LatencyMeasurementApplication)
