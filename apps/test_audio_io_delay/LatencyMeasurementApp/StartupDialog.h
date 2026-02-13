#pragma once
#include <juce_core/juce_core.h>
#include <juce_gui_basics/juce_gui_basics.h>
#include <juce_audio_devices/juce_audio_devices.h>
#include <juce_audio_utils/juce_audio_utils.h>

class StartupDialog : public juce::Component, public juce::Button::Listener {
public:
    StartupDialog(juce::AudioDeviceManager& deviceManager);
    
    juce::AudioDeviceManager::AudioDeviceSetup getDeviceSetup() const;
    bool wasOkClicked() const { return okClicked; }
    
    void resized() override;
    void paint(juce::Graphics& g) override;
    void buttonClicked(juce::Button* button) override;
    
private:
    juce::AudioDeviceManager& audioDeviceManager;
    bool okClicked = false;
    juce::Label titleLabel, instructionsLabel;
    juce::AudioDeviceSelectorComponent audioDeviceSelector;
    juce::TextButton okButton{"OK"}, cancelButton{"Cancel"};
    
    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(StartupDialog)
};
