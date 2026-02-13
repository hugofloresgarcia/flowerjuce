#pragma once
#include <juce_core/juce_core.h>
#include <juce_gui_basics/juce_gui_basics.h>
#include <juce_audio_devices/juce_audio_devices.h>
#include "LatencyMeasurementEngine.h"

class MainComponent : public juce::Component, public juce::Timer, public juce::Button::Listener {
public:
    MainComponent();
    ~MainComponent() override;
    
    void applyDeviceSetup(const juce::AudioDeviceManager::AudioDeviceSetup& setup);
    
    void paint(juce::Graphics& g) override;
    void resized() override;
    void timerCallback() override;
    void buttonClicked(juce::Button* button) override;
    
private:
    juce::AudioDeviceManager audioDeviceManager;
    std::unique_ptr<LatencyMeasurement::LatencyMeasurementEngine> engine;
    
    juce::Label titleLabel, statusLabel;
    juce::TextEditor instructionsText, resultsText;
    juce::TextButton startButton{"Start Test"};
    bool measurementInProgress = false;
    
    void startMeasurement();
    void displayResults(const LatencyMeasurement::LatencyResult& result);
    
    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(MainComponent)
};
