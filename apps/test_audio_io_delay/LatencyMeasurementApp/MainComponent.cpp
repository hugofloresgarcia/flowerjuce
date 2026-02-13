#include "MainComponent.h"

MainComponent::MainComponent()
    : titleLabel("", "Audio Latency Measurement"),
      statusLabel("", "Ready")
{
    titleLabel.setJustificationType(juce::Justification::centred);
    titleLabel.setFont(juce::FontOptions(24.0f).withStyle("Bold"));
    addAndMakeVisible(titleLabel);
    
    instructionsText.setMultiLine(true);
    instructionsText.setReadOnly(true);
    instructionsText.setText(
        "Instructions:\n\n"
        "1. Position speakers near microphone for acoustic feedback\n"
        "2. Click 'Start Test'\n"
        "3. A 1-second sweep tone will play\n"
        "4. Result shows round-trip latency\n\n"
        "Test takes ~2 seconds.");
    addAndMakeVisible(instructionsText);
    
    startButton.addListener(this);
    startButton.setColour(juce::TextButton::buttonColourId, juce::Colours::darkgreen);
    addAndMakeVisible(startButton);
    
    resultsText.setMultiLine(true);
    resultsText.setReadOnly(true);
    resultsText.setFont(juce::FontOptions(juce::Font::getDefaultMonospacedFontName(), 14.0f, 0));
    resultsText.setText("Results will appear here...");
    addAndMakeVisible(resultsText);
    
    statusLabel.setColour(juce::Label::textColourId, juce::Colours::lightblue);
    addAndMakeVisible(statusLabel);
    
    setSize(700, 500);
}

MainComponent::~MainComponent()
{
    stopTimer();
    audioDeviceManager.closeAudioDevice();
}

void MainComponent::applyDeviceSetup(const juce::AudioDeviceManager::AudioDeviceSetup& setup)
{
    audioDeviceManager.initialiseWithDefaultDevices(2, 2);
    
    // Find and set device type
    for (auto* type : audioDeviceManager.getAvailableDeviceTypes()) {
        auto outputNames = type->getDeviceNames(false);
        auto inputNames = type->getDeviceNames(true);
        if (outputNames.contains(setup.outputDeviceName) ||
            inputNames.contains(setup.inputDeviceName)) {
            audioDeviceManager.setCurrentAudioDeviceType(type->getTypeName(), false);
            break;
        }
    }
    
    auto error = audioDeviceManager.setAudioDeviceSetup(setup, true);
    if (error.isNotEmpty()) {
        juce::AlertWindow::showMessageBoxAsync(juce::AlertWindow::WarningIcon,
            "Error", "Failed to initialize: " + error);
        return;
    }
    
    // Enable all channels
    auto* device = audioDeviceManager.getCurrentAudioDevice();
    if (device) {
        juce::AudioDeviceManager::AudioDeviceSetup modSetup = setup;
        modSetup.inputChannels.clear();
        modSetup.outputChannels.clear();
        for (int i = 0; i < device->getInputChannelNames().size(); ++i) 
            modSetup.inputChannels.setBit(i, true);
        for (int i = 0; i < device->getOutputChannelNames().size(); ++i) 
            modSetup.outputChannels.setBit(i, true);
        modSetup.useDefaultInputChannels = modSetup.useDefaultOutputChannels = false;
        audioDeviceManager.setAudioDeviceSetup(modSetup, true);
    }
    
    engine = std::make_unique<LatencyMeasurement::LatencyMeasurementEngine>();
    if (device) engine->audioDeviceAboutToStart(device);
    audioDeviceManager.addAudioCallback(engine.get());
    
    if (device) {
        statusLabel.setText(juce::String("Device: ") + device->getName() + 
            " | " + juce::String(device->getCurrentSampleRate(), 0) + " Hz", 
            juce::dontSendNotification);
    }
}

void MainComponent::paint(juce::Graphics& g) { g.fillAll(juce::Colour(0xff1a1a1a)); }

void MainComponent::resized()
{
    auto b = getLocalBounds().reduced(20);
    titleLabel.setBounds(b.removeFromTop(40));
    b.removeFromTop(10);
    
    instructionsText.setBounds(b.removeFromTop(140));
    b.removeFromTop(10);
    startButton.setBounds(b.removeFromTop(40).reduced(200, 0));
    b.removeFromTop(20);
    
    statusLabel.setBounds(b.removeFromBottom(25));
    resultsText.setBounds(b);
}

void MainComponent::buttonClicked(juce::Button*) { startMeasurement(); }

void MainComponent::startMeasurement()
{
    if (!engine || measurementInProgress) return;
    
    resultsText.setText("Running test...");
    statusLabel.setText("Measurement in progress...", juce::dontSendNotification);
    startButton.setEnabled(false);
    measurementInProgress = true;
    
    if (engine->startMeasurement())
        startTimer(50);
    else {
        resultsText.setText("Error: Failed to start measurement");
        startButton.setEnabled(true);
        measurementInProgress = false;
    }
}

void MainComponent::timerCallback()
{
    if (engine && engine->isMeasurementComplete()) {
        stopTimer();
        auto result = engine->computeLatency();
        displayResults(result);
        startButton.setEnabled(true);
        measurementInProgress = false;
        statusLabel.setText(result.isValid ? "Complete" : "Check results", juce::dontSendNotification);
    }
}

void MainComponent::displayResults(const LatencyMeasurement::LatencyResult& result)
{
    juce::String text;
    
    if (result.isValid) {
        text << "=== MEASUREMENT SUCCESSFUL ===\n\n"
             << result.toString() << "\n\n"
             << "This is the total round-trip latency from output to input.";
    } else {
        text << "=== MEASUREMENT FAILED ===\n\n"
             << result.toString();
    }
    
    resultsText.setText(text);
}
