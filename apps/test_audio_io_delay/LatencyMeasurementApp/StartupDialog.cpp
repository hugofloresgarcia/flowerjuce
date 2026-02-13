#include "StartupDialog.h"

StartupDialog::StartupDialog(juce::AudioDeviceManager& deviceManager)
    : audioDeviceManager(deviceManager),
      titleLabel("", "Audio Latency Measurement"),
      instructionsLabel("", "Select audio devices. Place speakers near microphone for feedback loop."),
      audioDeviceSelector(deviceManager, 1, 2, 1, 2, false, false, false, false)
{
    titleLabel.setJustificationType(juce::Justification::centred);
    titleLabel.setFont(juce::FontOptions(20.0f).withStyle("Bold"));
    addAndMakeVisible(titleLabel);
    
    instructionsLabel.setJustificationType(juce::Justification::topLeft);
    instructionsLabel.setColour(juce::Label::textColourId, juce::Colours::lightgrey);
    addAndMakeVisible(instructionsLabel);
    
    addAndMakeVisible(audioDeviceSelector);
    
    okButton.addListener(this);
    cancelButton.addListener(this);
    addAndMakeVisible(okButton);
    addAndMakeVisible(cancelButton);
    
    setSize(600, 600);
}

void StartupDialog::resized()
{
    auto b = getLocalBounds().reduced(20);
    titleLabel.setBounds(b.removeFromTop(40));
    b.removeFromTop(10);
    instructionsLabel.setBounds(b.removeFromTop(40));
    b.removeFromTop(10);
    
    auto buttons = b.removeFromBottom(40);
    cancelButton.setBounds(buttons.removeFromRight(100).reduced(5));
    okButton.setBounds(buttons.removeFromRight(100).reduced(5));
    b.removeFromBottom(10);
    
    audioDeviceSelector.setBounds(b);
}

void StartupDialog::paint(juce::Graphics& g) { g.fillAll(juce::Colour(0xff2a2a2a)); }

void StartupDialog::buttonClicked(juce::Button* button)
{
    if (button == &okButton) {
        auto* device = audioDeviceManager.getCurrentAudioDevice();
        if (!device || device->getInputChannelNames().isEmpty() || device->getOutputChannelNames().isEmpty()) {
            juce::AlertWindow::showMessageBoxAsync(juce::AlertWindow::WarningIcon,
                "Error", "Please select both input and output devices.");
            return;
        }
        okClicked = true;
    }
    
    if (auto* dw = findParentComponentOfClass<juce::DialogWindow>())
        dw->exitModalState(button == &okButton ? 1 : 0);
}

juce::AudioDeviceManager::AudioDeviceSetup StartupDialog::getDeviceSetup() const
{
    juce::AudioDeviceManager::AudioDeviceSetup setup;
    audioDeviceManager.getAudioDeviceSetup(setup);
    return setup;
}
