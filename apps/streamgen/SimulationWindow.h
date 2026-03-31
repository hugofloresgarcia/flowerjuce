#pragma once

#include "StreamGenProcessor.h"

#include <juce_gui_basics/juce_gui_basics.h>
#include <functional>

namespace streamgen {

/// Pop-up window for loading and playing a saxophone simulation WAV file.
///
/// Streams the loaded audio into the input ring buffer at real-time rate,
/// replacing live mic input. Provides transport controls and speed multiplier.
class SimulationPanel : public juce::Component,
                        private juce::Timer {
public:
    /// Args:
    ///     processor: The audio processor to load simulation files into.
    explicit SimulationPanel(StreamGenProcessor& processor);
    ~SimulationPanel() override;

    void resized() override;
    void paint(juce::Graphics& g) override;

    /// Refresh file label and transport buttons from processor (e.g. after default load).
    void sync_from_processor();

private:
    void timerCallback() override;
    void load_file();
    void update_transport_state();

    StreamGenProcessor& m_processor;

    juce::TextButton m_load_button{"Load WAV..."};
    juce::TextButton m_play_button{"Play"};
    juce::TextButton m_pause_button{"Pause"};
    juce::TextButton m_stop_button{"Stop"};
    juce::ToggleButton m_loop_toggle{"Loop"};

    juce::Label m_speed_label;
    juce::Slider m_speed_slider;

    juce::Label m_file_label;
    juce::Label m_position_label;
    juce::Slider m_position_slider;

    juce::String m_loaded_filename;
};

/// DocumentWindow wrapper for the SimulationPanel.
class SimulationWindow : public juce::DocumentWindow {
public:
    /// Args:
    ///     processor: The audio processor to load simulation files into.
    explicit SimulationWindow(StreamGenProcessor& processor)
        : juce::DocumentWindow("simulation", juce::Colours::black,
                               juce::DocumentWindow::closeButton)
    {
        setContentOwned(new SimulationPanel(processor), true);
        setResizable(true, false);
        centreWithSize(420, 260);
        setUsingNativeTitleBar(true);
    }

    void closeButtonPressed() override { setVisible(false); }

    void sync_from_processor()
    {
        auto* panel = dynamic_cast<SimulationPanel*>(getContentComponent());
        if (panel != nullptr)
            panel->sync_from_processor();
    }
};

} // namespace streamgen
