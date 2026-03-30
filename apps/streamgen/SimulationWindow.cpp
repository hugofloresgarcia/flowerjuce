#include "SimulationWindow.h"

namespace streamgen {

SimulationPanel::SimulationPanel(StreamGenProcessor& processor)
    : m_processor(processor)
{
    m_load_button.onClick = [this]() { load_file(); };
    addAndMakeVisible(m_load_button);

    m_play_button.onClick = [this]()
    {
        m_processor.simulation_playing.store(true, std::memory_order_relaxed);
        update_transport_state();
    };
    addAndMakeVisible(m_play_button);

    m_pause_button.onClick = [this]()
    {
        m_processor.simulation_playing.store(false, std::memory_order_relaxed);
        update_transport_state();
    };
    addAndMakeVisible(m_pause_button);

    m_stop_button.onClick = [this]()
    {
        m_processor.simulation_playing.store(false, std::memory_order_relaxed);
        m_processor.simulation_position.store(0, std::memory_order_relaxed);
        update_transport_state();
    };
    addAndMakeVisible(m_stop_button);

    m_loop_toggle.setColour(juce::ToggleButton::textColourId, juce::Colour(0xffe0e0e0));
    m_loop_toggle.setToggleState(false, juce::dontSendNotification);
    m_loop_toggle.onClick = [this]()
    {
        m_processor.simulation_looping.store(m_loop_toggle.getToggleState(), std::memory_order_relaxed);
    };
    addAndMakeVisible(m_loop_toggle);

    m_speed_label.setText("Speed:", juce::dontSendNotification);
    m_speed_label.setColour(juce::Label::textColourId, juce::Colour(0xffe0e0e0));
    addAndMakeVisible(m_speed_label);

    m_speed_slider.setRange(0.25, 4.0, 0.25);
    m_speed_slider.setValue(1.0);
    m_speed_slider.setTextBoxStyle(juce::Slider::TextBoxRight, false, 40, 20);
    m_speed_slider.setColour(juce::Slider::textBoxTextColourId, juce::Colour(0xffe0e0e0));
    m_speed_slider.onValueChange = [this]()
    {
        m_processor.simulation_speed.store(
            static_cast<float>(m_speed_slider.getValue()), std::memory_order_relaxed);
    };
    addAndMakeVisible(m_speed_slider);

    m_file_label.setText("No file loaded", juce::dontSendNotification);
    m_file_label.setColour(juce::Label::textColourId, juce::Colour(0xffaaaaaa));
    addAndMakeVisible(m_file_label);

    m_position_label.setText("00:00.000 / 00:00.000", juce::dontSendNotification);
    m_position_label.setColour(juce::Label::textColourId, juce::Colour(0xffe0e0e0));
    m_position_label.setFont(juce::Font(juce::Font::getDefaultMonospacedFontName(), 12.0f, juce::Font::plain));
    addAndMakeVisible(m_position_label);

    m_position_slider.setRange(0.0, 1.0, 0.001);
    m_position_slider.setTextBoxStyle(juce::Slider::NoTextBox, true, 0, 0);
    m_position_slider.onDragEnd = [this]()
    {
        int64_t total = m_processor.simulation_total_samples.load(std::memory_order_relaxed);
        int64_t new_pos = static_cast<int64_t>(m_position_slider.getValue() * total);
        m_processor.simulation_position.store(new_pos, std::memory_order_relaxed);
    };
    addAndMakeVisible(m_position_slider);

    setSize(420, 260);
    startTimerHz(15);
    update_transport_state();
}

SimulationPanel::~SimulationPanel()
{
    stopTimer();
}

void SimulationPanel::resized()
{
    auto bounds = getLocalBounds().reduced(8);

    const int row_height = 26;
    const int spacing = 4;
    const int button_width = 60;

    // Row 1: File info
    auto row = bounds.removeFromTop(row_height);
    m_load_button.setBounds(row.removeFromLeft(100));
    row.removeFromLeft(spacing);
    m_file_label.setBounds(row);
    bounds.removeFromTop(spacing);

    // Row 2: Transport buttons
    row = bounds.removeFromTop(row_height);
    m_play_button.setBounds(row.removeFromLeft(button_width));
    row.removeFromLeft(spacing);
    m_pause_button.setBounds(row.removeFromLeft(button_width));
    row.removeFromLeft(spacing);
    m_stop_button.setBounds(row.removeFromLeft(button_width));
    row.removeFromLeft(spacing * 2);
    m_loop_toggle.setBounds(row.removeFromLeft(70));
    bounds.removeFromTop(spacing);

    // Row 3: Position slider + label
    row = bounds.removeFromTop(row_height);
    m_position_label.setBounds(row.removeFromRight(160));
    m_position_slider.setBounds(row);
    bounds.removeFromTop(spacing);

    // Row 4: Speed
    row = bounds.removeFromTop(row_height);
    m_speed_label.setBounds(row.removeFromLeft(50));
    m_speed_slider.setBounds(row);
}

void SimulationPanel::paint(juce::Graphics& g)
{
    g.fillAll(juce::Colour(0xff1a1a2e));
}

void SimulationPanel::timerCallback()
{
    int64_t pos = m_processor.simulation_position.load(std::memory_order_relaxed);
    int64_t total = m_processor.simulation_total_samples.load(std::memory_order_relaxed);

    if (total > 0)
    {
        double frac = static_cast<double>(pos) / static_cast<double>(total);
        m_position_slider.setValue(frac, juce::dontSendNotification);

        int sr = m_processor.current_sample_rate();
        juce::String pos_str = juce::String(format_time(pos, sr))
            + " / " + juce::String(format_time(total, sr));
        m_position_label.setText(pos_str, juce::dontSendNotification);
    }
}

void SimulationPanel::load_file()
{
    auto chooser = std::make_shared<juce::FileChooser>(
        "Load Simulation WAV", juce::File(), "*.wav;*.aif;*.aiff;*.flac");

    chooser->launchAsync(juce::FileBrowserComponent::openMode
                         | juce::FileBrowserComponent::canSelectFiles,
        [this, chooser](const juce::FileChooser& fc)
        {
            auto result = fc.getResult();
            if (result == juce::File())
            {
                DBG("SimulationPanel: file chooser cancelled");
                return;
            }

            if (m_processor.load_simulation_file(result))
            {
                m_loaded_filename = result.getFileName();
                m_file_label.setText(m_loaded_filename, juce::dontSendNotification);
                update_transport_state();
            }
        });
}

void SimulationPanel::update_transport_state()
{
    bool active = m_processor.simulation_active.load(std::memory_order_relaxed);
    bool playing = m_processor.simulation_playing.load(std::memory_order_relaxed);

    m_play_button.setEnabled(active && !playing);
    m_pause_button.setEnabled(active && playing);
    m_stop_button.setEnabled(active);
    m_position_slider.setEnabled(active);
}

} // namespace streamgen
