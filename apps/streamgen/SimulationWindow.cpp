#include "SimulationWindow.h"
#include "LayerCakeLookAndFeel.h"
#include "MusicalTime.h"

namespace streamgen {

SimulationPanel::SimulationPanel(StreamGenProcessor& processor)
    : m_processor(processor)
{
    LayerCakeLookAndFeel::setControlButtonType(m_load_button, LayerCakeLookAndFeel::ControlButtonType::Preset);
    m_load_button.onClick = [this]() { load_file(); };
    addAndMakeVisible(m_load_button);

    LayerCakeLookAndFeel::setControlButtonType(m_play_button, LayerCakeLookAndFeel::ControlButtonType::Clock);
    m_play_button.onClick = [this]()
    {
        if (m_processor.simulation_snap_to_bar_on_play.load(std::memory_order_relaxed)
            && m_processor.scheduler().musical_time_enabled.load(std::memory_order_relaxed))
            m_processor.snap_simulation_position_to_bar_grid();
        m_processor.simulation_playing.store(true, std::memory_order_relaxed);
        update_transport_state();
    };
    addAndMakeVisible(m_play_button);

    LayerCakeLookAndFeel::setControlButtonType(m_pause_button, LayerCakeLookAndFeel::ControlButtonType::Pattern);
    m_pause_button.onClick = [this]()
    {
        m_processor.simulation_playing.store(false, std::memory_order_relaxed);
        update_transport_state();
    };
    addAndMakeVisible(m_pause_button);

    LayerCakeLookAndFeel::setControlButtonType(m_stop_button, LayerCakeLookAndFeel::ControlButtonType::Record);
    m_stop_button.onClick = [this]()
    {
        m_processor.simulation_playing.store(false, std::memory_order_relaxed);
        m_processor.set_simulation_playback_sample(0);
        update_transport_state();
    };
    addAndMakeVisible(m_stop_button);

    m_loop_toggle.setToggleState(false, juce::dontSendNotification);
    m_loop_toggle.onClick = [this]()
    {
        m_processor.simulation_looping.store(m_loop_toggle.getToggleState(), std::memory_order_relaxed);
    };
    addAndMakeVisible(m_loop_toggle);

    m_snap_bar_toggle.setToggleState(
        m_processor.simulation_snap_to_bar_on_play.load(std::memory_order_relaxed),
        juce::dontSendNotification);
    m_snap_bar_toggle.onClick = [this]()
    {
        m_processor.simulation_snap_to_bar_on_play.store(
            m_snap_bar_toggle.getToggleState(),
            std::memory_order_relaxed);
    };
    addAndMakeVisible(m_snap_bar_toggle);

    m_speed_label.setText("Speed:", juce::dontSendNotification);
    addAndMakeVisible(m_speed_label);

    m_speed_slider.setRange(0.25, 4.0, 0.25);
    m_speed_slider.setValue(1.0);
    m_speed_slider.setTextBoxStyle(juce::Slider::TextBoxRight, false, 40, 20);
    m_speed_slider.onValueChange = [this]()
    {
        m_processor.simulation_speed.store(
            static_cast<float>(m_speed_slider.getValue()), std::memory_order_relaxed);
    };
    addAndMakeVisible(m_speed_slider);

    m_file_label.setText("No file loaded", juce::dontSendNotification);
    m_file_label.setColour(
        juce::Label::textColourId,
        getLookAndFeel().findColour(juce::Label::textColourId).withAlpha(0.55f));
    addAndMakeVisible(m_file_label);

    m_position_label.setText("00:00.000 / 00:00.000", juce::dontSendNotification);
    m_position_label.setFont(
        juce::Font(juce::FontOptions(juce::Font::getDefaultMonospacedFontName(), 12.0f, juce::Font::plain)));
    addAndMakeVisible(m_position_label);

    m_position_slider.setRange(0.0, 1.0, 0.001);
    m_position_slider.setTextBoxStyle(juce::Slider::NoTextBox, true, 0, 0);
    m_position_slider.onDragEnd = [this]()
    {
        const int64_t total = m_processor.simulation_total_samples.load(std::memory_order_relaxed);
        const int64_t new_pos = static_cast<int64_t>(m_position_slider.getValue() * static_cast<double>(total));
        m_processor.set_simulation_playback_sample(new_pos);
    };
    addAndMakeVisible(m_position_slider);

    setSize(420, 300);
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

    auto row = bounds.removeFromTop(row_height);
    m_load_button.setBounds(row.removeFromLeft(100));
    row.removeFromLeft(spacing);
    m_file_label.setBounds(row);
    bounds.removeFromTop(spacing);

    row = bounds.removeFromTop(row_height);
    m_play_button.setBounds(row.removeFromLeft(button_width));
    row.removeFromLeft(spacing);
    m_pause_button.setBounds(row.removeFromLeft(button_width));
    row.removeFromLeft(spacing);
    m_stop_button.setBounds(row.removeFromLeft(button_width));
    row.removeFromLeft(spacing * 2);
    m_loop_toggle.setBounds(row.removeFromLeft(70));
    row.removeFromLeft(spacing);
    m_snap_bar_toggle.setBounds(row.removeFromLeft(170));
    bounds.removeFromTop(spacing);

    row = bounds.removeFromTop(row_height);
    m_position_label.setBounds(row.removeFromRight(220));
    m_position_slider.setBounds(row);
    bounds.removeFromTop(spacing);

    row = bounds.removeFromTop(row_height);
    m_speed_label.setBounds(row.removeFromLeft(50));
    m_speed_slider.setBounds(row);
}

void SimulationPanel::paint(juce::Graphics& g)
{
    g.fillAll(getLookAndFeel().findColour(juce::ComboBox::backgroundColourId));
}

void SimulationPanel::timerCallback()
{
    const int64_t pos = m_processor.simulation_position.load(std::memory_order_relaxed);
    const int64_t total = m_processor.simulation_total_samples.load(std::memory_order_relaxed);

    if (total > 0)
    {
        const double frac = static_cast<double>(pos) / static_cast<double>(total);
        m_position_slider.setValue(frac, juce::dontSendNotification);

        const int sr = m_processor.current_sample_rate();
        juce::String pos_str = juce::String(format_time(pos, sr))
            + " / " + juce::String(format_time(total, sr));

        const auto& sched = m_processor.scheduler();
        if (sched.musical_time_enabled.load(std::memory_order_relaxed))
        {
            const float bpm = sched.bpm.load(std::memory_order_relaxed);
            const int bpb = juce::jmax(1, sched.time_sig_numerator.load(std::memory_order_relaxed));
            const double bpm_d = static_cast<double>(juce::jlimit(20.0f, 400.0f, bpm));
            pos_str += "  ";
            pos_str += juce::String(format_bar_beat(pos, sr, bpm_d, bpb));
        }

        m_position_label.setText(pos_str, juce::dontSendNotification);
    }
}

void SimulationPanel::load_file()
{
    auto chooser = std::make_shared<juce::FileChooser>(
        "Load Simulation WAV", juce::File(), "*.wav;*.aif;*.aiff;*.flac");

    chooser->launchAsync(
        juce::FileBrowserComponent::openMode | juce::FileBrowserComponent::canSelectFiles,
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

void SimulationPanel::sync_from_processor()
{
    juce::String name = m_processor.simulation_display_name();
    if (name.isNotEmpty())
    {
        m_loaded_filename = name;
        m_file_label.setText(m_loaded_filename, juce::dontSendNotification);
    }
    m_snap_bar_toggle.setToggleState(
        m_processor.simulation_snap_to_bar_on_play.load(std::memory_order_relaxed),
        juce::dontSendNotification);
    update_transport_state();
}

void SimulationPanel::update_transport_state()
{
    const bool has_buffer = m_processor.simulation_total_samples.load(std::memory_order_relaxed) > 0;
    const bool playing = m_processor.simulation_playing.load(std::memory_order_relaxed);

    m_play_button.setEnabled(has_buffer && !playing);
    m_pause_button.setEnabled(has_buffer && playing);
    m_stop_button.setEnabled(has_buffer);
    m_position_slider.setEnabled(has_buffer);
}

} // namespace streamgen
