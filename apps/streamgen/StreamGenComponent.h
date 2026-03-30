#pragma once

#include "StreamGenProcessor.h"
#include "InferenceWorker.h"
#include "WaveformTimelineComponent.h"
#include "StageTimingComponent.h"
#include "GenerationStatusComponent.h"
#include "ControlsComponent.h"
#include "MixerComponent.h"
#include "SimulationWindow.h"

#include <juce_gui_basics/juce_gui_basics.h>
#include <juce_audio_utils/juce_audio_utils.h>

#include <memory>
#include <string>

namespace streamgen {

/// Main dashboard component for StreamGen Live.
///
/// Composes all sub-components: two scrolling waveforms (sax + drums),
/// stage timing bars, generation status, controls, and mixer.
/// Runs a timer to poll the processor and update the UI.
class StreamGenComponent : public juce::Component,
                           private juce::Timer {
public:
    /// Args:
    ///     processor: The audio processor owning ring buffers and scheduler.
    ///     device_manager: JUCE audio device manager for the settings dialog.
    StreamGenComponent(StreamGenProcessor& processor,
                       juce::AudioDeviceManager& device_manager);

    ~StreamGenComponent() override;

    /// Load the inference pipeline. Must be called after construction.
    ///
    /// Args:
    ///     manifest_path: Path to zenon_pipeline_manifest.json.
    ///     use_cuda: Whether to use CUDA execution provider.
    ///     use_coreml: Whether to use CoreML execution provider (macOS).
    void load_pipeline(const std::string& manifest_path, bool use_cuda, bool use_coreml = false);

    void resized() override;
    void paint(juce::Graphics& g) override;

private:
    void timerCallback() override;
    void show_audio_settings();
    void show_simulation_window();
    void load_warm_start();

    StreamGenProcessor& m_processor;
    juce::AudioDeviceManager& m_device_manager;
    std::unique_ptr<InferenceWorker> m_worker;

    WaveformTimelineComponent m_sax_waveform;
    WaveformTimelineComponent m_drums_waveform;
    StageTimingComponent m_stage_timing;
    GenerationStatusComponent m_gen_status;
    ControlsComponent m_controls;
    MixerComponent m_mixer;

    std::unique_ptr<SimulationWindow> m_simulation_window;

    juce::Label m_title_label;
};

} // namespace streamgen
