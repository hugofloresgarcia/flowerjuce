#pragma once

#include "GenerationTimelineStore.h"
#include "StreamGenProcessor.h"
#include "InferenceWorker.h"
#include "WaveformTimelineComponent.h"
#include "StageTimingComponent.h"
#include "GenerationStatusComponent.h"
#include "ControlsComponent.h"
#include "MixerComponent.h"
#include "SimulationWindow.h"
#include "OperatorDashboardWindow.h"

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
    ///     use_mlx_vae: VAE via MLX Metal when sao_inference built with SAO_ENABLE_MLX.
    void load_pipeline(
        const std::string& manifest_path,
        bool use_cuda,
        bool use_coreml = false,
        bool use_mlx_vae = false);

    void resized() override;
    void paint(juce::Graphics& g) override;

private:
    void timerCallback() override;
    void show_audio_settings();
    void show_simulation_window();
    void show_operator_dashboard();
    void load_warm_start();
    void try_load_default_audio_from_repo(const juce::File& manifest_file);

    /// Set scheduler + UI to 120 BPM / 4/4 musical grid when default repo test assets are loaded.
    void apply_default_120bpm_grid_preset();

    /// Re-register the processor with the device manager after pipeline load and recover if I/O is down.
    void reattach_audio_callback_after_pipeline_load();

    /// Stop audio and the inference worker, clear timeline/rings/scheduler transport, then restart.
    void reset_session();

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
    std::unique_ptr<OperatorDashboardWindow> m_operator_window;

    juce::Label m_title_label;

    std::vector<JobTimelineRecord> m_timeline_paint_cache;
    std::vector<float> m_sax_wave_min;
    std::vector<float> m_sax_wave_max;
    std::vector<float> m_drums_wave_min;
    std::vector<float> m_drums_wave_max;
    std::vector<float> m_drums_wave_warm_min;
    std::vector<float> m_drums_wave_warm_max;
    std::vector<float> m_drums_wave_gen_min;
    std::vector<float> m_drums_wave_gen_max;
};

} // namespace streamgen
