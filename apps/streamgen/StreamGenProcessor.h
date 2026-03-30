#pragma once

#include "TimeRuler.h"
#include "GenerationScheduler.h"

#include <juce_audio_devices/juce_audio_devices.h>
#include <juce_audio_formats/juce_audio_formats.h>

#include <atomic>
#include <vector>
#include <mutex>

namespace streamgen {

/// Audio engine for StreamGen Live.
///
/// Owns the dual ring buffers (sax input + drums output), drives the audio
/// callback, and provides thread-safe access for the inference worker to
/// snapshot input and write output.
///
/// Threading contract:
///   - Audio thread calls audio_device_callback()
///   - Worker thread calls snapshot_input(), snapshot_output(), write_output()
///   - UI thread reads waveform data via get_recent_*() and reads atomics
class StreamGenProcessor : public juce::AudioIODeviceCallback {
public:
    static constexpr int NUM_CHANNELS = 2;
    static constexpr int RING_BUFFER_SECONDS = 30;

    StreamGenProcessor();
    ~StreamGenProcessor() override = default;

    /// Initialize with model constants. Must be called before audio starts.
    ///
    /// Args:
    ///     constants: Model timing parameters from Zenon config.
    void configure(const ModelConstants& constants);

    // --- juce::AudioIODeviceCallback ---
    void audioDeviceIOCallbackWithContext(
        const float* const* input_data,
        int num_input_channels,
        float* const* output_data,
        int num_output_channels,
        int num_samples,
        const juce::AudioIODeviceCallbackContext& context) override;

    void audioDeviceAboutToStart(juce::AudioIODevice* device) override;
    void audioDeviceStopped() override;

    // --- Simulation mode ---

    /// Load a WAV file for simulation. Called from the UI thread.
    /// Returns true on success.
    ///
    /// Args:
    ///     file: The WAV file to load.
    bool load_simulation_file(const juce::File& file);

    /// Clear the simulation buffer and revert to live mic input.
    void clear_simulation();

    std::atomic<bool> simulation_active{false};
    std::atomic<bool> simulation_playing{false};
    std::atomic<bool> simulation_looping{false};
    std::atomic<float> simulation_speed{1.0f};

    /// Current playback position within the simulation file (in samples).
    std::atomic<int64_t> simulation_position{0};

    /// Total length of the loaded simulation file (in samples).
    std::atomic<int64_t> simulation_total_samples{0};

    // --- Warm-start ---

    /// Load a WAV file as the warm-start drum track.
    /// Zero-padded at the front if shorter than model window.
    ///
    /// Args:
    ///     file: The WAV file to load.
    bool load_warm_start(const juce::File& file);

    std::atomic<bool> warm_start_playing{false};
    std::atomic<bool> warm_start_looping{true};

    // --- Worker thread interface ---

    /// Snapshot the most recent model_window_samples of sax input from the ring buffer.
    /// Called from the worker thread.
    ///
    /// Args:
    ///     window_start: Absolute sample position for the start of the snapshot.
    ///     num_samples: Number of samples to snapshot.
    ///
    /// Returns:
    ///     Stereo audio in row-major (2, num_samples) layout, or empty if not enough data.
    std::vector<float> snapshot_input(int64_t window_start, int64_t num_samples);

    /// Snapshot the most recent drums output from the ring buffer.
    /// Called from the worker thread.
    ///
    /// Args:
    ///     window_start: Absolute sample position for the start of the snapshot.
    ///     num_samples: Number of samples to snapshot.
    ///
    /// Returns:
    ///     Stereo audio in row-major (2, num_samples) layout, or empty if not enough data.
    std::vector<float> snapshot_output(int64_t window_start, int64_t num_samples);

    /// Write generated audio into the output ring buffer with overlap-add crossfade.
    /// Called from the worker thread.
    ///
    /// Args:
    ///     audio: Stereo audio in row-major (2, num_samples) layout.
    ///     start_sample: Absolute sample position where the generated portion begins.
    ///     num_samples: Number of samples in the generated portion.
    ///     crossfade_samples: Length of the crossfade region at the start.
    void write_output(const std::vector<float>& audio, int64_t start_sample,
                      int64_t num_samples, int crossfade_samples);

    // --- UI readout ---

    /// Copy recent waveform data for display. Returns samples for the last `duration_samples` samples.
    ///
    /// Args:
    ///     duration_samples: Number of recent samples to return.
    ///
    /// Returns:
    ///     Mono downmixed waveform data for display.
    std::vector<float> get_recent_input_waveform(int duration_samples);
    std::vector<float> get_recent_output_waveform(int duration_samples);

    // --- Gain controls ---
    std::atomic<float> sax_gain{1.0f};
    std::atomic<float> drums_gain{1.0f};

    /// Feed mono audio directly into the input ring buffer and advance the
    /// scheduler. Used by the CLI to bypass the audio device callback.
    ///
    /// Args:
    ///     mono_input: Mono audio samples.
    ///     num_samples: Number of samples to feed.
    void feed_audio(const float* mono_input, int num_samples);

    GenerationScheduler& scheduler() { return m_scheduler; }
    const ModelConstants& constants() const { return m_constants; }
    int current_sample_rate() const { return m_current_sample_rate; }

private:
    void write_sax_to_ring(const float* mono_input, int num_samples);
    void read_drums_from_ring(float* left, float* right, int num_samples);
    int64_t absolute_to_ring_index(int64_t absolute_sample) const;

    ModelConstants m_constants;
    GenerationScheduler m_scheduler;
    int m_current_sample_rate = 44100;

    // Ring buffers: interleaved [L, R, L, R, ...] for simplicity
    // Indexed by absolute_sample_pos % ring_buffer_size
    int m_ring_buffer_size = 0; // in frames (one frame = NUM_CHANNELS samples)
    std::vector<float> m_input_ring;   // sax input (stereo, duplicated from mono)
    std::vector<float> m_output_ring;  // drums output (stereo)

    // Simulation file data (mono, 44100 Hz, loaded on UI thread)
    std::mutex m_sim_mutex;
    std::vector<float> m_sim_audio;
    double m_sim_playback_pos = 0.0;

    // Warm-start data (stereo interleaved)
    std::mutex m_warm_mutex;
    std::vector<float> m_warm_audio;
    int64_t m_warm_length_frames = 0;
    int64_t m_warm_playback_pos = 0;

    juce::AudioFormatManager m_format_manager;
};

} // namespace streamgen
