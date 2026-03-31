#pragma once

#include "TimeRuler.h"
#include "GenerationScheduler.h"
#include "AudioThreadTelemetry.h"

#include <juce_audio_devices/juce_audio_devices.h>
#include <juce_audio_formats/juce_audio_formats.h>

#include <atomic>
#include <cstdint>
#include <mutex>
#include <vector>

namespace streamgen {

class GenerationTimelineStore;

/// Audio engine for StreamGen Live.
///
/// Owns the dual ring buffers (sax input + drums output), drives the audio
/// callback, and provides thread-safe access for the inference worker to
/// snapshot input and write output.
///
/// Threading contract:
///   - Audio thread calls audio_device_callback()
///   - Worker thread calls snapshot_input(), snapshot_output(), write_output()
///   - UI thread reads waveform buckets via fill_recent_*_waveform_buckets() and atomics
class StreamGenProcessor : public juce::AudioIODeviceCallback {
public:
    static constexpr int NUM_CHANNELS = 2;
    static constexpr int RING_BUFFER_SECONDS = 30;

    StreamGenProcessor();
    ~StreamGenProcessor() override;

    /// Clear timeline history, reset scheduler sample position and job queue, zero I/O rings,
    /// stop simulation playback and rewind transports, and reset audio telemetry.
    ///
    /// The audio callback must not be running (call after removeAudioCallback). The inference
    /// worker thread should be stopped so no job completes after this reset.
    void reset_timeline_and_transport();

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

    /// Short name of the last loaded simulation file (for UI). Empty if none.
    juce::String simulation_display_name() const;

    /// Set simulation file playback position (samples). Thread-safe for UI; matches `simulation_position`.
    void set_simulation_playback_sample(int64_t sample);

    /// Snap simulation playback to the start of the current bar in file-time (uses BPM and time signature).
    void snap_simulation_position_to_bar_grid();

    std::atomic<bool> simulation_playing{false};
    std::atomic<bool> simulation_looping{false};
    std::atomic<float> simulation_speed{1.0f};

    /// Current playback position within the simulation file (in samples).
    std::atomic<int64_t> simulation_position{0};

    /// Total length of the loaded simulation file (in samples).
    std::atomic<int64_t> simulation_total_samples{0};

    // --- Warm-start ---

    /// Load a WAV file as the warm-start drum track. Timeline positions use the **device** sample
    /// clock (same as the metronome); file frames are indexed with `timeline * file_sr / device_sr`
    /// so 120 BPM content authored at 44100 Hz stays on-grid when the interface runs at 48 kHz.
    ///
    /// Args:
    ///     file: The WAV file to load.
    ///     start_playback: If true, route warm-start audio to the output immediately.
    bool load_warm_start(const juce::File& file, bool start_playback = true);

    /// Route loaded warm-start audio to the output (no-op if no warm buffer loaded).
    void set_warm_start_playing(bool playing);

    /// True if a warm-start file has been loaded (length > 0).
    bool warm_start_has_audio() const;

    std::atomic<bool> warm_start_playing{false};
    std::atomic<bool> warm_start_looping{true};

    /// When true, output mixer repeats the last completed generation for ring frames that are still silent.
    std::atomic<bool> loop_last_generation{true};

    /// When true and musical time is enabled, simulation Play snaps file position to a bar boundary.
    std::atomic<bool> simulation_snap_to_bar_on_play{true};

    /// Metronome: quarter-note clicks when musical time is on (downbeat uses a louder impulse).
    std::atomic<bool> click_track_enabled{false};
    /// Linear gain 0..1 applied to click impulses on the main output bus.
    std::atomic<float> click_track_volume{0.35f};

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

    /// Fill per-pixel min/max buckets for the last `duration_samples` of audio (mono).
    ///
    /// Ring reads are subsampled within each bucket (bounded reads per column). Call from the message thread.
    /// `out_min` / `out_max` must hold `num_buckets` elements.
    ///
    /// Args:
    ///     duration_samples: Time window length in samples (e.g. sample_rate * visible_seconds).
    ///     num_buckets: Typically component width in logical pixels (one column per bucket).
    ///     out_min: Receives minimum sample per bucket.
    ///     out_max: Receives maximum sample per bucket.
    void fill_recent_input_waveform_buckets(int duration_samples, int num_buckets, float* out_min, float* out_max);

    /// Same as fill_recent_input_waveform_buckets for the drums monitor ring (stereo downmixed to mono).
    void fill_recent_output_waveform_buckets(int duration_samples, int num_buckets, float* out_min, float* out_max);

    /// Drums monitor split by source: warm-start file (1) vs model output_ring (2); see m_drums_origin_ring.
    /// Arrays must hold num_buckets elements. Same visible window as fill_recent_output_waveform_buckets.
    void fill_recent_drums_source_buckets(
        int duration_samples,
        int num_buckets,
        float* warm_min,
        float* warm_max,
        float* gen_min,
        float* gen_max);

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

    /// Native sample rate of the loaded simulation WAV (Hz), for UI time/bar readouts.
    int simulation_file_native_sample_rate_hz() const;

    /// Native sample rate of the loaded warm-start WAV (Hz).
    int warm_file_native_sample_rate_hz() const;

    /// RT-safe audio levels; read from UI/logger threads via copy_snapshot().
    AudioThreadTelemetry& audio_telemetry() { return m_audio_telemetry; }
    const AudioThreadTelemetry& audio_telemetry() const { return m_audio_telemetry; }

    /// Timeline history for generation markers (UI). Null before construction completes.
    GenerationTimelineStore* timeline_store() { return m_timeline.get(); }
    const GenerationTimelineStore* timeline_store() const { return m_timeline.get(); }

private:
    void rebuild_ring_buffers(int ring_sample_rate);
    void write_sax_to_ring(const float* mono_input, int num_samples);
    void write_sax_to_ring_at(const float* mono_input, int num_samples, int64_t abs_block_start);
    void read_drums_from_ring(float* left, float* right, int num_samples);
    void output_ring_sample_at(int64_t absolute_sample, float& out_left, float& out_right) const;
    void commit_last_generation_snapshot(
        const std::vector<float>& gen_row_major_stereo,
        int64_t output_start_sample,
        int64_t num_samples);
    int64_t absolute_to_ring_index(int64_t absolute_sample) const;

    void rebuild_click_impulses();
    void mix_click_track_into(float* left, float* right, int num_samples, int64_t block_start_sample);

    ModelConstants m_constants;
    std::unique_ptr<GenerationTimelineStore> m_timeline;
    GenerationScheduler m_scheduler;
    AudioThreadTelemetry m_audio_telemetry;
    int m_current_sample_rate = 44100;

    // Ring buffers: interleaved [L, R, L, R, ...] for simplicity
    // Indexed by absolute_sample_pos % ring_buffer_size
    int m_ring_buffer_size = 0; // in frames (one frame = NUM_CHANNELS samples)
    std::vector<float> m_input_ring;   // sax input (stereo, duplicated from mono)
    std::vector<float> m_output_ring;  // generated drums (worker writes; warm path reads only)
    /// Last drum bus (pre-drums_gain) for UI waveform — audio thread only, never touched by worker.
    std::vector<float> m_drums_monitor_ring;
    /// Per ring frame: 0 = silence, 1 = warm-start file, 2 = model (output_ring). Audio thread writes only.
    std::vector<uint8_t> m_drums_origin_ring;

    // Simulation file data (mono at native file rate). Phase keeps scrub offset vs timeline.
    std::mutex m_sim_mutex;
    std::vector<float> m_sim_audio;
    double m_sim_file_phase = 0.0;
    std::atomic<int> m_sim_native_sample_rate_hz{44100};
    juce::String m_simulation_display_name;

    // Warm-start data (stereo interleaved at native file rate)
    mutable std::mutex m_warm_mutex;
    std::vector<float> m_warm_audio;
    int64_t m_warm_length_frames = 0;
    std::atomic<int> m_warm_native_sample_rate_hz{44100};

    /// Grows in audioDeviceAboutToStart to max block size; simulation path fills then single ring write.
    std::vector<float> m_sim_callback_scratch;

    /// Last completed generation (row-major stereo: [L...][R...]), for loop-hold when ring is silent.
    mutable std::mutex m_last_gen_mutex;
    std::vector<float> m_last_gen_row_major;
    int64_t m_last_gen_output_start_sample = 0;
    int64_t m_last_gen_num_samples = 0;
    std::atomic<bool> m_last_gen_snapshot_valid{false};

    /// Short band-limited impulses for RT mixing (audio thread read-only after rebuild in aboutToStart).
    std::vector<float> m_click_impulse_beat;
    std::vector<float> m_click_impulse_downbeat;

    juce::AudioFormatManager m_format_manager;
};

} // namespace streamgen
