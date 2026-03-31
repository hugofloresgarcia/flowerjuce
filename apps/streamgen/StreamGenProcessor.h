#pragma once

#include "TimeRuler.h"
#include "GenerationScheduler.h"
#include "GenerationTimelineStore.h"

#include <juce_audio_devices/juce_audio_devices.h>
#include <juce_audio_formats/juce_audio_formats.h>

#include <atomic>
#include <cstdint>
#include <mutex>
#include <vector>

namespace streamgen {

/// One stereo sample from the drums output ring, with metadata for loop-hold substitution.
struct DrumsRingSample {
    float left = 0.0f;
    float right = 0.0f;
    /// True when `left`/`right` were taken from the last completed generation snapshot loop.
    bool from_last_gen_hold = false;
};

/// Audio engine for StreamGen Live.
///
/// Owns the dual ring buffers (**Python `streamgen_audio`** ring + generated **drums** ring),
/// drives the audio callback, and provides thread-safe access for the inference worker to
/// snapshot conditioning audio and write generated drums.
///
/// Threading contract:
///   - Audio thread calls audio_device_callback()
///   - Worker thread calls snapshot_streamgen_audio_for_vae(), snapshot_input_audio_for_vae(),
///     write_output(); raw ring snapshots remain for UI / CLI export
///   - UI thread reads waveform buckets via fill_recent_*_waveform_buckets() and atomics
class StreamGenProcessor : public juce::AudioIODeviceCallback {
public:
    static constexpr int NUM_CHANNELS = 2;
    static constexpr int RING_BUFFER_SECONDS = 30;

    StreamGenProcessor();
    ~StreamGenProcessor() override;

    /// Clear timeline history, reset scheduler sample position and job queue, zero I/O rings,
    /// and stop simulation playback and rewind transports.
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

    // --- Warmup audio ---

    /// Load a WAV file as the **drum** warmup stem (Python `input_audio`-style inpaint prefix).
    /// Used for monitoring on the main bus and, during inference, to fill DiT/VAE drum conditioning
    /// wherever the drums output ring and last-gen hold are still silent. Native file rate is preserved;
    /// resampling to the playback timeline matches `timeline * file_sr / playback_sr` (same as speaker path),
    /// then the worker resamples to the model/VAE rate when snapshotting.
    ///
    /// Args:
    ///     file: The WAV file to load.
    ///     start_playback: If true, route warmup audio to the output immediately.
    bool load_warmup_audio(const juce::File& file, bool start_playback = true);

    /// Route loaded warmup audio to the output (no-op if no warmup buffer loaded).
    void set_warmup_audio_playing(bool playing);

    /// True if warmup audio has been loaded (length > 0).
    bool warmup_audio_has_audio() const;

    std::atomic<bool> warmup_audio_playing{false};
    std::atomic<bool> warmup_audio_looping{true};

    /// When true, output mixer repeats the last completed generation for ring frames that are still silent.
    std::atomic<bool> loop_last_generation{true};

    /// True for the last processed audio block if any drums output sample used last-gen loop hold.
    /// Written by the audio callback only; read from the UI thread.
    std::atomic<bool> drums_output_from_last_gen_hold{false};

    /// When true and musical time is enabled, simulation Play snaps file position to a bar boundary.
    std::atomic<bool> simulation_snap_to_bar_on_play{true};

    /// Metronome: quarter-note clicks when musical time is on (downbeat uses a louder impulse).
    std::atomic<bool> click_track_enabled{false};
    /// Linear gain 0..1 applied to click impulses on the main output bus.
    std::atomic<float> click_track_volume{0.35f};

    // --- Worker thread interface ---

    /// Snapshot **`streamgen_audio`** ring only (playback clock), row-major stereo — Zenon `streamgen_audio` **before** model-rate resample.
    std::vector<float> snapshot_streamgen_audio(int64_t window_start, int64_t num_samples);

    /// Snapshot **generated drums** ring only (playback clock) — not Python `input_audio` (use `snapshot_input_audio_for_vae`).
    std::vector<float> snapshot_drums_output(int64_t window_start, int64_t num_samples);

    /// **`streamgen_audio`** for VAE (`prepare_audio`-equivalent): playback-clock window from `snapshot_streamgen_audio`, then resampled to `m_constants.sample_rate`.
    std::vector<float> snapshot_streamgen_audio_for_vae(int64_t window_start, int64_t num_samples);

    /// **`input_audio`** for VAE (drum inpaint stem; Python `input_audio_tensor`): for timeline indices
    /// in `[window_start, keep_end_sample)` only — ring + last-gen hold, else warmup when silent; samples
    /// at `abs_sample >= keep_end_sample` are **zeros** (no future drum conditioning). Resampled to
    /// `m_constants.sample_rate`. `keep_end_sample` is the same boundary as `GenerationJob::keep_end_sample`
    /// (first sample of the generated suffix in playback time).
    std::vector<float> snapshot_input_audio_for_vae(
        int64_t window_start,
        int64_t num_samples,
        int64_t keep_end_sample);

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
    /// Every sample in each bucket's time range is read from the ring (true min/max). Call from the message thread.
    /// `out_min` / `out_max` must hold `num_buckets` elements.
    ///
    /// Args:
    ///     duration_samples: Time window length in samples (e.g. sample_rate * visible_seconds).
    ///     num_buckets: Typically component width in logical pixels (one column per bucket).
    ///     out_min: Receives minimum sample per bucket.
    ///     out_max: Receives maximum sample per bucket.
    void fill_recent_streamgen_audio_waveform_buckets(int duration_samples, int num_buckets, float* out_min, float* out_max);

    /// Same as fill_recent_streamgen_audio_waveform_buckets for the **drums output / monitor** ring (stereo downmixed to mono).
    void fill_recent_drums_output_waveform_buckets(int duration_samples, int num_buckets, float* out_min, float* out_max);

    /// Drums monitor split by source: warmup (1), model from ring (2), loop-hold snapshot (3); see m_drums_origin_ring.
    /// Arrays must hold num_buckets elements. Same visible window as fill_recent_drums_output_waveform_buckets.
    /// When gen_land_jobs is non-null and non-empty, samples with origin gen are bucketed only if they fall inside a
    /// completed job land interval [output_start_sample(), output_start_sample() + gen_samples); otherwise they
    /// are omitted from gen (and hold/warm buckets) so the cyan trace aligns with scheduled lands.
    void fill_recent_drums_source_buckets(
        int duration_samples,
        int num_buckets,
        float* warm_min,
        float* warm_max,
        float* gen_min,
        float* gen_max,
        float* hold_min,
        float* hold_max,
        const std::vector<JobTimelineRecord>* gen_land_jobs = nullptr);

    /// Zero output and drums-monitor rings; clear last-generation hold snapshot. Message thread / UI.
    void clear_drums_output_buffers();

    // --- Gain controls ---
    std::atomic<float> streamgen_audio_gain{0.0f};
    std::atomic<float> drums_gain{1.0f};

    /// Feed mono **`streamgen_audio`** into `m_streamgen_audio_ring` and advance the scheduler (CLI / tests).
    void feed_streamgen_audio(const float* mono_input, int num_samples);

    GenerationScheduler& scheduler() { return m_scheduler; }
    const ModelConstants& constants() const { return m_constants; }
    int current_sample_rate() const { return m_current_sample_rate; }

    /// Native sample rate of the loaded simulation WAV (Hz), for UI time/bar readouts.
    int simulation_file_native_sample_rate_hz() const;

    /// Native sample rate of the loaded warmup-audio WAV (Hz).
    int warmup_audio_file_native_sample_rate_hz() const;

    /// Shift the drum warmup stem on the playback timeline so the last session sample reads the end of
    /// the warmup file (looping). `total_streamgen_samples` is the **streamgen_audio** session length in
    /// playback-clock samples (CLI: resampled `--input` length). `0` restores start-aligned warmup (default).
    ///
    /// Args:
    ///     total_streamgen_samples: Session span in samples; `0` disables end-alignment.
    void set_streamgen_session_total_samples_for_warmup_end_align(int64_t total_streamgen_samples);

    /// Timeline history for generation markers (UI). Null before construction completes.
    GenerationTimelineStore* timeline_store() { return m_timeline.get(); }
    const GenerationTimelineStore* timeline_store() const { return m_timeline.get(); }

    /// Drums output at `absolute_sample`: raw ring, or last-gen loop hold when the ring is silent and
    /// loop-last-gen is on. Hold never substitutes before `output_start_sample` of the committed snapshot.
    DrumsRingSample fetch_drums_ring_sample(int64_t absolute_sample) const;

private:
    void rebuild_ring_buffers(int ring_sample_rate);
    void write_streamgen_audio_to_ring(const float* mono_input, int num_samples);
    void write_streamgen_audio_to_ring_at(const float* mono_input, int num_samples, int64_t abs_block_start);
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
    int m_current_sample_rate = 44100;

    // Ring buffers: interleaved [L, R, L, R, ...] for simplicity
    // Indexed by absolute_sample_pos % ring_buffer_size
    int m_ring_buffer_size = 0; // in frames (one frame = NUM_CHANNELS samples)
    /// Python `streamgen_audio` timeline (stereo duplicated from mono).
    std::vector<float> m_streamgen_audio_ring;
    /// Generated **drums** bus (model output); warmup/monitor read via `fetch_drums_ring_sample`.
    std::vector<float> m_drums_output_ring;
    /// Last drum bus (pre-drums_gain) for UI waveform — audio thread only, never touched by worker.
    std::vector<float> m_drums_monitor_ring;
    /// Per ring frame: 0 = silence, 1 = warmup file, 2 = model from ring, 3 = loop-last hold snapshot. Audio thread only.
    std::vector<uint8_t> m_drums_origin_ring;

    // Simulation file data (mono at native file rate). Phase keeps scrub offset vs timeline.
    std::mutex m_sim_mutex;
    std::vector<float> m_sim_audio;
    double m_sim_file_phase = 0.0;
    std::atomic<int> m_sim_native_sample_rate_hz{44100};
    juce::String m_simulation_display_name;

    // Warmup audio (stereo interleaved at native file rate)
    mutable std::mutex m_warm_mutex;
    std::vector<float> m_warm_audio;
    int64_t m_warm_length_frames = 0;
    std::atomic<int> m_warm_native_sample_rate_hz{44100};
    /// When > 0, warmup file index uses `session_timeline + (warm_playback_span - this)` (see setter).
    std::atomic<int64_t> m_streamgen_session_total_samples_for_warmup_end_align{0};

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
