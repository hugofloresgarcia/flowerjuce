#pragma once

#include <cstdint>
#include <atomic>
#include <string>

namespace streamgen {

/// Model window and timing constants derived from the Zenon pipeline config.
struct ModelConstants {
    int sample_rate = 44100;
    int sample_size = 524288;       // ~11.9s @ 44100 Hz
    int latent_dim = 64;
    int latent_length = 256;
    int downsampling_ratio = 2048;  // sample_size / latent_length

    double window_seconds() const
    {
        return static_cast<double>(sample_size) / sample_rate;
    }
};

/// Describes a single generation job with absolute timeline positions.
///
/// All sample positions are absolute (from app start, monotonically increasing).
/// The audio callback uses these to know where to place output.
///
/// Timeline layout for one job:
///   [window_start ... tf_keep_end ... keep_end ... window_end]
///   |--- sax visible ---|
///   |---------- drum kept prefix ----------|--- generated suffix (model) ---|
/// When future_visibility_frames == 0, tf_keep_end == keep_end (Nithya's reference behaviour).
/// When future_visibility_frames < 0, tf_keep_end < keep_end and the gap is silence-padded
/// streamgen audio (right-pad in the encode buffer).
///
/// Decoded generation is written from output_start_sample() for playback; by default that is
/// keep_end_sample. output_delay_samples shifts landing later (schedule delay).
struct GenerationJob {
    int64_t job_id = -1;

    int64_t window_start_sample = 0;
    int64_t window_end_sample = 0;

    int64_t keep_end_sample = 0;

    /// Samples to place after keep_end where generation lands (from schedule delay).
    int64_t output_delay_samples = 0;

    float keep_ratio = 0.5f;
    int steps = 8;
    float cfg_scale = 7.0f;
    float seconds_total = 11.888616f;

    /// Signed offset (in latent frames) for tf_inpaint_mask vs inpaint_mask.
    /// 0 = no offset (tf == inpaint). Negative = sax visibility ends earlier than drum prefix.
    /// Positive lookahead is OOD for the current checkpoint and is clamped at 0 by the UI.
    int future_visibility_frames = 0;

    /// Absolute sample where decoded generation begins on the timeline.
    int64_t output_start_sample() const { return keep_end_sample + output_delay_samples; }

    /// Number of new audio samples this job produces (the generated suffix).
    int64_t output_length_samples() const { return window_end_sample - keep_end_sample; }

    /// Total window length in samples.
    int64_t window_length_samples() const { return window_end_sample - window_start_sample; }

    /// Absolute sample where the saxophone visibility window ends (tf_inpaint_mask boundary).
    /// `downsampling_ratio` should match the model config (typically 2048).
    int64_t tf_keep_end_sample(int downsampling_ratio) const
    {
        const int64_t shift = static_cast<int64_t>(future_visibility_frames)
                              * static_cast<int64_t>(downsampling_ratio);
        const int64_t lo = window_start_sample;
        const int64_t hi = window_end_sample;
        int64_t s = keep_end_sample + shift;
        if (s < lo) s = lo;
        if (s > hi) s = hi;
        return s;
    }
};

/// Snapshot of timing results from the most recent generation, in milliseconds.
struct StageTiming {
    double vae_encode_ms = 0.0;
    double t5_encode_ms = 0.0;
    double sampling_total_ms = 0.0;
    double vae_decode_ms = 0.0;
    double total_ms = 0.0;
    int steps = 0;
};

/// Atomic status visible to the UI thread.
struct GenerationStatus {
    std::atomic<int> queue_depth{0};
    std::atomic<int64_t> generation_count{0};
    std::atomic<double> last_latency_ms{0.0};
    std::atomic<int64_t> last_job_id{-1};
    std::atomic<bool> worker_busy{false};
};

/// Converts between absolute sample position and wall-clock seconds.
inline double samples_to_seconds(int64_t samples, int sample_rate)
{
    return static_cast<double>(samples) / sample_rate;
}

inline int64_t seconds_to_samples(double seconds, int sample_rate)
{
    return static_cast<int64_t>(seconds * sample_rate);
}

/// Format a sample position as MM:SS.mmm for UI display.
inline std::string format_time(int64_t samples, int sample_rate)
{
    double secs = samples_to_seconds(samples, sample_rate);
    int minutes = static_cast<int>(secs) / 60;
    double remainder = secs - minutes * 60.0;
    char buf[32];
    std::snprintf(buf, sizeof(buf), "%02d:%06.3f", minutes, remainder);
    return buf;
}

} // namespace streamgen
