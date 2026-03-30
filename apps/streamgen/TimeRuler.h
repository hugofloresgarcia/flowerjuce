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
///   [window_start ... keep_end ... window_end]
///   |--- kept prefix ---|--- generated suffix ---|
struct GenerationJob {
    int64_t job_id = -1;

    int64_t window_start_sample = 0;
    int64_t window_end_sample = 0;

    int64_t keep_end_sample = 0;

    float keep_ratio = 0.5f;
    int steps = 8;
    float cfg_scale = 7.0f;
    float seconds_total = 11.888616f;

    /// The absolute sample position where the generated (non-kept) portion begins.
    int64_t output_start_sample() const { return keep_end_sample; }

    /// Number of new audio samples this job produces (the generated suffix).
    int64_t output_length_samples() const { return window_end_sample - keep_end_sample; }

    /// Total window length in samples.
    int64_t window_length_samples() const { return window_end_sample - window_start_sample; }
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
