#pragma once

#include <cmath>
#include <cstdint>
#include <string>

namespace streamgen {

/// Musical time helpers: one beat = one quarter note at `bpm`. Beat 0 aligns with sample 0.

/// Args:
///     samples: Absolute sample index from session start.
///     sample_rate: Playback sample rate (Hz), must be > 0.
///     bpm: Tempo, must be > 0.
///
/// Returns:
///     Floating beat index (0 = downbeat at t=0).
inline double samples_to_beats(int64_t samples, int sample_rate, double bpm)
{
    if (sample_rate <= 0 || bpm <= 0.0)
        return 0.0;
    return static_cast<double>(samples) * bpm / (60.0 * static_cast<double>(sample_rate));
}

/// Args:
///     beats: Musical length in quarter-note beats.
///     sample_rate: Playback sample rate (Hz), must be > 0.
///     bpm: Tempo, must be > 0.
///
/// Returns:
///     Nearest sample count for `beats` (rounded).
inline int64_t beats_to_samples(double beats, int sample_rate, double bpm)
{
    if (sample_rate <= 0 || bpm <= 0.0)
        return 0;
    const double seconds = beats * (60.0 / bpm);
    return static_cast<int64_t>(std::llround(seconds * static_cast<double>(sample_rate)));
}

/// Smallest sample position >= `sample` whose beat index lies on a multiple of `grid_beats`.
///
/// Args:
///     sample: Current absolute sample.
///     sample_rate, bpm: As in samples_to_beats.
///     grid_beats: Grid period in beats, must be >= 1.
inline int64_t ceil_sample_to_beat_grid(int64_t sample, int sample_rate, double bpm, int grid_beats)
{
    if (sample_rate <= 0 || bpm <= 0.0 || grid_beats < 1)
        return sample;
    const double b = samples_to_beats(sample, sample_rate, bpm);
    const double g = static_cast<double>(grid_beats);
    const double ceiled_beats = std::ceil(b / g - 1e-12) * g;
    return beats_to_samples(ceiled_beats, sample_rate, bpm);
}

/// Human-readable bar|beat placement (1-indexed bar and beat within bar).
///
/// Args:
///     samples: Absolute sample index.
///     sample_rate, bpm: As above.
///     beats_per_bar: Time signature numerator, must be >= 1.
///
/// Returns:
///     String like "12|3" for bar 12, beat 3.
inline std::string format_bar_beat(int64_t samples, int sample_rate, double bpm, int beats_per_bar)
{
    if (beats_per_bar < 1)
        beats_per_bar = 1;
    const double beat_total = samples_to_beats(samples, sample_rate, bpm);
    const double bar_d = std::floor(beat_total / static_cast<double>(beats_per_bar));
    const int bar_idx = static_cast<int>(bar_d) + 1;
    double in_bar = beat_total - bar_d * static_cast<double>(beats_per_bar);
    if (in_bar < 0.0)
        in_bar = 0.0;
    const int beat_in_bar = static_cast<int>(std::floor(in_bar)) + 1;
    char buf[48];
    std::snprintf(buf, sizeof(buf), "%d|%d", bar_idx, beat_in_bar);
    return std::string(buf);
}

} // namespace streamgen
