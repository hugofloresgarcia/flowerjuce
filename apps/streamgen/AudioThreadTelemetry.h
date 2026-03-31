#pragma once

#include <atomic>
#include <cstdint>

namespace streamgen {

/// RT-safe telemetry updated only from the audio device callback (and feed_audio).
/// Threading: `record_block()` runs on the audio callback (and `feed_audio` for CLI).
/// `copy_snapshot()` runs on UI/logger threads. Single-writer atomics only — no DBG/heap in `record_block()`.
/// No locks, no heap, no logging. Readers use copy_snapshot() from non-RT threads.
class AudioThreadTelemetry {
public:
    /// Snapshot of values safe to read from UI or logger threads.
    struct Snapshot {
        uint64_t callback_count = 0;
        double last_block_input_rms = 0.0;
        double last_block_output_l_rms = 0.0;
        double last_block_output_r_rms = 0.0;
        double ema_input_rms = 0.0;
        double ema_output_rms = 0.0;
    };

    /// Record one audio block. Call once per callback after input/output samples are known.
    ///
    /// Args:
    ///     input_sum_sq_mono: Sum of squares of mono samples fed to the input ring.
    ///     output_sum_sq_l: Sum of squares of left output channel (post-mix).
    ///     output_sum_sq_r: Sum of squares of right output channel (post-mix).
    ///     num_samples: Block length (must be > 0).
    void record_block(double input_sum_sq_mono,
                      double output_sum_sq_l,
                      double output_sum_sq_r,
                      int num_samples);

    void copy_snapshot(Snapshot& out) const;

    /// Zero counters and EMA (e.g. after session reset). Safe from any non-audio thread
    /// while the audio callback is stopped.
    void reset_counters();

private:
    static double rms_from_sum_sq(double sum_sq, int n);
    static constexpr double kEmaAlpha = 0.15;

    std::atomic<uint64_t> m_callback_count{0};
    std::atomic<double> m_last_input_rms{0.0};
    std::atomic<double> m_last_out_l_rms{0.0};
    std::atomic<double> m_last_out_r_rms{0.0};
    std::atomic<double> m_ema_input_rms{0.0};
    std::atomic<double> m_ema_output_rms{0.0};
};

} // namespace streamgen
