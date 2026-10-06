#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace sao {

/// Which sa3-tap control the extractor produces. Mirrors stable_audio_3/data/features.py.
enum class ControlFeatureType {
    Rms,        ///< `rms_envelope_control`: 1 channel, RMS dBFS of the mono mix mapped to [0, 1].
    TriaFixed,  ///< `tria_control(norm="fixed")`: 2 channels, per-band dB z-scored against dataset stats.
    TriaEma,    ///< `tria_control(norm="ema")`: 2 channels, per-band dB z-scored against causal running stats.
};

/// Parameters of a control feature. Everything comes from the manifest (which copies the TriaStats JSON
/// and the feature constants); nothing is hardcoded in the extractor.
struct ControlFeatureConfig {
    ControlFeatureType type = ControlFeatureType::TriaFixed;
    int sample_rate = 44100;
    int hop = 4096;                 ///< samples per feature frame (= the latent hop)
    float floor_db = -80.0f;        ///< RMS floor in dBFS
    // TRIA only
    double split_hz = 0.0;          ///< crossover frequency
    int crossover_order = 4;        ///< even; order/2 cascaded biquads per band (4 = Linkwitz-Riley 4)
    std::array<double, 2> band_mean_db{{0.0, 0.0}};  ///< dataset mean per band (low, high)
    std::array<double, 2> band_std_db{{1.0, 1.0}};   ///< dataset std per band
    double min_std_db = 1.0;        ///< floor on the (running or dataset) std
    int n_levels = 33;              ///< quantisation levels
    // TRIA ema only
    double ema_tau_seconds = 0.0;   ///< time constant of the running statistics
    double ema_prior_frames = -1.0; ///< warm-start weight in frames; < 0 means tau_frames (the Python default)

    /// Parse the `type` string of the manifest: "rms", "tria_fixed" or "tria_ema". Throws on anything else.
    static ControlFeatureType parse_type(const std::string& name);
    /// Number of output channels for `type` (1 for Rms, 2 for Tria*).
    int channels() const;
};

/// Streaming, causal re-implementation of the sa3-tap control features.
///
/// Feed audio in chunks of any size with `push_*`; every time a hop boundary is crossed one frame of
/// `channels()` values is appended to an internal queue, read with `frames_available()` / `pop_frame()`.
/// Frame `t` depends only on samples before `(t+1)*hop`, exactly as the Python reference, and chunking
/// never changes the result: feeding one sample at a time or the whole clip at once gives identical frames.
///
/// Numerics follow stable_audio_3/data/features.py so quantised TRIA levels match the Python reference
/// exactly on real audio: mono mix in float32, crossover biquads in float64 with torchaudio's coefficient
/// formulas, unclamped IIR state and stage output clamped to [-1, 1] (lfilter's `clamp=True`), per-frame
/// energy of the float32-cast band signal, float32 dB / z-score / sigmoid, half-to-even rounding.
///
/// State that persists across calls (and across generated windows in the live loop): the biquad delay
/// lines, the partial-frame accumulators and, for TriaEma, the running statistics. `reset()` returns to
/// the state at construction. Not thread-safe; meant to run on the inference worker, never on the audio
/// thread (it allocates when the frame queue grows).
class ControlFeatureExtractor {
public:
    /// Throws std::invalid_argument on an inconsistent config (hop <= 0, odd crossover order, missing
    /// split_hz / tau for the TRIA types).
    explicit ControlFeatureExtractor(const ControlFeatureConfig& config);

    const ControlFeatureConfig& config() const { return m_config; }
    int channels() const { return m_channels; }

    /// Planar input: `channels[c][i]`, `n_channels` >= 1, `n_samples` >= 0. Channels are averaged to mono
    /// in float32 (torch `mean(dim=0)`).
    void push_planar(const float* const* channels, int n_channels, size_t n_samples);
    /// Interleaved input: `data[i * n_channels + c]`.
    void push_interleaved(const float* data, int n_channels, size_t n_samples);
    /// Mono input.
    void push_mono(const float* data, size_t n_samples);

    /// Emit the trailing partial frame (if any samples are pending) averaged over the samples it really
    /// has, as `block_rms_db` does for a finite clip. Only for offline / parity use: the live loop never
    /// ends a clip. Afterwards the partial accumulators are empty; filter and EMA state are kept.
    void flush_partial();

    /// Frames produced and not yet popped.
    size_t frames_available() const { return m_frames.size() / static_cast<size_t>(m_channels); }
    /// Total frames produced since construction / reset (popped or not).
    uint64_t frames_produced() const { return m_frames_produced; }
    /// Copy the oldest pending frame (`channels()` floats) into `out` and drop it. Returns false if none.
    bool pop_frame(float* out);
    /// Pending frames, row-major [frame][channel], without consuming them.
    const std::vector<float>& pending() const { return m_frames; }
    /// Drop all pending frames (state is kept).
    void clear_pending() { m_frames.clear(); }

    /// Back to the construction state: zero filter memory, empty accumulators, prior EMA statistics.
    void reset();

private:
    struct Biquad {
        double b0 = 0.0, b1 = 0.0, b2 = 0.0, a1 = 0.0, a2 = 0.0; // normalised by a0
        double x1 = 0.0, x2 = 0.0, y1 = 0.0, y2 = 0.0;
        /// One sample: returns the clamped output; the recursion state keeps the unclamped value.
        double process(double x);
        void clear() { x1 = x2 = y1 = y2 = 0.0; }
    };

    void process_mono_sample(float mono);
    void emit_frame(size_t count);
    float frame_db(double energy, size_t count) const;

    ControlFeatureConfig m_config;
    int m_channels = 1;
    int m_n_sections = 0;                   ///< biquads per band (crossover_order / 2)
    std::vector<Biquad> m_low;              ///< lowpass cascade
    std::vector<Biquad> m_high;             ///< highpass cascade
    std::array<double, 2> m_energy{{0.0, 0.0}};  ///< per-band sum of squares of the pending frame
    size_t m_count = 0;                     ///< samples in the pending frame
    // EMA state (TriaEma)
    double m_ema_alpha = 0.0;
    double m_ema_weight = 0.0;
    std::array<double, 2> m_ema_s1{{0.0, 0.0}};
    std::array<double, 2> m_ema_s2{{0.0, 0.0}};
    std::vector<float> m_frames;            ///< pending frames, [frame][channel]
    uint64_t m_frames_produced = 0;
};

} // namespace sao
