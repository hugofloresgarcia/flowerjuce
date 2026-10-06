#include "sao_inference/ControlFeatureExtractor.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace sao {

namespace {

constexpr double kPi = 3.14159265358979323846;

/// RBJ cookbook biquad as torchaudio.functional.{lowpass,highpass}_biquad computes it in float64,
/// then normalised by a0 as lfilter does. `highpass` selects the highpass numerator.
void make_biquad(double sample_rate, double cutoff_hz, double q, bool highpass,
                 double& b0, double& b1, double& b2, double& a1, double& a2)
{
    const double w0 = 2 * kPi * cutoff_hz / sample_rate;
    const double alpha = std::sin(w0) / 2 / q;
    const double c = std::cos(w0);
    double nb0, nb1;
    if (highpass) {
        nb0 = (1 + c) / 2;
        nb1 = -1 - c;
    } else {
        nb0 = (1 - c) / 2;
        nb1 = 1 - c;
    }
    const double nb2 = nb0;
    const double a0 = 1 + alpha;
    const double na1 = -2 * c;
    const double na2 = 1 - alpha;
    b0 = nb0 / a0;
    b1 = nb1 / a0;
    b2 = nb2 / a0;
    a1 = na1 / a0;
    a2 = na2 / a0;
}

inline float sigmoid32(float z)
{
    return 1.0f / (1.0f + std::exp(-z));
}

/// torch.round is half-to-even; std::nearbyint honours the default rounding mode (FE_TONEAREST).
inline float quantize_unit32(float x, int n_levels)
{
    const float steps = static_cast<float>(n_levels - 1);
    const float clamped = std::min(1.0f, std::max(0.0f, x));
    return std::nearbyint(clamped * steps) / steps;
}

} // namespace

ControlFeatureType ControlFeatureConfig::parse_type(const std::string& name)
{
    if (name == "rms") return ControlFeatureType::Rms;
    if (name == "tria_fixed") return ControlFeatureType::TriaFixed;
    if (name == "tria_ema") return ControlFeatureType::TriaEma;
    throw std::invalid_argument("unknown control feature type '" + name + "' (expected rms, tria_fixed, tria_ema)");
}

int ControlFeatureConfig::channels() const
{
    return type == ControlFeatureType::Rms ? 1 : 2;
}

double ControlFeatureExtractor::Biquad::process(double x)
{
    // torchaudio lfilter: FIR part b·x (conv in float64), then y[n] = f[n] - (y[n-2]·a2 + y[n-1]·a1).
    const double f = b0 * x + b1 * x1 + b2 * x2;
    const double y = f - (y2 * a2 + y1 * a1);
    x2 = x1;
    x1 = x;
    y2 = y1;
    y1 = y; // unclamped state
    return std::min(1.0, std::max(-1.0, y)); // lfilter(clamp=True) on the stage output
}

ControlFeatureExtractor::ControlFeatureExtractor(const ControlFeatureConfig& config)
    : m_config(config)
{
    if (m_config.hop <= 0) throw std::invalid_argument("ControlFeatureConfig.hop must be > 0");
    if (m_config.sample_rate <= 0) throw std::invalid_argument("ControlFeatureConfig.sample_rate must be > 0");
    m_channels = m_config.channels();
    if (m_config.type != ControlFeatureType::Rms) {
        if (!(m_config.split_hz > 0.0) || m_config.split_hz >= m_config.sample_rate / 2.0)
            throw std::invalid_argument("ControlFeatureConfig.split_hz must be in (0, nyquist)");
        if (m_config.crossover_order < 2 || m_config.crossover_order % 2 != 0)
            throw std::invalid_argument("ControlFeatureConfig.crossover_order must be an even number >= 2");
        if (m_config.n_levels < 2) throw std::invalid_argument("ControlFeatureConfig.n_levels must be >= 2");
        m_n_sections = m_config.crossover_order / 2;
        const double q = 1 / std::sqrt(2.0);
        m_low.assign(static_cast<size_t>(m_n_sections), Biquad{});
        m_high.assign(static_cast<size_t>(m_n_sections), Biquad{});
        for (int s = 0; s < m_n_sections; ++s) {
            auto& lo = m_low[static_cast<size_t>(s)];
            auto& hi = m_high[static_cast<size_t>(s)];
            make_biquad(m_config.sample_rate, m_config.split_hz, q, false, lo.b0, lo.b1, lo.b2, lo.a1, lo.a2);
            make_biquad(m_config.sample_rate, m_config.split_hz, q, true, hi.b0, hi.b1, hi.b2, hi.a1, hi.a2);
        }
    }
    if (m_config.type == ControlFeatureType::TriaEma) {
        if (!(m_config.ema_tau_seconds > 0.0))
            throw std::invalid_argument("ControlFeatureConfig.ema_tau_seconds must be > 0 for tria_ema");
    }
    reset();
}

void ControlFeatureExtractor::reset()
{
    for (auto& b : m_low) b.clear();
    for (auto& b : m_high) b.clear();
    m_energy = {{0.0, 0.0}};
    m_count = 0;
    m_frames.clear();
    m_frames_produced = 0;
    if (m_config.type == ControlFeatureType::TriaEma) {
        // ema_standardize: alpha = exp(-1/tau_frames); warm start with prior_frames (default tau_frames)
        const double tau_frames = m_config.ema_tau_seconds * m_config.sample_rate / m_config.hop;
        const double prior_frames = m_config.ema_prior_frames < 0.0 ? tau_frames : m_config.ema_prior_frames;
        m_ema_alpha = std::exp(-1.0 / tau_frames);
        m_ema_weight = prior_frames;
        for (int c = 0; c < 2; ++c) {
            const double mean0 = m_config.band_mean_db[static_cast<size_t>(c)];
            const double std0 = m_config.band_std_db[static_cast<size_t>(c)];
            m_ema_s1[static_cast<size_t>(c)] = prior_frames * mean0;
            m_ema_s2[static_cast<size_t>(c)] = prior_frames * (std0 * std0 + mean0 * mean0);
        }
    }
}

void ControlFeatureExtractor::push_planar(const float* const* channels, int n_channels, size_t n_samples)
{
    if (n_channels < 1) throw std::invalid_argument("push_planar: n_channels must be >= 1");
    for (size_t i = 0; i < n_samples; ++i) {
        float sum = 0.0f;
        for (int c = 0; c < n_channels; ++c) sum += channels[c][i];
        // torch.mean over the channel dim in float32: sum then divide
        process_mono_sample(n_channels == 1 ? sum : sum / static_cast<float>(n_channels));
    }
}

void ControlFeatureExtractor::push_interleaved(const float* data, int n_channels, size_t n_samples)
{
    if (n_channels < 1) throw std::invalid_argument("push_interleaved: n_channels must be >= 1");
    for (size_t i = 0; i < n_samples; ++i) {
        float sum = 0.0f;
        for (int c = 0; c < n_channels; ++c) sum += data[i * static_cast<size_t>(n_channels) + static_cast<size_t>(c)];
        process_mono_sample(n_channels == 1 ? sum : sum / static_cast<float>(n_channels));
    }
}

void ControlFeatureExtractor::push_mono(const float* data, size_t n_samples)
{
    for (size_t i = 0; i < n_samples; ++i) process_mono_sample(data[i]);
}

void ControlFeatureExtractor::process_mono_sample(float mono)
{
    if (m_config.type == ControlFeatureType::Rms) {
        const double v = static_cast<double>(mono);
        m_energy[0] += v * v;
    } else {
        double lo = static_cast<double>(mono);
        double hi = lo;
        for (int s = 0; s < m_n_sections; ++s) {
            lo = m_low[static_cast<size_t>(s)].process(lo);
            hi = m_high[static_cast<size_t>(s)].process(hi);
        }
        // block_rms_db casts the float64 band signal to float32 before squaring
        const double lo32 = static_cast<double>(static_cast<float>(lo));
        const double hi32 = static_cast<double>(static_cast<float>(hi));
        m_energy[0] += lo32 * lo32;
        m_energy[1] += hi32 * hi32;
    }
    if (++m_count == static_cast<size_t>(m_config.hop)) emit_frame(m_count);
}

void ControlFeatureExtractor::flush_partial()
{
    if (m_count > 0) emit_frame(m_count);
}

float ControlFeatureExtractor::frame_db(double energy, size_t count) const
{
    // float32 arithmetic as in block_rms_db: sqrt(energy / count), 20*log10(max(rms, 1e-12)), floor
    const float rms = std::sqrt(static_cast<float>(energy) / static_cast<float>(count));
    const float db = 20.0f * std::log10(std::max(rms, 1e-12f));
    return std::max(db, m_config.floor_db);
}

void ControlFeatureExtractor::emit_frame(size_t count)
{
    if (m_config.type == ControlFeatureType::Rms) {
        const float db = frame_db(m_energy[0], count);
        // normalize_db: ((db - floor) / (0 - floor)).clamp(0, 1)
        const float unit = (db - m_config.floor_db) / (0.0f - m_config.floor_db);
        m_frames.push_back(std::min(1.0f, std::max(0.0f, unit)));
    } else {
        float db[2] = {frame_db(m_energy[0], count), frame_db(m_energy[1], count)};
        float z[2];
        if (m_config.type == ControlFeatureType::TriaFixed) {
            for (int c = 0; c < 2; ++c) {
                // fixed_standardize in float32 with std floored at min_std
                const float mean = static_cast<float>(m_config.band_mean_db[static_cast<size_t>(c)]);
                const float std = std::max(static_cast<float>(m_config.band_std_db[static_cast<size_t>(c)]),
                                           static_cast<float>(m_config.min_std_db));
                z[c] = (db[c] - mean) / std;
            }
        } else {
            // ema_standardize in float64, result cast to float32
            m_ema_weight = m_ema_alpha * m_ema_weight + 1.0;
            for (int c = 0; c < 2; ++c) {
                const double xt = static_cast<double>(db[c]);
                auto& s1 = m_ema_s1[static_cast<size_t>(c)];
                auto& s2 = m_ema_s2[static_cast<size_t>(c)];
                s1 = m_ema_alpha * s1 + xt;
                s2 = m_ema_alpha * s2 + xt * xt;
                const double mu = s1 / m_ema_weight;
                const double var = std::max(s2 / m_ema_weight - mu * mu, 0.0);
                z[c] = static_cast<float>((xt - mu) / std::max(std::sqrt(var), m_config.min_std_db));
            }
        }
        for (int c = 0; c < 2; ++c) m_frames.push_back(quantize_unit32(sigmoid32(z[c]), m_config.n_levels));
    }
    m_energy = {{0.0, 0.0}};
    m_count = 0;
    ++m_frames_produced;
}

bool ControlFeatureExtractor::pop_frame(float* out)
{
    const size_t ch = static_cast<size_t>(m_channels);
    if (m_frames.size() < ch) return false;
    std::copy(m_frames.begin(), m_frames.begin() + static_cast<std::ptrdiff_t>(ch), out);
    m_frames.erase(m_frames.begin(), m_frames.begin() + static_cast<std::ptrdiff_t>(ch));
    return true;
}

} // namespace sao
