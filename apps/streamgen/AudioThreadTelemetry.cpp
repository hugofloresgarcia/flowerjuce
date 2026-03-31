#include "AudioThreadTelemetry.h"

#include <cmath>

namespace streamgen {

double AudioThreadTelemetry::rms_from_sum_sq(double sum_sq, int n)
{
    if (n <= 0 || sum_sq <= 0.0)
        return 0.0;
    return std::sqrt(sum_sq / static_cast<double>(n));
}

void AudioThreadTelemetry::record_block(
    double input_sum_sq_mono,
    double output_sum_sq_l,
    double output_sum_sq_r,
    int num_samples)
{
    m_callback_count.fetch_add(1, std::memory_order_relaxed);

    const double in_rms = rms_from_sum_sq(input_sum_sq_mono, num_samples);
    const double out_l = rms_from_sum_sq(output_sum_sq_l, num_samples);
    const double out_r = rms_from_sum_sq(output_sum_sq_r, num_samples);
    const double out_mean = (out_l + out_r) * 0.5;

    m_last_input_rms.store(in_rms, std::memory_order_relaxed);
    m_last_out_l_rms.store(out_l, std::memory_order_relaxed);
    m_last_out_r_rms.store(out_r, std::memory_order_relaxed);

    double prev_in = m_ema_input_rms.load(std::memory_order_relaxed);
    double prev_out = m_ema_output_rms.load(std::memory_order_relaxed);
    if (m_callback_count.load(std::memory_order_relaxed) == 1)
    {
        prev_in = in_rms;
        prev_out = out_mean;
    }
    prev_in = prev_in * (1.0 - kEmaAlpha) + in_rms * kEmaAlpha;
    prev_out = prev_out * (1.0 - kEmaAlpha) + out_mean * kEmaAlpha;
    m_ema_input_rms.store(prev_in, std::memory_order_relaxed);
    m_ema_output_rms.store(prev_out, std::memory_order_relaxed);
}

void AudioThreadTelemetry::copy_snapshot(Snapshot& out) const
{
    out.callback_count = m_callback_count.load(std::memory_order_relaxed);
    out.last_block_input_rms = m_last_input_rms.load(std::memory_order_relaxed);
    out.last_block_output_l_rms = m_last_out_l_rms.load(std::memory_order_relaxed);
    out.last_block_output_r_rms = m_last_out_r_rms.load(std::memory_order_relaxed);
    out.ema_input_rms = m_ema_input_rms.load(std::memory_order_relaxed);
    out.ema_output_rms = m_ema_output_rms.load(std::memory_order_relaxed);
}

void AudioThreadTelemetry::reset_counters()
{
    m_callback_count.store(0, std::memory_order_relaxed);
    m_last_input_rms.store(0.0, std::memory_order_relaxed);
    m_last_out_l_rms.store(0.0, std::memory_order_relaxed);
    m_last_out_r_rms.store(0.0, std::memory_order_relaxed);
    m_ema_input_rms.store(0.0, std::memory_order_relaxed);
    m_ema_output_rms.store(0.0, std::memory_order_relaxed);
}

} // namespace streamgen
