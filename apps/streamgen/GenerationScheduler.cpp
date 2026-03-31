#include "GenerationScheduler.h"
#include "MusicalTime.h"
#include "StreamGenDebugLog.h"
#include "GenerationTimelineStore.h"

#include <juce_core/juce_core.h>
#include <cmath>

namespace streamgen {

namespace {

float clamp_bpm(float v)
{
    if (v < 20.0f)
        return 20.0f;
    if (v > 400.0f)
        return 400.0f;
    return v;
}

} // namespace

void GenerationScheduler::configure(const ModelConstants& constants)
{
    m_constants = constants;
    streamgen_log("scheduler configure: model_sr=" + juce::String(constants.sample_rate)
        + " sample_size=" + juce::String(constants.sample_size)
        + " abs_pos=" + juce::String(m_absolute_sample_pos.load(std::memory_order_relaxed)));
}

void GenerationScheduler::set_playback_sample_rate(int hz)
{
    m_playback_sample_rate = hz > 0 ? hz : 0;
    streamgen_log("scheduler set_playback_sample_rate hz=" + juce::String(m_playback_sample_rate)
        + " effective=" + juce::String(effective_playback_rate_hz()));
}

void GenerationScheduler::reset_session()
{
    {
        std::lock_guard<std::mutex> lock(m_queue_mutex);
        m_queue.clear();
    }
    status.queue_depth.store(0, std::memory_order_relaxed);
    status.generation_count.store(0, std::memory_order_relaxed);
    status.last_latency_ms.store(0.0, std::memory_order_relaxed);
    status.last_job_id.store(-1, std::memory_order_relaxed);
    status.worker_busy.store(false, std::memory_order_relaxed);

    m_absolute_sample_pos.store(0, std::memory_order_relaxed);
    m_last_trigger_sample = 0;
    m_next_job_id.store(0, std::memory_order_relaxed);
    m_pending_keep_end_sample = -1;
    m_pending_snap_bpm = 0.0f;
    m_pending_snap_quantize = 0;

    streamgen_log("scheduler reset_session");
}

void GenerationScheduler::advance(int num_samples)
{
    int64_t old_pos = m_absolute_sample_pos.load(std::memory_order_relaxed);
    int64_t new_pos = old_pos + num_samples;
    m_absolute_sample_pos.store(new_pos, std::memory_order_relaxed);

    static int adv_log_counter = 0;
    const bool log_this = streamgen_log_audio_throttle(adv_log_counter);
    const bool gen_on = generation_enabled.load(std::memory_order_relaxed);
    const int rate = effective_playback_rate_hz();

    const bool musical = musical_time_enabled.load(std::memory_order_relaxed);
    float bpm_val = clamp_bpm(bpm.load(std::memory_order_relaxed));

    int64_t hop_samples = 0;
    if (gen_on)
    {
        if (musical)
        {
            float hb = hop_beats.load(std::memory_order_relaxed);
            if (hb < 0.25f)
                hb = 0.25f;
            hop_samples = beats_to_samples(static_cast<double>(hb), rate, static_cast<double>(bpm_val));
        }
        else
        {
            hop_samples = seconds_to_samples(
                static_cast<double>(hop_seconds.load(std::memory_order_relaxed)),
                rate
            );
        }
    }

    if (log_this && gen_on)
    {
        streamgen_log(juce::String::formatted(
            "scheduler advance: pos %lld -> %lld (n=%d) gen_en=%d musical=%d hop_smpl=%lld last_trig=%lld rate_hz=%d bpm=%.1f",
            static_cast<long long>(old_pos),
            static_cast<long long>(new_pos),
            num_samples,
            gen_on ? 1 : 0,
            musical ? 1 : 0,
            static_cast<long long>(hop_samples),
            static_cast<long long>(m_last_trigger_sample),
            rate,
            bpm_val));
    }

    if (!gen_on)
    {
        m_pending_keep_end_sample = -1;
        return;
    }

    if (!musical)
        m_pending_keep_end_sample = -1;

    if (musical && m_pending_keep_end_sample >= 0)
    {
        const int q_now = quantize_launch_beats.load(std::memory_order_relaxed);
        const bool params_stale = q_now == 0
            || std::abs(bpm_val - m_pending_snap_bpm) > 1.0e-4f
            || q_now != m_pending_snap_quantize;
        if (params_stale)
            m_pending_keep_end_sample = -1;
    }

    if (hop_samples <= 0)
    {
        if (log_this)
            streamgen_log("scheduler advance: hop_samples<=0, skip enqueue");
        return;
    }

    if (m_pending_keep_end_sample >= 0)
    {
        if (new_pos >= m_pending_keep_end_sample)
        {
            enqueue_job(m_pending_keep_end_sample);
            m_last_trigger_sample = m_pending_keep_end_sample;
            m_pending_keep_end_sample = -1;
        }
        return;
    }

    if (new_pos - m_last_trigger_sample < hop_samples)
        return;

    int64_t keep_end_sample = new_pos;
    if (musical)
    {
        const int q = quantize_launch_beats.load(std::memory_order_relaxed);
        if (q >= 1)
        {
            int64_t grid_keep = ceil_sample_to_beat_grid(new_pos, rate, static_cast<double>(bpm_val), q);
            if (grid_keep < new_pos)
                grid_keep = new_pos;
            if (grid_keep > new_pos)
            {
                m_pending_keep_end_sample = grid_keep;
                m_pending_snap_bpm = bpm_val;
                m_pending_snap_quantize = q;
                if (log_this)
                    streamgen_log(juce::String::formatted(
                        "scheduler quantize arm: pending_keep=%lld q=%d bpm=%.1f",
                        static_cast<long long>(grid_keep),
                        q,
                        bpm_val));
                return;
            }
            keep_end_sample = grid_keep;
        }
    }

    enqueue_job(keep_end_sample);
    m_last_trigger_sample = keep_end_sample;
}

void GenerationScheduler::enqueue_job(int64_t keep_end_sample)
{
    float kr = keep_ratio.load(std::memory_order_relaxed);
    const int rate = effective_playback_rate_hz();
    const bool musical = musical_time_enabled.load(std::memory_order_relaxed);
    float bpm_val = clamp_bpm(bpm.load(std::memory_order_relaxed));

    int64_t output_delay_smpl = 0;
    if (musical)
    {
        float delay_beats = schedule_delay_beats.load(std::memory_order_relaxed);
        if (delay_beats < 0.0f)
            delay_beats = 0.0f;
        output_delay_smpl = beats_to_samples(static_cast<double>(delay_beats), rate, static_cast<double>(bpm_val));
    }
    else
    {
        float delay_sec = schedule_delay_seconds.load(std::memory_order_relaxed);
        if (delay_sec < 0.0f)
            delay_sec = 0.0f;
        output_delay_smpl = seconds_to_samples(static_cast<double>(delay_sec), rate);
    }

    GenerationJob job;
    job.job_id = m_next_job_id.fetch_add(1, std::memory_order_relaxed);

    int64_t keep_samples = static_cast<int64_t>(m_constants.sample_size * kr);
    job.keep_end_sample = keep_end_sample;
    job.window_start_sample = keep_end_sample - keep_samples;
    job.window_end_sample = job.window_start_sample + m_constants.sample_size;
    job.output_delay_samples = output_delay_smpl;
    job.keep_ratio = kr;
    job.steps = steps.load(std::memory_order_relaxed);
    job.cfg_scale = cfg_scale.load(std::memory_order_relaxed);
    job.seconds_total = static_cast<float>(m_constants.window_seconds());

    {
        std::lock_guard<std::mutex> lock(m_queue_mutex);
        m_queue.push_back(job);
        status.queue_depth.store(static_cast<int>(m_queue.size()), std::memory_order_relaxed);
    }

    if (m_timeline != nullptr)
        m_timeline->record_scheduled(job);

    streamgen_log(juce::String::formatted(
        "scheduler enqueue job_id=%lld musical=%d keep_end=%lld win=[%lld,%lld) out_delay_smpl=%lld depth=%d bpm=%.1f",
        static_cast<long long>(job.job_id),
        musical ? 1 : 0,
        static_cast<long long>(job.keep_end_sample),
        static_cast<long long>(job.window_start_sample),
        static_cast<long long>(job.window_end_sample),
        static_cast<long long>(job.output_delay_samples),
        status.queue_depth.load(std::memory_order_relaxed),
        bpm_val));
}

bool GenerationScheduler::pop_job(GenerationJob& job)
{
    std::lock_guard<std::mutex> lock(m_queue_mutex);

    if (m_queue.empty())
        return false;

    // Skip stale jobs — only process the most recent one
    job = m_queue.back();
    m_queue.clear();
    status.queue_depth.store(0, std::memory_order_relaxed);

    streamgen_log(juce::String::formatted(
        "scheduler pop_job job_id=%lld keep_end=%lld (queue cleared to process latest)",
        static_cast<long long>(job.job_id),
        static_cast<long long>(job.keep_end_sample)));

    return true;
}

} // namespace streamgen
