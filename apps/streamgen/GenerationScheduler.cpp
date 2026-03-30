#include "GenerationScheduler.h"
#include <algorithm>

namespace streamgen {

void GenerationScheduler::configure(const ModelConstants& constants)
{
    m_constants = constants;
}

void GenerationScheduler::advance(int num_samples)
{
    int64_t old_pos = m_absolute_sample_pos.load(std::memory_order_relaxed);
    int64_t new_pos = old_pos + num_samples;
    m_absolute_sample_pos.store(new_pos, std::memory_order_relaxed);

    if (!generation_enabled.load(std::memory_order_relaxed))
        return;

    int64_t hop_samples = seconds_to_samples(
        hop_seconds.load(std::memory_order_relaxed),
        m_constants.sample_rate
    );

    if (hop_samples <= 0)
        return;

    if (new_pos - m_last_trigger_sample >= hop_samples)
    {
        enqueue_job();
        m_last_trigger_sample = new_pos;
    }
}

void GenerationScheduler::enqueue_job()
{
    int64_t current_pos = m_absolute_sample_pos.load(std::memory_order_relaxed);
    float kr = keep_ratio.load(std::memory_order_relaxed);

    GenerationJob job;
    job.job_id = m_next_job_id.fetch_add(1, std::memory_order_relaxed);

    // Prospective layout: keep_end = NOW, generated suffix extends into the future.
    // The kept prefix covers the most recent drums (continuity context).
    // The generated suffix produces drums the user will hear next.
    int64_t keep_samples = static_cast<int64_t>(m_constants.sample_size * kr);
    job.keep_end_sample = current_pos;
    job.window_start_sample = current_pos - keep_samples;
    job.window_end_sample = job.window_start_sample + m_constants.sample_size;
    job.keep_ratio = kr;
    job.steps = steps.load(std::memory_order_relaxed);
    job.cfg_scale = cfg_scale.load(std::memory_order_relaxed);
    job.seconds_total = static_cast<float>(m_constants.window_seconds());

    {
        std::lock_guard<std::mutex> lock(m_queue_mutex);
        m_queue.push_back(job);
        status.queue_depth.store(static_cast<int>(m_queue.size()), std::memory_order_relaxed);
    }
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

    return true;
}

} // namespace streamgen
