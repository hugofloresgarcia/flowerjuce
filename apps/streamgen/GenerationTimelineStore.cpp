#include "GenerationTimelineStore.h"

#include <algorithm>

namespace streamgen {

GenerationTimelineStore::GenerationTimelineStore()
{
    m_entries.reserve(k_max_entries);
}

int64_t GenerationTimelineStore::steady_now_ns()
{
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
               std::chrono::steady_clock::now().time_since_epoch())
        .count();
}

int64_t GenerationTimelineStore::system_now_ms()
{
    return std::chrono::duration_cast<std::chrono::milliseconds>(
               std::chrono::system_clock::now().time_since_epoch())
        .count();
}

int64_t GenerationTimelineStore::latest_relevant_sample(const JobTimelineRecord& r)
{
    const int64_t gen_len = r.job.window_end_sample - r.job.keep_end_sample;
    int64_t end = r.job.window_end_sample;
    int64_t land_end = r.job.output_start_sample() + gen_len;
    if (land_end > end)
        end = land_end;
    if (r.has_completed && r.gen_samples > 0)
    {
        int64_t actual_end = r.job.output_start_sample() + r.gen_samples;
        if (actual_end > end)
            end = actual_end;
    }
    return end;
}

void GenerationTimelineStore::prune_locked(int64_t min_sample_to_keep)
{
    while (m_entries.size() > k_max_entries)
        m_entries.erase(m_entries.begin());

    auto it = std::remove_if(
        m_entries.begin(),
        m_entries.end(),
        [min_sample_to_keep](const JobTimelineRecord& r)
        {
            return latest_relevant_sample(r) < min_sample_to_keep;
        });
    m_entries.erase(it, m_entries.end());
}

void GenerationTimelineStore::clear()
{
    std::lock_guard<std::mutex> lock(m_mutex);
    m_entries.clear();
}

void GenerationTimelineStore::record_scheduled(const GenerationJob& job)
{
    JobTimelineRecord rec;
    rec.job_id = job.job_id;
    rec.job = job;
    rec.scheduled_steady_ns = steady_now_ns();
    rec.scheduled_system_ms = system_now_ms();

    std::lock_guard<std::mutex> lock(m_mutex);
    m_entries.push_back(rec);
    while (m_entries.size() > k_max_entries)
        m_entries.erase(m_entries.begin());
}

void GenerationTimelineStore::record_completed(
    int64_t job_id,
    const GenerationJob& job,
    double inference_ms,
    int64_t gen_samples)
{
    std::lock_guard<std::mutex> lock(m_mutex);
    for (auto& e : m_entries)
    {
        if (e.job_id == job_id)
        {
            e.has_completed = true;
            e.job = job;
            e.completed_steady_ns = steady_now_ns();
            e.completed_system_ms = system_now_ms();
            e.inference_ms = inference_ms;
            e.gen_samples = gen_samples;
            return;
        }
    }
}

std::vector<JobTimelineRecord> GenerationTimelineStore::snapshot_intersecting(
    int64_t absolute_pos,
    int sample_rate,
    float visible_seconds)
{
    int64_t visible_samples = static_cast<int64_t>(static_cast<double>(sample_rate) * static_cast<double>(visible_seconds));
    const double past_d = static_cast<double>(k_timeline_playhead_past_fraction);
    int64_t past_span = static_cast<int64_t>(static_cast<double>(visible_samples) * past_d);
    int64_t future_span = visible_samples - past_span;
    int64_t window_start = absolute_pos - past_span;
    int64_t window_end = absolute_pos + future_span;
    int64_t margin = sample_rate * 2;

    std::lock_guard<std::mutex> lock(m_mutex);
    prune_locked(window_start - margin);

    std::vector<JobTimelineRecord> out;
    out.reserve(m_entries.size());
    for (const auto& e : m_entries)
    {
        int64_t lo = e.job.window_start_sample;
        int64_t hi = latest_relevant_sample(e);
        if (hi >= window_start && lo <= window_end)
            out.push_back(e);
    }
    return out;
}

} // namespace streamgen
