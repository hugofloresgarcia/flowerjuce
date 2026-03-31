#include "StreamGenProcessor.h"
#include "StreamGenDebugLog.h"
#include "GenerationTimelineStore.h"

#include <juce_audio_formats/juce_audio_formats.h>
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>

namespace streamgen {

namespace {

constexpr float k_drums_gen_amplitude_epsilon = 1e-6f;
constexpr std::uint8_t k_drums_origin_none = 0;
constexpr std::uint8_t k_drums_origin_warm = 1;
constexpr std::uint8_t k_drums_origin_gen = 2;

} // namespace

StreamGenProcessor::StreamGenProcessor()
{
    m_format_manager.registerBasicFormats();
    m_timeline = std::make_unique<GenerationTimelineStore>();
    m_scheduler.set_timeline_store(m_timeline.get());
}

StreamGenProcessor::~StreamGenProcessor() = default;

void StreamGenProcessor::rebuild_ring_buffers(int ring_sample_rate)
{
    if (ring_sample_rate <= 0)
        return;
    m_ring_buffer_size = ring_sample_rate * RING_BUFFER_SECONDS;
    m_input_ring.assign(static_cast<size_t>(m_ring_buffer_size) * NUM_CHANNELS, 0.0f);
    m_output_ring.assign(static_cast<size_t>(m_ring_buffer_size) * NUM_CHANNELS, 0.0f);
    m_drums_monitor_ring.assign(static_cast<size_t>(m_ring_buffer_size) * NUM_CHANNELS, 0.0f);
    m_drums_origin_ring.assign(static_cast<size_t>(m_ring_buffer_size), k_drums_origin_none);
    DBG("StreamGenProcessor: ring_buffer_size=" + juce::String(m_ring_buffer_size)
        + " frames @ " + juce::String(ring_sample_rate) + " Hz");
    streamgen_log("rebuild_ring_buffers: frames=" + juce::String(m_ring_buffer_size)
        + " rate_hz=" + juce::String(ring_sample_rate)
        + " abs_pos=" + juce::String(m_scheduler.absolute_sample_pos()));
}

void StreamGenProcessor::reset_timeline_and_transport()
{
    if (m_timeline != nullptr)
        m_timeline->clear();
    m_scheduler.reset_session();
    m_audio_telemetry.reset_counters();

    if (m_ring_buffer_size > 0)
    {
        std::fill(m_input_ring.begin(), m_input_ring.end(), 0.0f);
        std::fill(m_output_ring.begin(), m_output_ring.end(), 0.0f);
        std::fill(m_drums_monitor_ring.begin(), m_drums_monitor_ring.end(), 0.0f);
        std::fill(m_drums_origin_ring.begin(), m_drums_origin_ring.end(), k_drums_origin_none);
    }

    simulation_playing.store(false, std::memory_order_relaxed);
    simulation_position.store(0, std::memory_order_relaxed);
    {
        std::lock_guard<std::mutex> lock(m_sim_mutex);
        m_sim_playback_pos = 0.0;
    }

    {
        std::lock_guard<std::mutex> lock(m_warm_mutex);
        m_warm_playback_pos = 0;
    }

    streamgen_log("reset_timeline_and_transport done");
}

void StreamGenProcessor::configure(const ModelConstants& constants)
{
    m_constants = constants;
    m_scheduler.configure(constants);

    const int ring_sr = m_current_sample_rate > 0 ? m_current_sample_rate : constants.sample_rate;
    rebuild_ring_buffers(ring_sr);

    DBG("StreamGenProcessor configured: window=" + juce::String(constants.window_seconds(), 2) + "s"
        + " model_sr=" + juce::String(constants.sample_rate));
    streamgen_log("configure: model_sr=" + juce::String(constants.sample_rate)
        + " sample_size=" + juce::String(constants.sample_size)
        + " ring_sr=" + juce::String(ring_sr)
        + " abs_pos=" + juce::String(m_scheduler.absolute_sample_pos()));
}

void StreamGenProcessor::audioDeviceAboutToStart(juce::AudioIODevice* device)
{
    m_current_sample_rate = static_cast<int>(device->getCurrentSampleRate());
    m_scheduler.set_playback_sample_rate(m_current_sample_rate);
    rebuild_ring_buffers(m_current_sample_rate);
    DBG("StreamGenProcessor: audio device starting, sample_rate=" + juce::String(m_current_sample_rate));
    juce::String io_name = device != nullptr ? device->getName() : juce::String("<null>");
    streamgen_log("audioDeviceAboutToStart: device=" + io_name
        + " sr=" + juce::String(m_current_sample_rate)
        + " buf=" + juce::String(device != nullptr ? device->getCurrentBufferSizeSamples() : 0)
        + " sched_eff_hz=" + juce::String(m_scheduler.effective_playback_rate_hz()));
}

void StreamGenProcessor::audioDeviceStopped()
{
    DBG("StreamGenProcessor: audio device stopped");
    streamgen_log("audioDeviceStopped abs_pos=" + juce::String(m_scheduler.absolute_sample_pos()));
}

void StreamGenProcessor::audioDeviceIOCallbackWithContext(
    const float* const* input_data,
    int num_input_channels,
    float* const* output_data,
    int num_output_channels,
    int num_samples,
    const juce::AudioIODeviceCallbackContext&)
{
    juce::ScopedNoDenormals no_denormals;

    const int64_t pos_block_start = m_scheduler.absolute_sample_pos();
    juce::String in_path = "init";
    juce::String drums_path = "pending";

    float sax_g = sax_gain.load(std::memory_order_relaxed);
    float drums_g = drums_gain.load(std::memory_order_relaxed);

    double sum_in_sq = 0.0;

    // --- Write sax input to ring buffer ---
    if (simulation_playing.load(std::memory_order_relaxed)
        && simulation_total_samples.load(std::memory_order_relaxed) > 0)
    {
        // Simulation mode: read from loaded file
        float speed = simulation_speed.load(std::memory_order_relaxed);
        std::unique_lock<std::mutex> lock(m_sim_mutex, std::try_to_lock);

        if (!lock.owns_lock())
        {
            // Mutex held by UI thread during file load — output silence this block
            streamgen_log(juce::String::formatted(
                "audio: sim file load mutex busy | pos=%lld n=%d -> advance only, silence I/O",
                static_cast<long long>(pos_block_start),
                num_samples));
            float zeros[2048] = {};
            int remaining = num_samples;
            while (remaining > 0)
            {
                int chunk = std::min(remaining, 2048);
                write_sax_to_ring(zeros, chunk);
                remaining -= chunk;
            }
            if (num_output_channels >= 2 && output_data != nullptr)
            {
                std::memset(output_data[0], 0, static_cast<size_t>(num_samples) * sizeof(float));
                std::memset(output_data[1], 0, static_cast<size_t>(num_samples) * sizeof(float));
            }
            m_audio_telemetry.record_block(0.0, 0.0, 0.0, num_samples);
            m_scheduler.advance(num_samples);
            const int64_t pos_after_mux = m_scheduler.absolute_sample_pos();
            static int mux_trace_ctr = 0;
            if (streamgen_log_audio_throttle(mux_trace_ctr))
                streamgen_log(juce::String::formatted(
                    "audio summary (sim mutex stall): pos %lld+%d -> %lld",
                    static_cast<long long>(pos_block_start),
                    num_samples,
                    static_cast<long long>(pos_after_mux)));
            return;
        }

        in_path = "sim_file";

        float sim_buf[2048];
        int to_process = std::min(num_samples, 2048);

        for (int i = 0; i < to_process; ++i)
        {
            int64_t pos = static_cast<int64_t>(m_sim_playback_pos);
            int64_t total = static_cast<int64_t>(m_sim_audio.size());

            if (total == 0)
            {
                sim_buf[i] = 0.0f;
            }
            else if (pos >= total)
            {
                if (simulation_looping.load(std::memory_order_relaxed))
                {
                    m_sim_playback_pos = 0.0;
                    pos = 0;
                    sim_buf[i] = m_sim_audio[static_cast<size_t>(pos)];
                }
                else
                {
                    sim_buf[i] = 0.0f;
                    simulation_playing.store(false, std::memory_order_relaxed);
                }
            }
            else
            {
                sim_buf[i] = m_sim_audio[static_cast<size_t>(pos)];
            }

            m_sim_playback_pos += static_cast<double>(speed);
        }

        for (int i = 0; i < to_process; ++i)
        {
            float s = sim_buf[i];
            sum_in_sq += static_cast<double>(s) * static_cast<double>(s);
        }

        simulation_position.store(static_cast<int64_t>(m_sim_playback_pos), std::memory_order_relaxed);
        write_sax_to_ring(sim_buf, to_process);

        if (to_process < num_samples)
        {
            float zeros[2048] = {};
            write_sax_to_ring(zeros, num_samples - to_process);
        }
    }
    else
    {
        // Live mic input (mono channel 0)
        const float* mono_input = (input_data != nullptr && num_input_channels > 0)
            ? input_data[0] : nullptr;

        if (mono_input != nullptr)
        {
            in_path = "live_mic";
            for (int i = 0; i < num_samples; ++i)
            {
                float s = mono_input[i];
                sum_in_sq += static_cast<double>(s) * static_cast<double>(s);
            }
            write_sax_to_ring(mono_input, num_samples);
        }
        else
        {
            in_path = "live_zeros";
            float zeros[2048] = {};
            int remaining = num_samples;
            while (remaining > 0)
            {
                int chunk = std::min(remaining, 2048);
                write_sax_to_ring(zeros, chunk);
                remaining -= chunk;
            }
        }
    }

    // --- Read drums output from ring buffer ---
    if (num_output_channels >= 2 && output_data != nullptr)
    {
        // Check if warm-start should provide audio
        bool use_warm = warm_start_playing.load(std::memory_order_relaxed);
        bool warm_locked = false;
        std::unique_lock<std::mutex> warm_lock(m_warm_mutex, std::defer_lock);

        if (use_warm)
        {
            warm_locked = warm_lock.try_lock();
            if (!warm_locked)
                use_warm = false;
        }

        if (use_warm && warm_locked)
        {
            drums_path = "warm_plus_ring";
            const int64_t block_start = m_scheduler.absolute_sample_pos();

            for (int i = 0; i < num_samples; ++i)
            {
                float left = 0.0f;
                float right = 0.0f;
                bool consumed_warm_frame = false;

                if (m_warm_length_frames > 0 && m_warm_playback_pos < m_warm_length_frames)
                {
                    size_t idx = static_cast<size_t>(m_warm_playback_pos) * NUM_CHANNELS;
                    left = m_warm_audio[idx];
                    right = m_warm_audio[idx + 1];
                    consumed_warm_frame = true;
                    m_warm_playback_pos++;

                    if (m_warm_playback_pos >= m_warm_length_frames
                        && warm_start_looping.load(std::memory_order_relaxed))
                    {
                        m_warm_playback_pos = 0;
                    }
                }

                int64_t gen_abs = block_start + static_cast<int64_t>(i);
                int64_t ring_idx_gen = absolute_to_ring_index(gen_abs);
                float gen_left = m_output_ring[static_cast<size_t>(ring_idx_gen * NUM_CHANNELS)];
                float gen_right = m_output_ring[static_cast<size_t>(ring_idx_gen * NUM_CHANNELS + 1)];

                const bool has_generated = (std::fabs(gen_left) > k_drums_gen_amplitude_epsilon
                                            || std::fabs(gen_right) > k_drums_gen_amplitude_epsilon);
                std::uint8_t origin = k_drums_origin_none;
                if (has_generated)
                {
                    left = gen_left;
                    right = gen_right;
                    origin = k_drums_origin_gen;
                }
                else if (consumed_warm_frame)
                    origin = k_drums_origin_warm;

                int64_t mon_idx = absolute_to_ring_index(gen_abs);
                size_t mon_base = static_cast<size_t>(mon_idx * NUM_CHANNELS);
                m_drums_monitor_ring[mon_base] = left;
                m_drums_monitor_ring[mon_base + 1] = right;
                m_drums_origin_ring[static_cast<size_t>(mon_idx)] = origin;

                output_data[0][i] = left * drums_g;
                output_data[1][i] = right * drums_g;
            }
        }
        else
        {
            drums_path = use_warm ? "ring_only_warm_lock_fail" : "ring";
            const int64_t block_start = m_scheduler.absolute_sample_pos();
            read_drums_from_ring(output_data[0], output_data[1], num_samples);
            for (int i = 0; i < num_samples; ++i)
            {
                int64_t mon_idx = absolute_to_ring_index(block_start + static_cast<int64_t>(i));
                size_t mon_base = static_cast<size_t>(mon_idx * NUM_CHANNELS);
                float L = output_data[0][i];
                float R = output_data[1][i];
                m_drums_monitor_ring[mon_base] = L;
                m_drums_monitor_ring[mon_base + 1] = R;
                const std::uint8_t origin = (std::fabs(L) > k_drums_gen_amplitude_epsilon
                                             || std::fabs(R) > k_drums_gen_amplitude_epsilon)
                    ? k_drums_origin_gen
                    : k_drums_origin_none;
                m_drums_origin_ring[static_cast<size_t>(mon_idx)] = origin;
                output_data[0][i] = L * drums_g;
                output_data[1][i] = R * drums_g;
            }
        }

        // Mix sax passthrough into output
        if (sax_g > 0.0f)
        {
            int64_t current_pos = m_scheduler.absolute_sample_pos();
            for (int i = 0; i < num_samples; ++i)
            {
                int64_t abs_pos = current_pos + i;
                int64_t ring_idx = absolute_to_ring_index(abs_pos);
                float sax_sample = m_input_ring[static_cast<size_t>(ring_idx * NUM_CHANNELS)];
                output_data[0][i] += sax_sample * sax_g;
                output_data[1][i] += sax_sample * sax_g;
            }
        }
    }
    else if (num_output_channels >= 1 && output_data != nullptr)
    {
        drums_path = "out_mono_only";
        std::memset(output_data[0], 0, static_cast<size_t>(num_samples) * sizeof(float));
    }
    else
    {
        drums_path = "no_output";
    }

    double sum_out_l_sq = 0.0;
    double sum_out_r_sq = 0.0;
    if (num_output_channels >= 2 && output_data != nullptr)
    {
        for (int i = 0; i < num_samples; ++i)
        {
            float l = output_data[0][i];
            float r = output_data[1][i];
            sum_out_l_sq += static_cast<double>(l) * static_cast<double>(l);
            sum_out_r_sq += static_cast<double>(r) * static_cast<double>(r);
        }
    }

    m_audio_telemetry.record_block(sum_in_sq, sum_out_l_sq, sum_out_r_sq, num_samples);

    // Advance scheduler (checks hop trigger)
    m_scheduler.advance(num_samples);
    const int64_t pos_after = m_scheduler.absolute_sample_pos();

    static int audio_trace_counter = 0;
    if (streamgen_log_audio_throttle(audio_trace_counter))
    {
        const bool gen_en = m_scheduler.generation_enabled.load(std::memory_order_relaxed);
        streamgen_log("audio summary: pos=" + juce::String(pos_block_start) + "+" + juce::String(num_samples)
            + " -> " + juce::String(pos_after) + " | in=" + in_path + " drums=" + drums_path
            + " | in_ch=" + juce::String(num_input_channels) + " out_ch=" + juce::String(num_output_channels)
            + " | gen_en=" + juce::String(gen_en ? 1 : 0) + " sr_dev=" + juce::String(m_current_sample_rate)
            + " ring_frames=" + juce::String(m_ring_buffer_size) + " sax_g=" + juce::String(sax_g, 2)
            + " drums_g=" + juce::String(drums_g, 2));
    }
}

void StreamGenProcessor::feed_audio(const float* mono_input, int num_samples)
{
    streamgen_log("feed_audio n=" + juce::String(num_samples) + " pos0="
        + juce::String(m_scheduler.absolute_sample_pos()));
    double sum_in_sq = 0.0;
    for (int i = 0; i < num_samples; ++i)
    {
        float s = mono_input[i];
        sum_in_sq += static_cast<double>(s) * static_cast<double>(s);
    }
    write_sax_to_ring(mono_input, num_samples);
    m_audio_telemetry.record_block(sum_in_sq, 0.0, 0.0, num_samples);
    m_scheduler.advance(num_samples);
}

void StreamGenProcessor::write_sax_to_ring(const float* mono_input, int num_samples)
{
    int64_t abs_pos = m_scheduler.absolute_sample_pos();

    for (int i = 0; i < num_samples; ++i)
    {
        int64_t ring_idx = absolute_to_ring_index(abs_pos + i);
        size_t base = static_cast<size_t>(ring_idx * NUM_CHANNELS);
        float sample = mono_input[i];
        m_input_ring[base] = sample;
        m_input_ring[base + 1] = sample;
    }
}

void StreamGenProcessor::read_drums_from_ring(float* left, float* right, int num_samples)
{
    int64_t block_start = m_scheduler.absolute_sample_pos();

    for (int i = 0; i < num_samples; ++i)
    {
        int64_t abs_pos = block_start + i;
        int64_t ring_idx = absolute_to_ring_index(abs_pos);
        size_t base = static_cast<size_t>(ring_idx * NUM_CHANNELS);
        left[i] = m_output_ring[base];
        right[i] = m_output_ring[base + 1];
    }
}

int64_t StreamGenProcessor::absolute_to_ring_index(int64_t absolute_sample) const
{
    if (m_ring_buffer_size <= 0) return 0;
    int64_t idx = absolute_sample % m_ring_buffer_size;
    if (idx < 0) idx += m_ring_buffer_size;
    return idx;
}

// --- Worker thread interface ---

std::vector<float> StreamGenProcessor::snapshot_input(int64_t window_start, int64_t num_samples)
{
    std::vector<float> result(static_cast<size_t>(num_samples) * NUM_CHANNELS, 0.0f);

    for (int64_t i = 0; i < num_samples; ++i)
    {
        int64_t ring_idx = absolute_to_ring_index(window_start + i);
        size_t base = static_cast<size_t>(ring_idx * NUM_CHANNELS);
        // Row-major (2, N): left block then right block
        result[static_cast<size_t>(i)] = m_input_ring[base];
        result[static_cast<size_t>(num_samples + i)] = m_input_ring[base + 1];
    }

    return result;
}

std::vector<float> StreamGenProcessor::snapshot_output(int64_t window_start, int64_t num_samples)
{
    std::vector<float> result(static_cast<size_t>(num_samples) * NUM_CHANNELS, 0.0f);

    for (int64_t i = 0; i < num_samples; ++i)
    {
        int64_t ring_idx = absolute_to_ring_index(window_start + i);
        size_t base = static_cast<size_t>(ring_idx * NUM_CHANNELS);
        result[static_cast<size_t>(i)] = m_output_ring[base];
        result[static_cast<size_t>(num_samples + i)] = m_output_ring[base + 1];
    }

    return result;
}

void StreamGenProcessor::write_output(
    const std::vector<float>& audio,
    int64_t start_sample,
    int64_t num_samples,
    int crossfade_samples)
{
    assert(static_cast<int64_t>(audio.size()) >= num_samples * NUM_CHANNELS);
    crossfade_samples = std::min(crossfade_samples, static_cast<int>(num_samples));

    for (int64_t i = 0; i < num_samples; ++i)
    {
        int64_t ring_idx = absolute_to_ring_index(start_sample + i);
        size_t base = static_cast<size_t>(ring_idx * NUM_CHANNELS);

        float new_left = audio[static_cast<size_t>(i)];
        float new_right = audio[static_cast<size_t>(num_samples + i)];

        if (i < crossfade_samples)
        {
            float alpha = static_cast<float>(i) / static_cast<float>(crossfade_samples);
            float old_left = m_output_ring[base];
            float old_right = m_output_ring[base + 1];
            m_output_ring[base] = old_left * (1.0f - alpha) + new_left * alpha;
            m_output_ring[base + 1] = old_right * (1.0f - alpha) + new_right * alpha;
        }
        else
        {
            m_output_ring[base] = new_left;
            m_output_ring[base + 1] = new_right;
        }
    }
}

// --- UI waveform readout (bucketed min/max; subsampled ring reads per bucket) ---

namespace {

/// At most this many ring samples read per horizontal bucket (stride picks step size).
constexpr int k_max_ring_reads_per_waveform_bucket = 16;

} // namespace

void StreamGenProcessor::fill_recent_input_waveform_buckets(
    int duration_samples,
    int num_buckets,
    float* out_min,
    float* out_max)
{
    assert(out_min != nullptr && out_max != nullptr);
    if (duration_samples <= 0 || num_buckets <= 0)
        return;

    int64_t current_pos = m_scheduler.absolute_sample_pos();
    const int64_t past_span = static_cast<int64_t>(
        static_cast<double>(duration_samples) * static_cast<double>(k_timeline_playhead_past_fraction));
    int64_t start = current_pos - past_span;
    const float neg_inf = -std::numeric_limits<float>::infinity();
    const float pos_inf = std::numeric_limits<float>::infinity();

    for (int b = 0; b < num_buckets; ++b)
    {
        out_min[b] = pos_inf;
        out_max[b] = neg_inf;

        const int i0 = static_cast<int>((static_cast<int64_t>(b) * duration_samples) / num_buckets);
        int i1 = static_cast<int>((static_cast<int64_t>(b + 1) * duration_samples) / num_buckets);
        if (i1 <= i0)
            i1 = i0 + 1;
        const int span = i1 - i0;
        const int step = juce::jmax(1, (span + k_max_ring_reads_per_waveform_bucket - 1)
                                        / k_max_ring_reads_per_waveform_bucket);

        for (int i = i0; i < i1; i += step)
        {
            int64_t sample_abs = start + static_cast<int64_t>(i);
            if (sample_abs > current_pos)
            {
                float v = 0.0f;
                if (v < out_min[b])
                    out_min[b] = v;
                if (v > out_max[b])
                    out_max[b] = v;
                continue;
            }
            int64_t ring_idx = absolute_to_ring_index(sample_abs);
            float v = m_input_ring[static_cast<size_t>(ring_idx * NUM_CHANNELS)];
            if (v < out_min[b])
                out_min[b] = v;
            if (v > out_max[b])
                out_max[b] = v;
        }
    }

    for (int b = 0; b < num_buckets; ++b)
    {
        if (out_max[b] < out_min[b])
        {
            out_min[b] = 0.0f;
            out_max[b] = 0.0f;
        }
    }
}

void StreamGenProcessor::fill_recent_output_waveform_buckets(
    int duration_samples,
    int num_buckets,
    float* out_min,
    float* out_max)
{
    assert(out_min != nullptr && out_max != nullptr);
    if (duration_samples <= 0 || num_buckets <= 0)
        return;

    int64_t current_pos = m_scheduler.absolute_sample_pos();
    const int64_t past_span = static_cast<int64_t>(
        static_cast<double>(duration_samples) * static_cast<double>(k_timeline_playhead_past_fraction));
    int64_t start = current_pos - past_span;
    const float neg_inf = -std::numeric_limits<float>::infinity();
    const float pos_inf = std::numeric_limits<float>::infinity();

    for (int b = 0; b < num_buckets; ++b)
    {
        out_min[b] = pos_inf;
        out_max[b] = neg_inf;

        const int i0 = static_cast<int>((static_cast<int64_t>(b) * duration_samples) / num_buckets);
        int i1 = static_cast<int>((static_cast<int64_t>(b + 1) * duration_samples) / num_buckets);
        if (i1 <= i0)
            i1 = i0 + 1;
        const int span = i1 - i0;
        const int step = juce::jmax(1, (span + k_max_ring_reads_per_waveform_bucket - 1)
                                        / k_max_ring_reads_per_waveform_bucket);

        for (int i = i0; i < i1; i += step)
        {
            int64_t sample_abs = start + static_cast<int64_t>(i);
            if (sample_abs > current_pos)
            {
                float v = 0.0f;
                if (v < out_min[b])
                    out_min[b] = v;
                if (v > out_max[b])
                    out_max[b] = v;
                continue;
            }
            int64_t ring_idx = absolute_to_ring_index(sample_abs);
            size_t base = static_cast<size_t>(ring_idx * NUM_CHANNELS);
            float v = (m_drums_monitor_ring[base] + m_drums_monitor_ring[base + 1]) * 0.5f;
            if (v < out_min[b])
                out_min[b] = v;
            if (v > out_max[b])
                out_max[b] = v;
        }
    }

    for (int b = 0; b < num_buckets; ++b)
    {
        if (out_max[b] < out_min[b])
        {
            out_min[b] = 0.0f;
            out_max[b] = 0.0f;
        }
    }
}

void StreamGenProcessor::fill_recent_drums_source_buckets(
    int duration_samples,
    int num_buckets,
    float* warm_min,
    float* warm_max,
    float* gen_min,
    float* gen_max)
{
    assert(warm_min != nullptr && warm_max != nullptr && gen_min != nullptr && gen_max != nullptr);
    if (duration_samples <= 0 || num_buckets <= 0)
        return;

    int64_t current_pos = m_scheduler.absolute_sample_pos();
    const int64_t past_span = static_cast<int64_t>(
        static_cast<double>(duration_samples) * static_cast<double>(k_timeline_playhead_past_fraction));
    int64_t start = current_pos - past_span;
    const float neg_inf = -std::numeric_limits<float>::infinity();
    const float pos_inf = std::numeric_limits<float>::infinity();

    for (int b = 0; b < num_buckets; ++b)
    {
        warm_min[b] = pos_inf;
        warm_max[b] = neg_inf;
        gen_min[b] = pos_inf;
        gen_max[b] = neg_inf;

        const int i0 = static_cast<int>((static_cast<int64_t>(b) * duration_samples) / num_buckets);
        int i1 = static_cast<int>((static_cast<int64_t>(b + 1) * duration_samples) / num_buckets);
        if (i1 <= i0)
            i1 = i0 + 1;
        const int span = i1 - i0;
        const int step = juce::jmax(1, (span + k_max_ring_reads_per_waveform_bucket - 1)
                                        / k_max_ring_reads_per_waveform_bucket);

        for (int i = i0; i < i1; i += step)
        {
            int64_t sample_abs = start + static_cast<int64_t>(i);
            if (sample_abs > current_pos)
            {
                float v = 0.0f;
                if (v < warm_min[b])
                    warm_min[b] = v;
                if (v > warm_max[b])
                    warm_max[b] = v;
                if (v < gen_min[b])
                    gen_min[b] = v;
                if (v > gen_max[b])
                    gen_max[b] = v;
                continue;
            }
            int64_t ring_idx = absolute_to_ring_index(sample_abs);
            size_t base = static_cast<size_t>(ring_idx * NUM_CHANNELS);
            float v = (m_drums_monitor_ring[base] + m_drums_monitor_ring[base + 1]) * 0.5f;
            std::uint8_t origin = m_drums_origin_ring[static_cast<size_t>(ring_idx)];
            if (origin == k_drums_origin_warm)
            {
                if (v < warm_min[b])
                    warm_min[b] = v;
                if (v > warm_max[b])
                    warm_max[b] = v;
            }
            else if (origin == k_drums_origin_gen)
            {
                if (v < gen_min[b])
                    gen_min[b] = v;
                if (v > gen_max[b])
                    gen_max[b] = v;
            }
        }
    }

    for (int b = 0; b < num_buckets; ++b)
    {
        if (warm_max[b] < warm_min[b])
        {
            warm_min[b] = 0.0f;
            warm_max[b] = 0.0f;
        }
        if (gen_max[b] < gen_min[b])
        {
            gen_min[b] = 0.0f;
            gen_max[b] = 0.0f;
        }
    }
}

// --- File loading ---

bool StreamGenProcessor::load_simulation_file(const juce::File& file)
{
    std::unique_ptr<juce::AudioFormatReader> reader(
        m_format_manager.createReaderFor(file));

    if (reader == nullptr)
    {
        DBG("StreamGenProcessor: failed to load simulation file: " + file.getFullPathName());
        return false;
    }

    auto num_frames = static_cast<int64_t>(reader->lengthInSamples);
    juce::AudioBuffer<float> buffer(static_cast<int>(reader->numChannels), static_cast<int>(num_frames));
    reader->read(&buffer, 0, static_cast<int>(num_frames), 0, true, true);

    // Convert to mono
    std::vector<float> mono(static_cast<size_t>(num_frames));
    if (reader->numChannels == 1)
    {
        std::memcpy(mono.data(), buffer.getReadPointer(0),
                     static_cast<size_t>(num_frames) * sizeof(float));
    }
    else
    {
        const float* left = buffer.getReadPointer(0);
        const float* right = buffer.getReadPointer(1);
        for (int64_t i = 0; i < num_frames; ++i)
            mono[static_cast<size_t>(i)] = (left[i] + right[i]) * 0.5f;
    }

    {
        std::lock_guard<std::mutex> lock(m_sim_mutex);
        m_sim_audio = std::move(mono);
        m_sim_playback_pos = 0.0;
    }

    simulation_total_samples.store(num_frames, std::memory_order_relaxed);
    simulation_position.store(0, std::memory_order_relaxed);
    simulation_playing.store(false, std::memory_order_relaxed);
    m_simulation_display_name = file.getFileName();

    DBG("StreamGenProcessor: loaded simulation file: " + file.getFileName()
        + " (" + juce::String(num_frames) + " frames)");
    return true;
}

juce::String StreamGenProcessor::simulation_display_name() const
{
    return m_simulation_display_name;
}

void StreamGenProcessor::clear_simulation()
{
    simulation_playing.store(false, std::memory_order_relaxed);
    simulation_total_samples.store(0, std::memory_order_relaxed);
    simulation_position.store(0, std::memory_order_relaxed);
    m_simulation_display_name.clear();

    std::lock_guard<std::mutex> lock(m_sim_mutex);
    m_sim_audio.clear();
    m_sim_playback_pos = 0.0;

    DBG("StreamGenProcessor: simulation cleared, reverting to live mic");
}

bool StreamGenProcessor::load_warm_start(const juce::File& file, bool start_playback)
{
    std::unique_ptr<juce::AudioFormatReader> reader(
        m_format_manager.createReaderFor(file));

    if (reader == nullptr)
    {
        DBG("StreamGenProcessor: failed to load warm-start file: " + file.getFullPathName());
        return false;
    }

    auto num_frames = static_cast<int64_t>(reader->lengthInSamples);
    juce::AudioBuffer<float> buffer(static_cast<int>(reader->numChannels), static_cast<int>(num_frames));
    reader->read(&buffer, 0, static_cast<int>(num_frames), 0, true, true);

    int64_t target_frames = m_constants.sample_size;
    int64_t pad_frames = (num_frames < target_frames) ? (target_frames - num_frames) : 0;
    int64_t total_frames = pad_frames + num_frames;

    // Build stereo interleaved buffer [L, R, L, R, ...]
    std::vector<float> stereo(static_cast<size_t>(total_frames) * NUM_CHANNELS, 0.0f);

    for (int64_t i = 0; i < num_frames; ++i)
    {
        size_t dst = static_cast<size_t>(pad_frames + i) * NUM_CHANNELS;
        float left = buffer.getReadPointer(0)[i];
        float right = (reader->numChannels >= 2) ? buffer.getReadPointer(1)[i] : left;
        stereo[dst] = left;
        stereo[dst + 1] = right;
    }

    {
        std::lock_guard<std::mutex> lock(m_warm_mutex);
        m_warm_audio = std::move(stereo);
        m_warm_length_frames = total_frames;
        m_warm_playback_pos = 0;
    }

    warm_start_playing.store(start_playback, std::memory_order_relaxed);

    DBG("StreamGenProcessor: loaded warm-start file: " + file.getFileName()
        + " (" + juce::String(num_frames) + " frames, padded to "
        + juce::String(total_frames) + ")"
        + (start_playback ? ", playing" : ", idle"));
    return true;
}

void StreamGenProcessor::set_warm_start_playing(bool playing)
{
    if (!playing)
    {
        warm_start_playing.store(false, std::memory_order_relaxed);
        return;
    }
    std::lock_guard<std::mutex> lock(m_warm_mutex);
    if (m_warm_length_frames > 0)
        warm_start_playing.store(true, std::memory_order_relaxed);
}

bool StreamGenProcessor::warm_start_has_audio() const
{
    std::lock_guard<std::mutex> lock(m_warm_mutex);
    return m_warm_length_frames > 0;
}

} // namespace streamgen
