#include "StreamGenProcessor.h"
#include "StreamGenDebugLog.h"
#include "GenerationTimelineStore.h"
#include "MusicalTime.h"

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
constexpr std::uint8_t k_drums_origin_hold = 3;

bool sample_in_completed_gen_land(int64_t sample_abs, const std::vector<JobTimelineRecord>& jobs)
{
    for (const JobTimelineRecord& e : jobs)
    {
        if (!e.has_completed || e.gen_samples <= 0)
            continue;
        const int64_t lo = e.job.output_start_sample();
        const int64_t hi = lo + e.gen_samples;
        if (sample_abs >= lo && sample_abs < hi)
            return true;
    }
    return false;
}

inline int effective_device_hz(int current_sr, int model_sr)
{
    return current_sr > 0 ? current_sr : juce::jmax(1, model_sr);
}

/// Map timeline sample (device clock) to a fractional index in file space; `native_len` is frames.
inline void warm_timeline_to_stereo_linear(
    const std::vector<float>& warm,
    int64_t native_len,
    double file_hz,
    double dev_hz,
    int64_t timeline_sample,
    float& out_l,
    float& out_r)
{
    if (native_len <= 0 || file_hz <= 0.0 || dev_hz <= 0.0)
    {
        out_l = 0.0f;
        out_r = 0.0f;
        return;
    }
    const double idx_f = static_cast<double>(timeline_sample) * (file_hz / dev_hz);
    const double fl = std::floor(idx_f);
    const float frac = static_cast<float>(idx_f - fl);
    int64_t i0 = static_cast<int64_t>(fl) % native_len;
    if (i0 < 0)
        i0 += native_len;
    int64_t i1 = i0 + 1;
    i1 %= native_len;
    if (i1 < 0)
        i1 += native_len;
    const size_t b0 = static_cast<size_t>(i0) * static_cast<size_t>(StreamGenProcessor::NUM_CHANNELS);
    const size_t b1 = static_cast<size_t>(i1) * static_cast<size_t>(StreamGenProcessor::NUM_CHANNELS);
    const float l0 = warm[b0];
    const float r0 = warm[b0 + 1u];
    const float l1 = warm[b1];
    const float r1 = warm[b1 + 1u];
    out_l = l0 * (1.0f - frac) + l1 * frac;
    out_r = r0 * (1.0f - frac) + r1 * frac;
}

/// Length in playback-clock samples of one full loop of the warmup WAV (`native_len` frames at `file_hz`).
inline double warm_playback_span_samples(int64_t native_len, double file_hz, double playback_hz)
{
    if (native_len <= 0 || file_hz <= 0.0 || playback_hz <= 0.0)
        return 0.0;
    return static_cast<double>(native_len) * playback_hz / file_hz;
}

/// Maps session timeline sample to the warmup-file timeline so session end aligns with warmup end (looping).
inline int64_t warmup_timeline_for_session_end_align(
    int64_t session_timeline_sample,
    int64_t streamgen_session_total_samples,
    int64_t warm_native_frames,
    double warm_file_hz,
    double playback_hz)
{
    if (streamgen_session_total_samples <= 0 || warm_native_frames <= 0 || warm_file_hz <= 0.0
        || playback_hz <= 0.0)
        return session_timeline_sample;
    const double W = warm_playback_span_samples(warm_native_frames, warm_file_hz, playback_hz);
    if (W <= 0.0)
        return session_timeline_sample;
    const double shift = W - static_cast<double>(streamgen_session_total_samples);
    return session_timeline_sample + static_cast<int64_t>(std::llround(shift));
}

/// Resample stereo row-major (L block, R block), `num_samples` frames at playback_hz, to model_hz.
/// Output length stays `num_samples`; same wall-time span as the playback window mapped to model sample times.
void resample_row_major_playback_window_to_model_rate(
    const std::vector<float>& src_lr,
    int num_samples,
    double playback_hz,
    double model_hz,
    std::vector<float>& out_lr)
{
    assert(num_samples >= 1);
    assert(static_cast<int>(src_lr.size()) == num_samples * StreamGenProcessor::NUM_CHANNELS);
    out_lr.resize(static_cast<size_t>(num_samples) * static_cast<size_t>(StreamGenProcessor::NUM_CHANNELS));
    if (playback_hz <= 0.0 || model_hz <= 0.0)
    {
        std::memcpy(out_lr.data(), src_lr.data(), src_lr.size() * sizeof(float));
        return;
    }
    if (std::abs(playback_hz - model_hz) < 1.0e-3)
    {
        std::memcpy(out_lr.data(), src_lr.data(), src_lr.size() * sizeof(float));
        return;
    }
    const float* Ls = src_lr.data();
    const float* Rs = src_lr.data() + static_cast<size_t>(num_samples);
    float* Lo = out_lr.data();
    float* Ro = out_lr.data() + static_cast<size_t>(num_samples);
    const int last = num_samples - 1;
    for (int j = 0; j < num_samples; ++j)
    {
        const double src_f = static_cast<double>(j) * playback_hz / model_hz;
        const int i0 = static_cast<int>(std::floor(src_f));
        const float frac = static_cast<float>(src_f - std::floor(src_f));
        const int i0c = juce::jlimit(0, last, i0);
        const int i1c = juce::jlimit(0, last, i0 + 1);
        Lo[static_cast<size_t>(j)] =
            Ls[static_cast<size_t>(i0c)] * (1.0f - frac) + Ls[static_cast<size_t>(i1c)] * frac;
        Ro[static_cast<size_t>(j)] =
            Rs[static_cast<size_t>(i0c)] * (1.0f - frac) + Rs[static_cast<size_t>(i1c)] * frac;
    }
}

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
    m_streamgen_audio_ring.assign(static_cast<size_t>(m_ring_buffer_size) * NUM_CHANNELS, 0.0f);
    m_drums_output_ring.assign(static_cast<size_t>(m_ring_buffer_size) * NUM_CHANNELS, 0.0f);
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

    if (m_ring_buffer_size > 0)
    {
        std::fill(m_streamgen_audio_ring.begin(), m_streamgen_audio_ring.end(), 0.0f);
        std::fill(m_drums_output_ring.begin(), m_drums_output_ring.end(), 0.0f);
        std::fill(m_drums_monitor_ring.begin(), m_drums_monitor_ring.end(), 0.0f);
        std::fill(m_drums_origin_ring.begin(), m_drums_origin_ring.end(), k_drums_origin_none);
    }

    simulation_playing.store(false, std::memory_order_relaxed);
    simulation_position.store(0, std::memory_order_relaxed);
    {
        std::lock_guard<std::mutex> lock(m_sim_mutex);
        m_sim_file_phase = 0.0;
    }

    {
        std::lock_guard<std::mutex> lock(m_last_gen_mutex);
        m_last_gen_row_major.clear();
        m_last_gen_num_samples = 0;
        m_last_gen_output_start_sample = 0;
        m_last_gen_snapshot_valid.store(false, std::memory_order_relaxed);
    }

    drums_output_from_last_gen_hold.store(false, std::memory_order_relaxed);

    m_streamgen_session_total_samples_for_warmup_end_align.store(0, std::memory_order_relaxed);

    streamgen_log("reset_timeline_and_transport done");
}

void StreamGenProcessor::set_streamgen_session_total_samples_for_warmup_end_align(
    int64_t total_streamgen_samples)
{
    const int64_t v = total_streamgen_samples < 0 ? 0 : total_streamgen_samples;
    m_streamgen_session_total_samples_for_warmup_end_align.store(v, std::memory_order_relaxed);
}

void StreamGenProcessor::clear_drums_output_buffers()
{
    if (m_ring_buffer_size > 0)
    {
        std::fill(m_drums_output_ring.begin(), m_drums_output_ring.end(), 0.0f);
        std::fill(m_drums_monitor_ring.begin(), m_drums_monitor_ring.end(), 0.0f);
        std::fill(m_drums_origin_ring.begin(), m_drums_origin_ring.end(), k_drums_origin_none);
    }

    {
        std::lock_guard<std::mutex> lock(m_last_gen_mutex);
        m_last_gen_row_major.clear();
        m_last_gen_num_samples = 0;
        m_last_gen_output_start_sample = 0;
        m_last_gen_snapshot_valid.store(false, std::memory_order_relaxed);
    }

    drums_output_from_last_gen_hold.store(false, std::memory_order_relaxed);
    streamgen_log("clear_drums_output_buffers");
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
    rebuild_click_impulses();
    {
        const int blocksz = device != nullptr ? device->getCurrentBufferSizeSamples() : 0;
        const size_t cap = static_cast<size_t>(juce::jmax(1024, blocksz));
        if (m_sim_callback_scratch.size() < cap)
            m_sim_callback_scratch.assign(cap, 0.0f);
    }
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

    float streamgen_audio_g = streamgen_audio_gain.load(std::memory_order_relaxed);
    float drums_g = drums_gain.load(std::memory_order_relaxed);

    // --- Write streamgen_audio (Python) into ring ---
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
            int written = 0;
            while (written < num_samples)
            {
                int chunk = std::min(num_samples - written, 2048);
                write_streamgen_audio_to_ring_at(zeros, chunk, pos_block_start + static_cast<int64_t>(written));
                written += chunk;
            }
            if (num_output_channels >= 2 && output_data != nullptr)
            {
                std::memset(output_data[0], 0, static_cast<size_t>(num_samples) * sizeof(float));
                std::memset(output_data[1], 0, static_cast<size_t>(num_samples) * sizeof(float));
                mix_click_track_into(output_data[0], output_data[1], num_samples, pos_block_start);
            }
            m_scheduler.advance(num_samples);
            const int64_t pos_after_mux = m_scheduler.absolute_sample_pos();
            static int mux_trace_ctr = 0;
            if (streamgen_log_audio_throttle(mux_trace_ctr))
                streamgen_log(juce::String::formatted(
                    "audio summary (sim mutex stall): pos %lld+%d -> %lld",
                    static_cast<long long>(pos_block_start),
                    num_samples,
                    static_cast<long long>(pos_after_mux)));
            drums_output_from_last_gen_hold.store(false, std::memory_order_relaxed);
            return;
        }

        in_path = "sim_file";

        if (static_cast<int>(m_sim_callback_scratch.size()) < num_samples)
            m_sim_callback_scratch.assign(static_cast<size_t>(num_samples), 0.0f);
        float* sim_buf = m_sim_callback_scratch.data();

        const int dev_hz = effective_device_hz(m_current_sample_rate, m_constants.sample_rate);
        const double sim_hz = static_cast<double>(
            juce::jmax(1, m_sim_native_sample_rate_hz.load(std::memory_order_relaxed)));
        const double ratio = sim_hz / static_cast<double>(dev_hz);
        const double sp = static_cast<double>(speed);
        const int64_t total_frames = static_cast<int64_t>(m_sim_audio.size());
        const bool loop = simulation_looping.load(std::memory_order_relaxed);
        int64_t last_fi = 0;

        for (int i = 0; i < num_samples; ++i)
        {
            const int64_t timeline_i = pos_block_start + static_cast<int64_t>(i);
            double file_d = static_cast<double>(timeline_i) * ratio * sp + m_sim_file_phase;
            float s = 0.0f;

            if (total_frames <= 0)
            {
                s = 0.0f;
            }
            else if (loop)
            {
                file_d = std::fmod(file_d, static_cast<double>(total_frames));
                if (file_d < 0.0)
                    file_d += static_cast<double>(total_frames);
                last_fi = static_cast<int64_t>(std::floor(file_d));
                s = m_sim_audio[static_cast<size_t>(last_fi)];
            }
            else if (file_d >= static_cast<double>(total_frames))
            {
                s = 0.0f;
                simulation_playing.store(false, std::memory_order_relaxed);
            }
            else
            {
                last_fi = static_cast<int64_t>(std::floor(file_d));
                last_fi = juce::jlimit(static_cast<int64_t>(0), total_frames - 1, last_fi);
                s = m_sim_audio[static_cast<size_t>(last_fi)];
            }

            sim_buf[i] = s;
        }

        simulation_position.store(last_fi, std::memory_order_relaxed);
        write_streamgen_audio_to_ring_at(sim_buf, num_samples, pos_block_start);
    }
    else
    {
        // Live mic input (mono channel 0)
        const float* mono_input = (input_data != nullptr && num_input_channels > 0)
            ? input_data[0] : nullptr;

        if (mono_input != nullptr)
        {
            in_path = "live_mic";
            write_streamgen_audio_to_ring_at(mono_input, num_samples, pos_block_start);
        }
        else
        {
            in_path = "live_zeros";
            float zeros[2048] = {};
            int written = 0;
            while (written < num_samples)
            {
                int chunk = std::min(num_samples - written, 2048);
                write_streamgen_audio_to_ring_at(zeros, chunk, pos_block_start + static_cast<int64_t>(written));
                written += chunk;
            }
        }
    }

    // --- Read drums output from ring buffer ---
    if (num_output_channels >= 2 && output_data != nullptr)
    {
        bool block_drums_hold = false;

        // Check if warmup audio should provide output
        bool use_warm = warmup_audio_playing.load(std::memory_order_relaxed);
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
            const int dev_hz_w = effective_device_hz(m_current_sample_rate, m_constants.sample_rate);
            const double warm_hz = static_cast<double>(
                juce::jmax(1, m_warm_native_sample_rate_hz.load(std::memory_order_relaxed)));
            const int64_t warm_align_T = m_streamgen_session_total_samples_for_warmup_end_align.load(
                std::memory_order_relaxed);

            for (int i = 0; i < num_samples; ++i)
            {
                float left = 0.0f;
                float right = 0.0f;
                bool consumed_warm_frame = false;

                if (m_warm_length_frames > 0)
                {
                    const int64_t timeline_i = block_start + static_cast<int64_t>(i);
                    const int64_t warm_tl = warmup_timeline_for_session_end_align(
                        timeline_i,
                        warm_align_T,
                        m_warm_length_frames,
                        warm_hz,
                        static_cast<double>(dev_hz_w));
                    warm_timeline_to_stereo_linear(
                        m_warm_audio,
                        m_warm_length_frames,
                        warm_hz,
                        static_cast<double>(dev_hz_w),
                        warm_tl,
                        left,
                        right);
                    consumed_warm_frame = true;
                }

                int64_t gen_abs = block_start + static_cast<int64_t>(i);
                const DrumsRingSample gen_s = fetch_drums_ring_sample(gen_abs);
                const float gen_left = gen_s.left;
                const float gen_right = gen_s.right;

                const bool has_generated = (std::fabs(gen_left) > k_drums_gen_amplitude_epsilon
                                            || std::fabs(gen_right) > k_drums_gen_amplitude_epsilon);
                block_drums_hold |= (gen_s.from_last_gen_hold && has_generated);
                std::uint8_t origin = k_drums_origin_none;
                if (has_generated)
                {
                    left = gen_left;
                    right = gen_right;
                    origin = gen_s.from_last_gen_hold ? k_drums_origin_hold : k_drums_origin_gen;
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
            for (int i = 0; i < num_samples; ++i)
            {
                const DrumsRingSample s = fetch_drums_ring_sample(block_start + static_cast<int64_t>(i));
                block_drums_hold |= s.from_last_gen_hold;
                float L = s.left;
                float R = s.right;
                output_data[0][i] = L;
                output_data[1][i] = R;
                int64_t mon_idx = absolute_to_ring_index(block_start + static_cast<int64_t>(i));
                size_t mon_base = static_cast<size_t>(mon_idx * NUM_CHANNELS);
                m_drums_monitor_ring[mon_base] = L;
                m_drums_monitor_ring[mon_base + 1] = R;
                std::uint8_t origin = k_drums_origin_none;
                if (std::fabs(L) > k_drums_gen_amplitude_epsilon || std::fabs(R) > k_drums_gen_amplitude_epsilon)
                    origin = s.from_last_gen_hold ? k_drums_origin_hold : k_drums_origin_gen;
                m_drums_origin_ring[static_cast<size_t>(mon_idx)] = origin;
                output_data[0][i] = L * drums_g;
                output_data[1][i] = R * drums_g;
            }
        }

        drums_output_from_last_gen_hold.store(block_drums_hold, std::memory_order_relaxed);

        // Mix streamgen_audio passthrough into output
        if (streamgen_audio_g > 0.0f)
        {
            int64_t current_pos = m_scheduler.absolute_sample_pos();
            for (int i = 0; i < num_samples; ++i)
            {
                int64_t abs_pos = current_pos + i;
                int64_t ring_idx = absolute_to_ring_index(abs_pos);
                float streamgen_audio_sample =
                    m_streamgen_audio_ring[static_cast<size_t>(ring_idx * NUM_CHANNELS)];
                output_data[0][i] += streamgen_audio_sample * streamgen_audio_g;
                output_data[1][i] += streamgen_audio_sample * streamgen_audio_g;
            }
        }

        mix_click_track_into(output_data[0], output_data[1], num_samples, pos_block_start);
    }
    else if (num_output_channels >= 1 && output_data != nullptr)
    {
        drums_path = "out_mono_only";
        std::memset(output_data[0], 0, static_cast<size_t>(num_samples) * sizeof(float));
        drums_output_from_last_gen_hold.store(false, std::memory_order_relaxed);
    }
    else
    {
        drums_path = "no_output";
        drums_output_from_last_gen_hold.store(false, std::memory_order_relaxed);
    }

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
            + " ring_frames=" + juce::String(m_ring_buffer_size)
            + " streamgen_audio_g=" + juce::String(streamgen_audio_g, 2)
            + " drums_g=" + juce::String(drums_g, 2));
    }
}

void StreamGenProcessor::rebuild_click_impulses()
{
    const int sr = juce::jmax(1, m_current_sample_rate);
    const int len = juce::jlimit(32, 320, sr / 80);
    m_click_impulse_beat.assign(static_cast<size_t>(len), 0.0f);
    const double two_pi = 6.28318530717958647692;
    const double freq_hz = 2000.0;
    double abs_peak = 0.0;
    for (int i = 0; i < len; ++i)
    {
        const double t = static_cast<double>(i) / static_cast<double>(sr);
        const double env = std::exp(-t * 140.0);
        const float s = static_cast<float>(std::sin(two_pi * freq_hz * t) * env);
        m_click_impulse_beat[static_cast<size_t>(i)] = s;
        abs_peak = std::max(abs_peak, std::abs(static_cast<double>(s)));
    }
    const float inv_peak = abs_peak > 1.0e-8 ? static_cast<float>(1.0 / abs_peak) : 1.0f;
    for (float& v : m_click_impulse_beat)
        v *= inv_peak;
    m_click_impulse_downbeat = m_click_impulse_beat;
    for (float& v : m_click_impulse_downbeat)
        v *= 1.5f;
}

void StreamGenProcessor::mix_click_track_into(float* left, float* right, int num_samples,
                                             int64_t block_start_sample)
{
    if (left == nullptr || right == nullptr || num_samples <= 0)
        return;
    if (!click_track_enabled.load(std::memory_order_relaxed))
        return;
    if (!m_scheduler.musical_time_enabled.load(std::memory_order_relaxed))
        return;
    const float vol = click_track_volume.load(std::memory_order_relaxed);
    if (vol <= 1.0e-8f)
        return;
    const int sr = effective_device_hz(m_current_sample_rate, m_constants.sample_rate);
    float bpm = m_scheduler.bpm.load(std::memory_order_relaxed);
    bpm = juce::jlimit(20.0f, 400.0f, bpm);
    const int bpb = juce::jmax(1, m_scheduler.time_sig_numerator.load(std::memory_order_relaxed));
    const int64_t spb = beats_to_samples(1.0, sr, static_cast<double>(bpm));
    if (spb <= 0)
        return;
    const int ir_len = static_cast<int>(m_click_impulse_beat.size());
    if (ir_len <= 0)
        return;

    int64_t t = (block_start_sample + spb - 1) / spb;
    t *= spb;
    const int64_t block_end = block_start_sample + static_cast<int64_t>(num_samples);
    while (t < block_end)
    {
        const int offset = static_cast<int>(t - block_start_sample);
        const int64_t beat_idx = t / spb;
        const bool downbeat = (beat_idx % static_cast<int64_t>(bpb)) == 0;
        const std::vector<float>& ir = downbeat ? m_click_impulse_downbeat : m_click_impulse_beat;
        for (int k = 0; k < ir_len; ++k)
        {
            const int dst = offset + k;
            if (dst >= num_samples)
                break;
            const float s = vol * ir[static_cast<size_t>(k)];
            left[dst] += s;
            right[dst] += s;
        }
        t += spb;
    }
}

void StreamGenProcessor::feed_streamgen_audio(const float* mono_input, int num_samples)
{
    streamgen_log("feed_streamgen_audio n=" + juce::String(num_samples) + " pos0="
        + juce::String(m_scheduler.absolute_sample_pos()));
    write_streamgen_audio_to_ring(mono_input, num_samples);
    m_scheduler.advance(num_samples);
}

void StreamGenProcessor::write_streamgen_audio_to_ring_at(
    const float* mono_input,
    int num_samples,
    int64_t abs_block_start)
{
    for (int i = 0; i < num_samples; ++i)
    {
        int64_t ring_idx = absolute_to_ring_index(abs_block_start + static_cast<int64_t>(i));
        size_t base = static_cast<size_t>(ring_idx * NUM_CHANNELS);
        float sample = mono_input[i];
        m_streamgen_audio_ring[base] = sample;
        m_streamgen_audio_ring[base + 1] = sample;
    }
}

void StreamGenProcessor::write_streamgen_audio_to_ring(const float* mono_input, int num_samples)
{
    write_streamgen_audio_to_ring_at(mono_input, num_samples, m_scheduler.absolute_sample_pos());
}

void StreamGenProcessor::read_drums_from_ring(float* left, float* right, int num_samples)
{
    int64_t block_start = m_scheduler.absolute_sample_pos();

    for (int i = 0; i < num_samples; ++i)
    {
        int64_t abs_pos = block_start + i;
        const DrumsRingSample s = fetch_drums_ring_sample(abs_pos);
        left[i] = s.left;
        right[i] = s.right;
    }
}

DrumsRingSample StreamGenProcessor::fetch_drums_ring_sample(int64_t absolute_sample) const
{
    DrumsRingSample out;
    int64_t ring_idx = absolute_to_ring_index(absolute_sample);
    size_t base = static_cast<size_t>(ring_idx * NUM_CHANNELS);
    float L = m_drums_output_ring[base];
    float R = m_drums_output_ring[base + 1];
    const bool silent = (std::fabs(L) <= k_drums_gen_amplitude_epsilon
                         && std::fabs(R) <= k_drums_gen_amplitude_epsilon);
    const bool gen_off = !m_scheduler.generation_enabled.load(std::memory_order_relaxed);
    const bool hold_inactive = !loop_last_generation.load(std::memory_order_relaxed) || gen_off;
    if (!silent || hold_inactive)
    {
        out.left = L;
        out.right = R;
        out.from_last_gen_hold = false;
        return out;
    }

    std::lock_guard<std::mutex> lock(m_last_gen_mutex);
    if (!m_last_gen_snapshot_valid.load(std::memory_order_relaxed) || m_last_gen_num_samples <= 0)
    {
        out.left = L;
        out.right = R;
        out.from_last_gen_hold = false;
        return out;
    }

    // Do not loop the last-gen snapshot before its scheduled land time: negative `off` modulo `len`
    // would map to an interior sample of the new buffer and audibly place generation early (see plan).
    if (absolute_sample < m_last_gen_output_start_sample)
    {
        out.left = L;
        out.right = R;
        out.from_last_gen_hold = false;
        return out;
    }

    const int64_t len = m_last_gen_num_samples;
    int64_t off = absolute_sample - m_last_gen_output_start_sample;
    assert(off >= 0);
    // Loop phase within the snapshot when the ring is silent after `land_start` (including gaps past
    // the written [land_start, land_start + len) region).
    int64_t mod = off % len;
    if (mod < 0)
        mod += len;
    out.left = m_last_gen_row_major[static_cast<size_t>(mod)];
    out.right = m_last_gen_row_major[static_cast<size_t>(static_cast<size_t>(len) + static_cast<size_t>(mod))];
    out.from_last_gen_hold = true;
    return out;
}

void StreamGenProcessor::output_ring_sample_at(int64_t absolute_sample, float& out_left, float& out_right) const
{
    const DrumsRingSample s = fetch_drums_ring_sample(absolute_sample);
    out_left = s.left;
    out_right = s.right;
}

void StreamGenProcessor::commit_last_generation_snapshot(
    const std::vector<float>& gen_row_major_stereo,
    int64_t output_start_sample,
    int64_t num_samples)
{
    assert(static_cast<int64_t>(gen_row_major_stereo.size()) >= num_samples * NUM_CHANNELS);
    std::lock_guard<std::mutex> lock(m_last_gen_mutex);
    m_last_gen_row_major.assign(
        gen_row_major_stereo.begin(),
        gen_row_major_stereo.begin() + static_cast<size_t>(num_samples * NUM_CHANNELS));
    m_last_gen_output_start_sample = output_start_sample;
    m_last_gen_num_samples = num_samples;
    m_last_gen_snapshot_valid.store(num_samples > 0, std::memory_order_relaxed);
}

int64_t StreamGenProcessor::absolute_to_ring_index(int64_t absolute_sample) const
{
    if (m_ring_buffer_size <= 0) return 0;
    int64_t idx = absolute_sample % m_ring_buffer_size;
    if (idx < 0) idx += m_ring_buffer_size;
    return idx;
}

// --- Worker thread interface ---

std::vector<float> StreamGenProcessor::snapshot_streamgen_audio(int64_t window_start, int64_t num_samples)
{
    std::vector<float> result(static_cast<size_t>(num_samples) * NUM_CHANNELS, 0.0f);

    for (int64_t i = 0; i < num_samples; ++i)
    {
        int64_t ring_idx = absolute_to_ring_index(window_start + i);
        size_t base = static_cast<size_t>(ring_idx * NUM_CHANNELS);
        // Row-major (2, N): left block then right block
        result[static_cast<size_t>(i)] = m_streamgen_audio_ring[base];
        result[static_cast<size_t>(num_samples + i)] = m_streamgen_audio_ring[base + 1];
    }

    return result;
}

std::vector<float> StreamGenProcessor::snapshot_drums_output(int64_t window_start, int64_t num_samples)
{
    std::vector<float> result(static_cast<size_t>(num_samples) * NUM_CHANNELS, 0.0f);

    for (int64_t i = 0; i < num_samples; ++i)
    {
        int64_t ring_idx = absolute_to_ring_index(window_start + i);
        size_t base = static_cast<size_t>(ring_idx * NUM_CHANNELS);
        result[static_cast<size_t>(i)] = m_drums_output_ring[base];
        result[static_cast<size_t>(num_samples + i)] = m_drums_output_ring[base + 1];
    }

    return result;
}

std::vector<float> StreamGenProcessor::snapshot_streamgen_audio_for_vae(int64_t window_start, int64_t num_samples)
{
    assert(num_samples > 0);
    std::vector<float> raw = snapshot_streamgen_audio(window_start, num_samples);
    assert(static_cast<int64_t>(raw.size()) == num_samples * NUM_CHANNELS);
    const double playback_hz = static_cast<double>(m_scheduler.effective_playback_rate_hz());
    const double model_hz = static_cast<double>(m_constants.sample_rate);
    std::vector<float> out;
    resample_row_major_playback_window_to_model_rate(
        raw, static_cast<int>(num_samples), playback_hz, model_hz, out);
    return out;
}

std::vector<float> StreamGenProcessor::snapshot_input_audio_for_vae(
    int64_t window_start,
    int64_t num_samples,
    int64_t keep_end_sample)
{
    assert(num_samples > 0);
    assert(keep_end_sample >= window_start);
    assert(keep_end_sample <= window_start + num_samples);
    const int ns = static_cast<int>(num_samples);
    std::vector<float> raw(static_cast<size_t>(num_samples) * NUM_CHANNELS, 0.0f);
    const double playback_hz = static_cast<double>(m_scheduler.effective_playback_rate_hz());
    const double warm_file_hz =
        static_cast<double>(m_warm_native_sample_rate_hz.load(std::memory_order_relaxed));
    const int64_t warm_align_T = m_streamgen_session_total_samples_for_warmup_end_align.load(
        std::memory_order_relaxed);

    for (int64_t i = 0; i < num_samples; ++i)
    {
        const int64_t abs_sample = window_start + i;
        if (abs_sample >= keep_end_sample)
        {
            raw[static_cast<size_t>(i)] = 0.0f;
            raw[static_cast<size_t>(static_cast<size_t>(num_samples) + static_cast<size_t>(i))] = 0.0f;
            continue;
        }

        DrumsRingSample s = fetch_drums_ring_sample(abs_sample);
        float L = s.left;
        float R = s.right;
        if (std::fabs(L) <= k_drums_gen_amplitude_epsilon
            && std::fabs(R) <= k_drums_gen_amplitude_epsilon)
        {
            std::lock_guard<std::mutex> warm_lock(m_warm_mutex);
            if (m_warm_length_frames > 0)
            {
                const int64_t warm_tl = warmup_timeline_for_session_end_align(
                    abs_sample,
                    warm_align_T,
                    m_warm_length_frames,
                    warm_file_hz,
                    playback_hz);
                warm_timeline_to_stereo_linear(
                    m_warm_audio,
                    m_warm_length_frames,
                    warm_file_hz,
                    playback_hz,
                    warm_tl,
                    L,
                    R);
            }
        }
        raw[static_cast<size_t>(i)] = L;
        raw[static_cast<size_t>(static_cast<size_t>(num_samples) + static_cast<size_t>(i))] = R;
    }

    const double model_hz = static_cast<double>(m_constants.sample_rate);
    std::vector<float> out;
    resample_row_major_playback_window_to_model_rate(raw, ns, playback_hz, model_hz, out);
    return out;
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
            float old_left = m_drums_output_ring[base];
            float old_right = m_drums_output_ring[base + 1];
            m_drums_output_ring[base] = old_left * (1.0f - alpha) + new_left * alpha;
            m_drums_output_ring[base + 1] = old_right * (1.0f - alpha) + new_right * alpha;
        }
        else
        {
            m_drums_output_ring[base] = new_left;
            m_drums_output_ring[base + 1] = new_right;
        }
    }

    commit_last_generation_snapshot(audio, start_sample, num_samples);
}

void StreamGenProcessor::set_simulation_playback_sample(int64_t sample)
{
    std::lock_guard<std::mutex> lock(m_sim_mutex);
    const int64_t total = static_cast<int64_t>(m_sim_audio.size());
    int64_t clamped = sample;
    if (total > 0 && clamped >= total)
        clamped = total - 1;
    if (clamped < 0)
        clamped = 0;
    const int dev_hz = effective_device_hz(m_current_sample_rate, m_constants.sample_rate);
    const double sim_hz = static_cast<double>(
        juce::jmax(1, m_sim_native_sample_rate_hz.load(std::memory_order_relaxed)));
    const double ratio = sim_hz / static_cast<double>(dev_hz);
    const double sp = static_cast<double>(simulation_speed.load(std::memory_order_relaxed));
    const int64_t timeline_a = m_scheduler.absolute_sample_pos();
    m_sim_file_phase = static_cast<double>(clamped) - static_cast<double>(timeline_a) * ratio * sp;
    simulation_position.store(clamped, std::memory_order_relaxed);
}

void StreamGenProcessor::snap_simulation_position_to_bar_grid()
{
    const float bpm = m_scheduler.bpm.load(std::memory_order_relaxed);
    const int bpb = juce::jmax(1, m_scheduler.time_sig_numerator.load(std::memory_order_relaxed));
    if (!m_scheduler.musical_time_enabled.load(std::memory_order_relaxed))
        return;
    const double bpm_d = static_cast<double>(juce::jlimit(20.0f, 400.0f, bpm));
    const int sim_sr_int = juce::jmax(1, m_sim_native_sample_rate_hz.load(std::memory_order_relaxed));
    const int64_t bar_file = beats_to_samples(static_cast<double>(bpb), sim_sr_int, bpm_d);
    if (bar_file <= 0)
        return;

    std::lock_guard<std::mutex> lock(m_sim_mutex);
    if (m_sim_audio.empty())
        return;
    const int dev_hz = effective_device_hz(m_current_sample_rate, m_constants.sample_rate);
    const double sim_hz = static_cast<double>(sim_sr_int);
    const double ratio = sim_hz / static_cast<double>(dev_hz);
    const double sp = static_cast<double>(simulation_speed.load(std::memory_order_relaxed));
    const int64_t timeline_a = m_scheduler.absolute_sample_pos();
    const double fp = static_cast<double>(timeline_a) * ratio * sp + m_sim_file_phase;
    const int64_t fi = static_cast<int64_t>(std::floor(fp));
    const int64_t snapped_fi = (fi / bar_file) * bar_file;
    m_sim_file_phase += static_cast<double>(snapped_fi) - fp;
    const int64_t total = static_cast<int64_t>(m_sim_audio.size());
    const int64_t pos_store = juce::jlimit(static_cast<int64_t>(0), total - 1, snapped_fi);
    simulation_position.store(pos_store, std::memory_order_relaxed);
}

// --- UI waveform readout (bucketed min/max over full sample span per bucket) ---

void StreamGenProcessor::fill_recent_streamgen_audio_waveform_buckets(
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

        for (int i = i0; i < i1; ++i)
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
            float v = m_streamgen_audio_ring[static_cast<size_t>(ring_idx * NUM_CHANNELS)];
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

void StreamGenProcessor::fill_recent_drums_output_waveform_buckets(
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

        for (int i = i0; i < i1; ++i)
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
    float* gen_max,
    float* hold_min,
    float* hold_max,
    const std::vector<JobTimelineRecord>* gen_land_jobs)
{
    assert(warm_min != nullptr && warm_max != nullptr && gen_min != nullptr && gen_max != nullptr
           && hold_min != nullptr && hold_max != nullptr);
    if (duration_samples <= 0 || num_buckets <= 0)
        return;

    const bool land_filter = gen_land_jobs != nullptr && !gen_land_jobs->empty();

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
        hold_min[b] = pos_inf;
        hold_max[b] = neg_inf;

        const int i0 = static_cast<int>((static_cast<int64_t>(b) * duration_samples) / num_buckets);
        int i1 = static_cast<int>((static_cast<int64_t>(b + 1) * duration_samples) / num_buckets);
        if (i1 <= i0)
            i1 = i0 + 1;

        for (int i = i0; i < i1; ++i)
        {
            int64_t sample_abs = start + static_cast<int64_t>(i);
            if (sample_abs > current_pos)
            {
                const float v = 0.0f;
                if (v < warm_min[b])
                    warm_min[b] = v;
                if (v > warm_max[b])
                    warm_max[b] = v;
                if (v < gen_min[b])
                    gen_min[b] = v;
                if (v > gen_max[b])
                    gen_max[b] = v;
                if (v < hold_min[b])
                    hold_min[b] = v;
                if (v > hold_max[b])
                    hold_max[b] = v;
                continue;
            }
            int64_t ring_idx = absolute_to_ring_index(sample_abs);
            size_t base = static_cast<size_t>(ring_idx * NUM_CHANNELS);
            const float v = (m_drums_monitor_ring[base] + m_drums_monitor_ring[base + 1]) * 0.5f;
            const std::uint8_t origin = m_drums_origin_ring[static_cast<size_t>(ring_idx)];
            if (origin == k_drums_origin_warm)
            {
                if (v < warm_min[b])
                    warm_min[b] = v;
                if (v > warm_max[b])
                    warm_max[b] = v;
            }
            else if (origin == k_drums_origin_gen)
            {
                const bool in_land = !land_filter || sample_in_completed_gen_land(sample_abs, *gen_land_jobs);
                if (in_land)
                {
                    if (v < gen_min[b])
                        gen_min[b] = v;
                    if (v > gen_max[b])
                        gen_max[b] = v;
                }
            }
            else if (origin == k_drums_origin_hold)
            {
                if (v < hold_min[b])
                    hold_min[b] = v;
                if (v > hold_max[b])
                    hold_max[b] = v;
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
        if (hold_max[b] < hold_min[b])
        {
            hold_min[b] = 0.0f;
            hold_max[b] = 0.0f;
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
        m_sim_file_phase = 0.0;
    }

    const int wav_sr = reader->sampleRate > 0.0
        ? juce::jlimit(8000, 384000, static_cast<int>(std::lround(reader->sampleRate)))
        : 44100;
    m_sim_native_sample_rate_hz.store(wav_sr, std::memory_order_relaxed);

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
    m_sim_file_phase = 0.0;
    m_sim_native_sample_rate_hz.store(44100, std::memory_order_relaxed);

    DBG("StreamGenProcessor: simulation cleared, reverting to live mic");
}

bool StreamGenProcessor::load_warmup_audio(const juce::File& file, bool start_playback)
{
    std::unique_ptr<juce::AudioFormatReader> reader(
        m_format_manager.createReaderFor(file));

    if (reader == nullptr)
    {
        DBG("StreamGenProcessor: failed to load warmup audio file: " + file.getFullPathName());
        return false;
    }

    auto num_frames = static_cast<int64_t>(reader->lengthInSamples);
    juce::AudioBuffer<float> buffer(static_cast<int>(reader->numChannels), static_cast<int>(num_frames));
    reader->read(&buffer, 0, static_cast<int>(num_frames), 0, true, true);

    // One loop length = file length only. Padding to the model window (sample_size) inserted huge
    // trailing silence for short clips and broke alignment with the metronome (absolute_sample_pos
    // % loop vs beat grid).
    std::vector<float> stereo(static_cast<size_t>(num_frames) * NUM_CHANNELS, 0.0f);

    for (int64_t i = 0; i < num_frames; ++i)
    {
        size_t dst = static_cast<size_t>(i) * NUM_CHANNELS;
        float left = buffer.getReadPointer(0)[i];
        float right = (reader->numChannels >= 2) ? buffer.getReadPointer(1)[i] : left;
        stereo[dst] = left;
        stereo[dst + 1] = right;
    }

    const int wav_sr = reader->sampleRate > 0.0
        ? juce::jlimit(8000, 384000, static_cast<int>(std::lround(reader->sampleRate)))
        : 44100;
    m_warm_native_sample_rate_hz.store(wav_sr, std::memory_order_relaxed);

    {
        std::lock_guard<std::mutex> lock(m_warm_mutex);
        m_warm_audio = std::move(stereo);
        m_warm_length_frames = num_frames;
    }

    warmup_audio_playing.store(start_playback, std::memory_order_relaxed);

    DBG("StreamGenProcessor: loaded warmup audio file: " + file.getFileName()
        + " (" + juce::String(num_frames) + " frames)"
        + (start_playback ? ", playing" : ", idle"));
    return true;
}

void StreamGenProcessor::set_warmup_audio_playing(bool playing)
{
    if (!playing)
    {
        warmup_audio_playing.store(false, std::memory_order_relaxed);
        return;
    }
    std::lock_guard<std::mutex> lock(m_warm_mutex);
    if (m_warm_length_frames > 0)
        warmup_audio_playing.store(true, std::memory_order_relaxed);
}

bool StreamGenProcessor::warmup_audio_has_audio() const
{
    std::lock_guard<std::mutex> lock(m_warm_mutex);
    return m_warm_length_frames > 0;
}

int StreamGenProcessor::simulation_file_native_sample_rate_hz() const
{
    return juce::jmax(1, m_sim_native_sample_rate_hz.load(std::memory_order_relaxed));
}

int StreamGenProcessor::warmup_audio_file_native_sample_rate_hz() const
{
    return juce::jmax(1, m_warm_native_sample_rate_hz.load(std::memory_order_relaxed));
}

} // namespace streamgen
