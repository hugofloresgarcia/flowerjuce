#include "StreamGenProcessor.h"

#include <juce_audio_formats/juce_audio_formats.h>
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstring>

namespace streamgen {

StreamGenProcessor::StreamGenProcessor()
{
    m_format_manager.registerBasicFormats();
}

void StreamGenProcessor::configure(const ModelConstants& constants)
{
    m_constants = constants;
    m_scheduler.configure(constants);

    m_ring_buffer_size = constants.sample_rate * RING_BUFFER_SECONDS;
    m_input_ring.resize(static_cast<size_t>(m_ring_buffer_size) * NUM_CHANNELS, 0.0f);
    m_output_ring.resize(static_cast<size_t>(m_ring_buffer_size) * NUM_CHANNELS, 0.0f);

    DBG("StreamGenProcessor configured: ring_buffer_size=" + juce::String(m_ring_buffer_size)
        + " frames, window=" + juce::String(constants.window_seconds(), 2) + "s");
}

void StreamGenProcessor::audioDeviceAboutToStart(juce::AudioIODevice* device)
{
    m_current_sample_rate = static_cast<int>(device->getCurrentSampleRate());
    DBG("StreamGenProcessor: audio device starting, sample_rate=" + juce::String(m_current_sample_rate));
}

void StreamGenProcessor::audioDeviceStopped()
{
    DBG("StreamGenProcessor: audio device stopped");
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

    float sax_g = sax_gain.load(std::memory_order_relaxed);
    float drums_g = drums_gain.load(std::memory_order_relaxed);

    // --- Write sax input to ring buffer ---
    if (simulation_active.load(std::memory_order_relaxed)
        && simulation_playing.load(std::memory_order_relaxed))
    {
        // Simulation mode: read from loaded file
        float speed = simulation_speed.load(std::memory_order_relaxed);
        std::unique_lock<std::mutex> lock(m_sim_mutex, std::try_to_lock);

        if (!lock.owns_lock())
        {
            // Mutex held by UI thread during file load — output silence this block
            float zeros[2048] = {};
            int remaining = num_samples;
            while (remaining > 0)
            {
                int chunk = std::min(remaining, 2048);
                write_sax_to_ring(zeros, chunk);
                remaining -= chunk;
            }
            m_scheduler.advance(num_samples);
            if (num_output_channels >= 2 && output_data != nullptr)
            {
                std::memset(output_data[0], 0, static_cast<size_t>(num_samples) * sizeof(float));
                std::memset(output_data[1], 0, static_cast<size_t>(num_samples) * sizeof(float));
            }
            return;
        }

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
            write_sax_to_ring(mono_input, num_samples);
        }
        else
        {
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
            for (int i = 0; i < num_samples; ++i)
            {
                float left = 0.0f;
                float right = 0.0f;

                if (m_warm_length_frames > 0 && m_warm_playback_pos < m_warm_length_frames)
                {
                    size_t idx = static_cast<size_t>(m_warm_playback_pos) * NUM_CHANNELS;
                    left = m_warm_audio[idx];
                    right = m_warm_audio[idx + 1];
                    m_warm_playback_pos++;

                    if (m_warm_playback_pos >= m_warm_length_frames
                        && warm_start_looping.load(std::memory_order_relaxed))
                    {
                        m_warm_playback_pos = 0;
                    }
                }

                // Generated audio in the ring buffer takes priority over warm-start
                int64_t abs_pos = m_scheduler.absolute_sample_pos()
                    - num_samples + i;
                int64_t ring_idx = absolute_to_ring_index(abs_pos);
                float gen_left = m_output_ring[static_cast<size_t>(ring_idx * NUM_CHANNELS)];
                float gen_right = m_output_ring[static_cast<size_t>(ring_idx * NUM_CHANNELS + 1)];

                bool has_generated = (gen_left != 0.0f || gen_right != 0.0f);
                if (has_generated)
                {
                    left = gen_left;
                    right = gen_right;
                }

                output_data[0][i] = left * drums_g;
                output_data[1][i] = right * drums_g;
            }
        }
        else
        {
            read_drums_from_ring(output_data[0], output_data[1], num_samples);
            for (int i = 0; i < num_samples; ++i)
            {
                output_data[0][i] *= drums_g;
                output_data[1][i] *= drums_g;
            }
        }

        // Mix sax passthrough into output
        if (sax_g > 0.0f)
        {
            int64_t current_pos = m_scheduler.absolute_sample_pos();
            for (int i = 0; i < num_samples; ++i)
            {
                int64_t abs_pos = current_pos - num_samples + i;
                int64_t ring_idx = absolute_to_ring_index(abs_pos);
                float sax_sample = m_input_ring[static_cast<size_t>(ring_idx * NUM_CHANNELS)];
                output_data[0][i] += sax_sample * sax_g;
                output_data[1][i] += sax_sample * sax_g;
            }
        }
    }
    else if (num_output_channels >= 1 && output_data != nullptr)
    {
        std::memset(output_data[0], 0, static_cast<size_t>(num_samples) * sizeof(float));
    }

    // Advance scheduler (checks hop trigger)
    m_scheduler.advance(num_samples);
}

void StreamGenProcessor::feed_audio(const float* mono_input, int num_samples)
{
    write_sax_to_ring(mono_input, num_samples);
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
    int64_t current_pos = m_scheduler.absolute_sample_pos();

    for (int i = 0; i < num_samples; ++i)
    {
        int64_t abs_pos = current_pos - num_samples + i;
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

// --- UI waveform readout ---

std::vector<float> StreamGenProcessor::get_recent_input_waveform(int duration_samples)
{
    int64_t current_pos = m_scheduler.absolute_sample_pos();
    int64_t start = current_pos - duration_samples;
    std::vector<float> result(static_cast<size_t>(duration_samples), 0.0f);

    for (int i = 0; i < duration_samples; ++i)
    {
        int64_t ring_idx = absolute_to_ring_index(start + i);
        result[static_cast<size_t>(i)] = m_input_ring[static_cast<size_t>(ring_idx * NUM_CHANNELS)];
    }

    return result;
}

std::vector<float> StreamGenProcessor::get_recent_output_waveform(int duration_samples)
{
    int64_t current_pos = m_scheduler.absolute_sample_pos();
    int64_t start = current_pos - duration_samples;
    std::vector<float> result(static_cast<size_t>(duration_samples), 0.0f);

    for (int i = 0; i < duration_samples; ++i)
    {
        int64_t ring_idx = absolute_to_ring_index(start + i);
        size_t base = static_cast<size_t>(ring_idx * NUM_CHANNELS);
        result[static_cast<size_t>(i)] = (m_output_ring[base] + m_output_ring[base + 1]) * 0.5f;
    }

    return result;
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
    simulation_active.store(true, std::memory_order_relaxed);
    simulation_playing.store(false, std::memory_order_relaxed);

    DBG("StreamGenProcessor: loaded simulation file: " + file.getFileName()
        + " (" + juce::String(num_frames) + " frames)");
    return true;
}

void StreamGenProcessor::clear_simulation()
{
    simulation_active.store(false, std::memory_order_relaxed);
    simulation_playing.store(false, std::memory_order_relaxed);

    std::lock_guard<std::mutex> lock(m_sim_mutex);
    m_sim_audio.clear();
    m_sim_playback_pos = 0.0;

    DBG("StreamGenProcessor: simulation cleared, reverting to live mic");
}

bool StreamGenProcessor::load_warm_start(const juce::File& file)
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

    warm_start_playing.store(true, std::memory_order_relaxed);

    DBG("StreamGenProcessor: loaded warm-start file: " + file.getFileName()
        + " (" + juce::String(num_frames) + " frames, padded to "
        + juce::String(total_frames) + ")");
    return true;
}

} // namespace streamgen
