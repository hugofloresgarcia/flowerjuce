/// StreamGen CLI — headless batch streaming for testing on CUDA servers.
///
/// Reuses StreamGenProcessor, InferenceWorker, and GenerationScheduler from the
/// GUI app. Feeds streamgen_audio through the same ring buffers and generation logic
/// synchronously (no threads, no audio device).
///
/// Usage:
///   StreamGenCLI --manifest path/to/manifest.json --input streamgen_audio.wav [options]
///
/// Audio timeline (matches Python `generate_streamgen_inpaint`):
///   - `--input` is **`streamgen_audio`** (Python): decoded to mono, resampled to manifest
///     `sample_rate`, then fed into `m_streamgen_audio_ring`. In CLI one tick = one model-rate sample.
///   - `--warmup-audio` seeds **`input_audio`** (Python drum stem) where the drums ring + hold are silent.
///     When warmup is loaded, its playback timeline is **end-aligned** with `--input`: the last session
///     sample reads the end of the warmup file (looping), so streaming conditions on “future” drums.
///   - Jobs snapshot `sample_size` playback-clock frames; the worker resamples to VAE rate. Missing
///     streamgen_audio ring data is **zeros**.

#include "StreamGenProcessor.h"
#include "InferenceWorker.h"
#include "TimeRuler.h"

#include "sao_inference/ZenonPipelineConfig.h"

#include <juce_audio_formats/juce_audio_formats.h>
#include <juce_core/juce_core.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

static void write_wav(
    const std::string& path,
    const std::vector<float>& audio,
    int sample_rate,
    int num_channels,
    int num_samples)
{
    std::vector<int16_t> pcm(num_channels * num_samples);
    float max_abs = 0.0f;
    for (float s : audio) {
        float a = std::abs(s);
        if (a > max_abs) max_abs = a;
    }
    float scale = (max_abs > 0.0f) ? (32767.0f / max_abs) : 1.0f;

    for (int s = 0; s < num_samples; ++s) {
        for (int c = 0; c < num_channels; ++c) {
            float val = audio[c * num_samples + s] * scale;
            val = std::max(-32768.0f, std::min(32767.0f, val));
            pcm[s * num_channels + c] = static_cast<int16_t>(val);
        }
    }

    int data_size = static_cast<int>(pcm.size()) * 2;
    int file_size = 36 + data_size;

    std::ofstream out(path, std::ios::binary);
    auto write_str = [&](const char* s) { out.write(s, 4); };
    auto write_i32 = [&](int32_t v) { out.write(reinterpret_cast<const char*>(&v), 4); };
    auto write_i16 = [&](int16_t v) { out.write(reinterpret_cast<const char*>(&v), 2); };

    write_str("RIFF");
    write_i32(file_size);
    write_str("WAVE");
    write_str("fmt ");
    write_i32(16);
    write_i16(1);
    write_i16(static_cast<int16_t>(num_channels));
    write_i32(sample_rate);
    write_i32(sample_rate * num_channels * 2);
    write_i16(static_cast<int16_t>(num_channels * 2));
    write_i16(16);
    write_str("data");
    write_i32(data_size);
    out.write(reinterpret_cast<const char*>(pcm.data()), data_size);
}

/// Load a WAV file as mono float at the file's native sample rate.
///
/// Args:
///     path: Path to the WAV file.
///     format_manager: JUCE format manager with registered formats.
///
/// Returns:
///     `{ mono_samples, native_sample_rate_hz }`.
///
/// Raises:
///     Asserts on failure.
static std::pair<std::vector<float>, double> load_wav_mono_native_rate(
    const std::string& path,
    juce::AudioFormatManager& format_manager)
{
    juce::File file(path);
    assert(file.existsAsFile() && ("File not found: " + path).c_str());

    std::unique_ptr<juce::AudioFormatReader> reader(
        format_manager.createReaderFor(file));
    assert(reader != nullptr && ("Failed to read: " + path).c_str());

    auto num_frames = static_cast<int>(reader->lengthInSamples);
    juce::AudioBuffer<float> buffer(static_cast<int>(reader->numChannels), num_frames);
    reader->read(&buffer, 0, num_frames, 0, true, true);

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
        for (int i = 0; i < num_frames; ++i)
            mono[static_cast<size_t>(i)] = (left[i] + right[i]) * 0.5f;
    }

    const double native_hz = reader->sampleRate > 0.0 ? static_cast<double>(reader->sampleRate) : 44100.0;
    std::cout << "  Loaded " << path << " (" << num_frames << " samples, "
              << reader->numChannels << "ch, "
              << native_hz << " Hz native)" << std::endl;
    return {std::move(mono), native_hz};
}

/// Linearly resample mono audio from source_hz to target_hz (same clock the model expects).
///
/// Args:
///     input: Mono PCM at `source_hz`.
///     source_hz: Input rate in Hz; must be > 0.
///     target_hz: Manifest / VAE rate in Hz; must be > 0.
///
/// Returns:
///     Mono PCM at `target_hz`. Empty if `input` is empty. Copy of `input` if rates match.
static std::vector<float> resample_mono_linear(
    const std::vector<float>& input,
    double source_hz,
    double target_hz)
{
    assert(source_hz > 0.0 && target_hz > 0.0);
    if (input.empty())
        return {};
    if (std::abs(source_hz - target_hz) < 1.0e-3)
        return input;

    const double ratio = source_hz / target_hz;
    const auto out_len = static_cast<size_t>(std::max(
        1.0,
        std::ceil(static_cast<double>(input.size()) / ratio)));
    std::vector<float> output(out_len);
    const size_t last_in = input.size() - 1u;

    for (size_t j = 0; j < out_len; ++j)
    {
        const double src_f = static_cast<double>(j) * ratio;
        const auto i0 = static_cast<size_t>(std::floor(src_f));
        const float frac = static_cast<float>(src_f - std::floor(src_f));
        if (i0 >= last_in)
            output[j] = input[last_in];
        else
            output[j] = input[i0] * (1.0f - frac) + input[i0 + 1u] * frac;
    }

    std::cout << "  Resampled mono: " << input.size() << " @ " << source_hz << " Hz -> "
              << output.size() << " @ " << target_hz << " Hz" << std::endl;
    return output;
}

struct TimingEntry {
    int64_t gen_id;
    double vae_encode_ms;
    double t5_ms;
    double sampling_ms;
    double vae_decode_ms;
    double total_ms;
};

static void write_timing_json(
    const std::string& path,
    const std::vector<TimingEntry>& entries,
    float hop_s,
    float keep_ratio,
    int steps,
    float cfg,
    float schedule_delay_s,
    const std::string& prompt,
    const std::string& input_file,
    double input_duration_s)
{
    std::ofstream out(path);
    out << "{\n";

    out << "  \"config\": {"
        << " \"hop_s\": " << hop_s
        << ", \"keep_ratio\": " << keep_ratio
        << ", \"steps\": " << steps
        << ", \"cfg\": " << cfg
        << ", \"schedule_delay_s\": " << schedule_delay_s
        << ", \"prompt\": \"" << prompt << "\""
        << " },\n";

    out << "  \"input\": {"
        << " \"file\": \"" << input_file << "\""
        << ", \"duration_s\": " << input_duration_s
        << " },\n";

    out << "  \"generations\": [\n";
    for (size_t i = 0; i < entries.size(); ++i)
    {
        const auto& e = entries[i];
        out << "    { \"id\": " << e.gen_id
            << ", \"vae_encode_ms\": " << e.vae_encode_ms
            << ", \"t5_ms\": " << e.t5_ms
            << ", \"sampling_ms\": " << e.sampling_ms
            << ", \"vae_decode_ms\": " << e.vae_decode_ms
            << ", \"total_ms\": " << e.total_ms
            << " }";
        if (i + 1 < entries.size()) out << ",";
        out << "\n";
    }
    out << "  ],\n";

    double total_ms = 0.0;
    for (const auto& e : entries) total_ms += e.total_ms;
    double avg_ms = entries.empty() ? 0.0 : total_ms / static_cast<double>(entries.size());

    out << "  \"summary\": {"
        << " \"total_generations\": " << entries.size()
        << ", \"total_ms\": " << total_ms
        << ", \"avg_ms\": " << avg_ms
        << " }\n";

    out << "}\n";
}

static void print_usage(const char* prog)
{
    std::cerr << "Usage: " << prog << " [options]\n"
              << "  --manifest FILE          Path to zenon_pipeline_manifest.json (REQUIRED)\n"
              << "  --input FILE             streamgen_audio WAV (Python streamgen_audio) (REQUIRED)\n"
              << "  --warmup-audio FILE      Warmup-audio drums WAV\n"
              << "  --output FILE            Output drums WAV (default: output_streamgen.wav)\n"
              << "  --timing-json FILE       Write per-generation timing JSON\n"
              << "  --prompt TEXT             Text prompt (default: percussion)\n"
              << "  --hop SECONDS            Override hop (default: model window * (1 - keep ratio))\n"
              << "  --keep-ratio FLOAT       Inpaint keep ratio 0.0-1.0 (default: 0.5)\n"
              << "  --steps N                Sampling steps (default: 8)\n"
              << "  --cfg SCALE              CFG scale (default: 7.0)\n"
              << "  --schedule-delay SEC     Seconds after keep_end where output lands (default: 0)\n"
              << "  --seed N                 Random seed (default: 42)\n"
              << "  --crossfade-ms N         Crossfade in milliseconds (default: 100)\n"
              << "  --cuda                   Use CUDA execution provider\n"
              << "  --coreml                 Use CoreML execution provider (macOS)\n"
              << "  --migraphx               Use MIGraphX execution provider (Linux/ROCm)\n"
              << "  --mlx-vae                Zenon VAE via MLX Metal (requires SAO_ENABLE_MLX build)\n";
}

int main(int argc, char* argv[])
{
    juce::ScopedJuceInitialiser_GUI juce_init;

    std::string manifest_path;
    std::string input_file;
    std::string warmup_audio_file;
    std::string output_file = "output_streamgen.wav";
    std::string timing_json_file;
    std::string prompt = "percussion";
    bool hop_explicit = false;
    float hop_override_seconds = 0.0f;
    float keep_ratio = 0.5f;
    int steps = 8;
    float cfg_scale = 7.0f;
    float schedule_delay_seconds = 0.0f;
    uint32_t seed = 42;
    int crossfade_ms = 100;
    bool use_cuda = false;
    bool use_coreml = false;
    bool use_migraphx = false;
    bool use_mlx_vae = false;

    for (int i = 1; i < argc; ++i)
    {
        std::string arg = argv[i];
        if (arg == "--manifest" && i + 1 < argc) manifest_path = argv[++i];
        else if (arg == "--input" && i + 1 < argc) input_file = argv[++i];
        else if (arg == "--warmup-audio" && i + 1 < argc) warmup_audio_file = argv[++i];
        else if (arg == "--output" && i + 1 < argc) output_file = argv[++i];
        else if (arg == "--timing-json" && i + 1 < argc) timing_json_file = argv[++i];
        else if (arg == "--prompt" && i + 1 < argc) prompt = argv[++i];
        else if (arg == "--hop" && i + 1 < argc)
        {
            hop_explicit = true;
            hop_override_seconds = std::stof(argv[++i]);
        }
        else if (arg == "--keep-ratio" && i + 1 < argc) keep_ratio = std::stof(argv[++i]);
        else if (arg == "--steps" && i + 1 < argc) steps = std::stoi(argv[++i]);
        else if (arg == "--cfg" && i + 1 < argc) cfg_scale = std::stof(argv[++i]);
        else if (arg == "--schedule-delay" && i + 1 < argc)
            schedule_delay_seconds = std::stof(argv[++i]);
        else if (arg == "--seed" && i + 1 < argc) seed = static_cast<uint32_t>(std::stoul(argv[++i]));
        else if (arg == "--crossfade-ms" && i + 1 < argc) crossfade_ms = std::stoi(argv[++i]);
        else if (arg == "--cuda") use_cuda = true;
        else if (arg == "--coreml") use_coreml = true;
        else if (arg == "--migraphx") use_migraphx = true;
        else if (arg == "--mlx-vae") use_mlx_vae = true;
        else if (arg == "--help" || arg == "-h") { print_usage(argv[0]); return 0; }
        else { std::cerr << "Unknown argument: " << arg << std::endl; print_usage(argv[0]); return 1; }
    }

    assert(!manifest_path.empty() && "Must provide --manifest");
    assert(!input_file.empty() && "Must provide --input");

    sao::ZenonPipelineConfig zen_config = sao::ZenonPipelineConfig::load(manifest_path);
    const int model_sample_rate = zen_config.sample_rate;

    std::cout << "=== StreamGen CLI ===" << std::endl;
    std::cout << "Manifest: " << manifest_path << std::endl;
    std::cout << "Input:    " << input_file << std::endl;
    std::cout << "Output:   " << output_file << std::endl;
    std::cout << "Model SR: " << model_sample_rate << " Hz" << std::endl;
    std::cout << "Prompt:   " << prompt << std::endl;
    std::cout << "Keep: " << keep_ratio << ", Steps: " << steps << ", CFG: " << cfg_scale
              << ", Schedule delay: " << schedule_delay_seconds << "s" << std::endl;

    // --- Load input audio and resample to manifest sample_rate (VAE + ring timeline) ---
    juce::AudioFormatManager format_manager;
    format_manager.registerBasicFormats();

    auto mono_native = load_wav_mono_native_rate(input_file, format_manager);
    std::vector<float> input_mono = resample_mono_linear(
        mono_native.first,
        mono_native.second,
        static_cast<double>(model_sample_rate));
    int64_t total_input_samples = static_cast<int64_t>(input_mono.size());
    double input_duration_s = static_cast<double>(total_input_samples)
        / static_cast<double>(model_sample_rate);

    // --- Set up processor and worker ---
    streamgen::StreamGenProcessor processor;
    streamgen::InferenceWorker worker(processor);

    worker.load_pipeline(manifest_path, use_cuda, use_coreml, use_mlx_vae, use_migraphx);
    worker.set_prompt(prompt);

    auto& scheduler = processor.scheduler();
    scheduler.keep_ratio.store(keep_ratio, std::memory_order_relaxed);
    if (hop_explicit)
        scheduler.hop_seconds.store(hop_override_seconds, std::memory_order_relaxed);
    else
        scheduler.sync_hop_seconds_to_keep_ratio();
    scheduler.steps.store(steps, std::memory_order_relaxed);
    scheduler.cfg_scale.store(cfg_scale, std::memory_order_relaxed);
    scheduler.schedule_delay_seconds.store(schedule_delay_seconds, std::memory_order_relaxed);
    scheduler.generation_enabled.store(true, std::memory_order_relaxed);

    std::cout << "Hop:      " << scheduler.hop_seconds.load(std::memory_order_relaxed) << "s"
              << (hop_explicit ? " (--hop)" : " (window*(1-keep))") << std::endl;

    const int crossfade_samples = crossfade_ms * model_sample_rate / 1000;
    worker.crossfade_samples.store(crossfade_samples, std::memory_order_relaxed);

    // --- Load warmup audio (optional) ---
    if (!warmup_audio_file.empty())
    {
        juce::File ws_file(warmup_audio_file);
        assert(ws_file.existsAsFile() && "Warmup audio file not found");
        processor.load_warmup_audio(ws_file);
        processor.set_streamgen_session_total_samples_for_warmup_end_align(total_input_samples);
        std::cout << "Warmup audio loaded: " << warmup_audio_file
                  << " (end-aligned with input, " << total_input_samples << " samples)" << std::endl;
    }

    // --- Streaming loop ---
    std::cout << "\nStarting streaming loop..." << std::endl;
    auto t_start = std::chrono::steady_clock::now();

    std::vector<TimingEntry> timing_entries;
    constexpr int BLOCK_SIZE = 4096;
    int64_t pos = 0;
    int gen_count = 0;

    while (pos < total_input_samples)
    {
        int64_t remaining = total_input_samples - pos;
        int chunk = static_cast<int>(std::min(static_cast<int64_t>(BLOCK_SIZE), remaining));
        processor.feed_streamgen_audio(&input_mono[static_cast<size_t>(pos)], chunk);
        pos += chunk;

        streamgen::GenerationJob job;
        while (scheduler.pop_job(job))
        {
            worker.process_job(job);
            gen_count++;

            auto timing = worker.last_timing();
            timing_entries.push_back({
                job.job_id,
                timing.vae_encode_ms,
                timing.t5_encode_ms,
                timing.sampling_total_ms,
                timing.vae_decode_ms,
                timing.total_ms
            });

            double progress = 100.0 * static_cast<double>(pos) / static_cast<double>(total_input_samples);
            std::cout << "  Gen #" << gen_count
                      << " (job " << job.job_id << ")"
                      << " " << timing.total_ms << "ms"
                      << "  [" << progress << "%]"
                      << std::endl;
        }
    }

    auto t_end = std::chrono::steady_clock::now();
    double total_wall_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();

    std::cout << "\nStreaming complete: " << gen_count << " generations in "
              << total_wall_ms << "ms" << std::endl;

    // --- Extract output from ring buffer ---
    int64_t output_samples = total_input_samples;
    auto output_waveform = processor.snapshot_drums_output(0, output_samples);

    // output_waveform is row-major (2, N)
    constexpr int NUM_CHANNELS = 2;
    int num_output_samples = static_cast<int>(output_samples);

    std::cout << "Writing " << num_output_samples << " samples to " << output_file << std::endl;
    write_wav(output_file, output_waveform, model_sample_rate, NUM_CHANNELS, num_output_samples);

    // --- Timing JSON ---
    if (!timing_json_file.empty())
    {
        write_timing_json(timing_json_file, timing_entries,
                          scheduler.hop_seconds.load(std::memory_order_relaxed),
                          keep_ratio, steps, cfg_scale, schedule_delay_seconds,
                          prompt, input_file, input_duration_s);
        std::cout << "Timing report written to " << timing_json_file << std::endl;
    }

    std::cout << "Done!" << std::endl;
    return 0;
}
