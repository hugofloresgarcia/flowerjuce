#include "sao_inference/ZenonPipeline.h"
#include "sao_inference/ZenonPipelineConfig.h"
#include "sao_inference/Tokenizer.h"
#include <cnpy.h>

#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
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

static void write_timing_json(const std::string& path, const sao::ZenonTimingReport& t)
{
    std::ofstream out(path);
    out << "{\n";
    out << "  \"model_load_ms\": " << t.model_load_ms << ",\n";
    out << "  \"vae_encode_ms\": " << t.vae_encode_ms << ",\n";
    out << "  \"t5_encode_ms\": " << t.t5_encode_ms << ",\n";
    out << "  \"conditioning_ms\": " << t.conditioning_ms << ",\n";
    out << "  \"sampling_total_ms\": " << t.sampling_total_ms << ",\n";
    out << "  \"sampling_steps\": [";
    for (size_t i = 0; i < t.sampling_step_ms.size(); ++i) {
        if (i > 0) out << ", ";
        out << t.sampling_step_ms[i];
    }
    out << "],\n";
    out << "  \"vae_decode_ms\": " << t.vae_decode_ms << ",\n";
    out << "  \"total_ms\": " << t.total_ms << "\n";
    out << "}\n";
}

static std::vector<float> load_npy_float(const std::string& path)
{
    auto arr = cnpy::npy_load(path);
    size_t total = 1;
    for (auto d : arr.shape) total *= d;
    const float* data = arr.data<float>();
    return std::vector<float>(data, data + total);
}

static std::vector<int64_t> load_npy_int64_from_float(const std::string& path)
{
    auto arr = cnpy::npy_load(path);
    size_t total = 1;
    for (auto d : arr.shape) total *= d;
    const float* data = arr.data<float>();
    std::vector<int64_t> result(total);
    for (size_t i = 0; i < total; ++i) {
        result[i] = static_cast<int64_t>(data[i]);
    }
    return result;
}

static void print_usage(const char* prog)
{
    std::cerr << "Usage: " << prog << " [options]\n"
              << "  --manifest FILE          Path to zenon_pipeline_manifest.json (REQUIRED)\n"
              << "  --prompt TEXT            Text prompt (default: percussion)\n"
              << "  --duration SECONDS       Audio duration in seconds (default: 11)\n"
              << "  --keep-ratio FLOAT       Inpaint keep ratio 0.0-1.0 (default: 0.5)\n"
              << "  --output FILE            Output WAV file (default: output_zenon.wav)\n"
              << "  --timing-json FILE       Write timing report as JSON\n"
              << "  --seed N                 Random seed (default: 42)\n"
              << "  --steps N                Sampling steps (default: 50)\n"
              << "  --cfg SCALE              CFG scale (default: 7.0)\n"
              << "  --noise-npy FILE         Load starting noise from .npy\n"
              << "  --token-ids-npy FILE     Load token IDs from .npy\n"
              << "  --attention-mask-npy FILE Load attention mask from .npy\n"
              << "  --streamgen-latent-npy FILE  Load pre-encoded streamgen latent from .npy\n"
              << "  --input-latent-npy FILE  Load pre-encoded input latent from .npy\n"
              << "  --streamgen-audio-npy FILE   Load streamgen audio from .npy\n"
              << "  --input-audio-npy FILE   Load input audio from .npy\n"
              << "  --dump-steps-dir DIR     Dump per-step latents as .npy\n"
              << "  --cuda                   Use CUDA execution provider\n"
              << "  --coreml                 Use CoreML execution provider (macOS)\n"
              << "  --migraphx               Use MIGraphX execution provider (Linux/ROCm)\n"
              << "  --mlx-vae                Zenon VAE via MLX Metal (requires SAO_ENABLE_MLX build)\n"
              << "  --warmup-vae             After load, run one full VAE encode+decode warmup (offsets first-run JIT)\n";
}

int main(int argc, char* argv[])
{
    std::string manifest_path;
    std::string prompt = "percussion";
    float duration = 11.0f;
    float keep_ratio = 0.5f;
    std::string output_file = "output_zenon.wav";
    std::string timing_json_file;
    uint32_t seed = 42;
    int steps = 50;
    float cfg_scale = 7.0f;
    std::string noise_npy_file;
    std::string token_ids_npy_file;
    std::string attention_mask_npy_file;
    std::string streamgen_latent_npy_file;
    std::string input_latent_npy_file;
    std::string streamgen_audio_npy_file;
    std::string input_audio_npy_file;
    std::string dump_steps_dir;
    bool use_cuda = false;
    bool use_coreml = false;
    bool use_migraphx = false;
    bool use_mlx_vae = false;
    bool warmup_vae = false;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--manifest" && i + 1 < argc) {
            manifest_path = argv[++i];
        } else if (arg == "--prompt" && i + 1 < argc) {
            prompt = argv[++i];
        } else if (arg == "--duration" && i + 1 < argc) {
            duration = std::stof(argv[++i]);
        } else if (arg == "--keep-ratio" && i + 1 < argc) {
            keep_ratio = std::stof(argv[++i]);
        } else if (arg == "--output" && i + 1 < argc) {
            output_file = argv[++i];
        } else if (arg == "--timing-json" && i + 1 < argc) {
            timing_json_file = argv[++i];
        } else if (arg == "--seed" && i + 1 < argc) {
            seed = static_cast<uint32_t>(std::stoul(argv[++i]));
        } else if (arg == "--steps" && i + 1 < argc) {
            steps = std::stoi(argv[++i]);
        } else if (arg == "--cfg" && i + 1 < argc) {
            cfg_scale = std::stof(argv[++i]);
        } else if (arg == "--noise-npy" && i + 1 < argc) {
            noise_npy_file = argv[++i];
        } else if (arg == "--token-ids-npy" && i + 1 < argc) {
            token_ids_npy_file = argv[++i];
        } else if (arg == "--attention-mask-npy" && i + 1 < argc) {
            attention_mask_npy_file = argv[++i];
        } else if (arg == "--streamgen-latent-npy" && i + 1 < argc) {
            streamgen_latent_npy_file = argv[++i];
        } else if (arg == "--input-latent-npy" && i + 1 < argc) {
            input_latent_npy_file = argv[++i];
        } else if (arg == "--streamgen-audio-npy" && i + 1 < argc) {
            streamgen_audio_npy_file = argv[++i];
        } else if (arg == "--input-audio-npy" && i + 1 < argc) {
            input_audio_npy_file = argv[++i];
        } else if (arg == "--dump-steps-dir" && i + 1 < argc) {
            dump_steps_dir = argv[++i];
        } else if (arg == "--cuda") {
            use_cuda = true;
        } else if (arg == "--coreml") {
            use_coreml = true;
        } else if (arg == "--migraphx") {
            use_migraphx = true;
        } else if (arg == "--mlx-vae") {
            use_mlx_vae = true;
        } else if (arg == "--warmup-vae") {
            warmup_vae = true;
        } else if (arg == "--help" || arg == "-h") {
            print_usage(argv[0]);
            return 0;
        } else {
            std::cerr << "Unknown argument: " << arg << std::endl;
            print_usage(argv[0]);
            return 1;
        }
    }

    assert(!manifest_path.empty() && "Must provide --manifest path");

    std::cout << "=== Zenon Inference CLI ===" << std::endl;
    std::cout << "Manifest: " << manifest_path << std::endl;
    std::cout << "Prompt: " << prompt << std::endl;
    std::cout << "Duration: " << duration << "s, Keep ratio: " << keep_ratio << std::endl;
    std::cout << "Steps: " << steps << ", CFG: " << cfg_scale << std::endl;

    auto config = sao::ZenonPipelineConfig::load(manifest_path);
    config.use_cuda = use_cuda;
    config.use_coreml = use_coreml;
    config.use_migraphx = use_migraphx;
    config.use_mlx_vae = use_mlx_vae;

    sao::ZenonPipeline pipeline(config);
    if (warmup_vae)
        pipeline.warmup_vae();

    // --- Tokenization ---
    std::vector<int64_t> input_ids;
    std::vector<int64_t> attention_mask;

    if (!token_ids_npy_file.empty()) {
        std::cout << "Loading token IDs from " << token_ids_npy_file << std::endl;
        input_ids = load_npy_int64_from_float(token_ids_npy_file);
        assert(!attention_mask_npy_file.empty());
        auto mask_f = load_npy_float(attention_mask_npy_file);
        attention_mask.resize(mask_f.size());
        for (size_t i = 0; i < mask_f.size(); ++i)
            attention_mask[i] = static_cast<int64_t>(mask_f[i]);
    } else {
        std::string tokenizer_path = config.t5_onnx_path;
        auto last_slash = tokenizer_path.rfind('/');
        std::string tokenizer_dir = (last_slash != std::string::npos) ? tokenizer_path.substr(0, last_slash) : ".";
        std::string sp_path = tokenizer_dir + "/t5_tokenizer.model";
        std::cout << "Tokenizing with SentencePiece: " << sp_path << std::endl;
        sao::Tokenizer tokenizer(sp_path, 64);
        auto tokenized = tokenizer.tokenize(prompt);
        input_ids = tokenized.input_ids;
        attention_mask = tokenized.attention_mask;
    }

    // --- Audio inputs ---
    std::vector<float> streamgen_latent;
    std::vector<float> input_latent;
    std::vector<float> streamgen_audio;
    std::vector<float> input_audio;

    if (!streamgen_latent_npy_file.empty()) {
        std::cout << "Loading streamgen latent from " << streamgen_latent_npy_file << std::endl;
        streamgen_latent = load_npy_float(streamgen_latent_npy_file);
    } else if (!streamgen_audio_npy_file.empty()) {
        std::cout << "Loading streamgen audio from " << streamgen_audio_npy_file << std::endl;
        streamgen_audio = load_npy_float(streamgen_audio_npy_file);
    } else {
        std::cerr << "Must provide --streamgen-latent-npy or --streamgen-audio-npy" << std::endl;
        return 1;
    }

    if (!input_latent_npy_file.empty()) {
        std::cout << "Loading input latent from " << input_latent_npy_file << std::endl;
        input_latent = load_npy_float(input_latent_npy_file);
    } else if (!input_audio_npy_file.empty()) {
        std::cout << "Loading input audio from " << input_audio_npy_file << std::endl;
        input_audio = load_npy_float(input_audio_npy_file);
    } else {
        std::cerr << "Must provide --input-latent-npy or --input-audio-npy" << std::endl;
        return 1;
    }

    // --- Noise ---
    std::vector<float> external_noise;
    if (!noise_npy_file.empty()) {
        std::cout << "Loading noise from " << noise_npy_file << std::endl;
        external_noise = load_npy_float(noise_npy_file);
    }

    // --- Generate ---
    auto audio_out = pipeline.generate(
        input_ids, attention_mask,
        duration,
        streamgen_latent, input_latent,
        streamgen_audio, input_audio,
        keep_ratio, seed, steps, cfg_scale,
        [](int step, float t, const std::vector<float>& x) {
            float max_val = 0.0f;
            for (float v : x) {
                float a = std::abs(v);
                if (a > max_val) max_val = a;
            }
            std::cout << "  Step " << step << ": t=" << t
                      << " |x|_max=" << max_val << std::endl;
        },
        external_noise,
        dump_steps_dir
    );

    // --- Write output ---
    constexpr int NUM_CHANNELS = 2;
    int num_samples = static_cast<int>(audio_out.size()) / NUM_CHANNELS;
    std::cout << "Writing " << num_samples << " samples to " << output_file << std::endl;
    write_wav(output_file, audio_out, config.sample_rate, NUM_CHANNELS, num_samples);

    if (!timing_json_file.empty()) {
        write_timing_json(timing_json_file, pipeline.timing());
        std::cout << "Timing report written to " << timing_json_file << std::endl;
    }

    std::cout << "Done!" << std::endl;
    return 0;
}
