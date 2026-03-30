#include "sao_inference/Pipeline.h"
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

static void write_timing_json(const std::string& path, const sao::TimingReport& t)
{
    std::ofstream out(path);
    out << "{\n";
    out << "  \"model_load_ms\": " << t.model_load_ms << ",\n";
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
              << "  --prompt TEXT            Text prompt for audio generation\n"
              << "  --duration SECONDS       Audio duration in seconds (default: 11)\n"
              << "  --output FILE            Output WAV file (default: output.wav)\n"
              << "  --timing-json FILE       Write timing report as JSON\n"
              << "  --seed N                 Random seed (default: 42)\n"
              << "  --steps N                Sampling steps (default: 8)\n"
              << "  --cfg SCALE              CFG scale (default: 7.0)\n"
              << "  --models-dir DIR         Directory containing ONNX models (default: models/onnx)\n"
              << "  --weights-dir DIR        Directory containing weights (default: models/weights)\n"
              << "  --noise-npy FILE         Load starting noise from .npy (overrides --seed)\n"
              << "  --token-ids-npy FILE     Load token IDs from .npy (overrides --prompt)\n"
              << "  --attention-mask-npy FILE Load attention mask from .npy\n"
              << "  --dump-steps-dir DIR     Dump per-step latents as .npy to this directory\n"
              << "  --cuda                   Use CUDA execution provider\n"
              << "  --coreml                 Use CoreML execution provider (macOS)\n";
}

int main(int argc, char* argv[])
{
    std::string prompt = "128 BPM tech house drum loop";
    float duration = 11.0f;
    std::string output_file = "output.wav";
    std::string timing_json_file;
    uint32_t seed = 42;
    int steps = 8;
    float cfg_scale = 7.0f;
    std::string models_dir = "models/onnx";
    std::string weights_dir = "models/weights";
    std::string noise_npy_file;
    std::string token_ids_npy_file;
    std::string attention_mask_npy_file;
    std::string dump_steps_dir;
    bool use_cuda = false;
    bool use_coreml = false;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--prompt" && i + 1 < argc) {
            prompt = argv[++i];
        } else if (arg == "--duration" && i + 1 < argc) {
            duration = std::stof(argv[++i]);
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
        } else if (arg == "--models-dir" && i + 1 < argc) {
            models_dir = argv[++i];
        } else if (arg == "--weights-dir" && i + 1 < argc) {
            weights_dir = argv[++i];
        } else if (arg == "--noise-npy" && i + 1 < argc) {
            noise_npy_file = argv[++i];
        } else if (arg == "--token-ids-npy" && i + 1 < argc) {
            token_ids_npy_file = argv[++i];
        } else if (arg == "--attention-mask-npy" && i + 1 < argc) {
            attention_mask_npy_file = argv[++i];
        } else if (arg == "--dump-steps-dir" && i + 1 < argc) {
            dump_steps_dir = argv[++i];
        } else if (arg == "--cuda") {
            use_cuda = true;
        } else if (arg == "--coreml") {
            use_coreml = true;
        } else if (arg == "--help" || arg == "-h") {
            print_usage(argv[0]);
            return 0;
        } else {
            std::cerr << "Unknown argument: " << arg << std::endl;
            print_usage(argv[0]);
            return 1;
        }
    }

    std::cout << "=== SAO Inference CLI ===" << std::endl;
    std::cout << "Prompt: " << prompt << std::endl;
    std::cout << "Duration: " << duration << "s" << std::endl;
    std::cout << "Steps: " << steps << ", CFG: " << cfg_scale << std::endl;

    sao::PipelineConfig config;
    config.t5_onnx_path = models_dir + "/t5_encoder.onnx";
    config.dit_onnx_path = models_dir + "/dit_step.onnx";
    config.vae_onnx_path = models_dir + "/vae_decoder.onnx";
    config.number_embedder_weights_dir = weights_dir + "/number_embedder";
    config.vae_scale = 1.0f;
    config.use_cuda = use_cuda;
    config.use_coreml = use_coreml;
    config.sample_rate = 44100;
    config.latent_length = 256;

    sao::Pipeline pipeline(config);

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
        std::string tokenizer_path = models_dir + "/t5_tokenizer.model";
        std::cout << "Tokenizing with SentencePiece: " << tokenizer_path << std::endl;
        sao::Tokenizer tokenizer(tokenizer_path, 64);
        auto tokenized = tokenizer.tokenize(prompt);
        input_ids = tokenized.input_ids;
        attention_mask = tokenized.attention_mask;
    }

    std::cout << "Token IDs (" << input_ids.size() << "): ";
    for (size_t i = 0; i < input_ids.size() && input_ids[i] != 0; ++i) {
        if (i > 0) std::cout << ", ";
        std::cout << input_ids[i];
    }
    std::cout << std::endl;

    // --- Noise ---
    std::vector<float> external_noise;
    if (!noise_npy_file.empty()) {
        std::cout << "Loading noise from " << noise_npy_file << std::endl;
        external_noise = load_npy_float(noise_npy_file);
    }

    // --- Generate ---
    auto audio = pipeline.generate(
        input_ids, attention_mask,
        duration, seed, steps, cfg_scale,
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
    int num_samples = static_cast<int>(audio.size()) / NUM_CHANNELS;
    std::cout << "Writing " << num_samples << " samples to " << output_file << std::endl;
    write_wav(output_file, audio, config.sample_rate, NUM_CHANNELS, num_samples);

    if (!timing_json_file.empty()) {
        write_timing_json(timing_json_file, pipeline.timing());
        std::cout << "Timing report written to " << timing_json_file << std::endl;
    }

    std::cout << "Done!" << std::endl;
    return 0;
}
