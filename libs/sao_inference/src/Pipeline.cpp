#include "sao_inference/Pipeline.h"
#include <cnpy.h>
#include <chrono>
#include <random>
#include <cassert>
#include <iostream>
#include <iomanip>

namespace sao {

using Clock = std::chrono::high_resolution_clock;

static double elapsed_ms(Clock::time_point start)
{
    auto now = Clock::now();
    return std::chrono::duration<double, std::milli>(now - start).count();
}

static void save_npy_float(const std::string& path, const std::vector<float>& data,
                           const std::vector<size_t>& shape)
{
    cnpy::npy_save(path, data.data(), shape, "w");
}

Pipeline::Pipeline(const PipelineConfig& config)
    : m_config(config)
{
    auto t0 = Clock::now();
    std::cout << "[sao::Pipeline] Loading models..." << std::endl;
    m_t5 = std::make_unique<T5Encoder>(config.t5_onnx_path, config.use_cuda, config.use_coreml, config.use_migraphx);
    m_dit = std::make_unique<DiTModel>(config.dit_onnx_path, config.use_cuda, config.use_coreml, config.use_migraphx);
    // CoreML cannot handle the VAE (input dim > CoreML's 16384 limit)
    m_vae = std::make_unique<VAEDecoder>(config.vae_onnx_path, config.vae_scale, config.use_cuda, false, config.use_migraphx);
    m_number_embedder = std::make_unique<NumberEmbedder>(config.number_embedder_weights_dir);
    m_timing.model_load_ms = elapsed_ms(t0);
    std::cout << "[sao::Pipeline] All models loaded in " << m_timing.model_load_ms << " ms" << std::endl;
}

std::vector<float> Pipeline::generate(
    const std::vector<int64_t>& input_ids,
    const std::vector<int64_t>& attention_mask,
    float seconds_total,
    uint32_t seed,
    int steps,
    float cfg_scale,
    StepCallback callback,
    const std::vector<float>& external_noise,
    const std::string& dump_steps_dir)
{
    constexpr int EMBED_DIM = 768;
    constexpr int T5_SEQ_LEN = 64;

    auto total_start = Clock::now();

    // Step 1: T5 encode
    auto t0 = Clock::now();
    auto t5_embeddings = m_t5->encode(input_ids, attention_mask);

    for (int i = 0; i < T5_SEQ_LEN; ++i) {
        float mask_val = static_cast<float>(attention_mask[i]);
        for (int j = 0; j < EMBED_DIM; ++j) {
            t5_embeddings[i * EMBED_DIM + j] *= mask_val;
        }
    }
    m_timing.t5_encode_ms = elapsed_ms(t0);

    // Step 2-3: Number embedding + conditioning assembly
    t0 = Clock::now();
    auto seconds_embed = m_number_embedder->embed(seconds_total);
    auto conditioning = assemble_conditioning(
        t5_embeddings, T5_SEQ_LEN,
        seconds_embed, EMBED_DIM
    );
    m_timing.conditioning_ms = elapsed_ms(t0);

    // Step 4: Noise — use external if provided, otherwise generate with RNG
    int latent_size = m_config.latent_length * 64;
    std::vector<float> noise;
    if (!external_noise.empty()) {
        assert(static_cast<int>(external_noise.size()) == latent_size);
        noise = external_noise;
        std::cout << "[sao::Pipeline] Using external noise (size=" << noise.size() << ")" << std::endl;
    } else {
        noise.resize(latent_size);
        std::mt19937 rng(seed);
        std::normal_distribution<float> dist(0.0f, 1.0f);
        for (auto& n : noise) {
            n = dist(rng);
        }
        std::cout << "[sao::Pipeline] Generated noise with seed=" << seed << std::endl;
    }

    // Step 5: Sample with per-step timing
    m_timing.sampling_step_ms.clear();
    m_timing.sampling_step_ms.reserve(steps);
    t0 = Clock::now();

    SamplerConfig sampler_config;
    sampler_config.steps = steps;
    sampler_config.cfg_scale = cfg_scale;
    sampler_config.latent_channels = 64;
    sampler_config.latent_length = m_config.latent_length;

    auto sampled = sample_euler_cfg(*m_dit, noise, conditioning, sampler_config,
        [&](int step, float t, const std::vector<float>& x) {
            m_timing.sampling_step_ms.push_back(elapsed_ms(t0));
            t0 = Clock::now();

            if (!dump_steps_dir.empty()) {
                std::ostringstream fname;
                fname << dump_steps_dir << "/step_" << std::setfill('0') << std::setw(2) << step << "_x_after.npy";
                save_npy_float(fname.str(), x, {1, 64, static_cast<size_t>(m_config.latent_length)});
            }

            if (callback) callback(step, t, x);
        }
    );
    m_timing.sampling_total_ms = 0.0;
    for (double s : m_timing.sampling_step_ms) m_timing.sampling_total_ms += s;

    // Step 6: VAE decode
    t0 = Clock::now();
    auto audio = m_vae->decode(sampled, m_config.latent_length);
    m_timing.vae_decode_ms = elapsed_ms(t0);

    m_timing.total_ms = elapsed_ms(total_start);

    std::cout << "[sao::Pipeline] Timing breakdown:" << std::endl;
    std::cout << "  T5 encode:    " << m_timing.t5_encode_ms << " ms" << std::endl;
    std::cout << "  Conditioning: " << m_timing.conditioning_ms << " ms" << std::endl;
    std::cout << "  Sampling:     " << m_timing.sampling_total_ms << " ms (" << steps << " steps)" << std::endl;
    for (int i = 0; i < static_cast<int>(m_timing.sampling_step_ms.size()); ++i) {
        std::cout << "    Step " << i << ": " << m_timing.sampling_step_ms[i] << " ms" << std::endl;
    }
    std::cout << "  VAE decode:   " << m_timing.vae_decode_ms << " ms" << std::endl;
    std::cout << "  Total:        " << m_timing.total_ms << " ms" << std::endl;

    return audio;
}

} // namespace sao
