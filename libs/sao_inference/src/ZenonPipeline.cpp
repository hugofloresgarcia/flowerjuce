#include "sao_inference/ZenonPipeline.h"
#if defined(SAO_ENABLE_MLX)
#include "sao_inference/MlxZenonVae.h"
#endif
#include "sao_inference/OnnxVaeDecoder.h"
#include "sao_inference/OnnxVaeEncoder.h"
#include <cnpy.h>
#include <chrono>
#include <cmath>
#include <random>
#include <cassert>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <stdexcept>

namespace sao {

using Clock = std::chrono::high_resolution_clock;

static double elapsed_ms(Clock::time_point start)
{
    return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}

/// Populates `out` with mean and population standard deviation of `v`.
static void compute_mean_std(const std::vector<float>& v, ZenonLatentStats& out)
{
    if (v.empty())
    {
        out.mean = 0.0;
        out.std_dev = 0.0;
        return;
    }
    double sum = 0.0;
    for (float x : v)
        sum += static_cast<double>(x);
    out.mean = sum / static_cast<double>(v.size());
    double var = 0.0;
    for (float x : v)
    {
        double d = static_cast<double>(x) - out.mean;
        var += d * d;
    }
    out.std_dev = std::sqrt(var / static_cast<double>(v.size()));
}

static void save_npy_float(const std::string& path, const std::vector<float>& data,
                           const std::vector<size_t>& shape)
{
    cnpy::npy_save(path, data.data(), shape, "w");
}

ZenonPipeline::ZenonPipeline(const ZenonPipelineConfig& config)
    : m_config(config)
{
    auto t0 = Clock::now();
    if (m_verbose)
        std::cout << "[ZenonPipeline] Loading models..." << std::endl;

    m_t5 = std::make_unique<T5Encoder>(config.t5_onnx_path, config.use_cuda, config.use_coreml);
    m_dit = std::make_unique<DiTInpaintModel>(config.dit_onnx_path, config.use_cuda, config.use_coreml);

    if (config.use_mlx_vae)
    {
#if !defined(SAO_ENABLE_MLX)
        throw std::runtime_error(
            "ZenonPipeline: use_mlx_vae is set but sao_inference was built without SAO_ENABLE_MLX=ON");
#else
        auto mlx_bundle =
            load_mlx_vae_bundle(config.mlx_vae_weights_path, config.mlx_vae_config_path);
        m_vae_encoder = std::make_unique<MlxVaeEncoder>(mlx_bundle);
        m_vae_decoder = std::make_unique<MlxVaeDecoder>(mlx_bundle, config.vae_scale);
#endif
    }
    else
    {
        // CoreML cannot handle the VAE (input dim 524288 > CoreML's 16384 limit)
        m_vae_encoder = std::make_unique<OnnxVaeEncoder>(
            config.vae_encoder_onnx_path, config.use_cuda, false);
        m_vae_decoder = std::make_unique<OnnxVaeDecoder>(
            config.vae_decoder_onnx_path, config.vae_scale, config.use_cuda, false);
    }
    m_number_embedder = std::make_unique<NumberEmbedder>(config.number_embedder_weights_dir);

    m_timing.model_load_ms = elapsed_ms(t0);
    if (m_verbose)
        std::cout << "[ZenonPipeline] All models loaded in " << m_timing.model_load_ms << " ms" << std::endl;
}

std::vector<float> ZenonPipeline::generate(
    const std::vector<int64_t>& input_ids,
    const std::vector<int64_t>& attention_mask,
    float seconds_total,
    const std::vector<float>& streamgen_latent_in,
    const std::vector<float>& input_latent_in,
    const std::vector<float>& streamgen_audio,
    const std::vector<float>& input_audio,
    float keep_ratio,
    uint32_t seed,
    int steps,
    float cfg_scale,
    StepCallback callback,
    const std::vector<float>& external_noise,
    const std::string& dump_steps_dir,
    const std::vector<float>* precomputed_t5_masked,
    PhaseProgressCallback phase_callback)
{
    int C = m_config.latent_dim;
    int T = m_config.latent_length;
    int latent_size = C * T;
    int cond_dim = m_config.cond_token_dim;
    constexpr int T5_SEQ_LEN = 64;

    m_diagnostics = ZenonDiagnostics{};

    if (phase_callback)
        phase_callback(ZenonInferencePhase::Encode, 0);

    auto total_start = Clock::now();

    // ---- Step 1-2: VAE encode (or use pre-encoded latents) ----
    auto t0 = Clock::now();

    std::vector<float> streamgen_latent;
    std::vector<float> input_latent;

    if (!streamgen_latent_in.empty()) {
        streamgen_latent = streamgen_latent_in;
        if (m_verbose)
            std::cout << "[ZenonPipeline] Using pre-encoded streamgen latent" << std::endl;
    } else {
        assert(!streamgen_audio.empty());
        streamgen_latent = m_vae_encoder->encode(streamgen_audio, m_config.sample_size, C);
        if (m_verbose)
            std::cout << "[ZenonPipeline] Encoded streamgen audio -> latent" << std::endl;
    }

    if (!input_latent_in.empty()) {
        input_latent = input_latent_in;
        if (m_verbose)
            std::cout << "[ZenonPipeline] Using pre-encoded input latent" << std::endl;
    } else {
        assert(!input_audio.empty());
        input_latent = m_vae_encoder->encode(input_audio, m_config.sample_size, C);
        if (m_verbose)
            std::cout << "[ZenonPipeline] Encoded input audio -> latent" << std::endl;
    }

    assert(static_cast<int>(streamgen_latent.size()) == latent_size);
    assert(static_cast<int>(input_latent.size()) == latent_size);

    compute_mean_std(streamgen_latent, m_diagnostics.streamgen_latent);
    compute_mean_std(input_latent, m_diagnostics.input_latent);

    m_timing.vae_encode_ms = elapsed_ms(t0);

    // ---- Step 3: Build inpaint mask ----
    int keep_frames = static_cast<int>(T * keep_ratio);
    std::vector<float> inpaint_mask(T, 0.0f);
    for (int i = 0; i < keep_frames; ++i) {
        inpaint_mask[i] = 1.0f;
    }

    // ---- Step 4: Build inpaint_masked_input ----
    std::vector<float> inpaint_masked_input(latent_size);
    for (int c = 0; c < C; ++c) {
        for (int t_idx = 0; t_idx < T; ++t_idx) {
            inpaint_masked_input[c * T + t_idx] = input_latent[c * T + t_idx] * inpaint_mask[t_idx];
        }
    }

    // ---- Step 5: T5 encode (or use caller-supplied masked embeddings) ----
    const std::vector<float>* t5_masked_ptr = nullptr;
    if (precomputed_t5_masked != nullptr) {
        const int expected_t5 = T5_SEQ_LEN * cond_dim;
        assert(static_cast<int>(precomputed_t5_masked->size()) == expected_t5);
        t5_masked_ptr = precomputed_t5_masked;
        m_timing.t5_encode_ms = 0.0;
    } else {
        t0 = Clock::now();
        m_last_masked_t5 = m_t5->encode(input_ids, attention_mask);
        for (int i = 0; i < T5_SEQ_LEN; ++i) {
            float mask_val = static_cast<float>(attention_mask[i]);
            for (int j = 0; j < cond_dim; ++j) {
                m_last_masked_t5[static_cast<size_t>(i * cond_dim + j)] *= mask_val;
            }
        }
        m_timing.t5_encode_ms = elapsed_ms(t0);
        t5_masked_ptr = &m_last_masked_t5;
    }

    // ---- Step 6-7: Number embedding + conditioning assembly ----
    t0 = Clock::now();
    auto seconds_embed = m_number_embedder->embed(seconds_total);

    auto conditioning = assemble_inpaint_conditioning(
        *t5_masked_ptr, T5_SEQ_LEN,
        seconds_embed, cond_dim,
        streamgen_latent, inpaint_mask, inpaint_masked_input,
        m_config.input_add_keys, C, T
    );
    m_timing.conditioning_ms = elapsed_ms(t0);

    // ---- Step 8: Noise ----
    std::vector<float> noise;
    if (!external_noise.empty()) {
        assert(static_cast<int>(external_noise.size()) == latent_size);
        noise = external_noise;
        if (m_verbose)
            std::cout << "[ZenonPipeline] Using external noise" << std::endl;
    } else {
        noise.resize(latent_size);
        std::mt19937 rng(seed);
        std::normal_distribution<float> dist(0.0f, 1.0f);
        for (auto& n : noise) {
            n = dist(rng);
        }
        if (m_verbose)
            std::cout << "[ZenonPipeline] Generated noise with seed=" << seed << std::endl;
    }

    compute_mean_std(noise, m_diagnostics.noise);

    // ---- Step 9: Sample ----
    m_timing.sampling_step_ms.clear();
    m_timing.sampling_step_ms.reserve(static_cast<size_t>(steps));
    m_diagnostics.diffusion_steps.clear();
    if (m_collect_step_diagnostics)
        m_diagnostics.diffusion_steps.reserve(static_cast<size_t>(steps));
    t0 = Clock::now();

    SamplerConfig sampler_config;
    sampler_config.steps = steps;
    sampler_config.cfg_scale = cfg_scale;
    sampler_config.latent_channels = C;
    sampler_config.latent_length = T;

    auto sampled = sample_euler_cfg_inpaint(*m_dit, noise, conditioning, sampler_config, m_config,
        [&](int step, float t, const std::vector<float>& x) {
            if (phase_callback)
                phase_callback(ZenonInferencePhase::DiT, step + 1);

            m_timing.sampling_step_ms.push_back(elapsed_ms(t0));
            t0 = Clock::now();

            if (m_collect_step_diagnostics)
            {
                ZenonDiffusionStepStats step_stats;
                step_stats.step_index = step;
                step_stats.t_curr = t;
                ZenonLatentStats x_stats;
                compute_mean_std(x, x_stats);
                step_stats.mean_x = x_stats.mean;
                step_stats.std_x = x_stats.std_dev;
                m_diagnostics.diffusion_steps.push_back(step_stats);
            }

            if (!dump_steps_dir.empty()) {
                std::ostringstream fname;
                fname << dump_steps_dir << "/step_" << std::setfill('0') << std::setw(2) << step << "_x_after.npy";
                save_npy_float(fname.str(), x, {1, static_cast<size_t>(C), static_cast<size_t>(T)});
            }

            if (callback) callback(step, t, x);
        }
    );
    m_timing.sampling_total_ms = 0.0;
    for (double s : m_timing.sampling_step_ms) m_timing.sampling_total_ms += s;

    // ---- Step 10: VAE decode ----
    if (phase_callback)
        phase_callback(ZenonInferencePhase::Decode, 0);

    t0 = Clock::now();
    auto audio = m_vae_decoder->decode(sampled, T);
    m_timing.vae_decode_ms = elapsed_ms(t0);

    m_timing.total_ms = elapsed_ms(total_start);

    compute_mean_std(sampled, m_diagnostics.sampled_latent);

    if (m_verbose)
    {
        std::cout << "[ZenonPipeline] Timing breakdown:" << std::endl;
        std::cout << "  VAE encode:   " << m_timing.vae_encode_ms << " ms" << std::endl;
        std::cout << "  T5 encode:    " << m_timing.t5_encode_ms << " ms" << std::endl;
        std::cout << "  Conditioning: " << m_timing.conditioning_ms << " ms" << std::endl;
        std::cout << "  Sampling:     " << m_timing.sampling_total_ms << " ms (" << steps << " steps)" << std::endl;
        for (int i = 0; i < static_cast<int>(m_timing.sampling_step_ms.size()); ++i)
            std::cout << "    Step " << i << ": " << m_timing.sampling_step_ms[i] << " ms" << std::endl;
        std::cout << "  VAE decode:   " << m_timing.vae_decode_ms << " ms" << std::endl;
        std::cout << "  Total:        " << m_timing.total_ms << " ms" << std::endl;
    }

    return audio;
}

} // namespace sao
