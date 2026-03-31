#include "sao_inference/RectifiedFlowSampler.h"
#include <cassert>
#include <cmath>
#include <cstring>
#include <iostream>

namespace sao {

std::vector<float> build_time_schedule(int steps, float sigma_max)
{
    float logsnr_max;
    if (sigma_max < 1.0f) {
        logsnr_max = std::log(((1.0f - sigma_max) / sigma_max) + 1e-6f);
    } else {
        logsnr_max = -6.0f;
    }

    std::vector<float> t(steps + 1);
    for (int i = 0; i <= steps; ++i) {
        float logsnr = logsnr_max + (2.0f - logsnr_max) * static_cast<float>(i) / static_cast<float>(steps);
        t[i] = 1.0f / (1.0f + std::exp(logsnr)); // sigmoid(-logsnr)
    }
    t[0] = sigma_max;
    t[steps] = 0.0f;

    return t;
}

std::vector<float> sample_euler_cfg(
    DiTModel& dit,
    const std::vector<float>& noise,
    const Conditioning& conditioning,
    const SamplerConfig& config,
    StepCallback callback)
{
    int C = config.latent_channels;
    int T = config.latent_length;
    int latent_size = C * T;
    int embed_dim = 768;
    int seq_len = conditioning.cross_attn_seq_len;

    assert(static_cast<int>(noise.size()) == latent_size);

    auto t_schedule = build_time_schedule(config.steps, config.sigma_max);

    std::vector<float> x = noise;

    std::vector<float> batch_x(static_cast<size_t>(2 * latent_size));
    std::vector<float> batch_t(2);
    std::vector<float> batch_cross_attn(static_cast<size_t>(2 * seq_len * embed_dim), 0.0f);
    const size_t cross_attn_elems = static_cast<size_t>(seq_len * embed_dim);
    std::memcpy(
        batch_cross_attn.data(),
        conditioning.cross_attn_cond.data(),
        cross_attn_elems * sizeof(float));

    std::vector<float> batch_global(static_cast<size_t>(2 * embed_dim));
    const size_t global_bytes = static_cast<size_t>(embed_dim) * sizeof(float);
    std::memcpy(batch_global.data(), conditioning.global_embed.data(), global_bytes);
    std::memcpy(batch_global.data() + embed_dim, conditioning.global_embed.data(), global_bytes);

    std::vector<float> v(static_cast<size_t>(latent_size));

    for (int i = 0; i < config.steps; ++i) {
        float t_curr = t_schedule[i];
        float t_prev = t_schedule[i + 1];
        float dt = t_prev - t_curr;

        const size_t latent_bytes = static_cast<size_t>(latent_size) * sizeof(float);
        std::memcpy(batch_x.data(), x.data(), latent_bytes);
        std::memcpy(batch_x.data() + latent_size, x.data(), latent_bytes);

        batch_t[0] = t_curr;
        batch_t[1] = t_curr;

        // Run DiT with batch=2 (second half of batch_cross_attn stays zero = null conditioning)
        auto batch_output = dit.forward(
            batch_x, batch_t, batch_cross_attn, batch_global,
            2, seq_len
        );

        // Split output: first half is conditioned, second half is unconditioned
        // Apply CFG: v = uncond + cfg_scale * (cond - uncond)
        for (int j = 0; j < latent_size; ++j) {
            float cond_out = batch_output[j];
            float uncond_out = batch_output[latent_size + j];
            v[j] = uncond_out + config.cfg_scale * (cond_out - uncond_out);
        }

        // Euler step: x = x + dt * v
        for (int j = 0; j < latent_size; ++j) {
            x[j] += dt * v[j];
        }

        if (callback) {
            callback(i, t_curr, x);
        }
    }

    return x;
}

} // namespace sao
