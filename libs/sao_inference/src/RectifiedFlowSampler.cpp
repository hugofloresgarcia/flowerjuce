#include "sao_inference/RectifiedFlowSampler.h"
#include <cmath>
#include <cassert>
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

    for (int i = 0; i < config.steps; ++i) {
        float t_curr = t_schedule[i];
        float t_prev = t_schedule[i + 1];
        float dt = t_prev - t_curr;

        // --- CFG batch doubling ---
        // Stack x twice for batch=2 (cond, uncond)
        std::vector<float> batch_x(2 * latent_size);
        std::copy(x.begin(), x.end(), batch_x.begin());
        std::copy(x.begin(), x.end(), batch_x.begin() + latent_size);

        // Stack timestep
        std::vector<float> batch_t = {t_curr, t_curr};

        // Stack cross_attn: cond + null (zeros)
        std::vector<float> batch_cross_attn(2 * seq_len * embed_dim, 0.0f);
        std::copy(conditioning.cross_attn_cond.begin(),
                  conditioning.cross_attn_cond.end(),
                  batch_cross_attn.begin());
        // Second half is already zeros (null conditioning)

        // Stack global_embed: same for both (not nulled for global)
        std::vector<float> batch_global(2 * embed_dim);
        std::copy(conditioning.global_embed.begin(),
                  conditioning.global_embed.end(),
                  batch_global.begin());
        std::copy(conditioning.global_embed.begin(),
                  conditioning.global_embed.end(),
                  batch_global.begin() + embed_dim);

        // Run DiT with batch=2
        auto batch_output = dit.forward(
            batch_x, batch_t, batch_cross_attn, batch_global,
            2, seq_len
        );

        // Split output: first half is conditioned, second half is unconditioned
        // Apply CFG: v = uncond + cfg_scale * (cond - uncond)
        std::vector<float> v(latent_size);
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
