#include "sao_inference/InpaintSampler.h"
#include <cmath>
#include <cassert>
#include <iostream>

namespace sao {

void apply_distribution_shift(
    std::vector<float>& schedule,
    const DistributionShiftConfig& config,
    int seq_len)
{
    assert(config.enabled);

    float length_ratio = static_cast<float>(seq_len - config.min_length)
                       / static_cast<float>(config.max_length - config.min_length);
    float mu = -(config.base_shift + (config.max_shift - config.base_shift) * length_ratio);
    float sigma = 1.0f;

    float exp_mu = std::exp(mu);

    for (size_t i = 0; i < schedule.size(); ++i) {
        float t = schedule[i];
        if (t <= 0.0f || t >= 1.0f) continue;

        float inv_term = std::pow(1.0f / (1.0f - t) - 1.0f, sigma);
        schedule[i] = 1.0f - exp_mu / (exp_mu + inv_term);
    }
}

std::vector<float> sample_euler_cfg_inpaint(
    DiTInpaintModel& dit,
    const std::vector<float>& noise,
    const InpaintConditioning& conditioning,
    const SamplerConfig& config,
    const ZenonPipelineConfig& pipeline_config,
    StepCallback callback)
{
    int C = config.latent_channels;
    int T = config.latent_length;
    int latent_size = C * T;
    int cond_dim = pipeline_config.cond_token_dim;
    int seq_len = conditioning.cross_attn_seq_len;

    assert(static_cast<int>(noise.size()) == latent_size);

    auto t_schedule = build_time_schedule(config.steps, config.sigma_max);

    if (pipeline_config.dist_shift.enabled) {
        apply_distribution_shift(t_schedule, pipeline_config.dist_shift, T);
    }

    std::vector<float> x = noise;

    for (int i = 0; i < config.steps; ++i) {
        float t_curr = t_schedule[i];
        float t_prev = t_schedule[i + 1];
        float dt = t_prev - t_curr;

        std::vector<float> batch_x(2 * latent_size);
        std::copy(x.begin(), x.end(), batch_x.begin());
        std::copy(x.begin(), x.end(), batch_x.begin() + latent_size);

        std::vector<float> batch_t = {t_curr, t_curr};

        std::vector<float> batch_cross_attn(2 * seq_len * cond_dim, 0.0f);
        std::copy(conditioning.cross_attn_cond.begin(),
                  conditioning.cross_attn_cond.end(),
                  batch_cross_attn.begin());

        std::vector<float> batch_global(2 * cond_dim);
        std::copy(conditioning.global_embed.begin(),
                  conditioning.global_embed.end(),
                  batch_global.begin());
        std::copy(conditioning.global_embed.begin(),
                  conditioning.global_embed.end(),
                  batch_global.begin() + cond_dim);

        std::vector<InputAddTensor> batch_input_add;
        for (const auto& add : conditioning.input_add) {
            InputAddTensor batch_add;
            batch_add.name = add.name;
            batch_add.channels = add.channels;
            int add_size = add.channels * T;
            batch_add.data.resize(2 * add_size);
            std::copy(add.data.begin(), add.data.end(), batch_add.data.begin());
            std::copy(add.data.begin(), add.data.end(), batch_add.data.begin() + add_size);
            batch_input_add.push_back(std::move(batch_add));
        }

        auto batch_output = dit.forward(
            batch_x, batch_t, batch_cross_attn, batch_global,
            batch_input_add, 2, seq_len, C, T, cond_dim
        );

        std::vector<float> v(latent_size);
        for (int j = 0; j < latent_size; ++j) {
            float cond_out = batch_output[j];
            float uncond_out = batch_output[latent_size + j];
            v[j] = uncond_out + config.cfg_scale * (cond_out - uncond_out);
        }

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
