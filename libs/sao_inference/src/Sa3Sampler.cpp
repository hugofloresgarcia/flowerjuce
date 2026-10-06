#include "sao_inference/Sa3Sampler.h"

#include <cmath>
#include <cstring>
#include <stdexcept>

namespace sao {

std::vector<float> build_logsnr_schedule(int steps, float sigma_max, const LogSnrScheduleParams& p, int seq_len)
{
    if (steps < 1) throw std::invalid_argument("build_logsnr_schedule: steps must be >= 1");
    if (seq_len < 1) seq_len = 1; // build_schedule clamps the sequence length to >= 1
    const float logsnr_start = p.anchor_logsnr - p.rate * std::log2(static_cast<float>(seq_len) / p.anchor_length);
    std::vector<float> out(static_cast<size_t>(steps + 1));
    for (int i = 0; i <= steps; ++i) {
        // torch.linspace(sigma_max, 0, steps+1)[i] = sigma_max + i * (0 - sigma_max) / steps
        const float t = sigma_max + static_cast<float>(i) * (0.0f - sigma_max) / static_cast<float>(steps);
        float v;
        if (t <= 0.0f) {
            v = 0.0f;
        } else if (t >= 1.0f) {
            v = 1.0f;
        } else {
            const float logsnr = p.logsnr_end - t * (p.logsnr_end - logsnr_start);
            v = 1.0f / (1.0f + std::exp(logsnr)); // sigmoid(-logsnr)
        }
        out[static_cast<size_t>(i)] = v;
    }
    out[0] = sigma_max;
    return out;
}

void apply_single_axis_cfg(const std::vector<float>& x, const std::vector<float>& v_cond, const std::vector<float>& v_uncond,
                           float t, float cfg_scale, float apg_scale, std::vector<float>& v_out)
{
    const size_t n = x.size();
    if (v_cond.size() != n || v_uncond.size() != n) throw std::invalid_argument("apply_single_axis_cfg: size mismatch");
    if (!(t > 0.0f)) throw std::invalid_argument("apply_single_axis_cfg: t must be > 0 (the model is never evaluated at t = 0)");
    v_out.resize(n);

    // Pass 1: norms and the projection coefficient, accumulated in double.
    double norm_sq = 0.0;
    double dot = 0.0;
    for (size_t i = 0; i < n; ++i) {
        const float cond_den = x[i] - t * v_cond[i];
        const float uncond_den = x[i] - t * v_uncond[i];
        const float diff = cond_den - uncond_den;
        norm_sq += static_cast<double>(cond_den) * static_cast<double>(cond_den);
        dot += static_cast<double>(diff) * static_cast<double>(cond_den);
    }
    // F.normalize(v1): v1 / max(||v1||, eps); parallel = <diff, n> n = (dot / ||v1||^2) * cond_den
    const double norm = std::max(std::sqrt(norm_sq), 1e-12);
    const double coeff = dot / (norm * norm);
    const float g = cfg_scale - 1.0f;

    for (size_t i = 0; i < n; ++i) {
        const float cond_den = x[i] - t * v_cond[i];
        const float uncond_den = x[i] - t * v_uncond[i];
        const float diff = cond_den - uncond_den;
        const float parallel = static_cast<float>(coeff * static_cast<double>(cond_den));
        const float orth = diff - parallel;
        const float cfg_diff = apg_scale * orth + (1.0f - apg_scale) * diff;
        const float cfg_den = cond_den + g * cfg_diff;
        v_out[i] = (x[i] - cfg_den) / t;
    }
}

std::vector<float> sample_euler_single_axis_cfg(
    Sa3DiTModel& dit,
    const Sa3DiTModel::Dims& dims,
    const std::vector<float>& noise,
    const Sa3BranchInputs& cond,
    const Sa3BranchInputs& uncond,
    const Sa3SamplerConfig& config,
    Sa3StepCallback callback,
    Sa3StepTrace trace)
{
    if (dims.batch != 2) throw std::invalid_argument("sample_euler_single_axis_cfg: the DiT graph must have batch 2");
    const size_t latent = static_cast<size_t>(dims.channels) * static_cast<size_t>(dims.frames);
    if (noise.size() != latent) throw std::invalid_argument("sample_euler_single_axis_cfg: noise size mismatch");

    // Batch the two branches once: entry 0 = cond, entry 1 = uncond.
    auto batch2 = [](const std::vector<float>& a, const std::vector<float>& b) {
        if (a.size() != b.size()) throw std::invalid_argument("sample_euler_single_axis_cfg: branch tensor size mismatch");
        std::vector<float> out(a.size() * 2);
        std::memcpy(out.data(), a.data(), a.size() * sizeof(float));
        std::memcpy(out.data() + a.size(), b.data(), b.size() * sizeof(float));
        return out;
    };
    const auto b_cross = batch2(cond.cross_attn_cond, uncond.cross_attn_cond);
    const auto b_global = batch2(cond.global_embed, uncond.global_embed);
    const auto b_local = batch2(cond.local_add_cond, uncond.local_add_cond);
    const auto b_control = batch2(cond.control, uncond.control);
    const auto b_tf = batch2(cond.tf_inpaint_mask, uncond.tf_inpaint_mask);

    const auto schedule = build_logsnr_schedule(config.steps, config.sigma_max, config.schedule, dims.frames);
    std::vector<float> x = noise;
    std::vector<float> b_x(latent * 2);
    std::vector<float> b_t(2);
    std::vector<float> v_cond(latent), v_uncond(latent), v_cfg(latent);

    for (int i = 0; i < config.steps; ++i) {
        const float t_curr = schedule[static_cast<size_t>(i)];
        const float t_next = schedule[static_cast<size_t>(i + 1)];
        std::memcpy(b_x.data(), x.data(), latent * sizeof(float));
        std::memcpy(b_x.data() + latent, x.data(), latent * sizeof(float));
        b_t[0] = t_curr;
        b_t[1] = t_curr;
        const auto v = dit.forward(b_x, b_t, b_cross, b_global, b_local, b_control, b_tf, dims);
        std::memcpy(v_cond.data(), v.data(), latent * sizeof(float));
        std::memcpy(v_uncond.data(), v.data() + latent, latent * sizeof(float));
        apply_single_axis_cfg(x, v_cond, v_uncond, t_curr, config.cfg_scale, config.apg_scale, v_cfg);
        if (trace) trace(i, t_curr, x, v_cond, v_uncond, v_cfg);
        const float dt = t_next - t_curr;
        for (size_t j = 0; j < latent; ++j) x[j] += dt * v_cfg[j];
        if (callback) callback(i, t_curr, x);
    }
    return x;
}

} // namespace sao
