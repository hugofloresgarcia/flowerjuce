#pragma once

#include "Sa3DiTModel.h"

#include <functional>
#include <vector>

namespace sao {

/// LogSNRShift parameters (stable_audio_3/inference/distribution_shift.py); manifest `sampler.schedule`.
struct LogSnrScheduleParams {
    float rate = 0.0f;
    float anchor_logsnr = -6.2f;
    float logsnr_end = 2.0f;
    float anchor_length = 2000.0f;
};

/// `build_schedule(steps, sigma_max, dist_shift=LogSNRShift)` in float32:
///   t = linspace(sigma_max, 0, steps+1); logsnr_start = anchor_logsnr - rate*log2(seq_len/anchor_length);
///   logsnr = logsnr_end - t*(logsnr_end - logsnr_start); t' = sigmoid(-logsnr); t'(t<=0)=0, t'(t>=1)=1; t'[0] = sigma_max.
/// Returns steps+1 values, decreasing to 0.
std::vector<float> build_logsnr_schedule(int steps, float sigma_max, const LogSnrScheduleParams& params, int seq_len);

/// The per-branch DiT conditioning (one batch entry, flat row-major).
struct Sa3BranchInputs {
    std::vector<float> cross_attn_cond;  // [S, cond_dim]
    std::vector<float> global_embed;     // [global_dim]
    std::vector<float> local_add_cond;   // [local_add_channels, T]
    std::vector<float> control;          // [control_channels, T]
    std::vector<float> tf_inpaint_mask;  // [T]
};

struct Sa3SamplerConfig {
    int steps = 8;
    float cfg_scale = 4.0f;
    float apg_scale = 1.0f;   ///< 1 = only the component of the guidance orthogonal to the conditional denoised estimate
    float sigma_max = 1.0f;
    LogSnrScheduleParams schedule;
};

/// Called after each Euler step with the updated latent (flat [C, T]).
using Sa3StepCallback = std::function<void(int step, float t_curr, const std::vector<float>& x)>;
/// Called before the Euler update with everything the step computed (parity dumps, diagnostics).
using Sa3StepTrace = std::function<void(int step, float t_curr, const std::vector<float>& x_before,
                                        const std::vector<float>& v_cond, const std::vector<float>& v_uncond,
                                        const std::vector<float>& v_cfg)>;

/// Single-axis CFG with APG in denoised space, as `DiffusionTransformer.forward` applies it for `rf_denoiser`
/// (stable_audio_3/models/dit.py), but between the two branches the pipeline chose (conditional, guided-axis-nulled):
///   cond_den = x - t v_cond;  uncond_den = x - t v_uncond;  diff = cond_den - uncond_den
///   diff_orth = diff - <diff, n> n   with n = cond_den / max(||cond_den||, 1e-12)   (over all C*T values)
///   cfg_diff  = apg_scale * diff_orth + (1 - apg_scale) * diff
///   v = (x - (cond_den + (cfg_scale - 1) cfg_diff)) / t
/// All vectors flat [C, T]; writes `v_out`.
void apply_single_axis_cfg(const std::vector<float>& x, const std::vector<float>& v_cond, const std::vector<float>& v_uncond,
                           float t, float cfg_scale, float apg_scale, std::vector<float>& v_out);

/// Euler rectified-flow loop with the two-branch CFG above. `noise` is flat [C, T] and becomes x at t = sigma_max.
/// Each step runs the DiT once with batch 2 = (cond, uncond). Returns the final latent, flat [C, T].
std::vector<float> sample_euler_single_axis_cfg(
    Sa3DiTModel& dit,
    const Sa3DiTModel::Dims& dims,
    const std::vector<float>& noise,
    const Sa3BranchInputs& cond,
    const Sa3BranchInputs& uncond,
    const Sa3SamplerConfig& config,
    Sa3StepCallback callback = nullptr,
    Sa3StepTrace trace = nullptr);

} // namespace sao
