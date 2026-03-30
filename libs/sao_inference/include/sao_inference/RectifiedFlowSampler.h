#pragma once

#include "DiTModel.h"
#include "ConditioningAssembler.h"
#include <vector>
#include <functional>

namespace sao {

/// Configuration for the rectified flow sampler.
struct SamplerConfig {
    int steps = 8;
    float cfg_scale = 7.0f;
    float sigma_max = 1.0f;
    int latent_channels = 64;
    int latent_length = 256;
};

/// Callback invoked after each sampling step.
///
/// Args:
///     step: Current step index.
///     t_curr: Current timestep.
///     x: Current latent state, flat (1, C, T).
using StepCallback = std::function<void(int step, float t_curr, const std::vector<float>& x)>;

/// Build the time schedule for rectified flow sampling.
///
/// logsnr = linspace(logsnr_max, 2, steps+1)
/// t = sigmoid(-logsnr)
/// t[0] = sigma_max, t[-1] = 0
///
/// Args:
///     steps: Number of sampling steps.
///     sigma_max: Maximum sigma (typically 1.0).
///
/// Returns:
///     Time schedule of length (steps+1,).
std::vector<float> build_time_schedule(int steps, float sigma_max);

/// Run the rectified flow Euler sampling loop with CFG.
///
/// For each step:
///   1. Stack x (cond) + x (uncond) into batch=2
///   2. Stack conditioning (cond) + null conditioning into batch=2
///   3. Call DiT once with batch=2
///   4. Split output: cond_out, uncond_out
///   5. Apply CFG: v = uncond_out + cfg_scale * (cond_out - uncond_out)
///   6. Euler step: x = x + dt * v
///
/// Args:
///     dit: The DiT model to call each step.
///     noise: Initial noise, flat (1, C, T).
///     conditioning: Assembled conditioning tensors.
///     config: Sampler configuration.
///     callback: Optional callback invoked after each step.
///
/// Returns:
///     Final latent, flat (1, C, T).
std::vector<float> sample_euler_cfg(
    DiTModel& dit,
    const std::vector<float>& noise,
    const Conditioning& conditioning,
    const SamplerConfig& config,
    StepCallback callback = nullptr
);

} // namespace sao
