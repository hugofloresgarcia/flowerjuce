#pragma once

#include "DiTInpaintModel.h"
#include "InpaintConditioningAssembler.h"
#include "ZenonPipelineConfig.h"
#include "RectifiedFlowSampler.h"
#include <vector>
#include <functional>

namespace sao {

/// Apply distribution shift to a time schedule.
///
/// Mirrors DistributionShift.time_shift() from
/// sat-zenon/stable_audio_tools/inference/sampling.py (lines 33-41).
///
/// Formula:
///     mu = -(base_shift + (max_shift - base_shift) * (seq_len - min_length) / (max_length - min_length))
///     t_out = 1 - exp(mu) / (exp(mu) + (1/(1-t) - 1)^sigma)
///
/// Args:
///     schedule: Time schedule to shift, modified in-place.
///     config: Distribution shift parameters from manifest.
///     seq_len: Latent sequence length (latent_length from manifest).
void apply_distribution_shift(
    std::vector<float>& schedule,
    const DistributionShiftConfig& config,
    int seq_len
);

/// Run the rectified flow Euler sampling loop with CFG and input_add.
///
/// Same algorithm as sample_euler_cfg but with:
///   - input_add tensors duplicated for CFG batch=2 (same values for cond
///     and uncond, per DiffusionTransformer.forward in Python)
///   - Optional distribution shift on the time schedule
///   - Config-driven dimensions from the manifest
///
/// Args:
///     dit: The DiT model with input_add support.
///     noise: Initial noise, flat (1, C, T).
///     conditioning: Assembled inpaint conditioning (includes input_add).
///     config: Sampler configuration.
///     pipeline_config: Zenon pipeline configuration (for dims + dist shift).
///     callback: Optional callback invoked after each step.
///
/// Returns:
///     Final latent, flat (1, C, T).
std::vector<float> sample_euler_cfg_inpaint(
    DiTInpaintModel& dit,
    const std::vector<float>& noise,
    const InpaintConditioning& conditioning,
    const SamplerConfig& config,
    const ZenonPipelineConfig& pipeline_config,
    StepCallback callback = nullptr
);

} // namespace sao
