#include "sao_inference/Sa3Pipeline.h"

#include "sao_inference/OnnxVaeDecoder.h"
#include "sao_inference/OnnxVaeEncoder.h"

#include <cnpy.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>
#include <stdexcept>

namespace sao {

namespace {

using Clock = std::chrono::high_resolution_clock;

double elapsed_ms(Clock::time_point start)
{
    return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}

void mean_std(const std::vector<float>& v, Sa3LatentStats& out)
{
    out = Sa3LatentStats{};
    if (v.empty()) return;
    double sum = 0.0;
    for (float x : v) sum += x;
    out.mean = sum / static_cast<double>(v.size());
    double var = 0.0;
    for (float x : v) {
        const double d = x - out.mean;
        var += d * d;
    }
    out.std_dev = std::sqrt(var / static_cast<double>(v.size()));
}

[[noreturn]] void fail(const std::string& what)
{
    throw std::runtime_error("Sa3Pipeline: " + what);
}

/// Pads with zeros or truncates stereo audio, flat (2, n) -> flat (2, target).
std::vector<float> fit_stereo(const std::vector<float>& audio, size_t target)
{
    if (audio.size() % 2 != 0) fail("context_audio must be stereo, flat (2, N)");
    const size_t n = audio.size() / 2;
    std::vector<float> out(2 * target, 0.0f);
    const size_t m = std::min(n, target);
    for (size_t c = 0; c < 2; ++c) std::copy(audio.begin() + static_cast<std::ptrdiff_t>(c * n), audio.begin() + static_cast<std::ptrdiff_t>(c * n + m), out.begin() + static_cast<std::ptrdiff_t>(c * target));
    return out;
}

} // namespace

CfgAxis parse_cfg_axis(const std::string& name)
{
    if (name == "control") return CfgAxis::Control;
    if (name == "context") return CfgAxis::Context;
    throw std::invalid_argument("unknown cfg axis '" + name + "' (expected control or context)");
}

const char* cfg_axis_name(CfgAxis axis)
{
    return axis == CfgAxis::Control ? "control" : "context";
}

Sa3Pipeline::Sa3Pipeline(const Sa3PipelineConfig& config)
    : m_config(config)
{
    const auto t0 = Clock::now();
    if (m_verbose) std::cout << "[Sa3Pipeline] Loading " << m_config.arm << " (step " << m_config.global_step << ")..." << std::endl;
    const bool cuda = m_config.use_cuda, coreml = m_config.use_coreml, mgx = m_config.use_migraphx;
    m_text = std::make_unique<T5Encoder>(m_config.text.onnx, cuda, coreml, mgx, m_config.text.migraphx_cache_dir);
    m_dit = std::make_unique<Sa3DiTModel>(m_config.dit_onnx(m_config.dit.batch), m_config.dit.inputs, m_config.dit.output,
                                          m_config.dit.migraphx_cache_dir, cuda, coreml, mgx);
    // CoreML cannot take the full-window VAE graphs (sequence length limit); Zenon made the same choice.
    m_vae_encoder = std::make_unique<OnnxVaeEncoder>(m_config.vae.encoder_onnx, cuda, false, mgx, m_config.vae.migraphx_cache_dir);
    m_vae_decoder = std::make_unique<OnnxVaeDecoder>(m_config.vae.decoder_onnx, m_config.vae.scale, cuda, false, mgx, m_config.vae.migraphx_cache_dir);
    m_number_embedder = std::make_unique<NumberEmbedder>(m_config.number_embedder.weights_dir);
    if (m_number_embedder->features() != m_config.dit.global_cond_dim) fail("number embedder features != dit.global_cond_dim");
    m_seconds_embed = m_number_embedder->embed(m_config.window.seconds_total);
    m_timing.model_load_ms = elapsed_ms(t0);
    if (m_verbose) std::cout << "[Sa3Pipeline] All models loaded in " << m_timing.model_load_ms << " ms" << std::endl;
}

Sa3DiTModel::Dims Sa3Pipeline::dit_dims() const
{
    Sa3DiTModel::Dims d;
    d.batch = m_config.dit.batch;
    d.channels = m_config.latent_dim;
    d.frames = m_config.window.frames;
    d.cross_tokens = m_config.dit.cross_attn_tokens;
    d.cond_dim = m_config.dit.cond_token_dim;
    d.global_dim = m_config.dit.global_cond_dim;
    d.local_add_channels = m_config.dit.local_add_channels;
    d.control_channels = m_config.control.dim;
    return d;
}

std::vector<float> Sa3Pipeline::generate(const Sa3GenerateInputs& in)
{
    const int C = m_config.latent_dim;
    const int T = m_config.window.frames;
    const int L = m_config.text.max_length;
    const int D = m_config.text.dim;
    const int K = m_config.control.dim;
    const size_t latent = static_cast<size_t>(C) * static_cast<size_t>(T);
    const size_t samples = static_cast<size_t>(m_config.window.samples);
    const auto dims = dit_dims();

    if (static_cast<int>(in.input_ids.size()) != L || static_cast<int>(in.attention_mask.size()) != L) fail("input_ids / attention_mask must have text.max_length entries");
    if (in.control_frames.size() != static_cast<size_t>(K) * static_cast<size_t>(T)) fail("control_frames must be [control.dim, window.frames]");
    if (!in.context_latent.empty() && in.context_latent.size() != latent) fail("context_latent must be [latent_dim, window.frames]");
    if (in.steps < 1) fail("steps must be >= 1");
    if (in.keep_ratio < 0.0f || in.keep_ratio > 1.0f) fail("keep_ratio must be in [0, 1]");

    m_diagnostics = Sa3Diagnostics{};
    if (in.phase_callback) in.phase_callback(Sa3InferencePhase::Encode, 0);
    const auto total_start = Clock::now();

    // ---- 1. context latent --------------------------------------------------------------------------------
    auto t0 = Clock::now();
    std::vector<float> context_latent;
    if (!in.context_latent.empty()) {
        context_latent = in.context_latent;
        m_timing.vae_encode_ms = 0.0;
    } else if (!in.context_audio.empty()) {
        const std::vector<float> audio = in.context_audio.size() == 2 * samples ? in.context_audio : fit_stereo(in.context_audio, samples);
        if (m_verbose) std::cout << "[Sa3Pipeline] VAE-encoding context audio..." << std::flush;
        context_latent = m_vae_encoder->encode(audio, static_cast<int>(samples), C);
        if (context_latent.size() != latent) fail("VAE encoder returned an unexpected latent size");
        m_timing.vae_encode_ms = elapsed_ms(t0);
        if (m_verbose) std::cout << " done (" << m_timing.vae_encode_ms << " ms)" << std::endl;
    } else {
        context_latent.assign(latent, 0.0f); // the context null: nothing known yet
        m_timing.vae_encode_ms = 0.0;
    }
    mean_std(context_latent, m_diagnostics.context_latent);

    // ---- 2. masks -----------------------------------------------------------------------------------------
    const int keep_frames = static_cast<int>(static_cast<float>(T) * in.keep_ratio);
    const int tf_keep_frames = std::clamp(keep_frames + in.future_visibility_frames, 0, T);
    m_diagnostics.keep_frames = keep_frames;
    m_diagnostics.tf_keep_frames = tf_keep_frames;
    std::vector<float> inpaint_mask(static_cast<size_t>(T), 0.0f), tf_mask(static_cast<size_t>(T), 0.0f);
    std::fill(inpaint_mask.begin(), inpaint_mask.begin() + keep_frames, 1.0f);
    std::fill(tf_mask.begin(), tf_mask.begin() + tf_keep_frames, 1.0f);

    // ---- 4. text ------------------------------------------------------------------------------------------
    const std::vector<float>* text = nullptr;
    if (in.precomputed_text != nullptr) {
        if (in.precomputed_text->size() != static_cast<size_t>(L) * static_cast<size_t>(D)) fail("precomputed_text must be [max_length, dim]");
        text = in.precomputed_text;
        m_timing.text_encode_ms = 0.0;
    } else {
        t0 = Clock::now();
        if (m_verbose) std::cout << "[Sa3Pipeline] Text-encoding prompt..." << std::flush;
        m_last_text = m_text->encode(in.input_ids, in.attention_mask);
        if (m_last_text.size() != static_cast<size_t>(L) * static_cast<size_t>(D)) fail("text encoder returned an unexpected size");
        m_timing.text_encode_ms = elapsed_ms(t0);
        if (m_verbose) std::cout << " done (" << m_timing.text_encode_ms << " ms)" << std::endl;
        text = &m_last_text;
    }

    // ---- 5-6. conditioning: the conditional branch and the guided-axis-nulled branch -----------------------
    t0 = Clock::now();
    Sa3BranchInputs cond;
    cond.cross_attn_cond.reserve(static_cast<size_t>(L + 1) * static_cast<size_t>(D));
    cond.cross_attn_cond.insert(cond.cross_attn_cond.end(), text->begin(), text->end());
    cond.cross_attn_cond.insert(cond.cross_attn_cond.end(), m_seconds_embed.begin(), m_seconds_embed.end());
    cond.global_embed = m_seconds_embed;
    cond.local_add_cond.resize(static_cast<size_t>(m_config.dit.local_add_channels) * static_cast<size_t>(T));
    std::copy(inpaint_mask.begin(), inpaint_mask.end(), cond.local_add_cond.begin());          // channel 0: inpaint_mask
    for (int c = 0; c < C; ++c)                                                                // channels 1..C: masked context
        for (int t = 0; t < T; ++t)
            cond.local_add_cond[static_cast<size_t>(1 + c) * T + t] = context_latent[static_cast<size_t>(c) * T + t] * inpaint_mask[static_cast<size_t>(t)];
    cond.control.resize(static_cast<size_t>(K) * static_cast<size_t>(T));
    for (int k = 0; k < K; ++k)
        for (int t = 0; t < T; ++t)
            cond.control[static_cast<size_t>(k) * T + t] = in.control_frames[static_cast<size_t>(k) * T + t] * tf_mask[static_cast<size_t>(t)];
    cond.tf_inpaint_mask = tf_mask;

    Sa3BranchInputs uncond = cond;
    if (in.cfg_axis == CfgAxis::Control) {
        for (int k = 0; k < K; ++k)
            for (int t = 0; t < T; ++t)
                uncond.control[static_cast<size_t>(k) * T + t] = m_config.control.null_value * tf_mask[static_cast<size_t>(t)];
    } else {
        std::fill(uncond.local_add_cond.begin() + T, uncond.local_add_cond.end(), 0.0f); // zero latent, mask kept
    }
    mean_std(cond.control, m_diagnostics.control);
    m_timing.conditioning_ms = elapsed_ms(t0);

    // ---- 7. noise + sampling ------------------------------------------------------------------------------
    std::vector<float> noise;
    if (!in.external_noise.empty()) {
        if (in.external_noise.size() != latent) fail("external_noise must be [latent_dim, frames]");
        noise = in.external_noise;
    } else {
        noise.resize(latent);
        std::mt19937 rng(in.seed);
        std::normal_distribution<float> dist(0.0f, 1.0f);
        for (auto& n : noise) n = dist(rng);
    }
    mean_std(noise, m_diagnostics.noise);

    Sa3SamplerConfig sampler;
    sampler.steps = in.steps;
    sampler.cfg_scale = in.cfg_scale;
    sampler.apg_scale = m_config.cfg.apg_scale;
    sampler.sigma_max = m_config.sampler.sigma_max;
    sampler.schedule.rate = m_config.sampler.schedule.rate;
    sampler.schedule.anchor_logsnr = m_config.sampler.schedule.anchor_logsnr;
    sampler.schedule.logsnr_end = m_config.sampler.schedule.logsnr_end;
    sampler.schedule.anchor_length = m_config.sampler.schedule.anchor_length;

    m_timing.sampling_step_ms.clear();
    m_timing.sampling_step_ms.reserve(static_cast<size_t>(in.steps));
    m_diagnostics.diffusion_steps.clear();
    if (m_verbose)
        std::cout << "[Sa3Pipeline] Sampling " << in.steps << " Euler steps, cfg " << in.cfg_scale << " on " << cfg_axis_name(in.cfg_axis)
                  << " (apg " << sampler.apg_scale << "), keep " << keep_frames << " fr, tf " << tf_keep_frames << " fr" << std::endl;
    t0 = Clock::now();
    m_last_sampled = sample_euler_single_axis_cfg(
        *m_dit, dims, noise, cond, uncond, sampler,
        [&](int step, float t, const std::vector<float>& x) {
            if (in.phase_callback) in.phase_callback(Sa3InferencePhase::DiT, step + 1);
            const double step_ms = elapsed_ms(t0);
            m_timing.sampling_step_ms.push_back(step_ms);
            t0 = Clock::now();
            if (m_verbose) std::cout << "[Sa3Pipeline]   step " << step + 1 << "/" << in.steps << " t=" << t << " (" << step_ms << " ms)" << std::endl;
            if (m_collect_step_diagnostics) {
                Sa3DiffusionStepStats s;
                s.step_index = step;
                s.t_curr = t;
                Sa3LatentStats xs;
                mean_std(x, xs);
                s.mean_x = xs.mean;
                s.std_x = xs.std_dev;
                m_diagnostics.diffusion_steps.push_back(s);
            }
            if (!in.dump_steps_dir.empty()) {
                std::ostringstream name;
                name << in.dump_steps_dir << "/step_" << std::setfill('0') << std::setw(2) << step << "_x_after.npy";
                cnpy::npy_save(name.str(), x.data(), {1, static_cast<size_t>(C), static_cast<size_t>(T)}, "w");
            }
            if (in.step_callback) in.step_callback(step, t, x);
        },
        in.step_trace);
    m_timing.sampling_total_ms = 0.0;
    for (double s : m_timing.sampling_step_ms) m_timing.sampling_total_ms += s;
    mean_std(m_last_sampled, m_diagnostics.sampled_latent);
    m_last_cond = std::move(cond);
    m_last_uncond = std::move(uncond);

    // ---- 8. decode ----------------------------------------------------------------------------------------
    if (in.phase_callback) in.phase_callback(Sa3InferencePhase::Decode, 0);
    t0 = Clock::now();
    auto audio = m_vae_decoder->decode(m_last_sampled, T);
    if (audio.size() != 2 * samples) fail("VAE decoder returned an unexpected audio size");
    m_timing.vae_decode_ms = elapsed_ms(t0);
    m_timing.total_ms = elapsed_ms(total_start);
    if (m_verbose) {
        std::cout << "[Sa3Pipeline] Done in " << m_timing.total_ms << " ms: encode " << m_timing.vae_encode_ms << ", text " << m_timing.text_encode_ms
                  << ", cond " << m_timing.conditioning_ms << ", sampling " << m_timing.sampling_total_ms << " (" << in.steps << " steps), decode "
                  << m_timing.vae_decode_ms << " ms" << std::endl;
    }
    return audio;
}

void Sa3Pipeline::warmup()
{
    const auto t0 = Clock::now();
    const auto dims = dit_dims();
    const size_t latent = static_cast<size_t>(dims.channels) * static_cast<size_t>(dims.frames);
    Sa3BranchInputs zero;
    zero.cross_attn_cond.assign(static_cast<size_t>(dims.cross_tokens) * static_cast<size_t>(dims.cond_dim), 0.0f);
    zero.global_embed.assign(static_cast<size_t>(dims.global_dim), 0.0f);
    zero.local_add_cond.assign(static_cast<size_t>(dims.local_add_channels) * static_cast<size_t>(dims.frames), 0.0f);
    zero.control.assign(static_cast<size_t>(dims.control_channels) * static_cast<size_t>(dims.frames), 0.0f);
    zero.tf_inpaint_mask.assign(static_cast<size_t>(dims.frames), 0.0f);
    Sa3SamplerConfig one;
    one.steps = 1;
    one.cfg_scale = 1.0f;
    std::vector<float> noise(latent, 0.0f);
    const auto x = sample_euler_single_axis_cfg(*m_dit, dims, noise, zero, zero, one);
    m_vae_decoder->decode(x, dims.frames);
    std::vector<int64_t> ids(static_cast<size_t>(m_config.text.max_length), m_config.text.tokenizer.pad_id), mask(ids.size(), 0);
    m_text->encode(ids, mask);
    if (m_verbose) std::cout << "[Sa3Pipeline] warmup (DiT + decoder + text) in " << elapsed_ms(t0) << " ms" << std::endl;
}

} // namespace sao
