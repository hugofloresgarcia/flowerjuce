#include "sao_inference/Sa3PipelineConfig.h"

#include <nlohmann/json.hpp>

#include <filesystem>
#include <fstream>
#include <stdexcept>

namespace sao {

namespace {

using nlohmann::json;
namespace fs = std::filesystem;

constexpr const char* kSchema = "sa3_pipeline_manifest/1";

[[noreturn]] void fail(const std::string& what)
{
    throw std::runtime_error("Sa3PipelineConfig: " + what);
}

/// `obj[key]`, failing with the dotted path when the key is missing.
const json& field(const json& obj, const char* key, const std::string& where)
{
    if (!obj.is_object() || !obj.contains(key)) fail("missing field " + where + "." + key);
    return obj.at(key);
}

template <typename T>
T get(const json& obj, const char* key, const std::string& where)
{
    const json& v = field(obj, key, where);
    try {
        return v.get<T>();
    } catch (const json::exception& e) {
        fail("field " + where + "." + key + " has the wrong type (" + e.what() + ")");
    }
}

std::string resolve_path(const json& obj, const char* key, const std::string& where, const fs::path& base, bool must_exist)
{
    const fs::path p = fs::absolute(base / get<std::string>(obj, key, where)).lexically_normal();
    if (must_exist && !fs::exists(p)) fail(where + "." + key + " does not exist: " + p.string());
    return p.string();
}

void require(bool ok, const std::string& what)
{
    if (!ok) fail(what);
}

} // namespace

Sa3PipelineConfig Sa3PipelineConfig::load(const std::string& path)
{
    std::ifstream f(path);
    if (!f) fail("cannot open " + path);
    json j;
    try {
        j = json::parse(f);
    } catch (const json::exception& e) {
        fail(path + " is not valid JSON: " + e.what());
    }

    Sa3PipelineConfig c;
    c.manifest_path = fs::absolute(path).lexically_normal().string();
    const fs::path base = fs::path(c.manifest_path).parent_path();
    c.manifest_dir = base.string();

    c.schema = get<std::string>(j, "schema", "");
    if (c.schema != kSchema) fail("schema '" + c.schema + "' is not '" + kSchema + "'");
    c.arm = get<std::string>(j, "arm", "");
    c.model_config_source = resolve_path(j, "model_config_source", "", base, false);
    c.checkpoint_path = resolve_path(j, "checkpoint_path", "", base, false);
    c.global_step = get<long long>(j, "global_step", "");
    c.sample_rate = get<int>(j, "sample_rate", "");
    c.audio_channels = get<int>(j, "audio_channels", "");
    c.latent_dim = get<int>(j, "latent_dim", "");
    c.downsampling_ratio = get<int>(j, "downsampling_ratio", "");
    require(c.sample_rate > 0 && c.audio_channels > 0 && c.latent_dim > 0 && c.downsampling_ratio > 0, "non-positive audio/latent dims");

    {
        const json& w = field(j, "window", "");
        c.window.frames = get<int>(w, "frames", "window");
        c.window.samples = get<int>(w, "samples", "window");
        c.window.seconds_total = get<float>(w, "seconds_total", "window");
        const auto fvs = get<std::vector<float>>(w, "future_visibility_seconds", "window");
        const auto fvf = get<std::vector<int>>(w, "future_visibility_frames", "window");
        require(fvs.size() == 2 && fvf.size() == 2, "window.future_visibility_* must have 2 entries");
        c.window.future_visibility_seconds = {{fvs[0], fvs[1]}};
        c.window.future_visibility_frames = {{fvf[0], fvf[1]}};
        require(c.window.frames > 0 && c.window.samples == c.window.frames * c.downsampling_ratio, "window.samples != frames * downsampling_ratio");
    }
    {
        const json& t = field(j, "text", "");
        c.text.onnx = resolve_path(t, "onnx", "text", base, true);
        c.text.max_length = get<int>(t, "max_length", "text");
        c.text.dim = get<int>(t, "dim", "text");
        c.text.migraphx_cache_dir = resolve_path(t, "migraphx_cache_dir", "text", base, false);
        const json& tk = field(t, "tokenizer", "text");
        c.text.tokenizer_model = resolve_path(tk, "model", "text.tokenizer", base, true);
        c.text.tokenizer.max_length = get<int>(tk, "max_length", "text.tokenizer");
        c.text.tokenizer.add_bos = get<bool>(tk, "add_bos", "text.tokenizer");
        c.text.tokenizer.add_eos = get<bool>(tk, "add_eos", "text.tokenizer");
        c.text.tokenizer.pad_id = get<int>(tk, "pad_id", "text.tokenizer");
        c.text.tokenizer.bos_id = get<int>(tk, "bos_id", "text.tokenizer");
        c.text.tokenizer.eos_id = get<int>(tk, "eos_id", "text.tokenizer");
        require(c.text.tokenizer.max_length == c.text.max_length, "text.tokenizer.max_length != text.max_length");
        require(get<bool>(t, "padding_applied_in_graph", "text"), "text.padding_applied_in_graph must be true for this pipeline");
    }
    {
        const json& n = field(j, "number_embedder", "");
        c.number_embedder.weights_dir = resolve_path(n, "weights_dir", "number_embedder", base, true);
        c.number_embedder.min_val = get<float>(n, "min_val", "number_embedder");
        c.number_embedder.max_val = get<float>(n, "max_val", "number_embedder");
        c.number_embedder.dim = get<int>(n, "dim", "number_embedder");
        c.number_embedder.features = get<int>(n, "features", "number_embedder");
        require(get<std::string>(n, "fourier_features_type", "number_embedder") == "expo", "number_embedder.fourier_features_type must be expo");
        require(c.window.seconds_total >= c.number_embedder.min_val && c.window.seconds_total <= c.number_embedder.max_val,
                "window.seconds_total outside the number embedder range");
    }
    {
        const json& d = field(j, "dit", "");
        for (const auto& [k, v] : field(d, "onnx_by_batch", "dit").items()) {
            const int b = std::stoi(k);
            const fs::path p = fs::absolute(base / v.get<std::string>()).lexically_normal();
            if (!fs::exists(p)) fail("dit.onnx_by_batch[" + k + "] does not exist: " + p.string());
            c.dit.onnx_by_batch[b] = p.string();
        }
        c.dit.batch = get<int>(d, "batch", "dit");
        require(c.dit.onnx_by_batch.count(c.dit.batch) == 1, "dit.onnx_by_batch has no graph for dit.batch");
        c.dit.inputs = get<std::vector<std::string>>(d, "inputs", "dit");
        c.dit.output = get<std::string>(d, "output", "dit");
        c.dit.io_channels = get<int>(d, "io_channels", "dit");
        c.dit.embed_dim = get<int>(d, "embed_dim", "dit");
        c.dit.depth = get<int>(d, "depth", "dit");
        c.dit.cond_token_dim = get<int>(d, "cond_token_dim", "dit");
        c.dit.global_cond_dim = get<int>(d, "global_cond_dim", "dit");
        c.dit.cross_attn_tokens = get<int>(d, "cross_attn_tokens", "dit");
        c.dit.local_add_channels = get<int>(d, "local_add_channels", "dit");
        c.dit.migraphx_cache_dir = resolve_path(d, "migraphx_cache_dir", "dit", base, false);
        int sum = 0;
        for (const json& e : field(d, "local_add_layout", "dit")) {
            LocalAddEntry entry;
            entry.id = get<std::string>(e, "id", "dit.local_add_layout[]");
            entry.channels = get<int>(e, "channels", "dit.local_add_layout[]");
            sum += entry.channels;
            c.dit.local_add_layout.push_back(entry);
        }
        require(sum == c.dit.local_add_channels, "dit.local_add_layout channels do not sum to dit.local_add_channels");
        require(c.dit.io_channels == c.latent_dim, "dit.io_channels != latent_dim");
        require(c.dit.cross_attn_tokens == c.text.max_length + 1, "dit.cross_attn_tokens != text.max_length + 1");
        require(c.dit.cond_token_dim == c.text.dim && c.dit.global_cond_dim == c.number_embedder.features, "dit cond dims disagree with text/number embedders");
        require(c.dit.inputs.size() == 7 && !c.dit.output.empty(), "dit.inputs must list the 7 graph inputs");
    }
    {
        const json& ct = field(j, "control", "");
        c.control.id = get<std::string>(ct, "id", "control");
        c.control.dim = get<int>(ct, "dim", "control");
        c.control.null_value = get<float>(ct, "null_value", "control");
        const json& fe = field(ct, "feature", "control");
        ControlFeatureConfig& fc = c.control.feature;
        fc.type = ControlFeatureConfig::parse_type(get<std::string>(fe, "type", "control.feature"));
        fc.sample_rate = get<int>(fe, "sample_rate", "control.feature");
        fc.hop = get<int>(fe, "hop", "control.feature");
        fc.floor_db = get<float>(fe, "floor_db", "control.feature");
        fc.split_hz = get<double>(fe, "split_hz", "control.feature");
        fc.crossover_order = get<int>(fe, "crossover_order", "control.feature");
        const auto mean = get<std::vector<double>>(fe, "band_mean_db", "control.feature");
        const auto sd = get<std::vector<double>>(fe, "band_std_db", "control.feature");
        require(mean.size() == 2 && sd.size() == 2, "control.feature.band_* must have 2 entries");
        fc.band_mean_db = {{mean[0], mean[1]}};
        fc.band_std_db = {{sd[0], sd[1]}};
        fc.min_std_db = get<double>(fe, "min_std_db", "control.feature");
        fc.n_levels = get<int>(fe, "n_levels", "control.feature");
        fc.ema_tau_seconds = get<double>(fe, "ema_tau_seconds", "control.feature");
        const json& prior = field(fe, "ema_prior_frames", "control.feature");
        fc.ema_prior_frames = prior.is_null() ? -1.0 : prior.get<double>();
        require(fc.sample_rate == c.sample_rate && fc.hop == c.downsampling_ratio, "control.feature sample_rate/hop disagree with the model");
        require(fc.channels() == c.control.dim, "control.feature.type channel count != control.dim");
    }
    {
        const json& v = field(j, "vae", "");
        c.vae.decoder_onnx = resolve_path(v, "decoder_onnx", "vae", base, true);
        c.vae.encoder_onnx = resolve_path(v, "encoder_onnx", "vae", base, true);
        c.vae.latent_dim = get<int>(v, "latent_dim", "vae");
        c.vae.scale = get<float>(v, "scale", "vae");
        c.vae.migraphx_cache_dir = resolve_path(v, "migraphx_cache_dir", "vae", base, false);
        require(c.vae.latent_dim == c.latent_dim, "vae.latent_dim != latent_dim");
    }
    {
        const json& s = field(j, "sampler", "");
        c.sampler.type = get<std::string>(s, "type", "sampler");
        c.sampler.steps_default = get<int>(s, "steps_default", "sampler");
        c.sampler.sigma_max = get<float>(s, "sigma_max", "sampler");
        c.sampler.sigma_min = get<float>(s, "sigma_min", "sampler");
        c.sampler.diffusion_objective = get<std::string>(s, "diffusion_objective", "sampler");
        const json& sc = field(s, "schedule", "sampler");
        require(get<std::string>(sc, "type", "sampler.schedule") == "logsnr", "sampler.schedule.type must be logsnr");
        c.sampler.schedule.rate = get<float>(sc, "rate", "sampler.schedule");
        c.sampler.schedule.anchor_logsnr = get<float>(sc, "anchor_logsnr", "sampler.schedule");
        c.sampler.schedule.logsnr_end = get<float>(sc, "logsnr_end", "sampler.schedule");
        c.sampler.schedule.anchor_length = get<float>(sc, "anchor_length", "sampler.schedule");
        require(c.sampler.type == "euler", "sampler.type must be euler");
        require(c.sampler.diffusion_objective == "rf_denoiser", "sampler.diffusion_objective must be rf_denoiser");
        require(c.sampler.steps_default > 0, "sampler.steps_default must be > 0");
    }
    {
        const json& g = field(j, "cfg", "");
        c.cfg.mode = get<std::string>(g, "mode", "cfg");
        c.cfg.axes = get<std::vector<std::string>>(g, "axes", "cfg");
        c.cfg.axis_default = get<std::string>(g, "axis_default", "cfg");
        c.cfg.scale_default = get<float>(g, "scale_default", "cfg");
        c.cfg.apg_scale = get<float>(g, "apg_scale", "cfg");
        require(c.cfg.mode == "single_axis_apg", "cfg.mode must be single_axis_apg");
        bool found = false;
        for (const auto& a : c.cfg.axes) found = found || a == c.cfg.axis_default;
        require(found, "cfg.axis_default is not one of cfg.axes");
        require(c.dit.batch == 2, "single_axis_apg needs dit.batch == 2");
    }
    return c;
}

const std::string& Sa3PipelineConfig::dit_onnx(int batch) const
{
    const auto it = dit.onnx_by_batch.find(batch);
    if (it == dit.onnx_by_batch.end()) fail("no DiT graph exported for batch " + std::to_string(batch));
    return it->second;
}

} // namespace sao
