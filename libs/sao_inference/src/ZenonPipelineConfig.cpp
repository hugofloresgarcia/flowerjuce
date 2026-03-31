#include "sao_inference/ZenonPipelineConfig.h"
#include <nlohmann/json.hpp>
#include <fstream>
#include <cassert>
#include <iostream>
#include <filesystem>

namespace sao {

static MaskRule parse_mask_rule(const std::string& rule)
{
    if (rule == "pass_through") return MaskRule::pass_through;
    if (rule == "multiply_by_mask") return MaskRule::multiply_by_mask;
    if (rule == "multiply_by_complement") return MaskRule::multiply_by_complement;

    std::cerr << "[ZenonPipelineConfig] Unknown mask_rule '" << rule
              << "', defaulting to multiply_by_complement" << std::endl;
    return MaskRule::multiply_by_complement;
}

ZenonPipelineConfig ZenonPipelineConfig::load(const std::string& manifest_path)
{
    std::ifstream file(manifest_path);
    assert(file.is_open());

    auto j = nlohmann::json::parse(file);
    std::filesystem::path manifest_dir = std::filesystem::path(manifest_path).parent_path();

    ZenonPipelineConfig config;
    config.manifest_dir = manifest_dir.string();

    config.sample_rate = j.value("sample_rate", 44100);
    config.sample_size = j.value("sample_size", 524288);
    config.latent_dim = j.value("latent_dim", 64);
    config.latent_length = j.value("latent_length", 256);
    config.downsampling_ratio = j.value("downsampling_ratio", 2048);
    config.cond_token_dim = j.value("cond_token_dim", 768);
    config.global_cond_dim = j.value("global_cond_dim", 768);
    config.embed_dim = j.value("embed_dim", 1024);

    if (j.contains("cross_attn_cond_ids")) {
        for (auto& id : j["cross_attn_cond_ids"]) {
            config.cross_attn_cond_ids.push_back(id.get<std::string>());
        }
    }
    if (j.contains("global_cond_ids")) {
        for (auto& id : j["global_cond_ids"]) {
            config.global_cond_ids.push_back(id.get<std::string>());
        }
    }

    if (j.contains("input_add_keys")) {
        for (auto& key_obj : j["input_add_keys"]) {
            InputAddKeyDescriptor desc;
            desc.name = key_obj.at("name").get<std::string>();
            desc.channels = key_obj.at("channels").get<int>();
            desc.mask_rule = parse_mask_rule(key_obj.value("mask_rule", "multiply_by_complement"));
            config.input_add_keys.push_back(desc);
        }
    }

    if (j.contains("distribution_shift") && !j["distribution_shift"].is_null()) {
        auto& ds = j["distribution_shift"];
        config.dist_shift.enabled = true;
        config.dist_shift.base_shift = ds.value("base_shift", 0.5f);
        config.dist_shift.max_shift = ds.value("max_shift", 1.15f);
        config.dist_shift.min_length = ds.value("min_length", 256);
        config.dist_shift.max_length = ds.value("max_length", 4096);
    }

    config.diffusion_objective = j.value("diffusion_objective", "rectified_flow");
    config.vae_scale = j.value("vae_scale", 1.0f);

    if (j.contains("onnx_files")) {
        auto& onnx = j["onnx_files"];
        auto resolve = [&](const std::string& rel) -> std::string {
            return (manifest_dir / rel).string();
        };
        config.t5_onnx_path = resolve(onnx.value("t5_encoder", "onnx/t5_encoder.onnx"));
        config.dit_onnx_path = resolve(onnx.value("dit", "onnx/zenon_dit.onnx"));
        config.vae_encoder_onnx_path = resolve(onnx.value("vae_encoder", "onnx/zenon_vae_encoder.onnx"));
        config.vae_decoder_onnx_path = resolve(onnx.value("vae_decoder", "onnx/zenon_vae_decoder.onnx"));
    }

    std::string num_embed_dir = j.value("number_embedder_weights_dir", "weights/number_embedder_zenon");
    config.number_embedder_weights_dir = (manifest_dir / num_embed_dir).string();

    config.mlx_vae_weights_path = (manifest_dir / "mlx" / "oobleck_vae.safetensors").string();
    config.mlx_vae_config_path = (manifest_dir / "mlx" / "vae_config.json").string();
    if (j.contains("mlx_vae") && j["mlx_vae"].is_object())
    {
        const auto& mv = j["mlx_vae"];
        if (mv.contains("weights"))
        {
            config.mlx_vae_weights_path =
                (manifest_dir / mv.at("weights").get<std::string>()).string();
        }
        if (mv.contains("config"))
        {
            config.mlx_vae_config_path =
                (manifest_dir / mv.at("config").get<std::string>()).string();
        }
    }

    std::cout << "[ZenonPipelineConfig] Loaded from " << manifest_path << std::endl;
    std::cout << "  sample_rate=" << config.sample_rate
              << " latent_dim=" << config.latent_dim
              << " latent_length=" << config.latent_length
              << " input_add_keys=" << config.input_add_keys.size()
              << " dist_shift=" << (config.dist_shift.enabled ? "yes" : "no")
              << std::endl;

    return config;
}

} // namespace sao
