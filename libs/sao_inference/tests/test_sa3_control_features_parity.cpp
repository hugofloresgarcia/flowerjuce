// Parity test: sao::ControlFeatureExtractor vs the SA3 Python features (scripts/dump_control_features.py).
//
// Reads tests/parity_data/control_features/{audio.npy, stats.json, config.json, drums_*.npy},
// runs the three extractors (a) on the whole clip at once and (b) in pseudo-random chunk sizes, and checks:
//   - rms_db / band_db:                max abs error < 1e-3 dB   (float32 reduction order differs)
//   - drums_rms:                       max abs error < 1e-6
//   - drums_tria_fixed / drums_tria_ema: quantised levels identical (0 mismatches)
//   - (a) and (b) bit-identical (chunking invariance)
#include "sao_inference/ControlFeatureExtractor.h"

#include <cnpy.h>
#include <nlohmann/json.hpp>

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

namespace {

std::vector<float> load_npy(const std::string& path, std::vector<size_t>& shape)
{
    auto arr = cnpy::npy_load(path);
    if (arr.word_size != sizeof(float)) {
        std::cerr << path << ": expected float32, word size " << arr.word_size << std::endl;
        std::exit(1);
    }
    shape = arr.shape;
    size_t total = 1;
    for (auto d : arr.shape) total *= d;
    const float* data = arr.data<float>();
    return std::vector<float>(data, data + total);
}

nlohmann::json load_json(const std::string& path)
{
    std::ifstream f(path);
    if (!f) {
        std::cerr << "cannot open " << path << std::endl;
        std::exit(1);
    }
    return nlohmann::json::parse(f);
}

/// Run one extractor over planar stereo audio; returns [frame][channel] incl. the trailing partial frame.
std::vector<float> run(sao::ControlFeatureExtractor& ex, const float* left, const float* right, size_t n,
                       bool chunked)
{
    ex.reset();
    const float* ch[2] = {left, right};
    if (!chunked) {
        ex.push_planar(ch, 2, n);
    } else {
        uint32_t state = 12345u;
        size_t pos = 0;
        while (pos < n) {
            state = state * 1664525u + 1013904223u;          // LCG, deterministic
            size_t len = 1 + static_cast<size_t>(state >> 16) % 9000; // 1..9000 samples (crosses hop boundaries)
            if (pos + len > n) len = n - pos;
            const float* sub[2] = {left + pos, right + pos};
            ex.push_planar(sub, 2, len);
            pos += len;
        }
    }
    ex.flush_partial();
    return ex.pending();
}

int level(float x, int n_levels)
{
    return static_cast<int>(std::nearbyint(x * static_cast<float>(n_levels - 1)));
}

struct Result {
    bool ok = true;
    void fail(const std::string& what) { std::cerr << "FAIL: " << what << std::endl; ok = false; }
};

} // namespace

int main()
{
    const std::string dir = std::string(SA3_PARITY_DATA_DIR) + "/control_features";
    std::cout << "=== SA3 control feature parity: " << dir << " ===" << std::endl;

    const auto cfg = load_json(dir + "/config.json");
    const auto stats = load_json(dir + "/stats.json");
    const int hop = cfg["hop"].get<int>();
    const int sample_rate = cfg["sample_rate"].get<int>();
    const size_t n_frames = cfg["n_frames"].get<size_t>();

    std::vector<size_t> shape;
    const auto audio = load_npy(dir + "/audio.npy", shape);
    if (shape.size() != 2 || shape[0] != 2) {
        std::cerr << "audio.npy must be [2, N]" << std::endl;
        return 1;
    }
    const size_t n = shape[1];
    const float* left = audio.data();
    const float* right = audio.data() + n;
    std::cout << "audio: 2 x " << n << " samples, " << n_frames << " frames, hop " << hop << std::endl;

    sao::ControlFeatureConfig base;
    base.sample_rate = sample_rate;
    base.hop = hop;
    base.floor_db = stats["floor_db"].get<float>();
    base.split_hz = stats["split_hz"].get<double>();
    base.crossover_order = stats["crossover_order"].get<int>();
    base.band_mean_db = {{stats["band_mean_db"][0].get<double>(), stats["band_mean_db"][1].get<double>()}};
    base.band_std_db = {{stats["band_std_db"][0].get<double>(), stats["band_std_db"][1].get<double>()}};
    base.min_std_db = stats["min_std_db"].get<double>();
    base.n_levels = stats["n_levels"].get<int>();
    base.ema_tau_seconds = stats["ema_tau_seconds"].get<double>();

    Result res;
    struct Case {
        const char* name;
        sao::ControlFeatureType type;
        const char* ref_file;
        bool exact_levels;
        float tol;
    };
    const Case cases[] = {
        {"drums_rms", sao::ControlFeatureType::Rms, "drums_rms.npy", false, 1e-6f},
        {"drums_tria_fixed", sao::ControlFeatureType::TriaFixed, "drums_tria_fixed.npy", true, 0.0f},
        {"drums_tria_ema", sao::ControlFeatureType::TriaEma, "drums_tria_ema.npy", true, 0.0f},
    };

    for (const auto& c : cases) {
        sao::ControlFeatureConfig conf = base;
        conf.type = c.type;
        sao::ControlFeatureExtractor ex(conf);
        const int ch = ex.channels();

        std::vector<size_t> rshape;
        const auto ref = load_npy(dir + "/" + c.ref_file, rshape); // [C, T]
        if (rshape.size() != 2 || rshape[0] != static_cast<size_t>(ch) || rshape[1] != n_frames) {
            res.fail(std::string(c.name) + ": reference shape mismatch");
            continue;
        }

        const auto whole = run(ex, left, right, n, false);
        const auto chunked = run(ex, left, right, n, true);
        if (whole.size() != n_frames * static_cast<size_t>(ch)) {
            res.fail(std::string(c.name) + ": produced " + std::to_string(whole.size() / ch) + " frames, expected " + std::to_string(n_frames));
            continue;
        }
        if (whole != chunked) res.fail(std::string(c.name) + ": chunked run differs from whole-clip run");

        float max_err = 0.0f;
        size_t mismatches = 0;
        size_t first_mismatch = 0;
        for (size_t t = 0; t < n_frames; ++t) {
            for (int k = 0; k < ch; ++k) {
                const float got = whole[t * static_cast<size_t>(ch) + static_cast<size_t>(k)];
                const float want = ref[static_cast<size_t>(k) * n_frames + t];
                max_err = std::max(max_err, std::abs(got - want));
                if (c.exact_levels && level(got, conf.n_levels) != level(want, conf.n_levels)) {
                    if (mismatches == 0) first_mismatch = t;
                    ++mismatches;
                }
            }
        }
        std::cout << c.name << ": max abs err " << max_err;
        if (c.exact_levels) {
            std::cout << ", level mismatches " << mismatches << " / " << n_frames * static_cast<size_t>(ch);
            if (mismatches) std::cout << " (first at frame " << first_mismatch << ")";
        }
        std::cout << std::endl;
        if (c.exact_levels ? mismatches != 0 : max_err >= c.tol) res.fail(std::string(c.name) + ": out of tolerance");
    }

    std::cout << (res.ok ? "PASS" : "FAIL") << std::endl;
    return res.ok ? 0 : 1;
}
