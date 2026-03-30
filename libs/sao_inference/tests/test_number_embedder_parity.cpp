#include "sao_inference/NumberEmbedder.h"
#include <cnpy.h>
#include <cmath>
#include <iostream>
#include <cassert>
#include <string>

static float max_abs_error(const std::vector<float>& a, const float* b, size_t n)
{
    float max_err = 0.0f;
    for (size_t i = 0; i < n; ++i) {
        float err = std::abs(a[i] - b[i]);
        if (err > max_err) max_err = err;
    }
    return max_err;
}

int main()
{
    std::string weights_dir = WEIGHTS_DIR "/number_embedder";
    std::cout << "Loading NumberEmbedder from " << weights_dir << std::endl;

    sao::NumberEmbedder embedder(weights_dir);

    // Load test reference
    auto ref_input = cnpy::npy_load(weights_dir + "/test_input_normalized.npy");
    auto ref_output = cnpy::npy_load(weights_dir + "/test_output_embed.npy");

    float test_normalized = ref_input.data<float>()[0];
    // The embedder.embed() normalizes internally, so we need to reverse:
    float raw_value = test_normalized * (embedder.max_val() - embedder.min_val()) + embedder.min_val();
    std::cout << "Test value: " << raw_value << " (normalized: " << test_normalized << ")" << std::endl;

    auto result = embedder.embed(raw_value);

    size_t ref_size = ref_output.shape[0] * (ref_output.shape.size() > 1 ? ref_output.shape[1] : 1);
    assert(result.size() == ref_size);

    float err = max_abs_error(result, ref_output.data<float>(), ref_size);
    std::cout << "Max abs error: " << err << std::endl;

    constexpr float THRESHOLD = 1e-5f;
    if (err < THRESHOLD) {
        std::cout << "PASS (threshold=" << THRESHOLD << ")" << std::endl;
        return 0;
    } else {
        std::cerr << "FAIL: error " << err << " >= threshold " << THRESHOLD << std::endl;
        return 1;
    }
}
