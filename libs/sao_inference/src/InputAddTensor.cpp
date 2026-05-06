#include "sao_inference/InputAddTensor.h"

#include <cassert>
#include <cstring>

namespace sao {

std::vector<float> concat_input_add(
    const std::vector<InputAddTensor>& tensors,
    int latent_length)
{
    int total = total_channels(tensors);
    std::vector<float> out(static_cast<size_t>(total) * static_cast<size_t>(latent_length));

    // Channel-major layout: each per-key tensor is (1, channels_k, T) flat
    // row-major. Concatenating along channel axis is just a sequence of
    // memcpy of channels_k * T floats.
    size_t write_offset = 0;
    for (const auto& t : tensors) {
        const size_t block = static_cast<size_t>(t.channels) * static_cast<size_t>(latent_length);
        assert(t.data.size() == block);
        std::memcpy(out.data() + write_offset, t.data.data(), block * sizeof(float));
        write_offset += block;
    }
    assert(write_offset == out.size());
    return out;
}

} // namespace sao
