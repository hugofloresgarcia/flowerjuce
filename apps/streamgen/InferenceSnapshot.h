#pragma once

#include "TimeRuler.h"

#include <sao_inference/ZenonPipeline.h>

#include <cstddef>
#include <cstdint>

namespace streamgen {

/// Full forensic snapshot after the last completed `process_job` (worker thread writes, UI reads).
struct InferenceSnapshot {
    sao::ZenonTimingReport timing;
    sao::ZenonDiagnostics diagnostics;
    GenerationJob job;
    size_t t5_sequence_length = 0;
    uint64_t t5_attention_nonzero_positions = 0;
    double wall_clock_ms = 0.0;
};

} // namespace streamgen
