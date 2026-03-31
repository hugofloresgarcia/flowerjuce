#pragma once

#include "TimeRuler.h"

#include <cstdint>
#include <vector>

namespace streamgen {

/// Debug dump: writes pre-VAE tensors as float32 stereo WAV files **when** the environment
/// variable `STREAMGEN_PRE_VAE_WAV_DIR` is set to a directory path (created if missing).
///
/// Writes per job:
///   - `pre_vae_<jobid>_streamgen.wav` — `streamgen_audio` row-major (L block, R block), model rate
///   - `pre_vae_<jobid>_input.wav` — `input_audio` (drum stem / warmup), same layout and rate
///   - `pre_vae_<jobid>_meta.txt` — timeline + alignment notes
///   - `pre_vae_<jobid>_zenon_output.wav` — full stereo decode from `ZenonPipeline::generate()` (see below)
///
/// No-op if the env var is unset or empty. Intended for comparing streamgen vs. input stem
/// before the VAE encoder; both buffers are always the same length for a given job (same
/// snapshot window + resample path).
///
/// Args:
///     job: Generation job (timeline window in playback-clock samples).
///     model_sample_rate_hz: Manifest / VAE sample rate (e.g. 44100).
///     playback_sample_rate_hz: Device clock used for `window_*` samples (`effective_playback_rate_hz`).
///     streamgen_row_major_lr: Stereo row-major float, length `2 * num_frames`.
///     input_row_major_lr: Same layout and length as `streamgen_row_major_lr`.
void dump_pre_vae_wavs_if_enabled(
    const GenerationJob& job,
    double model_sample_rate_hz,
    int playback_sample_rate_hz,
    const std::vector<float>& streamgen_row_major_lr,
    const std::vector<float>& input_row_major_lr);

/// Same `STREAMGEN_PRE_VAE_WAV_DIR` gate. Writes `pre_vae_<jobid>_zenon_output.wav`: the pipeline
/// return value **before** overlap-add / ring write — full window `(2, model_num_frames)` row-major
/// at `model_sample_rate_hz` (VAE decode output, same layout as `InferenceWorker` uses for slicing).
///
/// Args:
///     model_num_frames: Manifest `sample_size` (e.g. 524288).
///     pipeline_output_row_major_lr: Length `2 * model_num_frames`.
void dump_zenon_pipeline_output_wav_if_enabled(
    const GenerationJob& job,
    double model_sample_rate_hz,
    int model_num_frames,
    const std::vector<float>& pipeline_output_row_major_lr);

} // namespace streamgen
