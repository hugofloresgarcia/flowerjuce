#pragma once

#include "TimeRuler.h"

#include <cstdint>
#include <string>
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

/// True when `STREAMGEN_PRE_VAE_WAV_DIR` is set and non-empty. Lets the worker skip the extra
/// snapshots `dump_job_conditioning_if_enabled` needs when no dump is requested.
bool pre_vae_dump_enabled();

/// Everything `dump_job_conditioning_if_enabled` records for one job besides audio.
struct JobConditioningDump {
    const std::vector<float>* inpaint_mask = nullptr;      ///< (latent_length,) in {0,1}; 1 = kept drum frame
    const std::vector<float>* tf_inpaint_mask = nullptr;   ///< (latent_length,) in {0,1}; 1 = sax visible
    const std::vector<float>* live_fraction = nullptr;     ///< (latent_length,) in [0,1]; share of frame from live perc
    int64_t live_context_samples = 0;                      ///< Playback samples of the kept prefix taken from live perc
    int64_t output_written_start_frame = 0;                ///< First model-rate frame of the output written to the ring
    bool percussion_as_context_at_dump = false;            ///< Toggle state when the job was processed
    int latent_length = 0;                                 ///< Manifest latent_length (T)
    int downsampling_ratio = 0;                            ///< Model samples per latent frame
    std::string prompt;                                    ///< Text prompt used for the job
};

/// Same `STREAMGEN_PRE_VAE_WAV_DIR` gate. Writes per job:
///   - `pre_vae_<jobid>_percussion.wav` — raw live percussion ring for the kept prefix (mono
///     duplicated to stereo, model rate, zeros after keep_end), whatever the Perc -> context toggle said.
///   - `pre_vae_<jobid>_cond.json` — job timeline, both masks, per-latent-frame live fraction.
///
/// Args:
///     job: Generation job (timeline window in playback-clock samples).
///     model_sample_rate_hz: Manifest / VAE sample rate (e.g. 44100).
///     playback_sample_rate_hz: Device clock used for `window_*` samples.
///     percussion_row_major_lr: Stereo row-major float, length `2 * num_frames`.
///     cond: Masks and metadata; all three vector pointers must be non-null and `latent_length` long.
/// Raises:
///     std::runtime_error when an output file cannot be written.
void dump_job_conditioning_if_enabled(
    const GenerationJob& job,
    double model_sample_rate_hz,
    int playback_sample_rate_hz,
    const std::vector<float>& percussion_row_major_lr,
    const JobConditioningDump& cond);

} // namespace streamgen
