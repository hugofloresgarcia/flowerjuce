#pragma once

#include "StreamGenProcessor.h"

#include <functional>
#include <juce_core/juce_core.h>

#include <memory>

namespace streamgen {

class InferenceWorker;

/// Background thread: appends timestamped telemetry lines to a file (not the audio thread).
class DiagnosticLogWriter : public juce::Thread {
public:
    /// Args:
    ///     processor: Audio engine (telemetry atomics).
    ///     get_worker: Returns inference worker or nullptr.
    ///     log_file: Destination file (append if exists per JUCE FileOutputStream semantics).
    DiagnosticLogWriter(StreamGenProcessor& processor,
                        std::function<InferenceWorker*()> get_worker,
                        const juce::File& log_file);

    ~DiagnosticLogWriter() override;

    void run() override;

private:
    StreamGenProcessor& m_processor;
    std::function<InferenceWorker*()> m_get_worker;
    juce::File m_log_file;
    std::unique_ptr<juce::FileOutputStream> m_stream;
};

} // namespace streamgen
