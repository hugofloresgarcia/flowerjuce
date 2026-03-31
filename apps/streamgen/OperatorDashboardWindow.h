#pragma once

#include <functional>
#include <juce_gui_basics/juce_gui_basics.h>
#include <memory>

#include "DiagnosticLogWriter.h"
#include "OperatorVizComponent.h"
#include "StreamGenProcessor.h"

namespace streamgen {

class InferenceWorker;

/// Forensic telemetry panel: threads, audio levels, scheduler, pipeline timings, latent stats.
class OperatorDashboardPanel : public juce::Component,
                               private juce::Timer {
public:
    /// Args:
    ///     processor: Audio engine (RT telemetry).
    ///     get_worker: Returns current inference worker or nullptr if pipeline not loaded.
    OperatorDashboardPanel(StreamGenProcessor& processor,
                           std::function<InferenceWorker*()> get_worker);

    ~OperatorDashboardPanel() override;

    void resized() override;

private:
    void timerCallback() override;
    void refresh_text();
    void on_trace_toggled();
    void on_file_log_toggled();
    void choose_log_file();

    StreamGenProcessor& m_processor;
    std::function<InferenceWorker*()> m_get_worker;

    OperatorVizComponent m_viz;
    juce::Viewport m_viewport;
    juce::TextEditor m_body;
    juce::ToggleButton m_trace_toggle{"Trace to console (1/s)"};
    juce::ToggleButton m_file_log_toggle{"Log to file"};
    juce::TextButton m_choose_log_button{"Choose log file..."};

    int m_trace_counter = 0;
    juce::File m_log_file;
    std::unique_ptr<DiagnosticLogWriter> m_log_writer;
};

/// Pop-up operator dashboard (close hides).
class OperatorDashboardWindow : public juce::DocumentWindow {
public:
    /// Args:
    ///     processor: Audio engine.
    ///     get_worker: Lazy lookup so the panel sees the worker after pipeline load.
    OperatorDashboardWindow(StreamGenProcessor& processor,
                            std::function<InferenceWorker*()> get_worker);

    void closeButtonPressed() override { setVisible(false); }
};

} // namespace streamgen
