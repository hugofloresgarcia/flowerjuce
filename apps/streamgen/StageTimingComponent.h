#pragma once

#include "TimeRuler.h"

#include <juce_gui_basics/juce_gui_basics.h>

namespace streamgen {

/// Displays per-stage inference timing as horizontal bars.
///
/// Shows VAE encode, T5 encode, DiT sampling, VAE decode, and total time
/// as colored bars with millisecond labels. Updates after each generation.
class StageTimingComponent : public juce::Component {
public:
    StageTimingComponent();

    /// Update the displayed timing values.
    ///
    /// Args:
    ///     timing: Stage timing from the most recent generation.
    void update(const StageTiming& timing);

    void paint(juce::Graphics& g) override;

private:
    StageTiming m_timing;
};

} // namespace streamgen
