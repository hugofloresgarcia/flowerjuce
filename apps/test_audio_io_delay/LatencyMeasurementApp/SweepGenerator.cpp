#include "SweepGenerator.h"

namespace LatencyMeasurement {

std::vector<float> SweepGenerator::generateLogSweep(double sampleRate, double startFreqHz,
    double endFreqHz, double durationSec, double volumeDB, int silenceSamples)
{
    int sweepSamples = static_cast<int>(durationSec * sampleRate);
    std::vector<float> signal(sweepSamples + silenceSamples, 0.0f);
    
    float amplitude = std::pow(10.0f, volumeDB / 20.0f);
    double logRatio = std::log(endFreqHz / startFreqHz);
    double phaseConstant = 2.0 * M_PI * startFreqHz * durationSec / logRatio;
    
    for (int i = 0; i < sweepSamples; ++i) {
        double t = static_cast<double>(i) / sampleRate;
        double phase = phaseConstant * (std::exp(t * logRatio / durationSec) - 1.0);
        signal[i] = amplitude * std::sin(phase);
    }
    return signal;
}

std::vector<float> SweepGenerator::generateDefaultSweep(double sampleRate)
{
    return generateLogSweep(sampleRate, 100.0, 10000.0, 1.0, -20.0, static_cast<int>(sampleRate));
}

}
