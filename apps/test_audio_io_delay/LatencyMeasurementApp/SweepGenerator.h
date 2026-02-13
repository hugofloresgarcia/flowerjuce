#pragma once
#include <vector>
#include <cmath>

namespace LatencyMeasurement {

class SweepGenerator {
public:
    static std::vector<float> generateLogSweep(double sampleRate, double startFreqHz, 
        double endFreqHz, double durationSec, double volumeDB, int silenceSamples);
    static std::vector<float> generateDefaultSweep(double sampleRate);
};

}
