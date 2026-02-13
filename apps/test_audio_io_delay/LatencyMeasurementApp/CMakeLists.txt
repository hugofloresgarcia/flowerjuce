cmake_minimum_required(VERSION 3.22)

# Create the GUI application
juce_add_gui_app(LatencyMeasurementApp
    COMPANY_NAME "Unsound"
    PRODUCT_NAME "Latency Measurement"
    VERSION 1.0.0
)

# Add source files
target_sources(LatencyMeasurementApp PRIVATE
    Main.cpp
    MainComponent.cpp
    MainComponent.h
    StartupDialog.cpp
    StartupDialog.h
    LatencyMeasurementEngine.cpp
    LatencyMeasurementEngine.h
    SweepGenerator.cpp
    SweepGenerator.h
)

# Link to flowerjuce library for utilities
target_link_libraries(LatencyMeasurementApp PRIVATE
    flowerjuce
)

# Link JUCE modules
target_link_libraries(LatencyMeasurementApp PRIVATE
    juce::juce_audio_basics
    juce::juce_audio_devices
    juce::juce_audio_formats
    juce::juce_audio_utils
    juce::juce_core
    juce::juce_data_structures
    juce::juce_dsp
    juce::juce_events
    juce::juce_graphics
    juce::juce_gui_basics
    juce::juce_gui_extra
)

# Compile definitions
target_compile_definitions(LatencyMeasurementApp PRIVATE
    JUCE_WEB_BROWSER=0
    JUCE_USE_CURL=0
    JUCE_APPLICATION_NAME_STRING="$<TARGET_PROPERTY:LatencyMeasurementApp,JUCE_PRODUCT_NAME>"
    JUCE_APPLICATION_VERSION_STRING="$<TARGET_PROPERTY:LatencyMeasurementApp,JUCE_VERSION>"
)
