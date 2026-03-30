# StreamGen Live — real-time generative drum accompaniment

juce_add_gui_app(StreamGenApp
    COMPANY_NAME "Unsound"
    PRODUCT_NAME "StreamGen"
    VERSION 0.1.0
    BUNDLE_ID "com.unsound.streamgen"
    MICROPHONE_PERMISSION_ENABLED TRUE
    MICROPHONE_PERMISSION_TEXT "StreamGen needs microphone access for live saxophone input"
)

target_sources(StreamGenApp PRIVATE
    Main.cpp
    StreamGenProcessor.h
    StreamGenProcessor.cpp
    StreamGenComponent.h
    StreamGenComponent.cpp
    InferenceWorker.h
    InferenceWorker.cpp
    GenerationScheduler.h
    GenerationScheduler.cpp
    TimeRuler.h
    WaveformTimelineComponent.h
    WaveformTimelineComponent.cpp
    StageTimingComponent.h
    StageTimingComponent.cpp
    GenerationStatusComponent.h
    GenerationStatusComponent.cpp
    ControlsComponent.h
    ControlsComponent.cpp
    MixerComponent.h
    MixerComponent.cpp
    SimulationWindow.h
    SimulationWindow.cpp
)

target_link_libraries(StreamGenApp PRIVATE
    sao_inference
    juce::juce_audio_basics
    juce::juce_audio_devices
    juce::juce_audio_formats
    juce::juce_audio_utils
    juce::juce_core
    juce::juce_events
    juce::juce_graphics
    juce::juce_gui_basics
    juce::juce_gui_extra
)

target_compile_definitions(StreamGenApp PRIVATE
    JUCE_WEB_BROWSER=0
    JUCE_USE_CURL=0
)
