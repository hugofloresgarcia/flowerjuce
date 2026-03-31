# StreamGen CLI — headless batch streaming for testing on CUDA servers
#
# Reuses the StreamGen GUI app's core engine (processor, worker, scheduler)
# but replaces the audio device + UI with file I/O.

juce_add_console_app(StreamGenCLI
    COMPANY_NAME "Unsound"
    PRODUCT_NAME "StreamGenCLI"
    VERSION 0.1.0
)

# Shared sources from the GUI app (engine, no UI components)
set(STREAMGEN_APP_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../streamgen)

target_sources(StreamGenCLI PRIVATE
    Main.cpp
    ${STREAMGEN_APP_DIR}/StreamGenDebugLog.h
    ${STREAMGEN_APP_DIR}/StreamGenProcessor.h
    ${STREAMGEN_APP_DIR}/StreamGenProcessor.cpp
    ${STREAMGEN_APP_DIR}/InferenceWorker.h
    ${STREAMGEN_APP_DIR}/InferenceWorker.cpp
    ${STREAMGEN_APP_DIR}/PreVaeDebugDump.h
    ${STREAMGEN_APP_DIR}/PreVaeDebugDump.cpp
    ${STREAMGEN_APP_DIR}/GenerationScheduler.h
    ${STREAMGEN_APP_DIR}/GenerationScheduler.cpp
    ${STREAMGEN_APP_DIR}/GenerationTimelineStore.h
    ${STREAMGEN_APP_DIR}/GenerationTimelineStore.cpp
    ${STREAMGEN_APP_DIR}/TimeRuler.h
)

target_include_directories(StreamGenCLI PRIVATE
    ${STREAMGEN_APP_DIR}
)

target_link_libraries(StreamGenCLI PRIVATE
    sao_inference
    juce::juce_audio_basics
    juce::juce_audio_devices
    juce::juce_audio_formats
    juce::juce_audio_utils
    juce::juce_core
    juce::juce_events
)

target_compile_definitions(StreamGenCLI PRIVATE
    JUCE_WEB_BROWSER=0
    JUCE_USE_CURL=0
)
