# WhAM — tape looper + VampNet (embedded ORT + Gradio fallback). Stub links vampnet_inference.

add_executable(WhamStub
    Main.cpp
)

target_include_directories(WhamStub PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../../libs/vampnet_inference/include
)

target_link_libraries(WhamStub PRIVATE
    vampnet_inference
)
