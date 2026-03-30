cmake_minimum_required(VERSION 3.22)

add_executable(zenon_cli Main.cpp)

target_link_libraries(zenon_cli PRIVATE sao_inference)
target_include_directories(zenon_cli PRIVATE ${cnpy_SOURCE_DIR})

target_compile_features(zenon_cli PRIVATE cxx_std_17)
