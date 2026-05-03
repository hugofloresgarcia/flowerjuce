cmake_minimum_required(VERSION 3.22)

add_executable(sao_cli Main.cpp)

target_link_libraries(sao_cli PRIVATE sao_inference cnpy-static)
target_include_directories(sao_cli PRIVATE "${SAO_CNPY_SOURCE_DIR}")

target_compile_features(sao_cli PRIVATE cxx_std_17)
