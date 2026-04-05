#include "vampnet_inference/VampNetInference.hpp"

#include <iostream>

int main(int argc, char* argv[]) {
    (void)argc;
    (void)argv;
    std::cout << vampnet::version_string() << "\n";
    vampnet::VampNetInference inf;
    std::cout << "ready=" << (inf.is_ready() ? "yes" : "no") << "\n";
    return 0;
}
