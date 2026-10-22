#include "experiment.hpp"

#include <exception>
#include <iostream>

int main(int argc, char** argv) {
    try {
        event_voxel::experiment::Experiment(event_voxel::experiment::parse(argc, argv)).run();
    } catch (const std::exception& error) {
        std::cerr << "ERROR: " << error.what() << '\n';
        return 1;
    }
}
