cmake_minimum_required(VERSION 3.20)
project(DistributedNeuralNetworkKernels CUDA CXX)

# Set C++ standard to C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Find CUDA 12+
find_package(CUDA 12.0 REQUIRED)

# Find MPI
find_package(MPI REQUIRED)

# Enable CUDA separable compilation for linking flexibility
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)

# Set target GPU compute capability
# GTX 1650 has compute capability 7.5
set(CMAKE_CUDA_ARCHITECTURES 75)

# Add test_device executable
add_executable(test_device test_device.cu)

# Set CUDA as the source language for linking
set_target_properties(test_device PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES 75
)

# Link against CUDA libraries
target_link_libraries(test_device 
    PRIVATE 
    CUDA::cudart
    CUDA::cuda_driver
)

# Optional: Link MPI if needed for GPU communication tests in the future
target_link_libraries(test_device PRIVATE MPI::MPI_CXX)

# Include directories for this project
target_include_directories(test_device PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

# Compiler flags for better diagnostics
if(MSVC)
    target_compile_options(test_device PRIVATE /W4)
else()
    target_compile_options(test_device PRIVATE -Wall -Wextra -Wpedantic)
endif()

# CUDA specific compiler flags
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -lineinfo")

# Print configuration info
message(STATUS "CUDA Compute Capability: 7.5 (NVIDIA GeForce GTX 1650)")
message(STATUS "CUDA Separable Compilation: Enabled")
message(STATUS "C++ Standard: 17")
message(STATUS "MPI Support: Enabled")
