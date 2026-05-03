cmake_minimum_required(VERSION 3.20)
project(DistributedNeuralNetworkTraining CUDA CXX)

# Set C++ standard to C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Find CUDA 12+ using new native CUDA support
find_package(CUDAToolkit REQUIRED)

# Disable CUDA separable compilation for simpler linking
set(CMAKE_CUDA_SEPARABLE_COMPILATION OFF)

# Set target GPU compute capability (GTX 1650 = 7.5)
set(CMAKE_CUDA_ARCHITECTURES 75)

# CUDA kernel library
add_library(cuda_kernels 
    kernels/matmul.cu
    kernels/activations.cu
)
set_target_properties(cuda_kernels PROPERTIES
    LANGUAGE CUDA
)
target_include_directories(cuda_kernels PRIVATE 
    ${CMAKE_CURRENT_SOURCE_DIR}/kernels
)
target_link_libraries(cuda_kernels PUBLIC CUDA::cudart)

# Engine library (with C++ sources only)
add_library(engine_lib
    engine/dense_layer.cpp
    engine/network.cpp
)
set_target_properties(engine_lib PROPERTIES
    LINKER_LANGUAGE CXX
)
target_include_directories(engine_lib PRIVATE 
    ${CMAKE_CURRENT_SOURCE_DIR}/engine
    ${CMAKE_CURRENT_SOURCE_DIR}/kernels
)
target_compile_definitions(engine_lib PRIVATE PARALLELNET_NO_MPI=1)
target_link_libraries(engine_lib PUBLIC cuda_kernels CUDA::cudart)

# Add test_network executable (C++ source only)
add_executable(test_network tests/test_network.cpp)
set_target_properties(test_network PROPERTIES
    LINKER_LANGUAGE CXX
)
target_include_directories(test_network PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/engine
    ${CMAKE_CURRENT_SOURCE_DIR}/kernels
)
target_compile_definitions(test_network PRIVATE PARALLELNET_NO_MPI=1)
target_link_libraries(test_network PRIVATE engine_lib cuda_kernels CUDA::cudart)

# Add test_exception executable (C++ source only)
add_executable(test_exception tests/test_exception.cpp)
set_target_properties(test_exception PROPERTIES
    LINKER_LANGUAGE CXX
)
target_include_directories(test_exception PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/kernels
)
target_link_libraries(test_exception PRIVATE cuda_kernels CUDA::cudart)

# Compiler flags
if(MSVC)
    target_compile_options(test_network PRIVATE /W4)
    target_compile_options(test_exception PRIVATE /W4)
else()
    target_compile_options(test_network PRIVATE -Wall -Wextra -Wpedantic)
    target_compile_options(test_exception PRIVATE -Wall -Wextra -Wpedantic)
endif()

# CUDA specific flags
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -lineinfo")

message(STATUS "CUDA Compute Capability: 7.5 (NVIDIA GeForce GTX 1650)")
message(STATUS "CUDA Separable Compilation: Enabled")
message(STATUS "C++ Standard: 17")
