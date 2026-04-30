#include "kernel_utils.cuh"

/**
 * @brief Simple hello kernel that prints thread index
 * Designed to run with 4 threads per block
 */
__global__ void hello_kernel() {
    int tid = threadIdx.x;
    if (tid < 4) {
        printf("Hello from thread %d (block %d)\n", tid, blockIdx.x);
    }
}

int main() {
    printf("CUDA Device Query and Hello Kernel Test\n");
    printf("========================================\n\n");
    
    // Get and print device information
    get_device_info();
    
    // Run hello kernel with 4 threads
    printf("Launching hello kernel with 4 threads...\n");
    hello_kernel<<<1, 4>>>();
    
    // Synchronize and check for errors
    CUDA_CHECK(cudaDeviceSynchronize());
    
    printf("\nHello kernel completed successfully!\n");
    
    return 0;
}
