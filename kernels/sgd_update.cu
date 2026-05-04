#include <cuda_runtime.h>

#include "kernel_utils.cuh"

extern "C" __global__ void sgd_update(float* weights, const float* gradients,
									float learning_rate, int n) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;
	for (int i = idx; i < n; i += stride) {
		weights[i] -= learning_rate * gradients[i];
	}
}

void sgd_update_gpu(float* weights, const float* gradients, float learning_rate, int n) {
	int block = 256;
	int grid = (n + block - 1) / block;
	sgd_update<<<grid, block>>>(weights, gradients, learning_rate, n);
	CUDA_CHECK(cudaGetLastError());
}
