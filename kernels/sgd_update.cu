#include <cuda_runtime.h>

extern "C" __global__ void sgd_update(float* weights, const float* gradients,
									float learning_rate, int n) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;
	for (int i = idx; i < n; i += stride) {
		weights[i] -= learning_rate * gradients[i];
	}
}
