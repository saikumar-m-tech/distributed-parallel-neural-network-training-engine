#include <cuda_runtime.h>

extern "C" __global__ void relu_forward(float* x, int n) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;
	for (int i = idx; i < n; i += stride) {
		float val = x[i];
		x[i] = val > 0.0f ? val : 0.0f;
	}
}

extern "C" __global__ void relu_backward(const float* x_pre, float* dx, int n) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;
	for (int i = idx; i < n; i += stride) {
		dx[i] = x_pre[i] > 0.0f ? dx[i] : 0.0f;
	}
}

extern "C" __global__ void softmax_forward(const float* in, float* out,
										int batch_size, int num_classes) {
	int sample = blockIdx.x;
	if (sample >= batch_size) {
		return;
	}

	const float* in_row = in + sample * num_classes;
	float* out_row = out + sample * num_classes;

	extern __shared__ float shared[];
	int tid = threadIdx.x;

	float local_max = -1.0e20f;
	for (int i = tid; i < num_classes; i += blockDim.x) {
		local_max = fmaxf(local_max, in_row[i]);
	}
	shared[tid] = local_max;
	__syncthreads();

	for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
		if (tid < stride) {
			shared[tid] = fmaxf(shared[tid], shared[tid + stride]);
		}
		__syncthreads();
	}

	float max_val = shared[0];
	float local_sum = 0.0f;
	for (int i = tid; i < num_classes; i += blockDim.x) {
		local_sum += expf(in_row[i] - max_val);
	}
	shared[tid] = local_sum;
	__syncthreads();

	for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
		if (tid < stride) {
			shared[tid] += shared[tid + stride];
		}
		__syncthreads();
	}

	float sum_val = shared[0];
	for (int i = tid; i < num_classes; i += blockDim.x) {
		out_row[i] = expf(in_row[i] - max_val) / sum_val;
	}
}

extern "C" __global__ void cross_entropy_loss(const float* probs, const int* labels,
										float* loss, int batch_size, int n_classes) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;

	float local_sum = 0.0f;
	for (int i = idx; i < batch_size; i += stride) {
		int label = labels[i];
		float prob = probs[i * n_classes + label];
		local_sum += -logf(prob);
	}

	extern __shared__ float shared[];
	shared[threadIdx.x] = local_sum;
	__syncthreads();

	for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
		if (threadIdx.x < offset) {
			shared[threadIdx.x] += shared[threadIdx.x + offset];
		}
		__syncthreads();
	}

	if (threadIdx.x == 0) {
		atomicAdd(loss, shared[0] / static_cast<float>(batch_size));
	}
}
