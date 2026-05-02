#include <cuda_runtime.h>

#include "kernel_utils.cuh"

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

extern "C" __global__ void bias_add(float* out, const float* bias,
									int batch, int features) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int total = batch * features;
	if (idx < total) {
		int feature = idx % features;
		out[idx] += bias[feature];
	}
}

extern "C" __global__ void sum_over_batch(const float* grad, float* db,
									int batch, int features) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < features) {
		float sum = 0.0f;
		int offset = idx;
		for (int b = 0; b < batch; ++b) {
			sum += grad[offset + b * features];
		}
		db[idx] = sum;
	}
}

extern "C" __global__ void transpose_matrix(const float* in, float* out,
									int rows, int cols) {
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	if (row < rows && col < cols) {
		out[col * rows + row] = in[row * cols + col];
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

extern "C" __global__ void grad_logits_kernel(const float* probs, const int* labels,
									float* grad_logits, int batch, int classes) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int total = batch * classes;
	if (idx < total) {
		int b = idx / classes;
		int o = idx % classes;
		float grad = probs[idx];
		if (o == labels[b]) {
			grad -= 1.0f;
		}
		grad_logits[idx] = grad / static_cast<float>(batch);
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

void relu_forward_gpu(float* x, int n) {
	int block = 256;
	int grid = (n + block - 1) / block;
	relu_forward<<<grid, block>>>(x, n);
	CUDA_CHECK(cudaGetLastError());
}

void relu_backward_gpu(const float* x_pre, float* dx, int n) {
	int block = 256;
	int grid = (n + block - 1) / block;
	relu_backward<<<grid, block>>>(x_pre, dx, n);
	CUDA_CHECK(cudaGetLastError());
}

void bias_add_gpu(float* out, const float* bias, int batch, int features) {
	int total = batch * features;
	int block = 256;
	int grid = (total + block - 1) / block;
	bias_add<<<grid, block>>>(out, bias, batch, features);
	CUDA_CHECK(cudaGetLastError());
}

void sum_over_batch_gpu(const float* grad, float* db, int batch, int features) {
	int block = 256;
	int grid = (features + block - 1) / block;
	sum_over_batch<<<grid, block>>>(grad, db, batch, features);
	CUDA_CHECK(cudaGetLastError());
}

void softmax_forward_gpu(const float* in, float* out, int batch_size, int num_classes) {
	int block = 256;
	softmax_forward<<<batch_size, block, block * sizeof(float)>>>(in, out, batch_size, num_classes);
	CUDA_CHECK(cudaGetLastError());
}

void grad_logits_gpu(const float* probs, const int* labels, float* grad_logits,
					int batch, int classes) {
	int total = batch * classes;
	int block = 256;
	int grid = (total + block - 1) / block;
	grad_logits_kernel<<<grid, block>>>(probs, labels, grad_logits, batch, classes);
	CUDA_CHECK(cudaGetLastError());
}

void transpose_matrix_gpu(const float* in, float* out, int rows, int cols) {
	dim3 block(16, 16);
	dim3 grid((cols + block.x - 1) / block.x, (rows + block.y - 1) / block.y);
	transpose_matrix<<<grid, block>>>(in, out, rows, cols);
	CUDA_CHECK(cudaGetLastError());
}
