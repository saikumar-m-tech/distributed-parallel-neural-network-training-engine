#include <cuda_runtime.h>

#include "kernel_utils.cuh"

namespace {

__global__ void bn_mean_var_kernel(
	const float* x,
	float* mean,
	float* inv_std,
	int batch,
	int features,
	float epsilon) {
	int feature = blockIdx.x;
	if (feature >= features) {
		return;
	}

	extern __shared__ float shared[];
	float* shared_sum = shared;
	float* shared_sumsq = shared + blockDim.x;

	float sum = 0.0f;
	float sumsq = 0.0f;
	for (int i = threadIdx.x; i < batch; i += blockDim.x) {
		float value = x[i * features + feature];
		sum += value;
		sumsq += value * value;
	}

	shared_sum[threadIdx.x] = sum;
	shared_sumsq[threadIdx.x] = sumsq;
	__syncthreads();

	for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
		if (threadIdx.x < stride) {
			shared_sum[threadIdx.x] += shared_sum[threadIdx.x + stride];
			shared_sumsq[threadIdx.x] += shared_sumsq[threadIdx.x + stride];
		}
		__syncthreads();
	}

	if (threadIdx.x == 0) {
		float mean_val = shared_sum[0] / static_cast<float>(batch);
		float var = shared_sumsq[0] / static_cast<float>(batch) - mean_val * mean_val;
		mean[feature] = mean_val;
		inv_std[feature] = rsqrtf(var + epsilon);
	}
}

__global__ void bn_forward_kernel(
	const float* x,
	float* y,
	float* x_hat,
	const float* mean,
	const float* inv_std,
	const float* gamma,
	const float* beta,
	int total,
	int features) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= total) {
		return;
	}
	int feature = idx % features;
	float normalized = (x[idx] - mean[feature]) * inv_std[feature];
	x_hat[idx] = normalized;
	y[idx] = normalized * gamma[feature] + beta[feature];
}

__global__ void bn_param_grad_kernel(
	const float* grad_out,
	const float* x_hat,
	float* dgamma,
	float* dbeta,
	int batch,
	int features) {
	int feature = blockIdx.x;
	if (feature >= features) {
		return;
	}

	extern __shared__ float shared[];
	float* shared_sum = shared;
	float* shared_sumxh = shared + blockDim.x;

	float sum = 0.0f;
	float sumxh = 0.0f;
	for (int i = threadIdx.x; i < batch; i += blockDim.x) {
		int idx = i * features + feature;
		float grad = grad_out[idx];
		sum += grad;
		sumxh += grad * x_hat[idx];
	}

	shared_sum[threadIdx.x] = sum;
	shared_sumxh[threadIdx.x] = sumxh;
	__syncthreads();

	for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
		if (threadIdx.x < stride) {
			shared_sum[threadIdx.x] += shared_sum[threadIdx.x + stride];
			shared_sumxh[threadIdx.x] += shared_sumxh[threadIdx.x + stride];
		}
		__syncthreads();
	}

	if (threadIdx.x == 0) {
		dbeta[feature] = shared_sum[0];
		dgamma[feature] = shared_sumxh[0];
	}
}

__global__ void bn_input_grad_kernel(
	const float* grad_out,
	const float* x_hat,
	const float* gamma,
	const float* inv_std,
	const float* dgamma,
	const float* dbeta,
	float* grad_in,
	int total,
	int batch,
	int features) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= total) {
		return;
	}
	int feature = idx % features;
	float n = static_cast<float>(batch);
	float grad = grad_out[idx];
	float term = n * grad - dbeta[feature] - x_hat[idx] * dgamma[feature];
	grad_in[idx] = (gamma[feature] * inv_std[feature] / n) * term;
}

} // namespace

void batch_norm_forward_gpu(
	const float* input,
	float* output,
	float* x_hat,
	float* mean,
	float* inv_std,
	const float* gamma,
	const float* beta,
	int batch,
	int features,
	float epsilon) {
	int block = 256;
	int grid = features;
	int shared = block * static_cast<int>(sizeof(float)) * 2;
	bn_mean_var_kernel<<<grid, block, shared>>>(input, mean, inv_std, batch, features, epsilon);
	CUDA_CHECK(cudaGetLastError());

	int total = batch * features;
	int grid_total = (total + block - 1) / block;
	bn_forward_kernel<<<grid_total, block>>>(
		input,
		output,
		x_hat,
		mean,
		inv_std,
		gamma,
		beta,
		total,
		features);
	CUDA_CHECK(cudaGetLastError());
}

void batch_norm_backward_gpu(
	const float* grad_out,
	const float* x_hat,
	const float* gamma,
	const float* inv_std,
	float* dgamma,
	float* dbeta,
	float* grad_in,
	int batch,
	int features) {
	int block = 256;
	int grid = features;
	int shared = block * static_cast<int>(sizeof(float)) * 2;
	bn_param_grad_kernel<<<grid, block, shared>>>(
		grad_out,
		x_hat,
		dgamma,
		dbeta,
		batch,
		features);
	CUDA_CHECK(cudaGetLastError());

	int total = batch * features;
	int grid_total = (total + block - 1) / block;
	bn_input_grad_kernel<<<grid_total, block>>>(
		grad_out,
		x_hat,
		gamma,
		inv_std,
		dgamma,
		dbeta,
		grad_in,
		total,
		batch,
		features);
	CUDA_CHECK(cudaGetLastError());
}
