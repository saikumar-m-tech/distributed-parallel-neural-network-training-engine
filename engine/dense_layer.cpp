#include "dense_layer.hpp"

#include <cmath>
#include <cstdio>
#include <random>

#include <cuda_runtime.h>

#include "kernel_utils.cuh"

void tiled_matmul_gpu(const float* A, const float* B, float* C,
					  int M, int N, int K);
void relu_forward_gpu(float* x, int n);
void relu_backward_gpu(const float* x_pre, float* dx, int n);
void bias_add_gpu(float* out, const float* bias, int batch, int features);
void sum_over_batch_gpu(const float* grad, float* db, int batch, int features);
void transpose_matrix_gpu(const float* in, float* out, int rows, int cols);

Dense::Dense(int in_features, int out_features)
	: in_features_(in_features),
	  out_features_(out_features),
	  weights_(static_cast<size_t>(out_features) * in_features),
	  bias_(static_cast<size_t>(out_features)),
	  dweights_(static_cast<size_t>(out_features) * in_features),
	  dbias_(static_cast<size_t>(out_features)),
	  cache_(),
	  last_batch_(0) {
	std::mt19937 rng(1337);
	float stddev = std::sqrt(2.0f / static_cast<float>(in_features_));
	std::normal_distribution<float> dist(0.0f, stddev);

	std::vector<float> host_weights(weights_.size());
	for (size_t i = 0; i < host_weights.size(); ++i) {
		float value = 0.0f;
		do {
			value = dist(rng);
		} while (value == 0.0f || std::fabs(value) > 0.2f);
		host_weights[i] = value;
	}

	std::vector<float> host_bias(bias_.size(), 0.0f);
	weights_.copy_from_host(host_weights.data(), host_weights.size());
	bias_.copy_from_host(host_bias.data(), host_bias.size());
}

void Dense::ensure_cache(size_t batch) {
	size_t input_size = batch * static_cast<size_t>(in_features_);
	size_t pre_relu_size = batch * static_cast<size_t>(out_features_);
	size_t grad_size = pre_relu_size;
	size_t grad_t_size = static_cast<size_t>(out_features_) * batch;
	size_t weights_t_size = static_cast<size_t>(in_features_) * out_features_;

	if (cache_.input_size != input_size) {
		cache_.input = FloatBuffer(input_size);
		cache_.input_size = input_size;
	}
	if (cache_.pre_relu_size != pre_relu_size) {
		cache_.pre_relu = FloatBuffer(pre_relu_size);
		cache_.pre_relu_size = pre_relu_size;
	}
	if (cache_.grad_size != grad_size) {
		cache_.grad_relu = FloatBuffer(grad_size);
		cache_.grad_size = grad_size;
	}
	if (cache_.grad_t_size != grad_t_size) {
		cache_.grad_relu_t = FloatBuffer(grad_t_size);
		cache_.grad_t_size = grad_t_size;
	}
	if (cache_.weights_t_size != weights_t_size) {
		cache_.weights_t = FloatBuffer(weights_t_size);
		cache_.weights_t_size = weights_t_size;
	}
}

void Dense::forward(const FloatBuffer& in, FloatBuffer& out) {
	if (in_features_ <= 0 || out_features_ <= 0) {
		return;
	}
	if (in.size() % static_cast<size_t>(in_features_) != 0) {
		fprintf(stderr, "Dense forward shape mismatch: input size %zu\n", in.size());
		return;
	}
	if (out.size() == 0 || out.size() % static_cast<size_t>(out_features_) != 0) {
		fprintf(stderr, "Dense forward shape mismatch: output size %zu\n", out.size());
		return;
	}
	static bool logged = false;
	if (!logged) {
		printf("running GPU forward\n");
		logged = true;
	}

	last_batch_ = in.size() / static_cast<size_t>(in_features_);
	ensure_cache(last_batch_);
	DenseCache& cache = cache_;
	CUDA_CHECK(cudaMemcpy(cache.input.data(), in.data(), sizeof(float) * cache.input_size,
					cudaMemcpyDeviceToDevice));

	transpose_matrix_gpu(weights_.data(), cache.weights_t.data(), out_features_, in_features_);
	tiled_matmul_gpu(cache.input.data(), cache.weights_t.data(), cache.pre_relu.data(),
					static_cast<int>(last_batch_), out_features_, in_features_);
	bias_add_gpu(cache.pre_relu.data(), bias_.data(), static_cast<int>(last_batch_), out_features_);

	CUDA_CHECK(cudaMemcpy(out.data(), cache.pre_relu.data(), sizeof(float) * cache.pre_relu_size,
					cudaMemcpyDeviceToDevice));
	relu_forward_gpu(out.data(), static_cast<int>(cache.pre_relu_size));
}

void Dense::backward(const FloatBuffer& grad_out, FloatBuffer& grad_in) {
	if (last_batch_ == 0) {
		fprintf(stderr, "Dense backward called without forward cache\n");
		return;
	}
	if (grad_out.size() != last_batch_ * static_cast<size_t>(out_features_)) {
		fprintf(stderr, "Dense backward shape mismatch: grad_out size %zu\n", grad_out.size());
		return;
	}
	if (grad_in.size() != last_batch_ * static_cast<size_t>(in_features_)) {
		fprintf(stderr, "Dense backward shape mismatch: grad_in size %zu\n", grad_in.size());
		return;
	}

	ensure_cache(last_batch_);
	DenseCache& cache = cache_;
	CUDA_CHECK(cudaMemcpy(cache.grad_relu.data(), grad_out.data(), sizeof(float) * cache.grad_size,
					cudaMemcpyDeviceToDevice));
	relu_backward_gpu(cache.pre_relu.data(), cache.grad_relu.data(), static_cast<int>(cache.grad_size));

	transpose_matrix_gpu(cache.grad_relu.data(), cache.grad_relu_t.data(),
					static_cast<int>(last_batch_), out_features_);
	tiled_matmul_gpu(cache.grad_relu_t.data(), cache.input.data(), dweights_.data(),
					out_features_, in_features_, static_cast<int>(last_batch_));

	sum_over_batch_gpu(cache.grad_relu.data(), dbias_.data(),
					static_cast<int>(last_batch_), out_features_);

	tiled_matmul_gpu(cache.grad_relu.data(), weights_.data(), grad_in.data(),
					static_cast<int>(last_batch_), in_features_, out_features_);
}

std::vector<FloatBuffer*> Dense::parameters() {
	return {&weights_, &bias_};
}

std::vector<FloatBuffer*> Dense::gradients() {
	return {&dweights_, &dbias_};
}

std::vector<const FloatBuffer*> Dense::parameters() const {
	return {&weights_, &bias_};
}

std::vector<const FloatBuffer*> Dense::gradients() const {
	return {&dweights_, &dbias_};
}

int Dense::in_features() const {
	return in_features_;
}

int Dense::out_features() const {
	return out_features_;
}
