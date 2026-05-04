#include "batch_norm_layer.hpp"

#include <algorithm>
#include <cstdio>
#include <vector>

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
	float epsilon);

void batch_norm_backward_gpu(
	const float* grad_out,
	const float* x_hat,
	const float* gamma,
	const float* inv_std,
	float* dgamma,
	float* dbeta,
	float* grad_in,
	int batch,
	int features);

BatchNorm::BatchNorm()
	: features_(0),
	  epsilon_(1e-5f),
	  gamma_(),
	  beta_(),
	  dgamma_(),
	  dbeta_(),
	  mean_(),
	  inv_std_(),
	  x_hat_(),
	  last_batch_(0) {}

BatchNorm::BatchNorm(int features, float epsilon)
	: features_(features),
	  epsilon_(epsilon),
	  gamma_(static_cast<size_t>(features)),
	  beta_(static_cast<size_t>(features)),
	  dgamma_(static_cast<size_t>(features)),
	  dbeta_(static_cast<size_t>(features)),
	  mean_(static_cast<size_t>(features)),
	  inv_std_(static_cast<size_t>(features)),
	  x_hat_(),
	  last_batch_(0) {
	std::vector<float> host_gamma(static_cast<size_t>(features), 1.0f);
	std::vector<float> host_beta(static_cast<size_t>(features), 0.0f);
	gamma_.copy_from_host(host_gamma.data(), host_gamma.size());
	beta_.copy_from_host(host_beta.data(), host_beta.size());
}

void BatchNorm::ensure_cache(size_t batch) {
	size_t total = batch * static_cast<size_t>(features_);
	if (x_hat_.size() != total) {
		x_hat_ = FloatBuffer(total);
	}
	if (mean_.size() != static_cast<size_t>(features_)) {
		mean_ = FloatBuffer(static_cast<size_t>(features_));
	}
	if (inv_std_.size() != static_cast<size_t>(features_)) {
		inv_std_ = FloatBuffer(static_cast<size_t>(features_));
	}
}

void BatchNorm::forward(const FloatBuffer& in, FloatBuffer& out) {
	if (features_ <= 0) {
		return;
	}
	if (in.size() % static_cast<size_t>(features_) != 0) {
		std::fprintf(stderr, "BatchNorm forward shape mismatch: input size %zu\n", in.size());
		return;
	}
	if (out.size() != in.size()) {
		std::fprintf(stderr, "BatchNorm forward shape mismatch: output size %zu\n", out.size());
		return;
	}

	last_batch_ = in.size() / static_cast<size_t>(features_);
	ensure_cache(last_batch_);

	batch_norm_forward_gpu(
		in.data(),
		out.data(),
		x_hat_.data(),
		mean_.data(),
		inv_std_.data(),
		gamma_.data(),
		beta_.data(),
		static_cast<int>(last_batch_),
		features_,
		epsilon_);
}

void BatchNorm::backward(const FloatBuffer& grad_out, FloatBuffer& grad_in) {
	if (last_batch_ == 0) {
		std::fprintf(stderr, "BatchNorm backward called without forward cache\n");
		return;
	}
	if (grad_out.size() != last_batch_ * static_cast<size_t>(features_)) {
		std::fprintf(stderr, "BatchNorm backward shape mismatch: grad_out size %zu\n", grad_out.size());
		return;
	}
	if (grad_in.size() != grad_out.size()) {
		std::fprintf(stderr, "BatchNorm backward shape mismatch: grad_in size %zu\n", grad_in.size());
		return;
	}

	batch_norm_backward_gpu(
		grad_out.data(),
		x_hat_.data(),
		gamma_.data(),
		inv_std_.data(),
		dgamma_.data(),
		dbeta_.data(),
		grad_in.data(),
		static_cast<int>(last_batch_),
		features_);
}

std::vector<FloatBuffer*> BatchNorm::parameters() {
	return {&gamma_, &beta_};
}

std::vector<FloatBuffer*> BatchNorm::gradients() {
	return {&dgamma_, &dbeta_};
}

std::vector<const FloatBuffer*> BatchNorm::parameters() const {
	return {&gamma_, &beta_};
}

std::vector<const FloatBuffer*> BatchNorm::gradients() const {
	return {&dgamma_, &dbeta_};
}

int BatchNorm::features() const {
	return features_;
}
