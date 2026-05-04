#include "network.hpp"

#include "../mpi/gradient_sync.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <random>
#include <chrono>

#include <cuda_runtime.h>

#include "kernel_utils.cuh"

void tiled_matmul_gpu(const float* A, const float* B, float* C,
					  int M, int N, int K);
void softmax_forward_gpu(const float* in, float* out, int batch_size, int num_classes);
void grad_logits_gpu(const float* probs, const int* labels, float* grad_logits,
					int batch, int classes);
void bias_add_gpu(float* out, const float* bias, int batch, int features);
void sum_over_batch_gpu(const float* grad, float* db, int batch, int features);
void transpose_matrix_gpu(const float* in, float* out, int rows, int cols);
void sgd_update_gpu(float* weights, const float* gradients, float learning_rate, int n);

Network::Network(std::vector<Dense> layers)
	: layers_(std::move(layers)),
	  last_probs_(),
	  last_labels_(),
	  out_weights_(),
	  out_bias_(),
	  dout_weights_(),
	  dout_bias_(),
	  cache_(),
	  last_batch_(0),
	  in_features_(0),
	  hidden_features_(0),
	  out_features_(0),
	  timing_() {
	if (layers_.size() < 2) {
		std::fprintf(stderr, "Network requires two Dense layers\n");
		return;
	}
	in_features_ = layers_[0].in_features();
	hidden_features_ = layers_[0].out_features();
	out_features_ = layers_[1].out_features();

	out_weights_ = FloatBuffer(static_cast<size_t>(out_features_) * hidden_features_);
	out_bias_ = FloatBuffer(static_cast<size_t>(out_features_));
	dout_weights_ = FloatBuffer(static_cast<size_t>(out_features_) * hidden_features_);
	dout_bias_ = FloatBuffer(static_cast<size_t>(out_features_));

	std::mt19937 rng(1337);
	float stddev = 0.7f / std::sqrt(static_cast<float>(hidden_features_));
	std::normal_distribution<float> dist(0.0f, stddev);

	std::vector<float> host_out_weights(static_cast<size_t>(out_features_) * hidden_features_);
	for (float& value : host_out_weights) {
		float sample = 0.0f;
		do {
			sample = dist(rng);
		} while (sample == 0.0f || std::fabs(sample) > 0.2f);
		value = sample;
	}
	std::vector<float> host_out_bias(static_cast<size_t>(out_features_), 0.0f);
	out_weights_.copy_from_host(host_out_weights.data(), host_out_weights.size());
	out_bias_.copy_from_host(host_out_bias.data(), host_out_bias.size());
	batch_norm_ = BatchNorm(hidden_features_);
}

void Network::ensure_cache(size_t batch) {
	const size_t hidden_size = batch * static_cast<size_t>(hidden_features_);
	const size_t logits_size = batch * static_cast<size_t>(out_features_);
	const size_t grad_logits_t_size = static_cast<size_t>(out_features_) * batch;
	const size_t out_weights_t_size = static_cast<size_t>(hidden_features_) * out_features_;

	if (cache_.hidden.size() != hidden_size) {
		cache_.hidden = FloatBuffer(hidden_size);
	}
	if (cache_.logits.size() != logits_size) {
		cache_.logits = FloatBuffer(logits_size);
	}
	if (cache_.probs.size() != logits_size) {
		cache_.probs = FloatBuffer(logits_size);
	}
	if (cache_.grad_logits.size() != logits_size) {
		cache_.grad_logits = FloatBuffer(logits_size);
	}
	if (cache_.grad_logits_t.size() != grad_logits_t_size) {
		cache_.grad_logits_t = FloatBuffer(grad_logits_t_size);
	}
	if (cache_.out_weights_t.size() != out_weights_t_size) {
		cache_.out_weights_t = FloatBuffer(out_weights_t_size);
	}
	cache_.cached_batch = batch;
}

float Network::forward(const FloatBuffer& input, const LabelBuffer& labels) {
	if (input.size() % static_cast<size_t>(in_features_) != 0) {
		std::fprintf(stderr, "Network forward input shape mismatch\n");
		return 0.0f;
	}

	last_batch_ = input.size() / static_cast<size_t>(in_features_);
	last_labels_.assign(labels.size(), 0);
	labels.copy_to_host(last_labels_.data(), last_labels_.size());
	const size_t hidden_size = last_batch_ * static_cast<size_t>(hidden_features_);
	const size_t logits_size = last_batch_ * static_cast<size_t>(out_features_);

	GpuTimer timer;
	timer.start();
	FloatBuffer hidden_buf(hidden_size);
	layers_[0].forward(input, hidden_buf);
	FloatBuffer norm_buf(hidden_size);
	batch_norm_.forward(hidden_buf, norm_buf);

	ensure_cache(last_batch_);
	CUDA_CHECK(cudaMemcpy(cache_.hidden.data(), norm_buf.data(),
					sizeof(float) * cache_.hidden.size(), cudaMemcpyDeviceToDevice));

	transpose_matrix_gpu(out_weights_.data(), cache_.out_weights_t.data(),
					out_features_, hidden_features_);
	tiled_matmul_gpu(cache_.hidden.data(), cache_.out_weights_t.data(), cache_.logits.data(),
					static_cast<int>(last_batch_), out_features_, hidden_features_);
	bias_add_gpu(cache_.logits.data(), out_bias_.data(), static_cast<int>(last_batch_), out_features_);

	softmax_forward_gpu(cache_.logits.data(), cache_.probs.data(),
					static_cast<int>(last_batch_), out_features_);

	last_probs_.assign(logits_size, 0.0f);
	cache_.probs.copy_to_host(last_probs_.data(), last_probs_.size());
	timer.stop();
	timing_.compute_ms += static_cast<double>(timer.elapsed_ms());

	float total_loss = 0.0f;
	for (size_t b = 0; b < last_batch_; ++b) {
		int label = last_labels_[b];
		float prob = last_probs_[b * static_cast<size_t>(out_features_) +
			static_cast<size_t>(label)];
		total_loss += -std::log(prob);
	}

	return total_loss / static_cast<float>(last_batch_);
}

void Network::backward() {
	if (last_batch_ == 0) {
		std::fprintf(stderr, "Network backward called before forward\n");
		return;
	}

	GpuTimer timer;
	timer.start();

	ensure_cache(last_batch_);

	LabelBuffer labels_buf(last_labels_.size());
	labels_buf.copy_from_host(last_labels_.data(), last_labels_.size());

	grad_logits_gpu(cache_.probs.data(), labels_buf.data(), cache_.grad_logits.data(),
				static_cast<int>(last_batch_), out_features_);

	transpose_matrix_gpu(cache_.grad_logits.data(), cache_.grad_logits_t.data(),
					static_cast<int>(last_batch_), out_features_);

	tiled_matmul_gpu(cache_.grad_logits_t.data(), cache_.hidden.data(), dout_weights_.data(),
					out_features_, hidden_features_, static_cast<int>(last_batch_));
	sum_over_batch_gpu(cache_.grad_logits.data(), dout_bias_.data(),
					static_cast<int>(last_batch_), out_features_);

	FloatBuffer hidden_grad_buf(cache_.hidden.size());
	tiled_matmul_gpu(cache_.grad_logits.data(), out_weights_.data(), hidden_grad_buf.data(),
					static_cast<int>(last_batch_), hidden_features_, out_features_);
	FloatBuffer bn_grad_buf(cache_.hidden.size());
	batch_norm_.backward(hidden_grad_buf, bn_grad_buf);
	FloatBuffer grad_in_buf(last_batch_ * static_cast<size_t>(in_features_));
	layers_[0].backward(bn_grad_buf, grad_in_buf);
	timer.stop();
	timing_.compute_ms += static_cast<double>(timer.elapsed_ms());
}

void Network::sgd_step(float learning_rate, GradientSync* sync) {
	auto params = layers_[0].parameters();
	auto grads = layers_[0].gradients();
	auto bn_params = batch_norm_.parameters();
	auto bn_grads = batch_norm_.gradients();
	if (sync != nullptr) {
		std::vector<float> host_dweights(grads[0]->size());
		std::vector<float> host_dbias(grads[1]->size());
		std::vector<float> host_dout_weights(dout_weights_.size());
		std::vector<float> host_dout_bias(dout_bias_.size());
		std::vector<float> host_dgamma(bn_grads[0]->size());
		std::vector<float> host_dbeta(bn_grads[1]->size());

		grads[0]->copy_to_host(host_dweights.data(), host_dweights.size());
		grads[1]->copy_to_host(host_dbias.data(), host_dbias.size());
		dout_weights_.copy_to_host(host_dout_weights.data(), host_dout_weights.size());
		dout_bias_.copy_to_host(host_dout_bias.data(), host_dout_bias.size());
		bn_grads[0]->copy_to_host(host_dgamma.data(), host_dgamma.size());
		bn_grads[1]->copy_to_host(host_dbeta.data(), host_dbeta.size());

		// Synchronize all gradient buffers across ranks before applying SGD.
		auto sync_start = std::chrono::high_resolution_clock::now();
		sync->allreduce_mean(host_dweights.data(), static_cast<int>(host_dweights.size()));
		auto sync_end = std::chrono::high_resolution_clock::now();
		timing_.sync_ms += std::chrono::duration<double, std::milli>(sync_end - sync_start).count();

		sync_start = std::chrono::high_resolution_clock::now();
		sync->allreduce_mean(host_dbias.data(), static_cast<int>(host_dbias.size()));
		sync_end = std::chrono::high_resolution_clock::now();
		timing_.sync_ms += std::chrono::duration<double, std::milli>(sync_end - sync_start).count();

		sync_start = std::chrono::high_resolution_clock::now();
		sync->allreduce_mean(host_dout_weights.data(), static_cast<int>(host_dout_weights.size()));
		sync_end = std::chrono::high_resolution_clock::now();
		timing_.sync_ms += std::chrono::duration<double, std::milli>(sync_end - sync_start).count();

		sync_start = std::chrono::high_resolution_clock::now();
		sync->allreduce_mean(host_dout_bias.data(), static_cast<int>(host_dout_bias.size()));
		sync_end = std::chrono::high_resolution_clock::now();
		timing_.sync_ms += std::chrono::duration<double, std::milli>(sync_end - sync_start).count();

		sync_start = std::chrono::high_resolution_clock::now();
		sync->allreduce_mean(host_dgamma.data(), static_cast<int>(host_dgamma.size()));
		sync_end = std::chrono::high_resolution_clock::now();
		timing_.sync_ms += std::chrono::duration<double, std::milli>(sync_end - sync_start).count();

		sync_start = std::chrono::high_resolution_clock::now();
		sync->allreduce_mean(host_dbeta.data(), static_cast<int>(host_dbeta.size()));
		sync_end = std::chrono::high_resolution_clock::now();
		timing_.sync_ms += std::chrono::duration<double, std::milli>(sync_end - sync_start).count();

		grads[0]->copy_from_host(host_dweights.data(), host_dweights.size());
		grads[1]->copy_from_host(host_dbias.data(), host_dbias.size());
		dout_weights_.copy_from_host(host_dout_weights.data(), host_dout_weights.size());
		dout_bias_.copy_from_host(host_dout_bias.data(), host_dout_bias.size());
		bn_grads[0]->copy_from_host(host_dgamma.data(), host_dgamma.size());
		bn_grads[1]->copy_from_host(host_dbeta.data(), host_dbeta.size());
	}

	GpuTimer timer;
	timer.start();
	sgd_update_gpu(params[0]->data(), grads[0]->data(), learning_rate,
					static_cast<int>(grads[0]->size()));
	sgd_update_gpu(params[1]->data(), grads[1]->data(), learning_rate,
					static_cast<int>(grads[1]->size()));
	sgd_update_gpu(out_weights_.data(), dout_weights_.data(), learning_rate,
					static_cast<int>(dout_weights_.size()));
	sgd_update_gpu(out_bias_.data(), dout_bias_.data(), learning_rate,
					static_cast<int>(dout_bias_.size()));
	sgd_update_gpu(bn_params[0]->data(), bn_grads[0]->data(), learning_rate,
					static_cast<int>(bn_grads[0]->size()));
	sgd_update_gpu(bn_params[1]->data(), bn_grads[1]->data(), learning_rate,
					static_cast<int>(bn_grads[1]->size()));
	timer.stop();
	timing_.compute_ms += static_cast<double>(timer.elapsed_ms());
}

void Network::reset_timers() {
	timing_.compute_ms = 0.0;
	timing_.sync_ms = 0.0;
}

Network::TimingStats Network::timing_ms() const {
	return timing_;
}

float Network::get_accuracy(const FloatBuffer& input, const LabelBuffer& labels) {
	if (input.size() % static_cast<size_t>(in_features_) != 0) {
		std::fprintf(stderr, "Network accuracy input shape mismatch\n");
		return 0.0f;
	}
	const size_t batch = input.size() / static_cast<size_t>(in_features_);
	std::vector<float> host_input(input.size());
	std::vector<int> host_labels(labels.size());
	input.copy_to_host(host_input.data(), host_input.size());
	labels.copy_to_host(host_labels.data(), host_labels.size());

	FloatBuffer input_buf(host_input.size());
	FloatBuffer hidden_buf(batch * static_cast<size_t>(hidden_features_));
	input_buf.copy_from_host(host_input.data(), host_input.size());
	layers_[0].forward(input_buf, hidden_buf);
	FloatBuffer norm_buf(hidden_buf.size());
	batch_norm_.forward(hidden_buf, norm_buf);

	ensure_cache(batch);
	last_probs_.assign(batch * static_cast<size_t>(out_features_), 0.0f);
	CUDA_CHECK(cudaMemcpy(cache_.hidden.data(), norm_buf.data(),
					sizeof(float) * cache_.hidden.size(), cudaMemcpyDeviceToDevice));
	transpose_matrix_gpu(out_weights_.data(), cache_.out_weights_t.data(),
					out_features_, hidden_features_);
	tiled_matmul_gpu(cache_.hidden.data(), cache_.out_weights_t.data(), cache_.logits.data(),
					static_cast<int>(batch), out_features_, hidden_features_);
	bias_add_gpu(cache_.logits.data(), out_bias_.data(), static_cast<int>(batch), out_features_);
	softmax_forward_gpu(cache_.logits.data(), cache_.probs.data(),
					static_cast<int>(batch), out_features_);
	cache_.probs.copy_to_host(last_probs_.data(), last_probs_.size());

	int correct = 0;
	for (size_t b = 0; b < batch; ++b) {
		float max_val = -1.0e20f;
		int max_idx = 0;
		for (int o = 0; o < out_features_; ++o) {
			float value = last_probs_[b * static_cast<size_t>(out_features_) +
				static_cast<size_t>(o)];
			if (value > max_val) {
				max_val = value;
				max_idx = o;
			}
		}
		if (max_idx == host_labels[b]) {
			++correct;
		}
	}

	return static_cast<float>(correct) / static_cast<float>(batch);
}

void Network::save_weights(const std::string& path) const {
	std::ofstream out(path, std::ios::binary);
	if (!out) {
		std::fprintf(stderr, "Failed to open %s for writing\n", path.c_str());
		return;
	}

	int header[3] = {in_features_, hidden_features_, out_features_};
	out.write(reinterpret_cast<const char*>(header), sizeof(header));

	auto params = layers_[0].parameters();
	std::vector<float> host_w0(params[0]->size());
	std::vector<float> host_b0(params[1]->size());
	params[0]->copy_to_host(host_w0.data(), host_w0.size());
	params[1]->copy_to_host(host_b0.data(), host_b0.size());
	auto bn_params = batch_norm_.parameters();
	std::vector<float> host_gamma(bn_params[0]->size());
	std::vector<float> host_beta(bn_params[1]->size());
	bn_params[0]->copy_to_host(host_gamma.data(), host_gamma.size());
	bn_params[1]->copy_to_host(host_beta.data(), host_beta.size());

	std::vector<float> host_w1(out_weights_.size());
	std::vector<float> host_b1(out_bias_.size());
	out_weights_.copy_to_host(host_w1.data(), host_w1.size());
	out_bias_.copy_to_host(host_b1.data(), host_b1.size());

	int sizes[4] = {
		static_cast<int>(host_w0.size()),
		static_cast<int>(host_b0.size()),
		static_cast<int>(host_w1.size()),
		static_cast<int>(host_b1.size())
	};
	out.write(reinterpret_cast<const char*>(sizes), sizeof(sizes));
	out.write(reinterpret_cast<const char*>(host_w0.data()), host_w0.size() * sizeof(float));
	out.write(reinterpret_cast<const char*>(host_b0.data()), host_b0.size() * sizeof(float));
	out.write(reinterpret_cast<const char*>(host_w1.data()), host_w1.size() * sizeof(float));
	out.write(reinterpret_cast<const char*>(host_b1.data()), host_b1.size() * sizeof(float));
	out.write(reinterpret_cast<const char*>(host_gamma.data()), host_gamma.size() * sizeof(float));
	out.write(reinterpret_cast<const char*>(host_beta.data()), host_beta.size() * sizeof(float));
}

void Network::load_weights(const std::string& path) {
	std::ifstream in(path, std::ios::binary);
	if (!in) {
		std::fprintf(stderr, "Failed to open %s for reading\n", path.c_str());
		return;
	}

	int header[3] = {0, 0, 0};
	in.read(reinterpret_cast<char*>(header), sizeof(header));
	if (header[0] != in_features_ || header[1] != hidden_features_ || header[2] != out_features_) {
		std::fprintf(stderr, "Weight file shape mismatch\n");
		return;
	}

	int sizes[4] = {0, 0, 0, 0};
	in.read(reinterpret_cast<char*>(sizes), sizeof(sizes));
	std::vector<float> host_w0(static_cast<size_t>(sizes[0]));
	std::vector<float> host_b0(static_cast<size_t>(sizes[1]));
	std::vector<float> host_w1(static_cast<size_t>(sizes[2]));
	std::vector<float> host_b1(static_cast<size_t>(sizes[3]));

	in.read(reinterpret_cast<char*>(host_w0.data()), host_w0.size() * sizeof(float));
	in.read(reinterpret_cast<char*>(host_b0.data()), host_b0.size() * sizeof(float));
	in.read(reinterpret_cast<char*>(host_w1.data()), host_w1.size() * sizeof(float));
	in.read(reinterpret_cast<char*>(host_b1.data()), host_b1.size() * sizeof(float));

	auto bn_params = batch_norm_.parameters();
	std::vector<float> host_gamma(bn_params[0]->size(), 1.0f);
	std::vector<float> host_beta(bn_params[1]->size(), 0.0f);
	std::streampos payload_pos = in.tellg();
	in.seekg(0, std::ios::end);
	std::streampos payload_end = in.tellg();
	in.seekg(payload_pos);
	const std::streamoff remaining = payload_end - payload_pos;
	const std::streamoff bn_bytes = static_cast<std::streamoff>(
		host_gamma.size() + host_beta.size()) * static_cast<std::streamoff>(sizeof(float));
	if (remaining >= bn_bytes) {
		in.read(reinterpret_cast<char*>(host_gamma.data()), host_gamma.size() * sizeof(float));
		in.read(reinterpret_cast<char*>(host_beta.data()), host_beta.size() * sizeof(float));
	}

	auto params = layers_[0].parameters();
	params[0]->copy_from_host(host_w0.data(), host_w0.size());
	params[1]->copy_from_host(host_b0.data(), host_b0.size());
	out_weights_.copy_from_host(host_w1.data(), host_w1.size());
	out_bias_.copy_from_host(host_b1.data(), host_b1.size());
	bn_params[0]->copy_from_host(host_gamma.data(), host_gamma.size());
	bn_params[1]->copy_from_host(host_beta.data(), host_beta.size());
}
