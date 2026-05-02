#include "network.hpp"

#include "../mpi/gradient_sync.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <memory>
#include <random>

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

namespace {
struct NetworkCache {
	FloatBuffer hidden;
	FloatBuffer logits;
	FloatBuffer probs;
	FloatBuffer grad_logits;
	FloatBuffer grad_logits_t;
	FloatBuffer out_weights_t;
	NetworkCache(size_t hidden_size, size_t logits_size, size_t out_weights_t_size)
		: hidden(hidden_size),
		  logits(logits_size),
		  probs(logits_size),
		  grad_logits(logits_size),
		  grad_logits_t(static_cast<size_t>(out_weights_t_size)),
		  out_weights_t(out_weights_t_size) {}
};

NetworkCache& get_cache(size_t batch, int hidden_features, int out_features) {
	static std::unique_ptr<NetworkCache> cache;
	static size_t cached_batch = 0;
	if (!cache || cached_batch != batch) {
		size_t hidden_size = batch * static_cast<size_t>(hidden_features);
		size_t logits_size = batch * static_cast<size_t>(out_features);
		size_t out_weights_t_size = static_cast<size_t>(hidden_features) * out_features;
		cache = std::make_unique<NetworkCache>(hidden_size, logits_size, out_weights_t_size);
		cached_batch = batch;
	}
	return *cache;
}

} // namespace

Network::Network(std::vector<Dense> layers)
	: layers_(std::move(layers)),
	  last_input_(),
	  last_hidden_(),
	  last_logits_(),
	  last_probs_(),
	  last_labels_(),
	  grad_logits_(),
	  grad_hidden_(),
	  out_weights_(),
	  out_bias_(),
	  dout_weights_(),
	  dout_bias_(),
	  last_batch_(0),
	  in_features_(0),
	  hidden_features_(0),
	  out_features_(0) {
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

	std::mt19937 rng(1234);
	float stddev = std::sqrt(2.0f / static_cast<float>(hidden_features_));
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
}

float Network::forward(const FloatBuffer& input, const LabelBuffer& labels) {
	if (input.size() % static_cast<size_t>(in_features_) != 0) {
		std::fprintf(stderr, "Network forward input shape mismatch\n");
		return 0.0f;
	}

	last_batch_ = input.size() / static_cast<size_t>(in_features_);
	last_labels_.assign(labels.size(), 0);
	labels.copy_to_host(last_labels_.data(), last_labels_.size());

	last_hidden_.assign(last_batch_ * static_cast<size_t>(hidden_features_), 0.0f);
	last_logits_.assign(last_batch_ * static_cast<size_t>(out_features_), 0.0f);
	last_probs_.assign(last_logits_.size(), 0.0f);
	grad_logits_.assign(last_logits_.size(), 0.0f);
	grad_hidden_.assign(last_hidden_.size(), 0.0f);

	FloatBuffer hidden_buf(last_hidden_.size());
	layers_[0].forward(input, hidden_buf);
	hidden_buf.copy_to_host(last_hidden_.data(), last_hidden_.size());

	NetworkCache& cache = get_cache(last_batch_, hidden_features_, out_features_);
	CUDA_CHECK(cudaMemcpy(cache.hidden.data(), hidden_buf.data(),
					sizeof(float) * cache.hidden.size(), cudaMemcpyDeviceToDevice));

	transpose_matrix_gpu(out_weights_.data(), cache.out_weights_t.data(),
					out_features_, hidden_features_);
	tiled_matmul_gpu(cache.hidden.data(), cache.out_weights_t.data(), cache.logits.data(),
					static_cast<int>(last_batch_), out_features_, hidden_features_);
	bias_add_gpu(cache.logits.data(), out_bias_.data(), static_cast<int>(last_batch_), out_features_);

	softmax_forward_gpu(cache.logits.data(), cache.probs.data(),
					static_cast<int>(last_batch_), out_features_);

	cache.logits.copy_to_host(last_logits_.data(), last_logits_.size());
	cache.probs.copy_to_host(last_probs_.data(), last_probs_.size());

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

	NetworkCache& cache = get_cache(last_batch_, hidden_features_, out_features_);

	LabelBuffer labels_buf(last_labels_.size());
	labels_buf.copy_from_host(last_labels_.data(), last_labels_.size());

	grad_logits_gpu(cache.probs.data(), labels_buf.data(), cache.grad_logits.data(),
				static_cast<int>(last_batch_), out_features_);

	transpose_matrix_gpu(cache.grad_logits.data(), cache.grad_logits_t.data(),
					static_cast<int>(last_batch_), out_features_);

	tiled_matmul_gpu(cache.grad_logits_t.data(), cache.hidden.data(), dout_weights_.data(),
					out_features_, hidden_features_, static_cast<int>(last_batch_));
	sum_over_batch_gpu(cache.grad_logits.data(), dout_bias_.data(),
					static_cast<int>(last_batch_), out_features_);

	FloatBuffer hidden_grad_buf(cache.hidden.size());
	tiled_matmul_gpu(cache.grad_logits.data(), out_weights_.data(), hidden_grad_buf.data(),
					static_cast<int>(last_batch_), hidden_features_, out_features_);
	FloatBuffer grad_in_buf(last_batch_ * static_cast<size_t>(in_features_));
	layers_[0].backward(hidden_grad_buf, grad_in_buf);
}

void Network::sgd_step(float learning_rate, GradientSync* sync) {
	auto params = layers_[0].parameters();
	auto grads = layers_[0].gradients();
	std::vector<float> host_weights(params[0]->size());
	std::vector<float> host_bias(params[1]->size());
	std::vector<float> host_dweights(grads[0]->size());
	std::vector<float> host_dbias(grads[1]->size());
	std::vector<float> host_out_weights(out_weights_.size());
	std::vector<float> host_out_bias(out_bias_.size());
	std::vector<float> host_dout_weights(dout_weights_.size());
	std::vector<float> host_dout_bias(dout_bias_.size());

	params[0]->copy_to_host(host_weights.data(), host_weights.size());
	params[1]->copy_to_host(host_bias.data(), host_bias.size());
	grads[0]->copy_to_host(host_dweights.data(), host_dweights.size());
	grads[1]->copy_to_host(host_dbias.data(), host_dbias.size());
	out_weights_.copy_to_host(host_out_weights.data(), host_out_weights.size());
	out_bias_.copy_to_host(host_out_bias.data(), host_out_bias.size());
	dout_weights_.copy_to_host(host_dout_weights.data(), host_dout_weights.size());
	dout_bias_.copy_to_host(host_dout_bias.data(), host_dout_bias.size());

	// Synchronize all gradient buffers across ranks before applying SGD.
	if (sync != nullptr) {
		sync->allreduce_mean(host_dweights.data(), static_cast<int>(host_dweights.size()));
		sync->allreduce_mean(host_dbias.data(), static_cast<int>(host_dbias.size()));
		sync->allreduce_mean(host_dout_weights.data(), static_cast<int>(host_dout_weights.size()));
		sync->allreduce_mean(host_dout_bias.data(), static_cast<int>(host_dout_bias.size()));
	}

	for (size_t i = 0; i < host_out_weights.size(); ++i) {
		host_out_weights[i] -= learning_rate * host_dout_weights[i];
	}
	for (size_t i = 0; i < host_out_bias.size(); ++i) {
		host_out_bias[i] -= learning_rate * host_dout_bias[i];
	}

	for (size_t i = 0; i < host_weights.size(); ++i) {
		host_weights[i] -= learning_rate * host_dweights[i];
	}
	for (size_t i = 0; i < host_bias.size(); ++i) {
		host_bias[i] -= learning_rate * host_dbias[i];
	}

	params[0]->copy_from_host(host_weights.data(), host_weights.size());
	params[1]->copy_from_host(host_bias.data(), host_bias.size());
	out_weights_.copy_from_host(host_out_weights.data(), host_out_weights.size());
	out_bias_.copy_from_host(host_out_bias.data(), host_out_bias.size());
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

	NetworkCache& cache = get_cache(batch, hidden_features_, out_features_);
	last_probs_.assign(batch * static_cast<size_t>(out_features_), 0.0f);
	CUDA_CHECK(cudaMemcpy(cache.hidden.data(), hidden_buf.data(),
					sizeof(float) * cache.hidden.size(), cudaMemcpyDeviceToDevice));
	transpose_matrix_gpu(out_weights_.data(), cache.out_weights_t.data(),
					out_features_, hidden_features_);
	tiled_matmul_gpu(cache.hidden.data(), cache.out_weights_t.data(), cache.logits.data(),
					static_cast<int>(batch), out_features_, hidden_features_);
	bias_add_gpu(cache.logits.data(), out_bias_.data(), static_cast<int>(batch), out_features_);
	softmax_forward_gpu(cache.logits.data(), cache.probs.data(),
					static_cast<int>(batch), out_features_);
	cache.probs.copy_to_host(last_probs_.data(), last_probs_.size());

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

	auto params = layers_[0].parameters();
	params[0]->copy_from_host(host_w0.data(), host_w0.size());
	params[1]->copy_from_host(host_b0.data(), host_b0.size());
	out_weights_.copy_from_host(host_w1.data(), host_w1.size());
	out_bias_.copy_from_host(host_b1.data(), host_b1.size());
}
