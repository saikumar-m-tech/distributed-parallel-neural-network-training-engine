#include "network.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <random>

namespace {

float relu(float x) {
	return x > 0.0f ? x : 0.0f;
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

	std::mt19937 rng(1234);
	float stddev = std::sqrt(2.0f / static_cast<float>(hidden_features_));
	std::normal_distribution<float> dist(0.0f, stddev);

	out_weights_.resize(static_cast<size_t>(out_features_) * hidden_features_);
	for (float& value : out_weights_) {
		float sample = 0.0f;
		do {
			sample = dist(rng);
		} while (sample == 0.0f || std::fabs(sample) > 0.2f);
		value = sample;
	}
	out_bias_.assign(static_cast<size_t>(out_features_), 0.0f);
	dout_weights_.assign(out_weights_.size(), 0.0f);
	dout_bias_.assign(out_bias_.size(), 0.0f);
}

float Network::forward(const FloatBuffer& input, const LabelBuffer& labels) {
	if (input.size() % static_cast<size_t>(in_features_) != 0) {
		std::fprintf(stderr, "Network forward input shape mismatch\n");
		return 0.0f;
	}

	last_batch_ = input.size() / static_cast<size_t>(in_features_);
	last_input_.assign(input.size(), 0.0f);
	input.copy_to_host(last_input_.data(), last_input_.size());

	last_labels_.assign(labels.size(), 0);
	labels.copy_to_host(last_labels_.data(), last_labels_.size());

	last_hidden_.assign(last_batch_ * static_cast<size_t>(hidden_features_), 0.0f);
	last_logits_.assign(last_batch_ * static_cast<size_t>(out_features_), 0.0f);
	last_probs_.assign(last_logits_.size(), 0.0f);
	grad_logits_.assign(last_logits_.size(), 0.0f);
	grad_hidden_.assign(last_hidden_.size(), 0.0f);

	// Forward pass: Dense -> ReLU handled by layer[0].
	FloatBuffer input_buf(last_input_.size());
	FloatBuffer hidden_buf(last_hidden_.size());
	input_buf.copy_from_host(last_input_.data(), last_input_.size());
	layers_[0].forward(input_buf, hidden_buf);
	hidden_buf.copy_to_host(last_hidden_.data(), last_hidden_.size());

	// Output layer: linear logits.
	for (size_t b = 0; b < last_batch_; ++b) {
		for (int o = 0; o < out_features_; ++o) {
			float sum = out_bias_[static_cast<size_t>(o)];
			size_t w_offset = static_cast<size_t>(o) * hidden_features_;
			size_t h_offset = b * static_cast<size_t>(hidden_features_);
			for (int h = 0; h < hidden_features_; ++h) {
				sum += last_hidden_[h_offset + static_cast<size_t>(h)] *
					out_weights_[w_offset + static_cast<size_t>(h)];
			}
			last_logits_[b * static_cast<size_t>(out_features_) + static_cast<size_t>(o)] = sum;
		}
	}

	float total_loss = 0.0f;
	for (size_t b = 0; b < last_batch_; ++b) {
		float max_val = -1.0e20f;
		for (int o = 0; o < out_features_; ++o) {
			float value = last_logits_[b * static_cast<size_t>(out_features_) +
				static_cast<size_t>(o)];
			max_val = std::max(max_val, value);
		}

		float sum_exp = 0.0f;
		for (int o = 0; o < out_features_; ++o) {
			float exp_val = std::exp(last_logits_[b * static_cast<size_t>(out_features_) +
				static_cast<size_t>(o)] - max_val);
			last_probs_[b * static_cast<size_t>(out_features_) + static_cast<size_t>(o)] = exp_val;
			sum_exp += exp_val;
		}

		for (int o = 0; o < out_features_; ++o) {
			last_probs_[b * static_cast<size_t>(out_features_) + static_cast<size_t>(o)] /= sum_exp;
		}

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

	std::fill(dout_weights_.begin(), dout_weights_.end(), 0.0f);
	std::fill(dout_bias_.begin(), dout_bias_.end(), 0.0f);
	std::fill(grad_hidden_.begin(), grad_hidden_.end(), 0.0f);

	for (size_t b = 0; b < last_batch_; ++b) {
		for (int o = 0; o < out_features_; ++o) {
			float prob = last_probs_[b * static_cast<size_t>(out_features_) +
				static_cast<size_t>(o)];
			float grad = prob;
			if (o == last_labels_[b]) {
				grad -= 1.0f;
			}
			grad /= static_cast<float>(last_batch_);
			grad_logits_[b * static_cast<size_t>(out_features_) + static_cast<size_t>(o)] = grad;
			dout_bias_[static_cast<size_t>(o)] += grad;
			for (int h = 0; h < hidden_features_; ++h) {
				size_t w_offset = static_cast<size_t>(o) * hidden_features_;
				dout_weights_[w_offset + static_cast<size_t>(h)] +=
					grad * last_hidden_[b * static_cast<size_t>(hidden_features_) +
						static_cast<size_t>(h)];
				grad_hidden_[b * static_cast<size_t>(hidden_features_) +
					static_cast<size_t>(h)] +=
					grad * out_weights_[w_offset + static_cast<size_t>(h)];
			}
		}
	}

	FloatBuffer hidden_grad_buf(grad_hidden_.size());
	hidden_grad_buf.copy_from_host(grad_hidden_.data(), grad_hidden_.size());
	FloatBuffer grad_in_buf(last_input_.size());
	layers_[0].backward(hidden_grad_buf, grad_in_buf);
}

void Network::sgd_step(float learning_rate) {
	for (size_t i = 0; i < out_weights_.size(); ++i) {
		out_weights_[i] -= learning_rate * dout_weights_[i];
	}
	for (size_t i = 0; i < out_bias_.size(); ++i) {
		out_bias_[i] -= learning_rate * dout_bias_[i];
	}

	auto params = layers_[0].parameters();
	auto grads = layers_[0].gradients();
	std::vector<float> host_weights(params[0]->size());
	std::vector<float> host_bias(params[1]->size());
	std::vector<float> host_dweights(grads[0]->size());
	std::vector<float> host_dbias(grads[1]->size());

	params[0]->copy_to_host(host_weights.data(), host_weights.size());
	params[1]->copy_to_host(host_bias.data(), host_bias.size());
	grads[0]->copy_to_host(host_dweights.data(), host_dweights.size());
	grads[1]->copy_to_host(host_dbias.data(), host_dbias.size());

	for (size_t i = 0; i < host_weights.size(); ++i) {
		host_weights[i] -= learning_rate * host_dweights[i];
	}
	for (size_t i = 0; i < host_bias.size(); ++i) {
		host_bias[i] -= learning_rate * host_dbias[i];
	}

	params[0]->copy_from_host(host_weights.data(), host_weights.size());
	params[1]->copy_from_host(host_bias.data(), host_bias.size());
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

	std::vector<float> hidden(batch * static_cast<size_t>(hidden_features_));
	hidden_buf.copy_to_host(hidden.data(), hidden.size());

	int correct = 0;
	for (size_t b = 0; b < batch; ++b) {
		float max_val = -1.0e20f;
		int max_idx = 0;
		for (int o = 0; o < out_features_; ++o) {
			float sum = out_bias_[static_cast<size_t>(o)];
			size_t w_offset = static_cast<size_t>(o) * hidden_features_;
			size_t h_offset = b * static_cast<size_t>(hidden_features_);
			for (int h = 0; h < hidden_features_; ++h) {
				sum += hidden[h_offset + static_cast<size_t>(h)] *
					out_weights_[w_offset + static_cast<size_t>(h)];
			}
			if (sum > max_val) {
				max_val = sum;
				max_idx = o;
			}
		}
		if (max_idx == host_labels[b]) {
			++correct;
		}
	}

	return static_cast<float>(correct) / static_cast<float>(batch);
}
