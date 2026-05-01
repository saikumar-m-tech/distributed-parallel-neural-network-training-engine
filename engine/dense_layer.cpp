#include "dense_layer.hpp"

#include <cmath>
#include <cstdio>
#include <random>

Dense::Dense(int in_features, int out_features)
	: in_features_(in_features),
	  out_features_(out_features),
	  weights_(static_cast<size_t>(out_features) * in_features),
	  bias_(static_cast<size_t>(out_features)),
	  dweights_(static_cast<size_t>(out_features) * in_features),
	  dbias_(static_cast<size_t>(out_features)),
	  last_input_(),
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

	last_batch_ = in.size() / static_cast<size_t>(in_features_);
	last_input_.assign(in.size(), 0.0f);
	in.copy_to_host(last_input_.data(), last_input_.size());

	std::vector<float> host_weights(weights_.size());
	std::vector<float> host_bias(bias_.size());
	weights_.copy_to_host(host_weights.data(), host_weights.size());
	bias_.copy_to_host(host_bias.data(), host_bias.size());

	std::vector<float> host_output(out.size(), 0.0f);
	for (size_t b = 0; b < last_batch_; ++b) {
		for (int o = 0; o < out_features_; ++o) {
			float sum = host_bias[static_cast<size_t>(o)];
			size_t w_offset = static_cast<size_t>(o) * in_features_;
			size_t in_offset = b * static_cast<size_t>(in_features_);
			for (int i = 0; i < in_features_; ++i) {
				sum += last_input_[in_offset + static_cast<size_t>(i)] *
					host_weights[w_offset + static_cast<size_t>(i)];
			}
			if (sum < 0.0f) {
				sum = 0.0f;
			}
			host_output[b * static_cast<size_t>(out_features_) + static_cast<size_t>(o)] = sum;
		}
	}

	out.copy_from_host(host_output.data(), host_output.size());
}

void Dense::backward(const FloatBuffer& grad_out, FloatBuffer& grad_in) {
	if (last_batch_ == 0 || last_input_.empty()) {
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

	std::vector<float> host_weights(weights_.size());
	weights_.copy_to_host(host_weights.data(), host_weights.size());

	std::vector<float> host_bias(bias_.size());
	bias_.copy_to_host(host_bias.data(), host_bias.size());

	std::vector<float> host_grad_out(grad_out.size());
	grad_out.copy_to_host(host_grad_out.data(), host_grad_out.size());

	std::vector<float> host_dweights(dweights_.size(), 0.0f);
	std::vector<float> host_dbias(dbias_.size(), 0.0f);
	std::vector<float> host_grad_in(grad_in.size(), 0.0f);

	for (size_t b = 0; b < last_batch_; ++b) {
		for (int o = 0; o < out_features_; ++o) {
			float pre_act = host_bias[static_cast<size_t>(o)];
			size_t w_offset = static_cast<size_t>(o) * in_features_;
			size_t in_offset = b * static_cast<size_t>(in_features_);
			for (int i = 0; i < in_features_; ++i) {
				pre_act += last_input_[in_offset + static_cast<size_t>(i)] *
					host_weights[w_offset + static_cast<size_t>(i)];
			}

			float grad_val = host_grad_out[b * static_cast<size_t>(out_features_) +
				static_cast<size_t>(o)];
			if (pre_act <= 0.0f) {
				grad_val = 0.0f;
			}

			host_dbias[static_cast<size_t>(o)] += grad_val;
			for (int i = 0; i < in_features_; ++i) {
				host_dweights[w_offset + static_cast<size_t>(i)] +=
					grad_val * last_input_[in_offset + static_cast<size_t>(i)];
				host_grad_in[in_offset + static_cast<size_t>(i)] +=
					grad_val * host_weights[w_offset + static_cast<size_t>(i)];
			}
		}
	}

	dweights_.copy_from_host(host_dweights.data(), host_dweights.size());
	dbias_.copy_from_host(host_dbias.data(), host_dbias.size());
	grad_in.copy_from_host(host_grad_in.data(), host_grad_in.size());
}

std::vector<FloatBuffer*> Dense::parameters() {
	return {&weights_, &bias_};
}

std::vector<FloatBuffer*> Dense::gradients() {
	return {&dweights_, &dbias_};
}

int Dense::in_features() const {
	return in_features_;
}

int Dense::out_features() const {
	return out_features_;
}
