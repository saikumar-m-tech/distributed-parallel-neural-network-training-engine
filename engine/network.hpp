#ifndef NETWORK_HPP
#define NETWORK_HPP

#include <string>
#include <vector>

#include "batch_norm_layer.hpp"
#include "dense_layer.hpp"

using LabelBuffer = GpuBuffer<int>;

class GradientSync;

class Network {
public:
	struct TimingStats {
		double compute_ms = 0.0;
		double sync_ms = 0.0;
	};

	explicit Network(std::vector<Dense> layers);

	float forward(const FloatBuffer& input, const LabelBuffer& labels);
	void backward();
	void sgd_step(float learning_rate, GradientSync* sync);
	void predict(const FloatBuffer& input, std::vector<int>& out_labels,
				std::vector<float>& out_probs);
	float get_accuracy(const FloatBuffer& input, const LabelBuffer& labels);
	void save_weights(const std::string& path) const;
	void load_weights(const std::string& path);
	void reset_timers();
	TimingStats timing_ms() const;

private:
	struct NetworkCache {
		FloatBuffer hidden;
		FloatBuffer logits;
		FloatBuffer probs;
		FloatBuffer grad_logits;
		FloatBuffer grad_logits_t;
		FloatBuffer out_weights_t;
		size_t cached_batch = 0;
	};

	void ensure_cache(size_t batch);

	std::vector<Dense> layers_;
	std::vector<float> last_probs_;
	std::vector<int> last_labels_;
	BatchNorm batch_norm_;
	FloatBuffer out_weights_;
	FloatBuffer out_bias_;
	FloatBuffer dout_weights_;
	FloatBuffer dout_bias_;
	NetworkCache cache_;
	size_t last_batch_;
	int in_features_;
	int hidden_features_;
	int out_features_;
	TimingStats timing_;
};

#endif // NETWORK_HPP
