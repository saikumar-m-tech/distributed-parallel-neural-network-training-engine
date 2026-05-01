#ifndef NETWORK_HPP
#define NETWORK_HPP

#include <string>
#include <vector>

#include "dense_layer.hpp"

using LabelBuffer = GpuBuffer<int>;

class GradientSync;

class Network {
public:
	explicit Network(std::vector<Dense> layers);

	float forward(const FloatBuffer& input, const LabelBuffer& labels);
	void backward();
	void sgd_step(float learning_rate, GradientSync* sync);
	float get_accuracy(const FloatBuffer& input, const LabelBuffer& labels);
	void save_weights(const std::string& path) const;
	void load_weights(const std::string& path);

private:
	std::vector<Dense> layers_;
	std::vector<float> last_input_;
	std::vector<float> last_hidden_;
	std::vector<float> last_logits_;
	std::vector<float> last_probs_;
	std::vector<int> last_labels_;
	std::vector<float> grad_logits_;
	std::vector<float> grad_hidden_;
	std::vector<float> out_weights_;
	std::vector<float> out_bias_;
	std::vector<float> dout_weights_;
	std::vector<float> dout_bias_;
	size_t last_batch_;
	int in_features_;
	int hidden_features_;
	int out_features_;
};

#endif // NETWORK_HPP
