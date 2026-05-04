#ifndef DENSE_LAYER_HPP
#define DENSE_LAYER_HPP

#include <vector>

#include "layer.hpp"

class Dense final : public Layer {
public:
	Dense(int in_features, int out_features);

	void forward(const FloatBuffer& in, FloatBuffer& out) override;
	void backward(const FloatBuffer& grad_out, FloatBuffer& grad_in) override;
	std::vector<FloatBuffer*> parameters() override;
	std::vector<FloatBuffer*> gradients() override;
	std::vector<const FloatBuffer*> parameters() const override;
	std::vector<const FloatBuffer*> gradients() const override;

	int in_features() const;
	int out_features() const;

private:
	int in_features_;
	int out_features_;
	FloatBuffer weights_;
	FloatBuffer bias_;
	FloatBuffer dweights_;
	FloatBuffer dbias_;
	size_t last_batch_;
};

#endif // DENSE_LAYER_HPP
