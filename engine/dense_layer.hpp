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
	struct DenseCache {
		FloatBuffer input;
		FloatBuffer pre_relu;
		FloatBuffer grad_relu;
		FloatBuffer grad_relu_t;
		FloatBuffer weights_t;
		size_t input_size = 0;
		size_t pre_relu_size = 0;
		size_t grad_size = 0;
		size_t grad_t_size = 0;
		size_t weights_t_size = 0;
	};

	void ensure_cache(size_t batch);

	int in_features_;
	int out_features_;
	FloatBuffer weights_;
	FloatBuffer bias_;
	FloatBuffer dweights_;
	FloatBuffer dbias_;
	DenseCache cache_;
	size_t last_batch_;
};

#endif // DENSE_LAYER_HPP
