#ifndef BATCH_NORM_LAYER_HPP
#define BATCH_NORM_LAYER_HPP

#include <vector>

#include "layer.hpp"

class BatchNorm final : public Layer {
public:
	BatchNorm();
	explicit BatchNorm(int features, float epsilon = 1e-5f);

	BatchNorm(const BatchNorm&) = delete;
	BatchNorm& operator=(const BatchNorm&) = delete;
	BatchNorm(BatchNorm&&) noexcept = default;
	BatchNorm& operator=(BatchNorm&&) noexcept = default;

	void forward(const FloatBuffer& in, FloatBuffer& out) override;
	void backward(const FloatBuffer& grad_out, FloatBuffer& grad_in) override;
	std::vector<FloatBuffer*> parameters() override;
	std::vector<FloatBuffer*> gradients() override;
	std::vector<const FloatBuffer*> parameters() const override;
	std::vector<const FloatBuffer*> gradients() const override;

	int features() const;

private:
	void ensure_cache(size_t batch);

	int features_;
	float epsilon_;
	FloatBuffer gamma_;
	FloatBuffer beta_;
	FloatBuffer dgamma_;
	FloatBuffer dbeta_;
	FloatBuffer mean_;
	FloatBuffer inv_std_;
	FloatBuffer x_hat_;
	size_t last_batch_;
};

#endif // BATCH_NORM_LAYER_HPP
