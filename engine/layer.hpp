#ifndef LAYER_HPP
#define LAYER_HPP

#include <vector>

#include "../kernels/kernel_utils.cuh"

using FloatBuffer = GpuBuffer<float>;

class Layer {
public:
	virtual void forward(const FloatBuffer& in, FloatBuffer& out) = 0;
	virtual void backward(const FloatBuffer& grad_out, FloatBuffer& grad_in) = 0;
	virtual std::vector<FloatBuffer*> parameters() = 0;
	virtual std::vector<FloatBuffer*> gradients() = 0;
	virtual std::vector<const FloatBuffer*> parameters() const = 0;
	virtual std::vector<const FloatBuffer*> gradients() const = 0;
	virtual ~Layer() = default;
};

#endif // LAYER_HPP
