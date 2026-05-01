#include <string>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include "../engine/network.hpp"

namespace py = pybind11;

class Trainer {
public:
	Trainer(int input_dim, int hidden_dim, int output_dim,
			float learning_rate, int mpi_rank, int world_size)
		: learning_rate_(learning_rate),
		  mpi_rank_(mpi_rank),
		  world_size_(world_size),
		  net_(build_layers(input_dim, hidden_dim, output_dim)) {}

	float train_step(py::array_t<float> X, py::array_t<int> y) {
		auto x = X.request();
		auto ybuf = y.request();
		if (x.ndim != 2) {
			throw std::runtime_error("X must be 2D");
		}
		if (ybuf.ndim != 1) {
			throw std::runtime_error("y must be 1D");
		}
		if (x.shape[0] != ybuf.shape[0]) {
			throw std::runtime_error("X and y batch sizes must match");
		}

		FloatBuffer input_buf(static_cast<size_t>(x.size));
		LabelBuffer label_buf(static_cast<size_t>(ybuf.size));

		// Copy contiguous host buffers into device storage.
		input_buf.copy_from_host(static_cast<float*>(x.ptr), static_cast<size_t>(x.size));
		label_buf.copy_from_host(static_cast<int*>(ybuf.ptr), static_cast<size_t>(ybuf.size));

		float loss = net_.forward(input_buf, label_buf);
		net_.backward();
		net_.sgd_step(learning_rate_, nullptr);
		return loss;
	}

	float get_accuracy(py::array_t<float> X, py::array_t<int> y) {
		auto x = X.request();
		auto ybuf = y.request();
		if (x.ndim != 2 || ybuf.ndim != 1 || x.shape[0] != ybuf.shape[0]) {
			throw std::runtime_error("Input shapes mismatch");
		}

		FloatBuffer input_buf(static_cast<size_t>(x.size));
		LabelBuffer label_buf(static_cast<size_t>(ybuf.size));
		input_buf.copy_from_host(static_cast<float*>(x.ptr), static_cast<size_t>(x.size));
		label_buf.copy_from_host(static_cast<int*>(ybuf.ptr), static_cast<size_t>(ybuf.size));

		return net_.get_accuracy(input_buf, label_buf);
	}

	void save_weights(const std::string& path) {
		net_.save_weights(path);
	}

	void load_weights(const std::string& path) {
		net_.load_weights(path);
	}

private:
	static std::vector<Dense> build_layers(int input_dim, int hidden_dim, int output_dim) {
		std::vector<Dense> layers;
		layers.emplace_back(input_dim, hidden_dim);
		layers.emplace_back(hidden_dim, output_dim);
		return layers;
	}

	float learning_rate_;
	int mpi_rank_;
	int world_size_;
	Network net_;
};

PYBIND11_MODULE(parallelnet_cpp, m) {
	py::class_<Trainer>(m, "Trainer")
		.def(py::init<int, int, int, float, int, int>())
		.def("train_step", &Trainer::train_step)
		.def("get_accuracy", &Trainer::get_accuracy)
		.def("save_weights", &Trainer::save_weights)
		.def("load_weights", &Trainer::load_weights);
}
